"""Cadrage intelligent 9:16 : facecam du streamer + zone de jeu detectee par analyse du mouvement.

Principe :
1. on detecte les visages stables (= webcams du streamer) ;
2. on calcule une carte de mouvement sur tout le clip (le jeu bouge en permanence,
   les decors / overlays du stream sont fixes), en masquant les webcams ;
3. la plus grande zone qui bouge = la scene de jeu (marche aussi pour les jeux mobiles
   affiches sur le cote, les jeux en plein ecran, les layouts a plusieurs cameras...) ;
4. on cadre cette zone au mieux pour le panneau du bas, avec un fond flou si besoin.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config
from .utils import log


@dataclass
class Layout:
    src_w: int
    src_h: int
    cam: tuple[int, int, int, int] | None      # zone webcam dans la source (x, y, w, h)
    game: tuple[int, int, int, int]            # zone de jeu cadree dans la source (x, y, w, h)
    cam_h: int                                  # hauteur du panneau webcam en sortie (0 si aucun)
    panel_y: int                                # panneau jeu : position y en sortie
    panel_h: int                                # panneau jeu : hauteur en sortie
    fg_w: int                                   # taille du jeu affiche dans le panneau (le reste = fond flou)
    fg_h: int
    fg_y: int                                   # y absolu du jeu en sortie

    @property
    def has_cam(self) -> bool:
        return self.cam is not None


# ------------------------------------------------------------------ lecture

def _read_pairs(video: Path, n: int = 28, gap_s: float = 0.25, scale_w: int = 384):
    """n paires d'images (t, t+gap) reparties sur le clip, en petit format gris, + images couleur pleine taille."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    gap = max(1, int(fps * gap_s))
    pairs, full = [], []
    for k in range(n):
        f0 = int((total - gap - 1) * (k + 0.5) / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        ok1, a = cap.read()
        for _ in range(gap - 1):
            cap.grab()
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        if k % 2 == 0:
            full.append(a)
        h, w = a.shape[:2]
        sh = int(h * scale_w / w)
        ga = cv2.cvtColor(cv2.resize(a, (scale_w, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (scale_w, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        pairs.append((ga, gb))
    cap.release()
    if not full:
        raise RuntimeError("Impossible de lire la video.")
    return pairs, full


# ------------------------------------------------------------------ visages

YUNET = config.ASSETS / "models" / "face_detection_yunet_2023mar.onnx"
_yunet = None


def _detect(fr: np.ndarray, h: int) -> list[tuple[int, int, int, int, float]]:
    """Visages d'une image : YuNet (fiable, visages reels) sinon Haar. (x, y, w, h, score)."""
    global _yunet
    if YUNET.exists():
        s = 960 / fr.shape[1]
        small = cv2.resize(fr, (960, int(fr.shape[0] * s)))
        if _yunet is None:
            _yunet = cv2.FaceDetectorYN.create(str(YUNET), "", (small.shape[1], small.shape[0]), 0.7, 0.3, 50)
        _yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = _yunet.detect(small)
        out = []
        for r in (faces if faces is not None else []):
            x, y, w, hh = (int(v / s) for v in r[:4])
            if hh >= h * 0.05:
                out.append((x, y, w, hh, float(r[14])))
        return out
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.equalizeHist(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    return [(*map(int, f), 0.8) for f in cascade.detectMultiScale(gray, 1.1, 6, minSize=(int(h * 0.05),) * 2,
                                                                     maxSize=(int(h * 0.5),) * 2)]


def _skin_ratio(fr: np.ndarray, box) -> float:
    x, y, w, h = box
    roi = fr[max(0, y + h // 4):y + h * 3 // 4, max(0, x + w // 4):x + w * 3 // 4]
    if roi.size == 0:
        return 0.0
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 180, 130))
    return float(m.mean() / 255)


def _liveness(frames: list[np.ndarray], box) -> float:
    """Une vraie personne bouge (yeux, bouche, tete) ; un dessin du decor reste identique."""
    x, y, w, h = box
    crops = []
    for fr in frames:
        c = fr[max(0, y):y + h, max(0, x):x + w]
        if c.size:
            crops.append(cv2.cvtColor(cv2.resize(c, (48, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    if len(crops) < 3:
        return 0.0
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(crops, crops[1:])]))


def _faces(frames: list[np.ndarray]) -> list[tuple[int, int, int, int, int]]:
    """Visages de vraies personnes, stables dans le clip : (x, y, w, h, nb_detections), le plus probable d'abord.

    On rejette les visages dessines du decor (personnages du jeu sur l'overlay) : ils ne bougent pas
    et n'ont pas une couleur de peau."""
    h = frames[0].shape[0]
    dets: list[tuple[int, int, int, int, float]] = []
    for fr in frames:
        dets.extend(_detect(fr, h))
    clusters: list[list[tuple]] = []
    for d in dets:
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        for cl in clusters:
            r = np.median(np.array([x[:4] for x in cl]), axis=0)
            if abs(r[0] + r[2] / 2 - cx) < r[2] * 0.8 and abs(r[1] + r[3] / 2 - cy) < r[3] * 0.8:
                cl.append(d)
                break
        else:
            clusters.append([d])
    out = []
    for cl in clusters:
        if len(cl) < max(3, len(frames) * 0.3):
            continue
        med = tuple(int(v) for v in np.median(np.array([x[:4] for x in cl]), axis=0))
        live = _liveness(frames, med)
        skin = float(np.median([_skin_ratio(fr, med) for fr in frames[::2]]))
        score = float(np.mean([x[4] for x in cl]))
        ok = live >= 4.0 and skin >= 0.12
        log.info("Visage %s : vu %d/%d, mouvement %.1f, peau %.0f%%, score %.2f -> %s", med, len(cl), len(frames),
                 live, skin * 100, score, "webcam" if ok else "rejete (decor)")
        if ok:
            out.append((*med, len(cl), live * skin * len(cl)))
    return [f[:5] for f in sorted(out, key=lambda f: -f[5])]


def _cam_rect(frames: list[np.ndarray], face, sw: int, sh: int) -> tuple[int, int, int, int] | None:
    """Rectangle de la webcam autour du visage : la zone qui 'vit' (change d'une image a l'autre)."""
    s = 384 / sw
    small = [cv2.cvtColor(cv2.resize(f, (384, int(sh * s))), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    acc = np.zeros_like(small[0])
    for a, b in zip(small, small[1:]):
        acc += (np.abs(a - b) > 8).astype(np.float32)
    live = (acc >= max(2, len(small) * 0.25)).astype(np.uint8)
    live = cv2.morphologyEx(live, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fx, fy, fw, fh = face[:4]
    cx, cy = int((fx + fw / 2) * s), int((fy + fh / 2) * s)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(live)
    lab = labels[min(cy, labels.shape[0] - 1), min(cx, labels.shape[1] - 1)]
    if lab == 0:
        return None
    x, y, w, h, area = stats[lab]
    rx, ry, rw, rh = (int(v / s) for v in (x, y, w, h))
    # rectangle plausible : contient le visage, pas tout l'ecran, pas minuscule
    if rw * rh > sw * sh * 0.45 or rw < fw * 1.5 or rh < fh * 1.3:
        return None
    return rx, ry, rw, rh


def _cam_in_rect(rect, face, aspect: float) -> tuple[int, int, int, int]:
    """Plus grand cadre au bon format dans le rectangle webcam, centre sur le visage."""
    rx, ry, rw, rh = rect
    fx, fy, fw, fh = face[:4]
    cw, ch = rw, int(rw / aspect)
    if ch > rh:
        ch, cw = rh, int(rh * aspect)
    cx = int(min(max(fx + fw / 2 - cw / 2, rx), rx + rw - cw))
    cy = int(min(max(fy + fh * 0.45 - ch * 0.42, ry), ry + rh - ch))
    return cx, cy, cw - cw % 2, ch - ch % 2

# ------------------------------------------------------------------ mouvement

def _motion_map(pairs, mask_boxes, scale: float) -> np.ndarray:
    # pour chaque pixel : dans combien de paires il bouge (+ un peu l'amplitude).
    # Le jeu bouge presque tout le temps ; un decor fixe jamais ; un flash une seule fois.
    acc = np.zeros_like(pairs[0][0], dtype=np.float32)
    for a, b in pairs:
        d = cv2.absdiff(a, b).astype(np.float32)
        d = np.minimum(d, 60)
        d[d < 6] = 0                        # bruit de compression
        acc += (d > 0) + d / 60
    m = cv2.GaussianBlur(acc, (0, 0), 3)
    for (x, y, w, h) in mask_boxes:
        x0, y0 = int(x * scale), int(y * scale)
        m[max(0, y0):y0 + int(h * scale), max(0, x0):x0 + int(w * scale)] = 0
    return m


def _game_box(m: np.ndarray) -> tuple[int, int, int, int] | None:
    """Plus grande zone en mouvement (somme du mouvement), en coordonnees de la carte."""
    hi = float(np.percentile(m, 99))
    if hi <= 0:
        return None
    binm = (m > hi * 0.18).astype(np.uint8)
    k = max(5, m.shape[1] // 28)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < m.size * 0.015:
            continue
        score = float(m[labels == i].sum())
        if score > best_score:
            best, best_score = (int(x), int(y), int(w), int(h)), score
    return best


def _cam_box(face, sw: int, sh: int, aspect: float) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = face[:4]
    cw = min(sw, int(fw * 4.2))  # une webcam de stream fait ~4 a 6 largeurs de visage
    ch = int(cw / aspect)
    if ch > sh:
        ch = sh
        cw = int(ch * aspect)
    cx = max(0, min(sw - cw, int(fx + fw / 2 - cw / 2)))
    cy = max(0, min(sh - ch, int(fy + fh * 0.45 - ch * 0.42)))
    return cx, cy, cw, ch


def _frame_game(box, motion: np.ndarray, scale: float, sw: int, sh: int,
                a_min: float, a_max: float) -> tuple[int, int, int, int]:
    """Cadre la zone de jeu avec un rapport largeur/hauteur compris entre a_min et a_max."""
    bx, by, bw, bh = (int(v / scale) for v in box)
    # petite marge pour ne pas couper les bords du jeu
    mx, my = int(bw * 0.03), int(bh * 0.03)
    bx, by = max(0, bx - mx), max(0, by - my)
    bw, bh = min(sw - bx, bw + 2 * mx), min(sh - by, bh + 2 * my)
    ar = bw / bh
    if ar < a_min:                      # jeu etroit (mobile, vertical) : on l'affiche en entier
        ch = bh
        cw = int(ch * a_min)
        if cw > sw:
            cw, ch = sw, int(sw / a_min)
        cx = bx + bw // 2 - cw // 2
        cy = by + bh // 2 - ch // 2
    elif ar > a_max:                    # jeu large (plein ecran) : on suit l'action
        ch = bh
        cw = int(ch * a_max)
        cols = motion.sum(axis=0)
        x0m, x1m = int(bx * scale), int((bx + bw - cw) * scale)
        wm = max(1, int(cw * scale))
        best_x, best_v = bx + (bw - cw) // 2, -1.0
        for xm in range(x0m, max(x0m, x1m) + 1, 2):
            v = float(cols[xm:xm + wm].sum())
            # leger bonus au centre : on evite de coller aux bords sans raison
            center = 1 - 0.15 * abs((xm + wm / 2) / scale - (bx + bw / 2)) / max(1, bw / 2)
            if v * center > best_v:
                best_v, best_x = v * center, int(xm / scale)
        cx, cy = best_x, by
    else:
        cx, cy, cw, ch = bx, by, bw, bh
    cw, ch = min(cw, sw), min(ch, sh)
    cx = max(0, min(sw - cw, cx))
    cy = max(0, min(sh - ch, cy))
    return cx, cy, cw - cw % 2, ch - ch % 2


def _fit(gw: int, gh: int, pw: int, ph: int) -> tuple[int, int]:
    s = min(pw / gw, ph / gh)
    w, h = int(gw * s), int(gh * s)
    return w - w % 2, h - h % 2


# ------------------------------------------------------------------ point d'entree

def compute_layout(video: Path, streamer: str = "") -> Layout:
    cfg = config.load()
    W, H = cfg["width"], cfg["height"]
    pairs, frames = _read_pairs(video)
    sh, sw = frames[0].shape[:2]
    scale = pairs[0][0].shape[1] / sw if pairs else 384 / sw
    widen = float(cfg.get("game_widen", 1.3))

    faces = [f for f in _faces(frames) if f[3] <= sh * 0.2]   # visage trop grand = camera plein ecran
    override = cfg.get("facecam_overrides", {}).get(streamer.lower())
    cam = None
    cam_h = cfg["facecam_height"]
    masks_extra: list = []
    if override:
        ox, oy, ow, oh = override
        cam = (int(ox * sw), int(oy * sh), int(ow * sw), int(oh * sh))
    elif faces:
        rect = _cam_rect(frames, faces[0], sw, sh)
        cam = _cam_in_rect(rect, faces[0], W / cam_h) if rect else _cam_box(faces[0], sw, sh, W / cam_h)
        if rect:
            masks_extra = [rect]

    # on masque TOUTES les webcams (visage + buste) pour l'analyse du mouvement
    masks = list(masks_extra)
    for f in faces[:1] + [f for f in faces[1:] if f[4] >= len(frames) * 0.6]:
        fx, fy, fw, fh = f[:4]
        masks.append((fx - fw, fy - fh, fw * 3, fh * 5))
    if override and cam:
        masks.append(cam)
    motion = _motion_map(pairs, masks, scale) if pairs else np.zeros((10, 10), np.float32)
    box = _game_box(motion)

    if cam:
        panel_y, panel_h = cam_h, H - cam_h
        a_min = W / panel_h
        a_max = a_min * widen
    else:
        cam_h = 0
        panel_y, panel_h = 0, H
        a_min, a_max = 0.75, 1.40      # sans webcam : jeu centre sur fond flou

    if box is None:
        log.info("Zone de jeu non detectee : cadrage centre")
        box = (0, 0, motion.shape[1], motion.shape[0])
    game = _frame_game(box, motion, scale, sw, sh, a_min, a_max)
    pw = W
    ph = panel_h if cam else int(W / a_min)  # sans webcam : le jeu prend toute la largeur
    fg_w, fg_h = _fit(game[2], game[3], pw, min(ph, panel_h))
    fg_y = panel_y + (panel_h - fg_h) // 2
    log.info("Webcam %s | zone de jeu %s -> affichee %dx%d", cam, game, fg_w, fg_h)
    return Layout(sw, sh, cam, game, cam_h, panel_y, panel_h, fg_w, fg_h, fg_y)


def debug_image(video: Path, out: Path, streamer: str = "") -> Layout:
    """Image de controle : rouge = webcam, vert = zone de jeu retenue."""
    lay = compute_layout(video, streamer)
    _, frames = _read_pairs(video, n=2)
    img = frames[0].copy()
    if lay.cam:
        x, y, w, h = lay.cam
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)
    x, y, w, h = lay.game
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
    cv2.imwrite(str(out), img)
    return lay
