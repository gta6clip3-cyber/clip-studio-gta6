"""Sous-titres animes facon TikTok (fichier .ass) + accroche du debut + credit streamer.

- un seul evenement par groupe de mots (pas de clignotement), groupes enchaines sans trou ;
- le mot prononce s'allume en douceur (transition couleur + leger zoom) ;
- une couleur par personne qui parle (0 = principale en blanc/jaune, les autres en cyan, rose, vert...)."""
from __future__ import annotations

import re
from pathlib import Path

from . import config


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


# couleurs ASS (&HBBGGRR) : (couleur de base, couleur du mot actif)
SPEAKER_COLORS = [
    ("&H00FFFFFF", "&H0000E5FF"),  # principal : blanc -> jaune
    ("&H00FFDC3D", "&H00FFFFFF"),  # cyan -> blanc
    ("&H00A25FFF", "&H00FFFFFF"),  # rose -> blanc
    ("&H006BFF7C", "&H00FFFFFF"),  # vert -> blanc
    ("&H001C9FFF", "&H00FFFFFF"),  # orange -> blanc
]
WEAK = set("""a an the to of and or but in on at for with my your his her our their its is are was were i you
    he she we they it that this so if as than then be""".split())


def english_words(segments: list[dict], translations: list[str] | None, t0: float, t1: float) -> list[dict]:
    """Liste de mots anglais horodates (relatifs au debut du clip).

    - si `translations` est fourni : on repartit les mots anglais sur la duree de chaque segment
      proportionnellement a leur longueur (le timing reste cale sur la parole d'origine) ;
    - sinon on utilise directement les mots de Whisper."""
    out: list[dict] = []
    for i, seg in enumerate(segments):
        if out:
            out[-1]["eos"] = True  # fin de phrase -> nouveau groupe de mots
        ws = seg.get("words") or []
        s_start = ws[0]["start"] if ws else seg["start"]
        s_end = ws[-1]["end"] if ws else seg["end"]
        if s_end < t0 or s_start > t1:
            continue
        if translations is None:
            for w in ws:
                if t0 <= w["start"] < t1:
                    out.append({"start": w["start"] - t0, "end": min(w["end"], t1) - t0, "word": w["word"],
                                "spk": seg.get("spk", 0)})
            continue
        words = translations[i].split()
        if not words:
            continue
        # on suit le rythme de la parole d'origine : les mots anglais sont poses sur les mots source
        src = [w for w in ws if w["end"] > w["start"]]
        if len(src) >= 2:
            dur_src = sum(w["end"] - w["start"] for w in src)
            total = sum(len(w) + 1 for w in words)
            cur_chars = 0.0
            for w in words:
                a = cur_chars / total
                b = (cur_chars + len(w) + 1) / total
                cur_chars += len(w) + 1
                ws_t, we_t = _map_progress(src, a, dur_src), _map_progress(src, b, dur_src)
                if t0 <= ws_t < t1:
                    out.append({"start": ws_t - t0, "end": min(max(we_t, ws_t + 0.08), t1) - t0, "word": w,
                                "spk": seg.get("spk", 0)})
            continue
        total = sum(len(w) + 1 for w in words)
        dur = max(0.3, s_end - s_start)
        cur = s_start
        for w in words:
            d = dur * (len(w) + 1) / total
            if t0 <= cur < t1:
                out.append({"start": cur - t0, "end": min(cur + d, t1) - t0, "word": w, "spk": seg.get("spk", 0)})
            cur += d
    return out


def _map_progress(src: list[dict], frac: float, dur_src: float) -> float:
    """Position temporelle correspondant a `frac` du temps de parole (en sautant les silences)."""
    target = frac * dur_src
    acc = 0.0
    for w in src:
        d = w["end"] - w["start"]
        if acc + d > target:
            return w["start"] + (target - acc)
        acc += d
    return src[-1]["end"]


def clean_words(words: list[dict]) -> list[dict]:
    """Nettoie : ordre chronologique, pas de chevauchement, pas de mots vides, ponctuation allegee."""
    out: list[dict] = []
    for w in sorted(words, key=lambda x: x["start"]):
        txt = w["word"].strip()
        if not re.search(r"[0-9A-Za-zÀ-ÿ]", txt):
            continue
        txt = re.sub(r"^[\"'«(\[]+|[\"'»)\]]+$", "", txt)
        eos = bool(w.get("eos")) or bool(re.search(r"[.,;:…]$", txt))
        txt = re.sub(r"[.,;:…]+$", "", txt)  # facon TikTok : on garde seulement ! et ?
        if not txt:
            continue
        w = dict(w, word=txt, end=max(w["end"], w["start"] + 0.08), eos=eos)
        if out:
            p = out[-1]
            if w["start"] < p["start"] + 0.04:
                w["start"] = p["start"] + 0.04
                w["end"] = max(w["end"], w["start"] + 0.08)
            if p["end"] > w["start"]:
                p["end"] = w["start"]
        out.append(w)
    return out


def make_chunks(words: list[dict], max_words: int = 3, max_chars: int = 20) -> list[list[dict]]:
    """Groupes de mots lisibles : coupe au changement de voix, aux pauses et a la ponctuation,
    et evite de finir un groupe sur un petit mot ("the", "a"...)."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []

    def flush():
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = []

    for w in words:
        if cur:
            chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            if (w.get("spk", 0) != cur[-1].get("spk", 0) or w["start"] - cur[-1]["end"] > 0.7
                    or len(cur) >= max_words or chars > max_chars):
                # ne pas laisser un petit mot pendre en fin de groupe
                if (len(cur) > 1 and cur[-1]["word"].lower() in WEAK and w.get("spk", 0) == cur[-1].get("spk", 0)
                        and w["start"] - cur[-1]["end"] < 0.4):
                    last = cur.pop()
                    flush()
                    cur = [last]
                    if len(cur) >= max_words or sum(len(x["word"]) + 1 for x in cur) + len(w["word"]) > max_chars:
                        flush()
                else:
                    flush()
        cur.append(w)
        if re.search(r"[!?]$", w["word"]) or w.get("eos"):
            flush()
    flush()
    return chunks


def chunk_times(chunks: list[list[dict]], duration: float, min_show: float = 0.55,
                bridge: float = 0.55, hold: float = 0.0) -> list[tuple[float, float]]:
    """Debut/fin d'affichage de chaque groupe : enchaine sans trou si la pause est courte,
    duree minimale pour ne pas 'flasher'."""
    times = []
    for i, c in enumerate(chunks):
        s = c[0]["start"]
        e = c[-1]["end"] + max(0.18, hold)
        nxt = chunks[i + 1][0]["start"] if i + 1 < len(chunks) else duration
        if nxt - e < bridge:
            e = nxt
        if e - s < min_show:
            e = min(s + min_show, max(nxt, s + 0.2))
        e = min(e, nxt, duration)
        times.append((s, max(e, s + 0.1)))
    return times


def fit_hook(text: str, font_name: str, max_w: int, size: int = 80, min_one_line: int = 64,
             min_size: int = 44) -> tuple[str, int]:
    """Phrase d'accroche qui tient dans la largeur : taille reduite si besoin, sinon coupee en 2 lignes
    equilibrees (ex. « RUBIUS CRIES LAUGHING AT LUCIA'S DIZZY OUTFIT » debordait de l'ecran).
    Retourne (texte avec \\N entre les lignes, taille de police)."""
    from .render import _font
    font = _font(font_name.lower() + ".ttf", size)

    def width(s: str) -> float:  # + 1 px d'espacement par lettre (Spacing du style)
        return font.getlength(s) + len(s)

    words = text.split()
    if not words:
        return "", size
    if width(text) <= max_w:
        return text, size
    def fit(lines: list[str], s: int) -> int:  # les largeurs ne sont pas tout a fait proportionnelles : on verifie
        while s > min_size:
            f = _font(font_name.lower() + ".ttf", s)
            if all(f.getlength(l) + len(l) <= max_w for l in lines):
                break
            s -= 1
        return s

    one = int(size * max_w / width(text))
    if one >= min_one_line or len(words) == 1:
        return text, fit([text], one)
    # 2 lignes : la coupure qui rend la plus longue ligne la plus courte possible
    best = min(range(1, len(words)), key=lambda i: max(width(" ".join(words[:i])), width(" ".join(words[i:]))))
    a, b = " ".join(words[:best]), " ".join(words[best:])
    two = min(size, int(size * max_w / max(width(a), width(b))))
    return f"{a}\\N{b}", fit([a, b], two)


def build_ass(path: Path, words: list[dict], duration: float, hook: str, streamer: str,
              sub_y: int, hook_y: int, credit_y: int, lanes: dict[int, int] | None = None) -> Path:
    """lanes = {voix: y} : quand plusieurs personnes parlent, chaque voix a sa propre ligne a l'ecran
    (on peut lire la question et la reponse en meme temps). Sinon tout est a `sub_y`."""
    cfg = config.load()
    W, H = cfg["width"], cfg["height"]
    font = cfg["subtitle_font"]
    size = int(cfg.get("subtitle_size", 68))
    outline = max(3, round(size / 14))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,{outline},3,5,60,60,0,1
Style: Hook,{cfg['title_font']},80,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,1,0,3,18,0,5,70,70,0,1
Style: Credit,Arial,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,9,30,30,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []
    words = clean_words(words)
    chunks = make_chunks(words)
    voices = sorted({c[0].get("spk", 0) for c in chunks})
    multi = len(voices) > 1 and bool(lanes)
    x = W // 2
    groups = [[c for c in chunks if c[0].get("spk", 0) == v] for v in voices] if multi else [chunks]
    for gi, group in enumerate(groups):
        # chaque ligne a son propre fil : une replique reste un peu affichee pendant la reponse de l'autre
        times = chunk_times(group, duration, hold=1.0 if multi else 0.0)
        for i, (chunk, (s, e)) in enumerate(zip(group, times)):
            spk = chunk[0].get("spk", 0)
            y = (lanes or {}).get(spk, sub_y) if multi else sub_y
            base, hl = SPEAKER_COLORS[spk % len(SPEAKER_COLORS)]
            prev_end = times[i - 1][1] if i else -1
            fade_in = 50 if s - prev_end < 0.05 else 90
            nxt = times[i + 1][0] if i + 1 < len(times) else duration + 1
            fade_out = 0 if nxt - e < 0.05 else 110
            parts = []
            for k, w in enumerate(chunk):
                a = max(0, int((w["start"] - s) * 1000))
                b = int(((chunk[k + 1]["start"] if k + 1 < len(chunk) else min(e, chunk[-1]["end"] + 0.2)) - s)
                        * 1000)
                b = max(b, a + 60)
                txt = _esc(w["word"].upper())
                parts.append(f"{{\\c{base}\\t({a},{a + 70},\\c{hl}\\fscx108\\fscy108)"
                             f"\\t({b},{b + 90},\\c{base}\\fscx100\\fscy100)}}{txt}")
            anim = f"\\move({x},{y + 14},{x},{y},0,120)\\fad({fade_in},{fade_out})"
            ev.append(f"Dialogue: {1 + gi},{_ts(s)},{_ts(e)},Sub,,0,0,0,,{{\\an5{anim}}}{' '.join(parts)}")
    if hook:
        # bandeau : 18 px de fond de chaque cote + 30 px de marge avec le bord de l'ecran
        text, hsize = fit_hook(_esc(hook.upper()), cfg["title_font"], W - 2 * (18 + 30))
        # 2 lignes : le bandeau grandit du cote oppose au « @streamer » pour ne pas le recouvrir
        grow = (1 if credit_y < hook_y else -1) * round(text.count("\\N") * hsize * 0.55)
        hold = float(cfg.get("hook_seconds", 10))  # duree d'affichage de la phrase d'accroche
        ev.append(f"Dialogue: 2,{_ts(0)},{_ts(min(hold, duration))},Hook,,0,0,0,,"
                  f"{{\\an5\\pos({W // 2},{hook_y + grow})\\fs{hsize}\\fscx60\\fscy60\\t(0,140,\\fscx100\\fscy100)"
                  f"\\fad(0,400)}}{text}")
    if streamer and cfg.get("credit_streamer", True):
        ev.append(f"Dialogue: 0,{_ts(0)},{_ts(duration)},Credit,,0,0,0,,"
                  f"{{\\an9\\pos({W - 30},{credit_y})}}@{_esc(streamer)}")
    path.write_text(header + "\n".join(ev) + "\n", encoding="utf-8-sig")
    return path
