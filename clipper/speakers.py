"""Qui parle quand ? (diarisation locale, gratuite : pyannote-segmentation + WeSpeaker en ONNX via sherpa-onnx).

Sert a colorer les sous-titres differemment pour chaque personne qui parle."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import config
from .utils import log

MODELS = config.ASSETS / "models"
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "wespeaker_en_voxceleb_resnet34_LM.onnx"

_sd = None
_sd_key = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return SEG_MODEL.exists() and EMB_MODEL.exists()


def _engine(threshold: float):
    global _sd, _sd_key
    if _sd is not None and _sd_key == threshold:
        return _sd
    import sherpa_onnx as so
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=4),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=4),
        clustering=so.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.25,
        min_duration_off=0.4,
    )
    if not cfg.validate():
        raise RuntimeError("Configuration de diarisation invalide")
    _sd = so.OfflineSpeakerDiarization(cfg)
    _sd_key = threshold
    return _sd


def diarize(pcm16k: np.ndarray, threshold: float | None = None) -> list[tuple[float, float, int]]:
    """Segments (debut, fin, locuteur) en secondes relatives au debut de `pcm16k` (mono 16 kHz float32)."""
    if not available() or len(pcm16k) < 16000 * 2:
        return []
    thr = threshold if threshold is not None else config.load().get("speaker_threshold", 0.62)
    with _lock:
        sd = _engine(thr)
        res = sd.process(np.ascontiguousarray(pcm16k, dtype=np.float32)).sort_by_start_time()
    return [(float(r.start), float(r.end), int(r.speaker)) for r in res]


def _speaker_at(segs: list[tuple[float, float, int]], a: float, b: float) -> int | None:
    best, best_ov = None, 0.0
    for s, e, spk in segs:
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is not None:
        return best
    # pas de recouvrement : segment le plus proche (< 0.6 s)
    mid = (a + b) / 2
    near = min(segs, key=lambda x: min(abs(mid - x[0]), abs(mid - x[1])), default=None)
    if near and min(abs(mid - near[0]), abs(mid - near[1])) < 0.6:
        return near[2]
    return None


def assign(words: list[dict], segs: list[tuple[float, float, int]], min_share: float = 0.12,
           min_seconds: float = 2.5) -> list[dict]:
    """Ajoute words[i]["spk"] : 0 = personne principale (celle qui parle le plus), 1, 2, ... = les autres.

    Les "locuteurs" qui parlent tres peu sont fusionnes avec leurs voisins (evite les fausses couleurs)."""
    if not words:
        return words
    if not segs:
        for w in words:
            w["spk"] = 0
        return words
    raw = [_speaker_at(segs, w["start"], max(w["end"], w["start"] + 0.05)) for w in words]
    # temps de parole par locuteur
    talk: dict[int, float] = {}
    for s, e, spk in segs:
        talk[spk] = talk.get(spk, 0.0) + (e - s)
    total = sum(talk.values()) or 1.0
    keep = {spk for spk, t in talk.items() if t / total >= min_share and t >= min_seconds}
    if not keep:
        keep = {max(talk, key=talk.get)}
    # mots d'un locuteur "fantome" ou sans locuteur -> heritent du mot precedent garde
    prev = None
    for i, spk in enumerate(raw):
        if spk in keep:
            prev = spk
        else:
            raw[i] = prev
    nxt = None
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] is None:
            raw[i] = nxt
        else:
            nxt = raw[i]
    # lissage : un changement de voix sur 1 seul mot au milieu d'une phrase = bruit
    for i in range(1, len(raw) - 1):
        if raw[i - 1] == raw[i + 1] and raw[i] != raw[i - 1]:
            raw[i] = raw[i - 1]
    # renumerotation : 0 = celui qui a le plus de mots
    counts: dict[int, int] = {}
    for spk in raw:
        if spk is not None:
            counts[spk] = counts.get(spk, 0) + 1
    order = sorted(counts, key=counts.get, reverse=True)
    remap = {spk: k for k, spk in enumerate(order)}
    for w, spk in zip(words, raw):
        w["spk"] = remap.get(spk, 0)
    return words


EMB_DIR = MODELS
DEFAULT_EMB = ["nemo_en_titanet_large"]
_exts: dict[str, object] = {}


def _extractor(name: str):
    if name not in _exts:
        import sherpa_onnx as so
        path = EMB_DIR / f"{name}.onnx"
        if not path.exists():
            path = EMB_DIR / "bench" / f"{name}.onnx"
        _exts[name] = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=4))
    return _exts[name]


def emb_models() -> list[str]:
    names = config.load().get("speaker_models") or DEFAULT_EMB
    ok = [n for n in names if (EMB_DIR / f"{n}.onnx").exists() or (EMB_DIR / "bench" / f"{n}.onnx").exists()]
    return ok or ["wespeaker_en_voxceleb_resnet34_LM"]


def _embed(pcm: np.ndarray, a: float, b: float, models: list[str] | None = None) -> np.ndarray | None:
    """Empreinte vocale du passage [a, b] : concatenation des empreintes de plusieurs modeles (plus fiable)."""
    x = pcm[int(max(0, a) * 16000):int(b * 16000)]
    if len(x) < 16000 * 0.35:
        return None
    parts = []
    for name in models or emb_models():
        with _lock:
            ext = _extractor(name)
            st = ext.create_stream()
            st.accept_waveform(16000, np.ascontiguousarray(x, dtype=np.float32))
            st.input_finished()
            e = np.asarray(ext.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n > 0:
            parts.append(e / n)
    if not parts:
        return None
    e = np.concatenate(parts)
    return e / np.linalg.norm(e)


def runs_of(words: list[dict], raw: list[int | None]) -> list[list[int]]:
    """Decoupe les mots en 'tours de parole' : fin de phrase, pause, ou changement de voix vu par pyannote."""
    runs: list[list[int]] = []
    for i, w in enumerate(words):
        if runs:
            p = words[runs[-1][-1]]
            long_run = p["end"] - words[runs[-1][0]]["start"] > 5.0
            if not (p.get("eos") or p["word"].rstrip()[-1:] in ".?!" or long_run or w["start"] - p["end"] > 0.5
                    or (raw[i] is not None and raw[i - 1] is not None and raw[i] != raw[i - 1])):
                runs[-1].append(i)
                continue
        runs.append([i])
    return runs


def _kmeans2(E: np.ndarray, W: np.ndarray, iters: int = 30, seeds: int = 12) -> np.ndarray:
    """K-moyennes spheriques a 2 groupes (plusieurs departs, on garde le meilleur)."""
    best, best_obj = np.zeros(len(E), dtype=int), -1e9
    rng = np.random.default_rng(0)
    for _ in range(seeds):
        C = E[rng.choice(len(E), 2, replace=False)]
        for _ in range(iters):
            lab = np.argmax(E @ C.T, 1)
            newC = []
            for g in range(2):
                m = lab == g
                v = (E[m] * W[m, None]).sum(0) if m.any() else C[g]
                newC.append(v / (np.linalg.norm(v) + 1e-9))
            C = np.array(newC)
        obj = float(np.sum(W * np.sum(E * C[lab], 1)))
        if obj > best_obj:
            best, best_obj = lab.copy(), obj
    return best


def choose(E: np.ndarray, W: np.ndarray, max_inter: float, min_talk: float = 3.0, min_runs: int = 2,
           max_voices: int = 4, times: np.ndarray | None = None, burst_inter: float = 0.15,
           loud: np.ndarray | None = None, max_louder_db: float = 2.5) -> np.ndarray:
    """Separe les voix par divisions successives : un groupe est coupe en deux seulement si les deux moities
    se ressemblent peu (similarite moyenne entre elles < max_inter) et parlent chacune assez longtemps.
    Une seule voix (meme avec des cris / rires) ne passe pas ce test ; deux voix differentes, si."""
    lab = np.zeros(len(E), dtype=int)
    queue = [0]
    while queue and len(set(lab.tolist())) < max_voices:
        g = queue.pop(0)
        idx = np.where(lab == g)[0]
        if len(idx) < 2 * min_runs:
            continue
        sub = _kmeans2(E[idx], W[idx])
        parts = [idx[sub == 0], idx[sub == 1]]
        if any(len(p) < min_runs or W[p].sum() < min_talk for p in parts):
            continue
        # une vraie 2e personne dit au moins 2 vraies phrases (>= 2 s) ; des exclamations courtes ("Non !", "Vite !")
        # dans le bruit du jeu donnent des empreintes trompeuses
        if any((W[p] >= 2.0).sum() < 2 for p in parts):
            continue
        inter = float((E[parts[0]] @ E[parts[1]].T).mean())
        # 3e voix et plus : il faut une difference encore plus nette
        if inter >= max_inter - 0.05 * (len(set(lab.tolist())) - 1):
            continue
        if times is not None and inter >= burst_inter:
            # une "2e voix" qui n'est qu'un seul bloc d'affilee (pas d'alternance) = le plus souvent la meme
            # personne qui crie / s'emballe : on exige alors des voix vraiment tres differentes
            minor = parts[0] if W[parts[0]].sum() < W[parts[1]].sum() else parts[1]
            order = idx[np.argsort(times[idx])]
            pos = np.where(np.isin(order, minor))[0]
            span = float(times[minor].max() - times[minor].min())
            major = parts[1] if minor is parts[0] else parts[0]
            louder = float(np.average(loud[minor], weights=W[minor]) - np.average(loud[major], weights=W[major])) \
                if loud is not None else 0.0
            log.info("Separation candidate : inter %.2f, %d tours (positions %s, sur %.0f s), %+.1f dB", inter,
                     len(minor), pos.tolist(), span, louder)
            if pos.max() - pos.min() + 1 <= len(minor) + 1 and span <= 20:
                continue
            if louder >= max_louder_db:
                continue  # la "2e voix" parle nettement plus fort = la meme personne qui s'emballe
        new = lab.max() + 1
        lab[parts[1]] = new
        queue += [g, new]
    return lab

def label_words(audio_path: Path, t0: float, t1: float, words: list[dict], opts: dict | None = None) -> list[dict]:
    """Etiquette chaque mot avec sa voix (temps relatifs a t0) : 0 = voix principale, 1, 2... = les autres."""
    from .transcribe import load_pcm
    if not words:
        return words
    cfg = config.load()
    o = {"max_inter": cfg.get("speaker_max_inter_similarity", 0.45), "veto": False, "models": None, **(opts or {})}
    try:
        if not available():
            raise RuntimeError("modeles absents")
        pcm = load_pcm(audio_path, t0, t1 - t0)
        segs = diarize(pcm) if o["veto"] or o.get("use_pyannote_turns", True) else []
        raw = [_speaker_at(segs, w["start"], max(w["end"], w["start"] + 0.05)) for w in words] if segs else \
            [None] * len(words)
        if o["veto"]:
            labeled = [r for r in raw if r is not None]
            if labeled and labeled.count(max(set(labeled), key=labeled.count)) / len(labeled) >= 0.97:
                raise RuntimeError("une seule voix entendue")
        runs = runs_of(words, raw)
        embs, weights, idx, starts, louds = [], [], [], [], []
        for r, run in enumerate(runs):
            a, b = words[run[0]]["start"], words[run[-1]]["end"]
            if b - a >= 0.6:
                e = _embed(pcm, a - 0.05, b + 0.05, o["models"])
                if e is not None:
                    embs.append(e)
                    weights.append(min(b - a, 5.0))
                    idx.append(r)
                    starts.append(a)
                    seg = pcm[int(max(0, a) * 16000):int(b * 16000)]
                    louds.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
        if len(embs) < 3:
            raise RuntimeError("pas assez de parole")
        E = np.array(embs)
        W = np.array(weights)
        lab = choose(E, W, o["max_inter"], times=np.array(starts), loud=np.array(louds),
                     burst_inter=o.get("burst_inter", cfg.get("speaker_burst_inter_similarity", 0.15)))
        keep = sorted(set(lab.tolist()))
        cent = {g: E[lab == g].mean(0) for g in keep}
        cent = {g: v / np.linalg.norm(v) for g, v in cent.items()}
        talk = {g: float(W[lab == g].sum()) for g in keep}
        run_lab = {r: int(g) for r, g in zip(idx, lab)}
        # tours courts : compares aux voix trouvees, sinon heritent du tour precedent
        prev = run_lab[idx[0]]
        for r, run in enumerate(runs):
            if r not in run_lab:
                g = prev
                if len(keep) > 1:
                    a, b = words[run[0]]["start"], words[run[-1]]["end"]
                    e = _embed(pcm, a - 0.15, b + 0.15, o["models"])
                    if e is not None:
                        sims = {k: float(e @ cent[k]) for k in keep}
                        best = max(sims, key=sims.get)
                        if sims[best] - sims[prev] > 0.04:
                            g = best
                run_lab[r] = g
            prev = run_lab[r]
        order = sorted(keep, key=lambda g: talk[g], reverse=True)
        remap = {g: k for k, g in enumerate(order)}
        for r, run in enumerate(runs):
            for i in run:
                words[i]["spk"] = remap[run_lab[r]]
        log.info("Voix : %d (tours de parole : %d)", len(keep), len(runs))
    except Exception as e:
        log.info("Une seule couleur de sous-titres (%s)", e)
        for w in words:
            w["spk"] = 0
    return words

def split_by_speaker(segs: list[dict], audio_path: Path, t0: float, t1: float) -> list[dict]:
    """Etiquette les mots d'origine (horaires exacts de Whisper) puis coupe chaque phrase a chaque changement
    de voix : chaque phrase renvoyee a une seule voix ("spk"), et sera traduite a part."""
    words, owner = [], []
    for si, s in enumerate(segs):
        ws = s.get("words") or []
        for k, w in enumerate(ws):
            words.append({"word": w["word"], "start": w["start"] - t0, "end": w["end"] - t0, "eos": k == len(ws) - 1})
            owner.append(si)
    if not words:
        return [dict(s, spk=0) for s in segs]
    label_words(audio_path, t0, t1, words)
    out: list[dict] = []
    for si, s in enumerate(segs):
        ws = s.get("words") or []
        labs = [w["spk"] for w, o in zip(words, owner) if o == si]
        if not ws:
            out.append(dict(s, spk=out[-1]["spk"] if out else 0))
            continue
        start = 0
        for k in range(1, len(ws) + 1):
            if k == len(ws) or labs[k] != labs[start]:
                part = ws[start:k]
                out.append({"start": part[0]["start"], "end": part[-1]["end"], "words": part, "spk": labs[start],
                            "text": " ".join(w["word"] for w in part) if k - start < len(ws) else s["text"]})
                start = k
    return out
