"""Fusion des signaux (chat + audio) pour trouver les moments forts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chat import _TOKEN, HYPE_WORDS, hype_weight

CHAT_DELAY = 6  # le chat reagit ~6 s apres l'evenement
# seuils des notes donnees a l'IA, mesures sur de vraies rediffusions
LOUD_Z = 1.8    # phrase criee = le 1 % des secondes les plus fortes (99e centile = 1,72 sur 2 rediffusions)
CHAT_Z = 2.5    # le chat s'emballe ; avec "2 personnes ecrivent la meme reaction", ~2 % des phrases sont annotees
AFTER = 6       # une reaction (cri, rire, chat) compte pour la phrase jusqu'a 6 s apres sa fin, pas plus tard


@dataclass
class Candidate:
    peak: float          # seconde du pic dans la source
    score: float
    start: float         # fenetre large envoyee a l'IA
    end: float
    reasons: str = ""


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return x
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")


def _rolling_z(x: np.ndarray, block: int = 300, min_scale: float = 1e-3,
               ignore_below: float | None = None) -> np.ndarray:
    """z-score robuste par rapport a la tendance locale (le chat grossit au fil du live).

    `min_scale` : plus petit ecart qui compte. Pour le chat = 1 message : sans ce plancher, dans un chat calme
    (mediane 0, ecart 0) un seul message isole valait un pic maximal ; avec, il faut 3 messages groupes pour z = 3.
    `ignore_below` : les secondes sous ce niveau (silence : micro coupe, ecran d'attente) ne servent pas de
    reference ; sinon, apres une longue pause, la parole normale passait pour un cri."""
    n = len(x)
    if n == 0:
        return x
    centers, meds, mads = [], [], []
    heard = x > ignore_below if ignore_below is not None else np.ones(n, dtype=bool)
    for s in range(0, n, block // 2):
        seg = x[max(0, s - block): s + block][heard[max(0, s - block): s + block]]
        if len(seg) < 30:  # presque que du silence ici : reference = tout ce qui s'entend dans le live
            seg = x[heard] if heard.any() else x
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med))) * 1.4826
        centers.append(s)
        meds.append(med)
        mads.append(max(mad, min_scale))
    idx = np.arange(n)
    med_i = np.interp(idx, centers, meds)
    mad_i = np.interp(idx, centers, mads)
    floor = max(float(np.percentile(mad_i, 25)), 1e-3)
    return (x - med_i) / np.maximum(mad_i, floor)


def chat_arrays(messages: list[tuple[float, str]], duration: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(duration, dtype=np.float32)
    hype = np.zeros(duration, dtype=np.float32)
    for t, text in messages:
        i = int(t)
        if 0 <= i < duration:
            count[i] += 1
            hype[i] += hype_weight(text)
    return count, hype


def score_timeline(loud_db: np.ndarray, messages: list[tuple[float, str]]) -> tuple[np.ndarray, dict]:
    duration = len(loud_db)
    loud = _smooth(loud_db, 3)
    audio_z = _rolling_z(loud, ignore_below=float(np.median(loud)) - 25 if duration else None)
    audio_z = np.clip(audio_z, -3, 6)
    parts = {"audio": audio_z}
    if len(messages) > 50:
        count, hype = chat_arrays(messages, duration)
        # planchers : 1 message / 1 message "hype" moyen (poids ~2), ramenes a la fenetre de lissage de 12 s
        cz = np.clip(_rolling_z(_smooth(count, 12), min_scale=1 / 12), -3, 8)
        hz = np.clip(_rolling_z(_smooth(hype, 12), min_scale=2 / 12), -3, 8)
        # recaler le chat sur le moment de l'evenement ; la fin est remplie de zeros (np.roll y recopiait le debut)
        tail = np.zeros(min(CHAT_DELAY, duration), dtype=cz.dtype)
        cz = np.concatenate([cz[CHAT_DELAY:], tail])
        hz = np.concatenate([hz[CHAT_DELAY:], tail])
        score = 0.40 * cz + 0.35 * hz + 0.25 * audio_z
        parts.update(chat=cz, hype=hz)
    else:
        score = audio_z
    return _smooth(score, 5), parts


def reactions(segs: list[dict], parts: dict, messages: list[tuple[float, str]]) -> list[str]:
    """Pour chaque phrase, ce que le texte seul ne montre pas : elle est criee, le chat explose juste apres.

    Ces notes sont ajoutees a la transcription lue par l'IA : elle "entend" le volume et "lit" le chat."""
    if not segs:
        return []
    lo, hi = segs[0]["start"] + CHAT_DELAY, segs[-1]["end"] + CHAT_DELAY + AFTER
    # messages ramenes a l'instant de l'evenement (comme parts["chat"] / parts["hype"])
    near = [(t - CHAT_DELAY, text) for t, text in messages if lo <= t <= hi] if "hype" in parts else []
    out = []
    for i, s in enumerate(segs):
        a = s["start"]
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else float("inf")
        b = min(nxt, s["end"] + AFTER)  # le court silence apres la phrase compte pour elle
        z = slice(int(a), max(int(a) + 1, int(b)))
        notes = ["LOUD"] if parts["audio"][z].max(initial=0) >= LOUD_Z else []
        if near and max(parts["hype"][z].max(initial=0), parts["chat"][z].max(initial=0)) >= CHAT_Z:
            said = Counter(w for t, text in near if a <= t < b
                           for w in set(_TOKEN.findall(text.lower())) & HYPE_WORDS.keys())
            top = [f"{w.upper()} x{n}" for w, n in said.most_common(3) if n >= 2]
            if top:
                notes.append("chat: " + ", ".join(top))
        out.append(" | ".join(notes))
    return out


def pick_candidates(score: np.ndarray, parts: dict, n: int, min_gap: int = 120,
                    pre: int = 55, post: int = 30) -> list[Candidate]:
    duration = len(score)
    order = np.argsort(score)[::-1]
    taken: list[int] = []
    out: list[Candidate] = []
    for i in order:
        i = int(i)
        if score[i] < 0.8 or len(out) >= n:
            break
        if i < 30 or any(abs(i - j) < min_gap for j in taken):
            continue
        taken.append(i)
        reasons = ", ".join(f"{k}={float(v[i]):.1f}" for k, v in parts.items())
        out.append(Candidate(peak=float(i), score=float(score[i]),
                             start=float(max(0, i - pre)), end=float(min(duration, i + post)),
                             reasons=reasons))
    return out
