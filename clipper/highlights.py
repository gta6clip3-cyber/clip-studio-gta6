"""Fusion des signaux (chat + audio) pour trouver les moments forts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chat import hype_weight

CHAT_DELAY = 6  # le chat reagit ~6 s apres l'evenement


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


def _rolling_z(x: np.ndarray, block: int = 300) -> np.ndarray:
    """z-score robuste par rapport a la tendance locale (le chat grossit au fil du live)."""
    n = len(x)
    if n == 0:
        return x
    centers, meds, mads = [], [], []
    for s in range(0, n, block // 2):
        seg = x[max(0, s - block): s + block]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med))) * 1.4826
        centers.append(s)
        meds.append(med)
        mads.append(max(mad, 1e-3))
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
    audio_z = _rolling_z(_smooth(loud_db, 3))
    audio_z = np.clip(audio_z, -3, 6)
    parts = {"audio": audio_z}
    if len(messages) > 50:
        count, hype = chat_arrays(messages, duration)
        cz = np.clip(_rolling_z(_smooth(count, 12)), -3, 8)
        hz = np.clip(_rolling_z(_smooth(hype, 12)), -3, 8)
        # recaler le chat sur le moment de l'evenement
        cz = np.roll(cz, -CHAT_DELAY)
        hz = np.roll(hz, -CHAT_DELAY)
        score = 0.40 * cz + 0.35 * hz + 0.25 * audio_z
        parts.update(chat=cz, hype=hz)
    else:
        score = audio_z
    return _smooth(score, 5), parts


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
