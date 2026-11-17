"""Analyse audio : volume par seconde (cris, rires, explosions)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .utils import CREATE_NO_WINDOW, tool

SR = 8000


def loudness_per_second(path: Path, cache: Path | None = None) -> np.ndarray:
    """Volume RMS (en dB) pour chaque seconde du fichier."""
    if cache and cache.exists():
        return np.load(cache)
    proc = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                             "-ar", str(SR), "-f", "s16le", "-"], stdout=subprocess.PIPE,
                            creationflags=CREATE_NO_WINDOW)
    values: list[float] = []
    chunk = SR * 2  # 1 seconde en int16
    buf = b""
    assert proc.stdout
    while True:
        data = proc.stdout.read(chunk * 60)
        if not data:
            break
        buf += data
        n = len(buf) // chunk
        if n:
            arr = np.frombuffer(buf[: n * chunk], dtype=np.int16).astype(np.float32).reshape(n, SR)
            rms = np.sqrt((arr ** 2).mean(axis=1)) + 1e-3
            values.extend((20 * np.log10(rms / 32768)).tolist())
            buf = buf[n * chunk:]
    proc.wait()
    out = np.array(values, dtype=np.float32)
    if cache:
        np.save(cache, out)
    return out


def loudness_of_pcm(pcm: np.ndarray) -> float:
    rms = float(np.sqrt((pcm.astype(np.float32) ** 2).mean()) + 1e-3)
    return 20 * np.log10(rms / 32768)
