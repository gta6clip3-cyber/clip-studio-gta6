"""Effets sonores generes localement (libres de droits, crees par l'app)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .config import ASSETS

SR = 48000
SFX_DIR = ASSETS / "sfx"
MUSIC_DIR = ASSETS / "music"


def _write(path: Path, x: np.ndarray) -> None:
    x = np.clip(x / (np.abs(x).max() + 1e-9) * 0.9, -1, 1)
    st = np.repeat((x * 32767).astype(np.int16)[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(st.tobytes())


def _env(n: int, attack: float, release: float) -> np.ndarray:
    a = int(SR * attack)
    env = np.ones(n)
    if a:
        env[:a] = np.linspace(0, 1, a)
    env[a:] = np.exp(-np.linspace(0, release, n - a))
    return env


def _whoosh() -> np.ndarray:
    n = int(SR * 0.55)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    # filtre passe-bande glissant (approx. par moyenne mobile variable)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    for k, c in enumerate(np.linspace(400, 5000, 8)):
        band = np.exp(-((freqs - c) / (c * 0.5)) ** 2)
        part = np.fft.irfft(spec * band, n)
        w = np.exp(-((t - k / 8) / 0.12) ** 2)
        out += part * w
    return out * np.sin(np.pi * t) ** 1.5


def _boom() -> np.ndarray:
    n = int(SR * 1.2)
    t = np.arange(n) / SR
    f = 85 * np.exp(-t * 3) + 38
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * np.exp(-t * 3.2)
    click = np.random.default_rng(2).standard_normal(n) * np.exp(-t * 60) * 0.5
    return np.tanh((body + click) * 2.5)


def _pop() -> np.ndarray:
    n = int(SR * 0.18)
    t = np.arange(n) / SR
    f = 1400 * np.exp(-t * 25) + 500
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, 18)


def _ding() -> np.ndarray:
    n = int(SR * 0.9)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 1318 * t) + 0.5 * np.sin(2 * np.pi * 2637 * t) + 0.25 * np.sin(2 * np.pi * 1976 * t)
    return tone * _env(n, 0.003, 6)


def ensure_sfx() -> dict[str, Path]:
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    gens = {"whoosh": _whoosh, "boom": _boom, "pop": _pop, "ding": _ding}
    out = {}
    for name, fn in gens.items():
        p = SFX_DIR / f"{name}.wav"
        if not p.exists():
            _write(p, fn())
        out[name] = p
    return out


def pick_music(seed: int) -> Path | None:
    files = sorted([p for p in MUSIC_DIR.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg")])
    if not files:
        return None
    return files[seed % len(files)]
