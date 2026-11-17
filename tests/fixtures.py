"""Genere des fichiers audio de test (voix de synthese Windows) : dialogues a plusieurs voix, francais, anglais."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from clipper.utils import tool

FIX = Path(__file__).resolve().parent / "_fixtures"
SR = 16000

# voix : (nom SAPI, transformation ffmpeg) -> 3 "personnes" bien distinctes
VOICES = {
    "A": ("Microsoft Hortense Desktop", "asetrate=22050*0.78,aresample=22050,atempo=1.20"),  # grave
    "B": ("Microsoft Zira Desktop", None),
    "C": ("Microsoft Hortense Desktop", "asetrate=22050*1.12,aresample=22050,atempo=0.92"),  # aigue
}


def _tts(text: str, voice: str, out: Path) -> None:
    if not shutil.which("powershell"):  # voix de synthese de Windows : ces tests-la sont sautes ailleurs
        pytest.skip("voix de synthese Windows absentes")
    t = text.replace("'", "''")
    ps = (f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.Rate = 1; "
          f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, "
          f"[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{out}', $f); $s.Speak('{t}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def _load(path: Path, af: str | None) -> np.ndarray:
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path)]
    if af:
        cmd += ["-af", af]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def dialogue(name: str, turns: list[tuple[str, str]], gap: float = 0.35, noise: float = 0.0) -> tuple[Path, list]:
    """turns = [(voix, texte)]. Retourne (wav, verite terrain [(debut, fin, voix)])."""
    FIX.mkdir(exist_ok=True)
    out = FIX / f"{name}.wav"
    truth_file = FIX / f"{name}.npy"
    pieces, truth, t = [], [], 0.6
    pieces.append(np.zeros(int(SR * 0.6), np.float32))
    for k, (v, text) in enumerate(turns):
        raw = FIX / f"{name}_{k}.wav"
        if not raw.exists():
            _tts(text, VOICES[v][0], raw)
        pcm = _load(raw, VOICES[v][1])
        nz = np.nonzero(np.abs(pcm) > 0.01)[0]
        if len(nz):
            pcm = pcm[max(0, nz[0] - 800): nz[-1] + 800]
        truth.append((t, t + len(pcm) / SR, v))
        pieces.append(pcm)
        pieces.append(np.zeros(int(SR * gap), np.float32))
        t += len(pcm) / SR + gap
    audio = np.concatenate(pieces)
    if noise:
        rng = np.random.default_rng(0)
        # bruit "jeu video" : bruit rose + bips
        white = rng.standard_normal(len(audio)).astype(np.float32)
        pink = np.convolve(white, np.ones(30) / 30, mode="same")
        tt = np.arange(len(audio)) / SR
        beeps = 0.3 * np.sin(2 * np.pi * 440 * tt) * (np.sin(2 * np.pi * 0.7 * tt) > 0.95)
        audio = audio + noise * (pink * 3 + beeps).astype(np.float32)
    audio = np.clip(audio, -1, 1)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    np.save(truth_file, np.array([(a, b, ord(v)) for a, b, v in truth]))
    return out, truth


FR_LINES = [
    "Attends attends, il arrive derrière toi, fais gaffe !",
    "Non mais c'est pas possible, il m'a encore tué ce mec.",
    "Vas-y fonce, on a le temps, on peut le faire !",
    "J'ai plus de munitions, quelqu'un peut me couvrir ?",
    "Oh la vache, t'as vu ce saut ? C'est incroyable !",
    "Bon, on se calme, on refait la stratégie.",
]
EN_LINES = [
    "Wait wait, he's right behind you, watch out!",
    "No way, this guy killed me again.",
    "Go go go, we still have time, we can do it!",
    "I'm out of ammo, can somebody cover me?",
]
