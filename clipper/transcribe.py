"""Transcription locale (faster-whisper sur GPU) avec horodatage mot par mot."""
from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from . import config
from .utils import CREATE_NO_WINDOW, log, tool

_model = None
_model_key = None
_lock = threading.Lock()


def _add_cuda_dlls() -> None:
    """Rend visibles les bibliotheques CUDA installees via pip (nvidia-cublas / nvidia-cudnn)."""
    if os.name != "nt":  # Linux : le systeme ne les cherche pas dans le venv, on les charge d'avance
        import ctypes
        libs = sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib/lib*.so.*"))
        for lib in libs + libs:  # 2 passes : certaines ont besoin d'une autre, chargee apres elles
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.exists():
        return
    for sub in base.iterdir():
        b = sub / "bin"
        if b.exists():
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(b))
            except (OSError, AttributeError):
                pass


_add_cuda_dlls()


def get_model():
    global _model, _model_key
    cfg = config.load()
    key = (cfg["whisper_model"], cfg["whisper_device"], cfg["whisper_compute_type"])
    if _model is not None and _model_key == key:
        return _model
    from faster_whisper import WhisperModel
    try:
        log.info("Chargement de Whisper %s (%s)...", key[0], key[1])
        _model = WhisperModel(key[0], device=key[1], compute_type=key[2])
    except Exception as e:
        log.warning("Whisper GPU indisponible (%s) -> CPU", e)
        _model = WhisperModel(key[0], device="cpu", compute_type="int8")
    _model_key = key
    return _model


def unload() -> None:
    """Libere la VRAM (pour laisser la place au modele de langage)."""
    global _model, _model_key
    with _lock:
        _model = None
        _model_key = None
        gc.collect()


def load_pcm(path: Path, start: float, dur: float) -> np.ndarray:
    p = subprocess.run([tool("ffmpeg"), "-v", "error", "-ss", f"{max(0, start):.3f}", "-t", f"{dur:.3f}",
                        "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    return np.frombuffer(p.stdout, dtype=np.float32)


def transcribe(path: Path, start: float, end: float, language: str | None = None,
               task: str = "transcribe") -> dict:
    """Transcrit [start, end] du fichier. Les temps retournes sont ceux de la source."""
    audio = load_pcm(path, start, end - start)
    with _lock:
        model = get_model()
        segments, info = model.transcribe(
            audio, language=language, task=task, beam_size=5, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        segs = []
        for s in segments:
            words = [{"start": round(w.start + start, 2), "end": round(w.end + start, 2), "word": w.word.strip()}
                     for w in (s.words or []) if w.word.strip()]
            segs.append({"start": round(s.start + start, 2), "end": round(s.end + start, 2),
                         "text": s.text.strip(), "words": words})
    return {"language": info.language, "segments": clean_segments(segs)}


def _norm(t: str) -> str:
    return re.sub(r"[^\w]+", " ", t.lower()).strip()


def clean_segments(segs: list[dict]) -> list[dict]:
    """Retire les hallucinations classiques de Whisper : phrase repetee a l'identique en un temps impossible,
    segments vides, textes parasites ("Sous-titres realises par...", "Merci d'avoir regarde")."""
    out: list[dict] = []
    junk = re.compile(r"sous-titr|amara\.org|merci d'avoir regard|thanks for watching|subtitles by", re.I)
    for s in segs:
        txt = s["text"].strip()
        if not txt or junk.search(txt):
            continue
        dur = max(0.05, s["end"] - s["start"])
        n = len(s["words"]) or len(txt.split())
        if out and _norm(txt) == _norm(out[-1]["text"]) and n >= 3 and n / dur > 5:
            continue  # repetition impossible a dire aussi vite = hallucination
        out.append(s)
    return out


def sentences(segs: list[dict], max_words: int = 28) -> list[dict]:
    """Regroupe les segments Whisper en vraies phrases (Whisper coupe souvent en plein milieu),
    pour une traduction propre et sans doublon."""
    out: list[dict] = []
    for s in segs:
        if out:
            p = out[-1]
            ends = p["text"].rstrip()[-1:] in ".!?…"
            gap = s["start"] - p["end"]
            n = len(p["words"]) + len(s["words"])
            if not ends and gap < 0.5 and n <= max_words:
                out[-1] = {"start": p["start"], "end": s["end"], "text": p["text"].rstrip() + " " + s["text"].lstrip(),
                           "words": p["words"] + s["words"]}
                continue
        out.append(dict(s))
    return out


def as_text(tr: dict, origin: float = 0.0, notes: list[str] | None = None) -> str:
    """Transcription horodatee ; `notes` (une par phrase, voir highlights.reactions) est ajoute en fin de ligne."""
    segs = tr["segments"]
    return "\n".join(f"[{s['start'] - origin:6.1f}-{s['end'] - origin:6.1f}] {s['text']}" + (f"  <<{n}>>" if n else "")
                     for s, n in zip(segs, notes or [""] * len(segs)))
