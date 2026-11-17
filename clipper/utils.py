"""Outils communs : ffmpeg, logs, sous-processus."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("clipper")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

for _noisy in ("httpx", "huggingface_hub", "urllib3", "faster_whisper"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

CREATE_NO_WINDOW =0x08000000 if os.name == "nt" else 0


def _find_winget_ffmpeg(name: str) -> str | None:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        for p in base.glob(f"Gyan.FFmpeg*/**/bin/{name}.exe"):
            return str(p)
    return None


def tool(name: str) -> str:
    """Trouve ffmpeg/ffprobe meme si le PATH n'est pas encore rafraichi."""
    p = shutil.which(name)
    if p:
        return p
    p = _find_winget_ffmpeg(name)
    if p:
        return p
    raise RuntimeError(f"{name} introuvable. Installe FFmpeg (winget install Gyan.FFmpeg).")


def venv_bin(name: str) -> str:
    """Executable installe dans le venv (yt-dlp, streamlink)."""
    scripts = Path(sys.executable).parent
    for cand in (scripts / f"{name}.exe", scripts / name):
        if cand.exists():
            return str(cand)
    p = shutil.which(name)
    if p:
        return p
    raise RuntimeError(f"{name} introuvable dans l'environnement Python.")


def run(cmd: list[str], check: bool = True, capture: bool = True, timeout: float | None = None,
        cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    log.debug("RUN %s", " ".join(map(str, cmd)))
    res = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True, encoding="utf-8",
                         errors="replace", timeout=timeout, creationflags=CREATE_NO_WINDOW, cwd=cwd)
    if check and res.returncode != 0:
        tail = (res.stderr or "")[-2500:]
        raise RuntimeError(f"Commande echouee ({Path(str(cmd[0])).name}): {tail}")
    return res


def probe_duration(path: str | Path) -> float:
    res = run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
               "-of", "json", str(path)])
    return float(json.loads(res.stdout)["format"]["duration"])


def probe_video(path: str | Path) -> tuple[int, int]:
    res = run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height", "-of", "json", str(path)])
    s = json.loads(res.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def probe_fps(path: str | Path) -> float:
    res = run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=avg_frame_rate", "-of", "json", str(path)])
    num, _, den = json.loads(res.stdout)["streams"][0]["avg_frame_rate"].partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return 60.0 if fps > 50 else 30.0


def slug(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text[:maxlen] or "clip"


def ffmpeg_path_escape(path: str | Path) -> str:
    """Echappe un chemin Windows pour les filtres ffmpeg (ass=...)."""
    p = str(Path(path).resolve()).replace("\\", "/")
    return p.replace(":", "\\:").replace("'", "\\'")
