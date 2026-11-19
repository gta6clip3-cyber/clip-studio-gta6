"""Recuperation des videos : infos, audio complet, sections en 1080p (Twitch / YouTube / Kick)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import log, run, tool, venv_bin


def platform_of(url: str) -> str:
    u = url.lower()
    if "twitch.tv" in u:
        return "twitch"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "kick.com" in u:
        return "kick"
    return "other"


def youtube_cookies() -> Path | None:
    """Fichier de cookies d'un compte YouTube (format Netscape « cookies.txt »), s'il y en a un.

    Sans lui, YouTube finit par bloquer les lives (« Sign in to confirm you're not a bot »). Par defaut
    data/youtube_cookies.txt ; un autre chemin peut etre mis dans Parametres (youtube_cookies_file)."""
    from . import config
    p = Path(config.load().get("youtube_cookies_file") or config.DATA / "youtube_cookies.txt")
    return p if p.is_file() and p.stat().st_size > 0 else None


def _cookie_copy(src: Path) -> Path:
    """yt-dlp reecrit le fichier de cookies en partant : plusieurs yt-dlp en meme temps (un par live) le
    corromperaient. Chacun recoit donc sa copie (dans work/, vide au demarrage de l'app)."""
    import shutil
    import uuid

    from . import config
    d = config.WORK / "cookies"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"yt_{uuid.uuid4().hex[:10]}.txt"
    shutil.copyfile(src, dst)
    return dst


def _ytdlp_base() -> list[str]:
    ff = Path(tool("ffmpeg")).parent
    base = [venv_bin("yt-dlp"), "--no-warnings", "--no-playlist", "--ffmpeg-location", str(ff)]
    cookies = youtube_cookies()
    if cookies:  # seuls les sites dont le domaine est dans le fichier (YouTube) recoivent ces cookies
        base += ["--cookies", str(_cookie_copy(cookies))]
    if _node():  # YouTube impose un defi JavaScript (resolu par yt-dlp-ejs) : sans lui, « No video formats found »
        base += ["--js-runtimes", "node"]
    return base


def _node() -> bool:
    import shutil
    return shutil.which("node") is not None


def streamlink_cookie_args(url: str) -> list[str]:
    """Cookies YouTube pour streamlink (il ne lit pas les fichiers cookies.txt : un --http-cookie par cookie)."""
    cookies = youtube_cookies()
    if not cookies or platform_of(url) != "youtube":
        return []
    args = []
    for line in cookies.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        parts = line.split("\t")
        if len(parts) == 7 and not line.startswith("#") and "youtube.com" in parts[0]:
            args += ["--http-cookie", f"{parts[5]}={parts[6].strip()}"]
    return args


def clean_name(name: str) -> str:
    """Nom du streamer sans les suffixes de chaines de rediffusion ("JL Tomy - Live" -> "JL Tomy")."""
    import re
    n, prev = name.strip(), None
    while n != prev:  # "Kameto - Replays et VODs" -> "Kameto"
        prev = n
        n = re.sub(r"\s*[-|:]?\s*\b(live|lives|replays?|vods?|rediff\w*|clips?|tv|officiel|official|yt|twitch|"
                   r"best of|et)\b\s*$", "", n, flags=re.I).strip(" -|:")
    return n.strip(" -|:") or name.strip()


def get_info(url: str) -> dict:
    res = run(_ytdlp_base() + ["-J", url], timeout=120)
    info = json.loads(res.stdout)
    title = info.get("title") or ""
    if info.get("is_live") and platform_of(url) == "twitch" and info.get("description"):
        title = info["description"]  # live Twitch : title = "Nom (live) date", le vrai titre est dans description
    return {
        "url": url,
        "platform": platform_of(url),
        "id": info.get("id"),
        "title": title,
        "streamer": clean_name(info.get("uploader") or info.get("channel") or info.get("uploader_id") or "streamer"),
        "streamer_id": info.get("uploader_id") or info.get("channel_id") or "",
        "is_live": bool(info.get("is_live")),
        "duration": float(info.get("duration") or 0),
        "webpage_url": info.get("webpage_url") or url,
        # indices pour reconnaitre le jeu (tags YouTube, categorie Twitch, description)
        "context": " | ".join(x for x in (
            ", ".join(info.get("categories") or []),
            ", ".join((info.get("tags") or [])[:15]),
            (info.get("description") or "")[:300].replace("\n", " "),
        ) if x),
    }


def download_audio(url: str, out_dir: Path) -> Path:
    """Telecharge uniquement l'audio (leger) pour analyser toute la VOD."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("audio.*"))
    if existing:
        return existing[0]
    tmpl = str(out_dir / "audio.%(ext)s")
    # -N 8 : 8 morceaux telecharges en meme temps (une rediffusion Twitch = des milliers de petits morceaux)
    run(_ytdlp_base() + ["-N", "8", "-f", "bestaudio/Audio_Only/worst", "-o", tmpl, url], timeout=None)
    files = [p for p in out_dir.glob("audio.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Telechargement audio echoue.")
    return files[0]


def download_section(url: str, start: float, end: float, out: Path) -> Path:
    """Telecharge une section video en <=1080p."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    section = f"*{max(0, start):.2f}-{end:.2f}"
    import time as _time
    for attempt in range(3):  # YouTube renvoie parfois un 403 passager : on reessaie avec une nouvelle URL
        try:
            run(_ytdlp_base() + [
                "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
                "--download-sections", section,
                "--merge-output-format", "mp4",
                "-o", str(out), url,
            ], timeout=1800)
            break
        except RuntimeError:
            for p in out.parent.glob(out.stem + ".*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            if attempt == 2:
                raise
            _time.sleep(4 * (attempt + 1))
    if not out.exists():
        cands = list(out.parent.glob(out.stem + ".*"))
        if cands:
            cands[0].rename(out)
    if not out.exists():
        raise RuntimeError("Telechargement de la section video echoue.")
    return out


def _pcm_bytes(path: Path, start: float, dur: float, sr: int = 8000) -> np.ndarray:
    import subprocess
    from .utils import CREATE_NO_WINDOW
    p = subprocess.run([tool("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                        "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)


def align_offset(audio_src: Path, src_time: float, clip: Path, search: float = 6.0) -> float:
    """Retourne a quel temps (dans `clip`) correspond `src_time` de l'audio source.

    Les sections telechargees commencent sur une keyframe, donc le debut exact varie
    de quelques secondes ; on le retrouve par correlation audio.
    """
    sr = 8000
    ref = _pcm_bytes(audio_src, src_time, 8.0, sr)
    hay = _pcm_bytes(clip, 0.0, 8.0 + 2 * search, sr)
    if len(ref) < sr or len(hay) < len(ref):
        return search
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    hay = (hay - hay.mean()) / (hay.std() + 1e-6)
    n = len(hay) + len(ref)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(hay, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    valid = corr[: len(hay) - len(ref) + 1]
    lag = int(np.argmax(valid))
    return lag / sr
