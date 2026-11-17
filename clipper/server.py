"""Serveur web local : interface + API."""
from __future__ import annotations

import json
import shutil
import threading

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, llm
from .jobs import Job, manager
from .utils import tool

app = FastAPI(title="Clip Studio")

# au demarrage : rien ne tourne encore, donc tout ce qui reste dans work/ est un reste
# d'une session precedente (crash, fermeture brutale) -> on supprime
for _old in config.WORK.iterdir():
    shutil.rmtree(_old, ignore_errors=True) if _old.is_dir() else _old.unlink(missing_ok=True)
for _d in config.OUTPUT.iterdir():
    if _d.is_dir() and not (_d / "clip.mp4").exists():
        shutil.rmtree(_d, ignore_errors=True)
WEB = config.ROOT / "web"
app.mount("/media", StaticFiles(directory=str(config.OUTPUT)), name="media")


class NewJob(BaseModel):
    url: str
    mode: str = "auto"       # auto | vod | live
    max_clips: int | None = None


class AutoReq(BaseModel):
    top_n: int = 3


class PublishReq(BaseModel):
    platforms: list[str] | None = None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    st = {}
    try:
        tool("ffmpeg")
        st["ffmpeg"] = True
    except RuntimeError:
        st["ffmpeg"] = False
    cfg = config.load()
    try:
        requests.get(cfg["ollama_url"] + "/api/tags", timeout=2)
        st["ollama"] = llm.available()
    except Exception:
        st["ollama"] = False
    st["twitch_api"] = bool(cfg["twitch_client_id"] and cfg["twitch_client_secret"])
    st["youtube_connected"] = (config.DATA / "youtube_token.json").exists()
    st["tiktok"] = bool(cfg["tiktok_access_token"])
    st["instagram"] = bool(cfg["instagram_access_token"] and cfg["instagram_user_id"])
    st["publish_mode"] = cfg["publish_mode"]
    return st


@app.post("/api/jobs")
def new_job(req: NewJob):
    from .live import run_live
    from .pipeline import run_vod
    from .sources import get_info

    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Lien invalide")
    mode = req.mode
    if mode == "auto":
        try:
            mode = "live" if get_info(url)["is_live"] else "vod"
        except Exception as e:
            raise HTTPException(400, f"Lien illisible : {str(e)[-300:]}")
    job = Job(kind=mode, url=url, options={"max_clips": req.max_clips})
    if mode == "live":
        manager.submit(job, run_live, queued=False)
    else:
        manager.submit(job, run_vod, queued=True)
    return job.to_dict()


@app.post("/api/auto")
def start_auto(req: AutoReq):
    from .trending import run_auto
    for j in manager.jobs.values():
        if j.kind == "auto" and j.status in ("en cours", "en attente"):
            raise HTTPException(400, "Le mode automatique tourne deja")
    job = Job(kind="auto", url="top GTA 6", options={"top_n": req.top_n})
    manager.submit(job, run_auto, queued=False)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return sorted((j.to_dict() for j in manager.jobs.values()), key=lambda d: d["created"], reverse=True)


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    manager.stop(job_id)
    return {"ok": True}


@app.get("/api/clips")
def clips():
    out = []
    for d in sorted(config.OUTPUT.iterdir(), reverse=True):
        m = d / "meta.json"
        if d.is_dir() and m.exists() and (d / "clip.mp4").exists():
            meta = json.loads(m.read_text(encoding="utf-8"))
            meta["folder"] = d.name
            out.append(meta)
    return out


@app.post("/api/clips/{folder}/publish")
def publish(folder: str, req: PublishReq):
    from .publish import publish_clip
    d = config.OUTPUT / folder
    if not (d / "meta.json").exists():
        raise HTTPException(404, "Clip introuvable")
    # Sans liste explicite : on suit les cases « Publier sur ... » des Paramètres.
    return publish_clip(d, req.platforms or None)


@app.delete("/api/clips/{folder}")
def delete_clip(folder: str):
    d = config.OUTPUT / folder
    if not d.exists() or d.parent != config.OUTPUT:
        raise HTTPException(404, "Clip introuvable")
    trash = config.ROOT / "corbeille"
    trash.mkdir(exist_ok=True)
    shutil.move(str(d), str(trash / folder))
    return {"ok": True}


@app.get("/api/trending")
def trending():
    from .trending import discover
    return discover()


@app.get("/api/config")
def get_config():
    return config.load()


@app.post("/api/config")
def set_config(values: dict):
    return config.save(values)


@app.post("/api/youtube/connect")
def youtube_connect():
    from .publish.youtube import credentials
    try:
        credentials(interactive=True)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/open-output")
def open_output():
    import os
    import subprocess
    if hasattr(os, "startfile"):
        os.startfile(str(config.OUTPUT))  # noqa: S606 - ouvre le dossier dans l'explorateur Windows
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(config.OUTPUT)])  # Linux
    return {"ok": True}
