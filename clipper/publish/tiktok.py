"""TikTok via la Content Posting API officielle (Direct Post)."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .. import config

API = "https://open.tiktokapis.com/v2"


def _refresh(cfg: dict) -> str:
    if not (cfg["tiktok_refresh_token"] and cfg["tiktok_client_key"] and cfg["tiktok_client_secret"]):
        return cfg["tiktok_access_token"]
    r = requests.post(f"{API}/oauth/token/", data={
        "client_key": cfg["tiktok_client_key"], "client_secret": cfg["tiktok_client_secret"],
        "grant_type": "refresh_token", "refresh_token": cfg["tiktok_refresh_token"]},
        headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=20)
    d = r.json()
    if "access_token" in d:
        config.save({"tiktok_access_token": d["access_token"],
                     "tiktok_refresh_token": d.get("refresh_token", cfg["tiktok_refresh_token"])})
        return d["access_token"]
    return cfg["tiktok_access_token"]


def upload(video: Path, caption: str) -> str:
    cfg = config.load()
    token = _refresh(cfg)
    if not token:
        raise RuntimeError("Jeton TikTok manquant (Parametres).")
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    size = video.stat().st_size
    chunk = 10 * 1024 * 1024
    if size <= chunk:
        chunk, count = size, 1
    else:
        count = size // chunk
    init = requests.post(f"{API}/post/publish/video/init/", headers=h, json={
        "post_info": {"title": caption[:2200], "privacy_level": cfg["tiktok_privacy"],
                      "disable_duet": False, "disable_comment": False, "disable_stitch": False,
                      "video_cover_timestamp_ms": 1500},
        "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk,
                        "total_chunk_count": count},
    }, timeout=30).json()
    if init.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok init : {init['error']}")
    upload_url = init["data"]["upload_url"]
    publish_id = init["data"]["publish_id"]
    with open(video, "rb") as f:
        for i in range(count):
            start = i * chunk
            end = size - 1 if i == count - 1 else start + chunk - 1
            f.seek(start)
            data = f.read(end - start + 1)
            r = requests.put(upload_url, data=data, headers={
                "Content-Type": "video/mp4", "Content-Length": str(len(data)),
                "Content-Range": f"bytes {start}-{end}/{size}"}, timeout=300)
            if r.status_code not in (200, 201, 206):
                raise RuntimeError(f"TikTok upload : HTTP {r.status_code} {r.text[:200]}")
    for _ in range(60):
        st = requests.post(f"{API}/post/publish/status/fetch/", headers=h,
                           json={"publish_id": publish_id}, timeout=20).json()
        status = st.get("data", {}).get("status")
        if status == "PUBLISH_COMPLETE":
            return f"tiktok publish_id {publish_id}"
        if status == "FAILED":
            raise RuntimeError(f"TikTok : {st['data'].get('fail_reason')}")
        time.sleep(5)
    return f"tiktok publish_id {publish_id} (en traitement)"
