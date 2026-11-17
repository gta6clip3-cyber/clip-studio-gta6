"""Instagram Reels via l'API Graph officielle (envoi direct du fichier, upload resumable)."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .. import config

GRAPH = "https://graph.facebook.com/v27.0"


def upload(video: Path, caption: str) -> str:
    cfg = config.load()
    uid, token = cfg["instagram_user_id"], cfg["instagram_access_token"]
    if not uid or not token:
        raise RuntimeError("Identifiants Instagram manquants (Parametres).")
    r = requests.post(f"{GRAPH}/{uid}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
        "share_to_feed": "true", "thumb_offset": "1500", "access_token": token}, timeout=30).json()
    if "id" not in r:
        raise RuntimeError(f"Instagram creation : {r}")
    cid = r["id"]
    size = video.stat().st_size
    with open(video, "rb") as f:
        up = requests.post(f"https://rupload.facebook.com/ig-api-upload/v27.0/{cid}", data=f, headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)}, timeout=600)
    if up.status_code != 200:
        raise RuntimeError(f"Instagram upload : {up.status_code} {up.text[:200]}")
    for _ in range(60):
        st = requests.get(f"{GRAPH}/{cid}", params={"fields": "status_code,status", "access_token": token},
                          timeout=20).json()
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code == "ERROR":
            raise RuntimeError(f"Instagram traitement : {st.get('status')}")
        time.sleep(5)
    pub = requests.post(f"{GRAPH}/{uid}/media_publish", data={"creation_id": cid, "access_token": token},
                        timeout=30).json()
    if "id" not in pub:
        raise RuntimeError(f"Instagram publication : {pub}")
    link = requests.get(f"{GRAPH}/{pub['id']}", params={"fields": "permalink", "access_token": token},
                        timeout=20).json().get("permalink", pub["id"])
    return link
