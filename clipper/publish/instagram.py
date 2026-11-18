"""Instagram Reels via l'API Graph officielle (envoi direct du fichier, upload resumable).

Avant l'envoi, le clip est reencode au format Reels strict (30 i/s, H.264 High, AAC 44,1 kHz,
GOP ferme, faststart) : Instagram refuse sinon certains fichiers avec « ProcessingFailedError ».
Un Reel peut durer de 3 s a 15 min via l'API.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .. import config
from ..utils import log, probe_duration, run, tool

GRAPH = "https://graph.facebook.com/v27.0"
RUPLOAD = "https://rupload.facebook.com/ig-api-upload/v27.0"
MIN_S, MAX_S = 3, 15 * 60
ATTEMPTS = 3


def prepare(video: Path) -> Path:
    """Copie du clip au format Reels strict (ig.mp4 a cote du clip, reutilisee si deja faite)."""
    out = video.with_name("ig.mp4")
    if out.exists() and out.stat().st_mtime >= video.stat().st_mtime:
        return out
    tmp = out.with_suffix(".tmp.mp4")
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(video),
         "-map", "0:v:0", "-map", "0:a:0?",
         "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-preset", "medium",
         "-crf", "20", "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p",
         "-r", "30", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0", "-flags", "+cgop",
         "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
         "-movflags", "+faststart", str(tmp)], timeout=1800)
    tmp.replace(out)
    return out


def _upload_once(uid: str, token: str, video: Path, caption: str) -> str:
    r = requests.post(f"{GRAPH}/{uid}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
        "share_to_feed": "true", "thumb_offset": "1500", "access_token": token}, timeout=30).json()
    if "id" not in r:
        raise RuntimeError(f"Instagram creation : {r}")
    cid = r["id"]
    log.info("Instagram conteneur %s", cid)

    def send() -> str:
        size = video.stat().st_size
        try:
            with open(video, "rb") as f:
                up = requests.post(f"{RUPLOAD}/{cid}", data=f, headers={
                    "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size),
                    "Content-Length": str(size)}, timeout=900)
        except requests.RequestException as e:
            return f"reseau : {e}"
        return "" if up.status_code == 200 else f"{up.status_code} {up.text[:200]}"

    # Au-dela de ~40 s, rupload repond souvent « 400 ProcessingFailedError » alors que la video
    # est bien recue : on juge sur l'etat du conteneur, et on renvoie au meme conteneur s'il stagne.
    up_err = send()
    if up_err:
        log.warning("Instagram upload (%s) : %s - on surveille le conteneur", cid, up_err)
    resends, stalled, start = 0, 0, time.time()
    while True:
        st = requests.get(f"{GRAPH}/{cid}", params={"fields": "status_code,status", "access_token": token},
                          timeout=20).json()
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram traitement : {st.get('status')} (upload : {up_err or 'ok'})")
        if time.time() - start > 600:
            raise RuntimeError(f"Instagram traitement : pas fini apres 10 min (upload : {up_err or 'ok'})")
        stalled += 1
        if up_err and stalled >= 12 and resends < 2:  # ~1 min sans avancer : on renvoie le fichier
            resends += 1
            stalled = 0
            up_err = send()
            log.warning("Instagram renvoi %d au conteneur %s : %s", resends, cid, up_err or "ok")
        time.sleep(5)
    pub = requests.post(f"{GRAPH}/{uid}/media_publish", data={"creation_id": cid, "access_token": token},
                        timeout=30).json()
    if "id" not in pub:
        raise RuntimeError(f"Instagram publication : {pub}")
    return requests.get(f"{GRAPH}/{pub['id']}", params={"fields": "permalink", "access_token": token},
                        timeout=20).json().get("permalink", pub["id"])


def upload(video: Path, caption: str) -> str:
    cfg = config.load()
    uid, token = cfg["instagram_user_id"], cfg["instagram_access_token"]
    if not uid or not token:
        raise RuntimeError("Identifiants Instagram manquants (Parametres).")
    dur = probe_duration(video)
    if not MIN_S <= dur <= MAX_S:
        raise RuntimeError(f"Instagram : un Reel doit durer entre 3 s et 15 min (ce clip : {dur:.0f} s).")
    ig = prepare(video)
    err: Exception | None = None
    for i in range(ATTEMPTS):
        try:
            return _upload_once(uid, token, ig, caption)  # nouveau conteneur a chaque essai
        except RuntimeError as e:
            err = e
            log.warning("Instagram essai %d/%d : %s", i + 1, ATTEMPTS, e)
            if "creation" in str(e) or "publication" in str(e):
                break  # erreur de compte/jeton : inutile de recommencer
            time.sleep(30)
    raise err  # type: ignore[misc]
