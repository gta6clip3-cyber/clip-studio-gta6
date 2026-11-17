"""YouTube Shorts via l'API officielle YouTube Data v3 (OAuth)."""
from __future__ import annotations

from pathlib import Path

from .. import config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
SECRET = config.DATA / "youtube_client_secret.json"
TOKEN = config.DATA / "youtube_token.json"


def credentials(interactive: bool = True):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError("YouTube non connecte (Parametres > Connecter YouTube).")
        if not SECRET.exists():
            raise RuntimeError(f"Fichier {SECRET.name} manquant dans le dossier data (voir le guide).")
        flow = InstalledAppFlow.from_client_secrets_file(str(SECRET), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
    TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload(video: Path, cover: Path, title: str, caption: str, tags: list[str]) -> str:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    cfg = config.load()
    yt = build("youtube", "v3", credentials=credentials(interactive=False), cache_discovery=False)
    t = title[:90]
    if "#shorts" not in t.lower():
        t = f"{t} #shorts"
    body = {
        "snippet": {"title": t[:100], "description": caption[:4900], "tags": tags[:15], "categoryId": "20"},
        "status": {"privacyStatus": cfg["youtube_privacy"], "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(cover))).execute()
    except Exception:
        pass  # miniatures personnalisees non supportees pour certains Shorts / comptes non verifies
    return f"https://youtube.com/shorts/{vid}"
