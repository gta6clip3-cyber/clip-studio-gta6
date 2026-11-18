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


PRIVACY_LABELS = {"PUBLIC_TO_EVERYONE": "Tout le monde", "MUTUAL_FOLLOW_FRIENDS": "Amis",
                  "FOLLOWER_OF_CREATOR": "Abonnés", "SELF_ONLY": "Moi uniquement"}


def creator_info() -> dict:
    """Compte qui va publier + options autorisees par TikTok (a afficher avant chaque publication)."""
    cfg = config.load()
    token = _refresh(cfg)
    if not token:
        raise RuntimeError("TikTok n'est pas connecte : lance « Connecter TikTok.bat ».")
    r = requests.post(f"{API}/post/publish/creator_info/query/", timeout=20, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}).json()
    if r.get("error", {}).get("code") not in (None, "ok"):
        code = r["error"].get("code")
        if code == "spam_risk_too_many_posts":
            raise RuntimeError("TikTok : limite de publications atteinte pour aujourd'hui, reessaie plus tard.")
        raise RuntimeError(f"TikTok : {r['error'].get('message') or code}")
    return r.get("data", {})


def check_options(opts: dict, info: dict, duration: float) -> dict:
    """Verifie les choix de l'utilisateur selon les regles TikTok (Direct Post). Renvoie post_info."""
    privacy = opts.get("privacy_level")
    if privacy not in (info.get("privacy_level_options") or []):
        raise ValueError("Choisis qui peut voir la vidéo.")
    brand_content, brand_organic = bool(opts.get("brand_content_toggle")), bool(opts.get("brand_organic_toggle"))
    if opts.get("commercial") and not (brand_content or brand_organic):
        raise ValueError("Contenu commercial : coche « Ta marque » et/ou « Contenu de marque ».")
    if brand_content and privacy == "SELF_ONLY":
        raise ValueError("Le contenu de marque ne peut pas être privé.")
    max_s = info.get("max_video_post_duration_sec")
    if max_s and duration > max_s:
        raise ValueError(f"Vidéo trop longue pour ce compte ({duration:.0f} s, maximum {max_s} s).")
    return {"privacy_level": privacy,
            "disable_comment": bool(info.get("comment_disabled")) or not opts.get("allow_comment"),
            "disable_duet": bool(info.get("duet_disabled")) or not opts.get("allow_duet"),
            "disable_stitch": bool(info.get("stitch_disabled")) or not opts.get("allow_stitch"),
            "brand_content_toggle": brand_content, "brand_organic_toggle": brand_organic}


def upload_draft(video: Path) -> str:
    """Depose la video dans les BROUILLONS TikTok de l'utilisateur (rien n'est publie) : il la publie ensuite
    lui-meme depuis l'appli TikTok. Autorise automatiquement ; demande la permission « video.upload »."""
    return upload(video, "", draft=True)


def upload(video: Path, caption: str, post_info: dict | None = None, draft: bool = False) -> str:
    """Publie sur TikTok. `post_info` = choix faits par l'utilisateur dans l'ecran de publication
    (sans lui : reglage des Parametres, uniquement pour les tests internes)."""
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
    body = {"source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk,
                            "total_chunk_count": count}}
    if draft:
        url, done = f"{API}/post/publish/inbox/video/init/", "SEND_TO_USER_INBOX"
    else:
        url, done = f"{API}/post/publish/video/init/", "PUBLISH_COMPLETE"
        body["post_info"] = {"title": caption[:2200], "video_cover_timestamp_ms": 1500,
                             **(post_info or {"privacy_level": cfg["tiktok_privacy"], "disable_duet": False,
                                              "disable_comment": False, "disable_stitch": False})}
    init = requests.post(url, headers=h, json=body, timeout=30).json()
    if init.get("error", {}).get("code") not in (None, "ok"):
        if init["error"].get("code") == "scope_not_authorized":
            raise RuntimeError("TikTok : ajoute la permission « video.upload » a ton app TikTok, "
                               "puis relance « Connecter TikTok.bat ».")
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
        if status == done:
            return f"tiktok {'brouillon' if draft else 'publish_id'} {publish_id}"
        if status == "FAILED":
            raise RuntimeError(f"TikTok : {st['data'].get('fail_reason')}")
        time.sleep(5)
    return f"tiktok publish_id {publish_id} (en traitement)"
