"""Decouverte des streamers GTA 6 les plus regardes + mode automatique."""
from __future__ import annotations

import time

import requests

from . import config
from .jobs import Job, manager

_token: dict = {"value": "", "exp": 0.0}


def _twitch_headers(cfg: dict) -> dict:
    if not cfg["twitch_client_id"] or not cfg["twitch_client_secret"]:
        raise RuntimeError("Identifiants Twitch manquants (Parametres > Twitch Client ID / Secret).")
    if time.time() > _token["exp"]:
        r = requests.post("https://id.twitch.tv/oauth2/token", params={
            "client_id": cfg["twitch_client_id"], "client_secret": cfg["twitch_client_secret"],
            "grant_type": "client_credentials"}, timeout=15)
        r.raise_for_status()
        d = r.json()
        _token.update(value=d["access_token"], exp=time.time() + d.get("expires_in", 3600) - 60)
    return {"Client-ID": cfg["twitch_client_id"], "Authorization": f"Bearer {_token['value']}"}


def _twitch_game_id(cfg: dict, h: dict) -> str:
    r = requests.get("https://api.twitch.tv/helix/games", params={"name": cfg["game_name"]}, headers=h, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError(f"Categorie Twitch '{cfg['game_name']}' introuvable.")
    return data[0]["id"]


def twitch_live(limit: int = 20) -> list[dict]:
    cfg = config.load()
    h = _twitch_headers(cfg)
    gid = _twitch_game_id(cfg, h)
    r = requests.get("https://api.twitch.tv/helix/streams", params={"game_id": gid, "first": limit},
                     headers=h, timeout=15)
    r.raise_for_status()
    return [{"platform": "twitch", "streamer": s["user_name"], "login": s["user_login"],
             "title": s["title"], "viewers": s["viewer_count"], "language": s["language"],
             "url": f"https://www.twitch.tv/{s['user_login']}", "live": True,
             "thumbnail": s["thumbnail_url"].replace("{width}", "320").replace("{height}", "180")}
            for s in r.json().get("data", [])]


def twitch_top_vods(limit: int = 20, period: str = "week") -> list[dict]:
    cfg = config.load()
    h = _twitch_headers(cfg)
    gid = _twitch_game_id(cfg, h)
    r = requests.get("https://api.twitch.tv/helix/videos", params={
        "game_id": gid, "first": limit, "sort": "views", "period": period, "type": "archive"},
        headers=h, timeout=15)
    r.raise_for_status()
    return [{"platform": "twitch", "streamer": v["user_name"], "login": v["user_login"], "title": v["title"],
             "viewers": v["view_count"], "language": v.get("language", ""), "url": v["url"], "live": False,
             "duration": v.get("duration", ""),
             "thumbnail": v["thumbnail_url"].replace("%{width}", "320").replace("%{height}", "180")}
            for v in r.json().get("data", [])]


def youtube_live(limit: int = 15) -> list[dict]:
    cfg = config.load()
    key = cfg["youtube_api_key"]
    if not key:
        raise RuntimeError("Cle API YouTube manquante (Parametres).")
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params={
        "part": "snippet", "q": "GTA 6", "eventType": "live", "type": "video", "order": "viewCount",
        "maxResults": limit, "key": key}, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    ids = ",".join(i["id"]["videoId"] for i in items)
    viewers = {}
    if ids:
        v = requests.get("https://www.googleapis.com/youtube/v3/videos", params={
            "part": "liveStreamingDetails", "id": ids, "key": key}, timeout=15).json()
        for it in v.get("items", []):
            viewers[it["id"]] = int(it.get("liveStreamingDetails", {}).get("concurrentViewers", 0) or 0)
    out = [{"platform": "youtube", "streamer": i["snippet"]["channelTitle"], "login": i["snippet"]["channelId"],
            "title": i["snippet"]["title"], "viewers": viewers.get(i["id"]["videoId"], 0), "language": "",
            "url": f"https://www.youtube.com/watch?v={i['id']['videoId']}", "live": True,
            "thumbnail": i["snippet"]["thumbnails"]["medium"]["url"]} for i in items]
    return sorted(out, key=lambda x: x["viewers"], reverse=True)


def discover() -> dict:
    res: dict = {"live": [], "vods": [], "errors": []}
    for name, fn, key in (("Twitch live", twitch_live, "live"), ("YouTube live", youtube_live, "live"),
                          ("Twitch VOD", twitch_top_vods, "vods")):
        try:
            res[key].extend(fn())
        except Exception as e:
            res["errors"].append(f"{name} : {e}")
    res["live"].sort(key=lambda x: x["viewers"], reverse=True)
    return res


def run_auto(job: Job) -> None:
    """Surveille en permanence les N plus gros lives GTA 6 et lance un job live pour chacun."""
    from .live import run_live
    top_n = int(job.options.get("top_n") or 3)
    watching: dict[str, Job] = {}
    job.say(f"Mode automatique : surveillance des {top_n} plus gros lives GTA 6")
    while not job.stopped:
        try:
            lives = discover()["live"]
        except Exception as e:
            job.say(f"Decouverte echouee : {e}")
            lives = []
        for k, j in list(watching.items()):
            if j.status in ("termine", "erreur", "arrete"):
                del watching[k]
        for s in lives[:top_n]:
            if s["url"] not in watching:
                lj = Job(kind="live", url=s["url"], options={"auto": True, "game": config.load()["game_name"]})
                manager.submit(lj, run_live, queued=False)
                watching[s["url"]] = lj
                job.say(f"-> {s['streamer']} ({s['viewers']} viewers) surveille")
        job.step = "Auto : " + ", ".join(j.streamer or j.url for j in watching.values())
        total = sum(len(j.clips) for j in watching.values())
        job.clips = [c for j in watching.values() for c in j.clips]
        for _ in range(120):  # re-verifie toutes les 10 minutes
            if job.stopped:
                break
            time.sleep(5)
        job.say(f"Auto : {total} clip(s) au total")
    for j in watching.values():
        manager.stop(j.id)
