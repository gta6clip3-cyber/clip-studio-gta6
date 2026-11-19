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


def youtube_live(limit: int = 15, query: str = "GTA 6") -> list[dict]:
    cfg = config.load()
    key = cfg["youtube_api_key"]
    if not key:
        raise RuntimeError("Cle API YouTube manquante (Parametres).")
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params={
        "part": "snippet", "q": query, "eventType": "live", "type": "video", "order": "viewCount",
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


def twitch_search(query: str, limit: int = 20) -> list[dict]:
    """Lives Twitch dont le nom de chaine correspond a la recherche (tous les jeux)."""
    cfg = config.load()
    h = _twitch_headers(cfg)
    r = requests.get("https://api.twitch.tv/helix/search/channels", params={
        "query": query, "live_only": "true", "first": limit}, headers=h, timeout=15)
    r.raise_for_status()
    found = [c for c in r.json().get("data", []) if c.get("is_live")]
    if not found:
        return []
    # la recherche ne donne ni les spectateurs ni l'apercu du live : on les demande pour toutes les chaines d'un coup
    st = requests.get("https://api.twitch.tv/helix/streams", params=[("user_id", c["id"]) for c in found] +
                      [("first", "100")], headers=h, timeout=15)
    st.raise_for_status()
    live = {s["user_id"]: s for s in st.json().get("data", [])}
    out = []
    for c in found:
        s = live.get(c["id"])
        if not s:
            continue
        out.append({"platform": "twitch", "streamer": c["display_name"], "login": c["broadcaster_login"],
                    "title": s["title"], "viewers": s["viewer_count"], "language": s.get("language", ""),
                    "game": s.get("game_name", ""), "url": f"https://www.twitch.tv/{c['broadcaster_login']}",
                    "live": True,
                    "thumbnail": s["thumbnail_url"].replace("{width}", "320").replace("{height}", "180")})
    return out


def search_lives(query: str, platform: str = "all") -> dict:
    """Recherche d'un live a clipper : nom du streamer ou mots du titre, sur Twitch et/ou YouTube.
    Une recherche YouTube coute 100 unites de quota (10 000 par jour) : elle n'est faite que sur demande."""
    query = query.strip()
    res: dict = {"live": [], "errors": []}
    if len(query) < 2:
        res["errors"].append("Tape au moins 2 lettres.")
        return res
    sources = []
    if platform in ("all", "twitch"):
        sources.append(("Twitch", lambda: twitch_search(query)))
    if platform in ("all", "youtube"):
        sources.append(("YouTube", lambda: youtube_live(10, query)))
    for name, fn in sources:
        try:
            res["live"].extend(fn())
        except Exception as e:
            res["errors"].append(f"{name} : {e}")
    res["live"].sort(key=lambda s: (match_rank(query, s), -s["viewers"]))
    return res


def match_rank(query: str, s: dict) -> int:
    """0 = c'est ce streamer, 1 = son nom contient la recherche, 2 = trouve par le titre ou autre.
    Les espaces et la casse ne comptent pas : « kai cenat » = « KaiCenat »."""
    import re
    norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())  # noqa: E731
    q = norm(query)
    names = [norm(s.get("streamer")), norm(s.get("login"))]
    if q in names:
        return 0
    return 1 if any(q in n for n in names if n) else 2


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


_yt_cache: dict = {"lives": [], "t": 0.0}


def auto_lives(yt_every_s: float = 1800) -> list[dict]:
    """Lives pour le mode auto : Twitch a chaque fois, YouTube au plus toutes les 30 min
    (une recherche YouTube coute 100 unites de quota sur 10 000 par jour)."""
    lives, errors = [], []
    try:
        lives += twitch_live()
    except Exception as e:
        errors.append(f"Twitch : {e}")
    if time.time() - _yt_cache["t"] > yt_every_s:
        try:
            _yt_cache.update(lives=youtube_live(), t=time.time())
        except Exception as e:
            errors.append(f"YouTube : {e}")
            _yt_cache["t"] = time.time() - yt_every_s + 600  # nouvel essai dans 10 min, pas a chaque tour
    lives += _yt_cache["lives"]
    if errors and not lives:
        raise RuntimeError(" ; ".join(errors))
    return sorted(lives, key=lambda x: x["viewers"], reverse=True)


def plan_rotation(watched: dict[str, dict], lives: list[dict], top_n: int, margin: float = 0.2,
                  min_watch_s: float = 900, now: float | None = None, skip: set | None = None) -> tuple[list, list]:
    """Decide quels lives arreter et lesquels lancer.

    watched : url -> {"since": debut du suivi}. lives : lives actuels, tries par audience.
    - une place libre (live termine) est remplie par le plus gros live non suivi ;
    - un live plus gros remplace le plus faible suivi s'il a au moins `margin` (20 %) de spectateurs en plus
      et si le plus faible est suivi depuis `min_watch_s` (sinon on changerait sans cesse sans rien clipper).
    Retourne (urls a arreter, lives a lancer)."""
    now = time.time() if now is None else now
    skip = skip or set()
    views = {s["url"]: s["viewers"] for s in lives}
    keep = dict(watched)
    stop, start = [], []
    for s in lives:
        if s["url"] in keep or s["url"] in skip:
            continue
        if len(keep) < top_n:
            keep[s["url"]] = {"since": now}
            start.append(s)
            continue
        # le plus faible des lives suivis (absent de la liste = hors du top : 0 spectateur)
        old = [u for u in keep if u in watched and now - watched[u]["since"] >= min_watch_s]
        if not old:
            break
        weak = min(old, key=lambda u: views.get(u, 0))
        if s["viewers"] < views.get(weak, 0) * (1 + margin) or s["viewers"] <= views.get(weak, 0):
            break  # liste triee : les suivants sont encore plus petits
        del keep[weak]
        stop.append(weak)
        keep[s["url"]] = {"since": now}
        start.append(s)
    return stop, start


def run_auto(job: Job) -> None:
    """Surveille en permanence les N plus gros lives GTA 6 : un live termine est remplace,
    un live nettement plus gros prend la place du plus petit."""
    from .live import run_live
    cfg = config.load()
    top_n = int(job.options.get("top_n") or 3)
    margin = float(cfg.get("auto_swap_margin", 0.2))
    min_watch = float(cfg.get("auto_min_watch_minutes", 15)) * 60
    every = float(cfg.get("auto_check_minutes", 5)) * 60
    watching: dict[str, Job] = {}
    since: dict[str, float] = {}
    mine_stopped: set[str] = set()   # lives arretes par la rotation (pas par toi)
    skip: set[str] = set()           # lives que tu as arretes a la main : on ne les relance pas
    cool: dict[str, float] = {}      # url -> pas relance avant cette heure
    old_clips: list[str] = []
    job.say(f"Mode automatique : surveillance des {top_n} plus gros lives GTA 6")
    last = 0.0
    while not job.stopped:
        freed = False
        for url, j in list(watching.items()):
            if j.status in ("termine", "erreur", "arrete"):
                if j.status == "arrete" and url not in mine_stopped:
                    skip.add(url)
                    job.say(f"{j.streamer or url} arrete a la main : il ne sera pas relance")
                elif url not in mine_stopped:
                    job.say(f"{j.streamer or url} : live termine, place liberee")
                    cool[url] = time.time() + 600  # flux coupe mais encore liste : pas relance avant 10 min
                old_clips += [c for c in j.clips if c not in old_clips]
                del watching[url]
                since.pop(url, None)
                mine_stopped.discard(url)
                freed = True
        if freed or time.time() - last >= every:
            last = time.time()
            try:
                lives = auto_lives()
            except Exception as e:
                job.say(f"Decouverte echouee : {e}")
                lives = []
            if lives:  # liste vide = API en panne : on ne touche a rien
                # un live en cours d'arret (il finit ses clips) ne compte plus comme une place prise
                active = {u: {"since": since[u]} for u in watching if u not in mine_stopped}
                now = time.time()
                wait = skip | mine_stopped | {u for u, t in cool.items() if t > now}
                stop, start = plan_rotation(active, lives, top_n, margin, min_watch, skip=wait)
                views = {s["url"]: s["viewers"] for s in lives}
                for url in stop:
                    j = watching[url]
                    mine_stopped.add(url)
                    manager.stop(j.id)  # les clips en cours de montage sont termines avant l'arret
                    job.say(f"<- {j.streamer or url} ({views.get(url, 0)} viewers) remplace par plus gros")
                for s in start:
                    lj = Job(kind="live", url=s["url"], options={"auto": True, "game": cfg["game_name"]})
                    manager.submit(lj, run_live, queued=False)
                    watching[s["url"]] = lj
                    since[s["url"]] = time.time()
                    job.say(f"-> {s['streamer']} ({s['viewers']} viewers) surveille")
        live_now = [j for u, j in watching.items() if u not in mine_stopped]
        job.step = "Auto : " + (", ".join(j.streamer or j.url for j in live_now) or "en recherche de lives")
        job.clips = old_clips + [c for j in watching.values() for c in j.clips if c not in old_clips]
        for _ in range(6):  # etat des lives verifie toutes les 30 s, recherche toutes les 5 min
            if job.stopped:
                break
            time.sleep(5)
    for j in watching.values():
        manager.stop(j.id)
    # on attend que chaque live ait fini et publie ses clips commences : l'app peut ensuite etre fermee sans perte
    while True:
        busy = [j for j in watching.values() if j.status in ("en cours", "en attente")]
        job.clips = old_clips + [c for j in watching.values() for c in j.clips if c not in old_clips]
        if not busy:
            break
        job.step = f"Arret : {len(busy)} live(s) finissent leurs clips en cours, ne ferme pas l'app..."
        time.sleep(5)
    job.say(f"Auto arrete : {len(job.clips)} clip(s) au total. Tu peux fermer l'app.")
