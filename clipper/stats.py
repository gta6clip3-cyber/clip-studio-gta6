"""Statistiques des comptes et des videos (YouTube, TikTok, Instagram), relevees toutes les 15 minutes.

Chaque releve est garde (data/stats.db) : on peut tracer les courbes d'abonnes et de vues dans le temps.
Une plateforme sans la bonne permission n'empeche pas les autres : son statut dit quoi faire."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import requests

from . import config
from .utils import log

import os  # noqa: E402

# CLIP_STATS_DB : autre base (apercu de l'interface avec des donnees de demonstration)
DB = Path(os.environ.get("CLIP_STATS_DB") or config.DATA / "stats.db")
PLATFORMS = ("youtube", "tiktok", "instagram")
_db_lock = threading.Lock()
_run_lock = threading.Lock()
_wake = threading.Event()
_started = False


class Missing(RuntimeError):
    """Connexion absente ou permission manquante : message = ce que l'utilisateur doit faire."""


# ------------------------------------------------------------------ base de donnees

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS accounts (ts INTEGER, platform TEXT, followers INTEGER,
                 views INTEGER, likes INTEGER, videos INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS videos (ts INTEGER, platform TEXT, video_id TEXT, views INTEGER,
                 likes INTEGER, comments INTEGER, shares INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS video_info (platform TEXT, video_id TEXT, title TEXT, url TEXT,
                 thumb TEXT, published_at INTEGER, folder TEXT, PRIMARY KEY (platform, video_id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS status (platform TEXT PRIMARY KEY, ok INTEGER, message TEXT,
                 ts INTEGER, account TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS i_acc ON accounts(platform, ts)")
    c.execute("CREATE INDEX IF NOT EXISTS i_vid ON videos(platform, video_id, ts)")
    return c


def _q(sql: str, args: tuple = ()) -> list[tuple]:
    with _db_lock:
        c = _conn()
        try:
            return c.execute(sql, args).fetchall()
        finally:
            c.close()


# ------------------------------------------------------------------ lien clip <-> video publiee

def _clips_index() -> dict:
    """Liens publies par l'app (meta.json) -> dossier du clip, pour relier une video a son clip."""
    idx = {"youtube": {}, "instagram": {}, "tiktok": []}
    for d in config.OUTPUT.iterdir():
        m = d / "meta.json"
        if not m.exists():
            continue
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pub = meta.get("published", {})
        link = str(pub.get("youtube", {}).get("link", ""))
        yt = re.search(r"(?:shorts/|v=|youtu\.be/)([\w-]{11})", link)
        if yt:
            idx["youtube"][yt.group(1)] = d.name
        link = str(pub.get("instagram", {}).get("link", ""))
        if link.startswith("http"):
            idx["instagram"][link.rstrip("/")] = d.name
        if pub.get("tiktok", {}).get("ok"):
            idx["tiktok"].append((_norm(meta.get("caption", "")), d.name))
    return idx


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()[:60]


# ------------------------------------------------------------------ lecteurs par plateforme

def _youtube() -> dict:
    from .publish import youtube as yt_pub
    if not yt_pub.TOKEN.exists():
        raise Missing("YouTube n'est pas connecte : Parametres > Connecter YouTube.")
    creds = yt_pub.credentials(interactive=False)
    if not creds.has_scopes([yt_pub.READ_SCOPE]):
        raise Missing("Reconnecte YouTube (Parametres > Connecter YouTube) pour autoriser la lecture des statistiques.")
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    ch = yt.channels().list(part="statistics,snippet,contentDetails", mine=True).execute()["items"][0]
    st = ch["statistics"]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, page = [], None
    while len(ids) < 200:
        r = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50, pageToken=page).execute()
        ids += [it["contentDetails"]["videoId"] for it in r.get("items", [])]
        page = r.get("nextPageToken")
        if not page:
            break
    vids = []
    for k in range(0, len(ids), 50):
        r = yt.videos().list(part="statistics,snippet", id=",".join(ids[k:k + 50])).execute()
        for v in r.get("items", []):
            s, sn = v.get("statistics", {}), v["snippet"]
            vids.append({"id": v["id"], "title": sn.get("title", ""), "url": f"https://youtube.com/shorts/{v['id']}",
                         "thumb": (sn.get("thumbnails", {}).get("high") or sn.get("thumbnails", {}).get("default") or {}).get("url", ""),
                         "published_at": _ts(sn.get("publishedAt")),
                         "views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                         "comments": int(s.get("commentCount", 0)), "shares": None})
    return {"account": ch["snippet"].get("title", ""), "followers": int(st.get("subscriberCount", 0)),
            "views": int(st.get("viewCount", 0)), "videos": vids}


def _tiktok() -> dict:
    cfg = config.load()
    if not cfg.get("tiktok_access_token"):
        raise Missing("TikTok n'est pas connecte : lance « Connecter TikTok.bat ».")
    from .publish.tiktok import _refresh
    token = _refresh(cfg)
    h = {"Authorization": f"Bearer {token}"}
    API = "https://open.tiktokapis.com/v2"
    u = requests.get(f"{API}/user/info/", headers=h, timeout=20, params={
        "fields": "display_name,follower_count,likes_count,video_count"}).json()
    if u.get("error", {}).get("code") not in (None, "ok"):
        if u["error"]["code"] == "scope_not_authorized":
            raise Missing("Ajoute les permissions « user.info.stats » et « video.list » a ton app TikTok, "
                          "puis relance « Connecter TikTok.bat ».")
        raise RuntimeError(f"TikTok : {u['error'].get('message')}")
    user = u["data"]["user"]
    vids, cursor = [], None
    fields = "id,title,video_description,create_time,cover_image_url,share_url,view_count,like_count,comment_count,share_count"
    for _ in range(5):  # 100 videos max
        body = {"max_count": 20, **({"cursor": cursor} if cursor else {})}
        r = requests.post(f"{API}/video/list/?fields={fields}", headers={**h, "Content-Type": "application/json"},
                          json=body, timeout=20).json()
        if r.get("error", {}).get("code") not in (None, "ok"):
            if r["error"]["code"] == "scope_not_authorized":
                raise Missing("Ajoute la permission « video.list » a ton app TikTok, puis relance « Connecter TikTok.bat ».")
            raise RuntimeError(f"TikTok : {r['error'].get('message')}")
        d = r.get("data", {})
        for v in d.get("videos", []):
            vids.append({"id": str(v["id"]), "title": v.get("title") or v.get("video_description", ""),
                         "url": v.get("share_url", ""), "thumb": v.get("cover_image_url", ""),
                         "published_at": int(v.get("create_time") or 0),
                         "views": int(v.get("view_count", 0)), "likes": int(v.get("like_count", 0)),
                         "comments": int(v.get("comment_count", 0)), "shares": int(v.get("share_count", 0))})
        if not d.get("has_more"):
            break
        cursor = d.get("cursor")
    return {"account": user.get("display_name", ""), "followers": int(user.get("follower_count", 0)),
            "views": sum(v["views"] for v in vids), "videos": vids}


_ig_insights_ok: bool | None = None


def _instagram() -> dict:
    global _ig_insights_ok
    cfg = config.load()
    uid, tok = cfg.get("instagram_user_id"), cfg.get("instagram_access_token")
    if not uid or not tok:
        raise Missing("Instagram n'est pas connecte : renseigne l'ID et le jeton dans les Parametres.")
    G = "https://graph.facebook.com/v27.0"
    acc = requests.get(f"{G}/{uid}", params={"fields": "username,followers_count,media_count", "access_token": tok},
                       timeout=20).json()
    if "error" in acc:
        if acc["error"].get("code") == 190:
            raise Missing("Le jeton Instagram a expire : genere-en un nouveau et colle-le dans les Parametres.")
        raise RuntimeError(f"Instagram : {acc['error'].get('message')}")
    media = requests.get(f"{G}/{uid}/media", timeout=20, params={
        "fields": "id,caption,timestamp,permalink,thumbnail_url,media_url,media_type,like_count,comments_count",
        "limit": 25, "access_token": tok}).json().get("data", [])
    vids, note = [], ""
    for m in media:
        v = {"id": m["id"], "title": (m.get("caption") or "").split("\n")[0][:120], "url": m.get("permalink", ""),
             "thumb": m.get("thumbnail_url") or m.get("media_url", ""), "published_at": _ts(m.get("timestamp")),
             "views": None, "likes": int(m.get("like_count", 0)), "comments": int(m.get("comments_count", 0)),
             "shares": None}
        if _ig_insights_ok is not False:
            ins = requests.get(f"{G}/{m['id']}/insights", params={"metric": "views,shares", "access_token": tok},
                               timeout=20).json()
            if "error" in ins:
                if ins["error"].get("code") in (10, 200):  # permission instagram_manage_insights absente
                    _ig_insights_ok = False
            else:
                _ig_insights_ok = True
                for it in ins.get("data", []):
                    val = (it.get("values") or [{}])[0].get("value")
                    if it.get("name") in ("views", "shares") and val is not None:
                        v[it["name"]] = int(val)
        vids.append(v)
    if _ig_insights_ok is False:
        note = ("Vues des Reels indisponibles : ajoute la permission « instagram_manage_insights » a ton app Meta, "
                "puis genere un nouveau jeton.")
    return {"account": acc.get("username", ""), "followers": int(acc.get("followers_count", 0)),
            "views": sum(v["views"] or 0 for v in vids), "videos": vids, "note": note}


READERS = {"youtube": _youtube, "tiktok": _tiktok, "instagram": _instagram}


def _ts(iso: str | None) -> int:
    if not iso:
        return 0
    from datetime import datetime
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


# ------------------------------------------------------------------ releve

def collect(platforms: tuple = PLATFORMS) -> dict:
    """Releve les statistiques de chaque plateforme et les enregistre. Renvoie le statut par plateforme."""
    with _run_lock:
        now = int(time.time())
        idx = _clips_index()
        result = {}
        for p in platforms:
            try:
                d = READERS[p]()
            except Missing as e:
                result[p] = {"ok": False, "message": str(e)}
            except Exception as e:  # reseau, quota... : on reessaiera au prochain releve
                log.warning("Statistiques %s : %s", p, e)
                result[p] = {"ok": False, "message": f"Erreur temporaire : {str(e)[:200]}"}
            else:
                rows = []
                for v in d["videos"]:
                    folder = None
                    if p == "youtube":
                        folder = idx["youtube"].get(v["id"])
                    elif p == "instagram":
                        folder = idx["instagram"].get(v["url"].rstrip("/"))
                    elif p == "tiktok" and v["title"]:
                        t = _norm(v["title"])
                        folder = next((f for cap, f in idx["tiktok"] if cap and (cap[:40] in t or t[:40] in cap)), None)
                    rows.append((p, v["id"], v["title"], v["url"], v["thumb"], v["published_at"], folder))
                with _db_lock:
                    c = _conn()
                    try:
                        c.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?)", (
                            now, p, d["followers"], d["views"], sum(v["likes"] or 0 for v in d["videos"]),
                            len(d["videos"])))
                        c.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?,?)", [
                            (now, p, v["id"], v["views"], v["likes"], v["comments"], v["shares"]) for v in d["videos"]])
                        c.executemany("""INSERT INTO video_info VALUES (?,?,?,?,?,?,?)
                            ON CONFLICT(platform, video_id) DO UPDATE SET title=excluded.title, url=excluded.url,
                            thumb=excluded.thumb, published_at=excluded.published_at,
                            folder=COALESCE(excluded.folder, video_info.folder)""", rows)
                        c.commit()
                    finally:
                        c.close()
                result[p] = {"ok": True, "message": d.get("note", ""), "account": d.get("account", "")}
            _write("""INSERT INTO status VALUES (?,?,?,?,?) ON CONFLICT(platform) DO UPDATE SET ok=excluded.ok,
                      message=excluded.message, ts=excluded.ts, account=excluded.account""",
                   (p, int(result[p]["ok"]), result[p]["message"], now, result[p].get("account", "")))
        return result


def _write(sql: str, args: tuple) -> None:
    with _db_lock:
        c = _conn()
        try:
            c.execute(sql, args)
            c.commit()
        finally:
            c.close()


def start_scheduler() -> None:
    """Releve en arriere-plan toutes les `stats_interval_minutes` minutes (et tout de suite au demarrage)."""
    global _started
    if _started or os.environ.get("CLIP_STATS_NO_SCHEDULER"):
        return
    _started = True

    def loop():
        while True:
            try:
                collect()
            except Exception:
                log.exception("releve des statistiques")
            minutes = max(5, int(config.load().get("stats_interval_minutes", 15)))
            _wake.wait(minutes * 60)
            _wake.clear()

    threading.Thread(target=loop, daemon=True, name="stats").start()


def refresh_now() -> None:
    _wake.set()


# ------------------------------------------------------------------ lecture pour l'interface

def _value_at(series: list[tuple], t: int) -> int | None:
    """Valeur de la serie (ts, valeur) la plus proche avant t (ou la premiere si rien avant)."""
    before = [v for ts, v in series if ts <= t and v is not None]
    if before:
        return before[-1]
    first = [v for ts, v in series if v is not None]
    return first[0] if first else None


def overview(days: int = 30) -> dict:
    now = int(time.time())
    since = now - days * 86400
    out = {"platforms": {}, "totals": {"followers": 0, "views": 0, "likes": 0, "videos": 0,
                                       "followers_24h": 0, "views_24h": 0}, "last_update": None}
    status = {r[0]: {"ok": bool(r[1]), "message": r[2], "ts": r[3], "account": r[4]}
              for r in _q("SELECT platform, ok, message, ts, account FROM status")}
    for p in PLATFORMS:
        rows = _q("SELECT ts, followers, views, likes, videos FROM accounts WHERE platform=? AND ts>=? ORDER BY ts",
                  (p, since))
        allrows = rows or _q("SELECT ts, followers, views, likes, videos FROM accounts WHERE platform=? "
                             "ORDER BY ts DESC LIMIT 1", (p,))
        last = allrows[-1] if allrows else None
        fol = [(r[0], r[1]) for r in rows]
        vws = [(r[0], r[2]) for r in rows]
        info = {"status": status.get(p, {"ok": False, "message": "Pas encore de releve.", "ts": None, "account": ""}),
                "followers": last[1] if last else None, "views": last[2] if last else None,
                "likes": last[3] if last else None, "videos": last[4] if last else None,
                "followers_24h": (last[1] - (_value_at(fol, now - 86400) or last[1])) if last else 0,
                "views_24h": (last[2] - (_value_at(vws, now - 86400) or last[2])) if last else 0,
                "series": [{"t": r[0], "followers": r[1], "views": r[2], "likes": r[3]} for r in rows]}
        out["platforms"][p] = info
        if last:
            for k in ("followers", "views", "likes", "videos", "followers_24h", "views_24h"):
                out["totals"][k] += info[k] or 0
            out["last_update"] = max(out["last_update"] or 0, last[0])
    out["daily_views"] = _daily_views(since)
    out["next_update"] = (out["last_update"] or now) + 60 * int(config.load().get("stats_interval_minutes", 15))
    return out


def _daily_views(since: int) -> list[dict]:
    """Vues gagnees par jour et par plateforme (difference entre le dernier releve du jour et celui de la veille)."""
    days: dict[str, dict] = {}
    for p in PLATFORMS:
        rows = _q("SELECT ts, views FROM accounts WHERE platform=? AND ts>=? ORDER BY ts", (p, since - 86400))
        last_by_day: dict[str, int] = {}
        for ts, v in rows:
            last_by_day[time.strftime("%Y-%m-%d", time.localtime(ts))] = v or 0
        prev = None
        for day in sorted(last_by_day):
            if prev is not None:
                days.setdefault(day, {"day": day})[p] = max(0, last_by_day[day] - prev)
            prev = last_by_day[day]
    return [days[d] for d in sorted(days)]


def videos() -> list[dict]:
    """Toutes les videos publiees, regroupees par clip quand on sait de quel clip elles viennent."""
    now = int(time.time())
    info = _q("SELECT platform, video_id, title, url, thumb, published_at, folder FROM video_info")
    groups: dict[str, dict] = {}
    for p, vid, title, url, thumb, pub, folder in info:
        hist = _q("SELECT ts, views, likes, comments, shares FROM videos WHERE platform=? AND video_id=? ORDER BY ts",
                  (p, vid))
        if not hist:
            continue
        last = hist[-1]
        vseries = [(h[0], h[1]) for h in hist]
        key = folder or f"{p}:{vid}"
        g = groups.setdefault(key, {"key": key, "folder": folder, "title": title, "thumb": thumb,
                                    "published_at": pub, "platforms": {}})
        if folder and (config.OUTPUT / folder / "cover.jpg").exists():
            g["thumb"] = f"/media/{folder}/cover.jpg"
            meta = config.OUTPUT / folder / "meta.json"
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                g.update(title=m.get("title", title), streamer=m.get("streamer"), score=m.get("score"),
                         duration=m.get("duration"))
            except (OSError, json.JSONDecodeError):
                pass
        g["published_at"] = min(x for x in (g["published_at"], pub) if x) if (g["published_at"] or pub) else 0
        g["platforms"][p] = {"id": vid, "url": url, "views": last[1], "likes": last[2], "comments": last[3],
                             "shares": last[4],
                             "views_24h": (last[1] - (_value_at(vseries, now - 86400) or last[1])) if last[1] is not None else None,
                             "spark": [h[1] for h in hist[-48:] if h[1] is not None]}
    out = []
    for g in groups.values():
        ps = g["platforms"].values()
        g["views"] = sum(x["views"] or 0 for x in ps)
        g["likes"] = sum(x["likes"] or 0 for x in ps)
        g["comments"] = sum(x["comments"] or 0 for x in ps)
        g["views_24h"] = sum(x["views_24h"] or 0 for x in ps)
        out.append(g)
    return sorted(out, key=lambda g: g["published_at"] or 0, reverse=True)


def video_detail(key: str) -> dict | None:
    g = next((v for v in videos() if v["key"] == key), None)
    if not g:
        return None
    series = {}
    for p, d in g["platforms"].items():
        hist = _q("SELECT ts, views, likes, comments, shares FROM videos WHERE platform=? AND video_id=? ORDER BY ts",
                  (p, d["id"]))
        series[p] = [{"t": h[0], "views": h[1], "likes": h[2], "comments": h[3], "shares": h[4]} for h in hist]
    g["series"] = series
    if g.get("folder"):
        try:
            g["meta"] = json.loads((config.OUTPUT / g["folder"] / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return g
