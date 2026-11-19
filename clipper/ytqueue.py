"""File de publication YouTube et Instagram : les clips du mode automatique ne partent plus tous d'un coup.

Une chaine neuve qui publie 10 Shorts en 4 h ressemble a un robot et YouTube la montre tres peu. Les clips
attendent donc dans une file et un seul part a chaque heure prevue (Parametres : youtube_slots, ex.
« 12:00,18:00,21:00 » = 3 par jour). Le bouton « Publier » d'un clip, lui, publie tout de suite."""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path

from . import config
from .utils import log

_started = False
_lock = threading.Lock()


PLATFORMS = ("youtube", "instagram")


def enabled(cfg: dict | None = None, platform: str = "youtube") -> bool:
    cfg = cfg or config.load()
    return bool(cfg.get(f"{platform}_queue", True)) and bool(slots(cfg))


def slots(cfg: dict | None = None) -> list[dt.time]:
    """Heures de publication, triees (« 12:00, 18:00,21h » -> [12:00, 18:00, 21:00])."""
    cfg = cfg or config.load()
    out = []
    for part in str(cfg.get("youtube_slots", "12:00,18:00,21:00")).replace("h", ":").split(","):
        part = part.strip().rstrip(":")
        if not part:
            continue
        h, _, m = part.partition(":")
        try:
            out.append(dt.time(int(h) % 24, int(m or 0) % 60))
        except ValueError:
            log.warning("heure de publication YouTube ignoree : %r", part)
    return sorted(set(out))


def add(folder: Path, platform: str = "youtube") -> None:
    """Met le clip en file (note dans meta.json, visible dans « Mes clips »)."""
    meta_file = folder / "meta.json"
    with _lock:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta.setdefault("published", {})[f"{platform}_queue"] = {"ok": True, "at": time.time(), "link": "en file"}
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def queued(platform: str = "youtube") -> list[Path]:
    """Clips en attente, le meilleur d'abord (note IA), puis le plus ancien."""
    items = []
    for d in config.OUTPUT.iterdir() if config.OUTPUT.exists() else []:
        m = d / "meta.json"
        if not m.exists() or not (d / "clip.mp4").exists():
            continue
        meta = json.loads(m.read_text(encoding="utf-8"))
        pub = meta.get("published", {})
        if f"{platform}_queue" in pub and not pub.get(platform, {}).get("ok"):
            items.append((-(meta.get("score") or 0), pub[f"{platform}_queue"].get("at", 0), d))
    return [d for *_, d in sorted(items)]


def due_slot(now: dt.datetime, cfg: dict) -> str:
    """Derniere heure prevue deja passee aujourd'hui (« 2026-11-19 18:00 »), ou "" avant la premiere."""
    passed = [s for s in slots(cfg) if s <= now.time()]
    return f"{now.date()} {passed[-1].strftime('%H:%M')}" if passed else ""


def _state_file(platform: str = "youtube") -> Path:
    return config.DATA / f"{platform}_queue.json"


def tick(now: dt.datetime | None = None, platform: str = "youtube") -> str | None:
    """Publie UN clip si une heure prevue est passee et pas encore servie. Une heure manquee pendant que
    l'app etait fermee n'est rattrapee qu'une fois (pas de rafale au redemarrage). Retourne le dossier publie."""
    cfg = config.load()
    if not enabled(cfg, platform):
        return None
    slot = due_slot(now or dt.datetime.now(), cfg)
    sf = _state_file(platform)
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    if not slot or state.get("last_slot") == slot:
        return None
    q = queued(platform)
    if not q:
        return None  # rien en attente : l'heure reste libre, le prochain clip pret partira tout de suite
    from .publish import publish_clip
    folder = q[0]
    res = publish_clip(folder, [platform])
    log.info("File %s (%s) : %s -> %s", platform, slot, folder.name, res)
    with _lock:
        meta_file = folder / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("published", {}).get(platform, {}).get("ok"):
            meta["published"].pop(f"{platform}_queue", None)
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # meme en cas d'erreur, l'heure est consommee : pas d'essais en boucle (le clip reste en file)
    sf.write_text(json.dumps({"last_slot": slot, "folder": folder.name}), encoding="utf-8")
    return folder.name


def status(platform: str = "youtube") -> dict:
    cfg = config.load()
    now = dt.datetime.now()
    nxt = next((s for s in slots(cfg) if s > now.time()), None)
    return {"enabled": enabled(cfg, platform), "slots": [s.strftime("%H:%M") for s in slots(cfg)],
            "waiting": [d.name for d in queued(platform)],
            "next": nxt.strftime("%H:%M") if nxt else (slots(cfg)[0].strftime("%H:%M") + " (demain)" if slots(cfg) else "")}


def start_scheduler() -> None:
    global _started
    if _started or os.environ.get("CLIP_STATS_NO_SCHEDULER"):  # copie de test : ne publie jamais
        return
    _started = True

    def loop():
        while True:
            for platform in PLATFORMS:
                try:
                    tick(platform=platform)
                except Exception:
                    log.exception("file de publication %s", platform)
            time.sleep(60)

    threading.Thread(target=loop, daemon=True).start()
