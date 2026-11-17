"""Publication automatique sur YouTube Shorts, TikTok et Instagram Reels."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from ..utils import log


def publish_clip(folder: Path, platforms: list[str] | None = None) -> dict:
    cfg = config.load()
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if platforms is None:
        platforms = [p for p in ("youtube", "tiktok", "instagram") if cfg.get(f"publish_{p}")]
    video = folder / "clip.mp4"
    cover = folder / "cover.jpg"
    results = {}
    for p in platforms:
        if meta.get("published", {}).get(p, {}).get("ok"):
            results[p] = "deja publie"
            continue
        try:
            if p == "youtube":
                from .youtube import upload
                link = upload(video, cover, meta["title"], meta["caption"], meta.get("hashtags", []))
            elif p == "tiktok":
                from .tiktok import upload
                link = upload(video, meta["caption"])
            elif p == "instagram":
                from .instagram import upload
                link = upload(video, meta["caption"])
            else:
                continue
            meta.setdefault("published", {})[p] = {"ok": True, "link": link, "at": time.time()}
            results[p] = link
        except Exception as e:
            log.exception("publication %s", p)
            meta.setdefault("published", {})[p] = {"ok": False, "error": str(e)[:400], "at": time.time()}
            results[p] = f"erreur : {e}"
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
