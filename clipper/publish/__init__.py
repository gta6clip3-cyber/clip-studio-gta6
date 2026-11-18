"""Publication automatique sur YouTube Shorts, TikTok et Instagram Reels."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from ..utils import log


def draft_clip(folder: Path) -> dict:
    """Depose le clip dans les brouillons TikTok (rien n'est publie) et le note dans meta.json."""
    from .tiktok import upload_draft
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    try:
        link = upload_draft(folder / "clip.mp4")
        meta.setdefault("published", {})["tiktok_draft"] = {"ok": True, "link": link, "at": time.time()}
        res = {"tiktok_draft": "dans tes brouillons TikTok"}
    except Exception as e:
        log.exception("brouillon tiktok")
        meta.setdefault("published", {})["tiktok_draft"] = {"ok": False, "error": str(e)[:400], "at": time.time()}
        res = {"tiktok_draft": f"erreur : {e}"}
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def publish_clip(folder: Path, platforms: list[str] | None = None, tiktok_post: dict | None = None,
                 caption: str | None = None) -> dict:
    """Publie le clip. TikTok exige que chaque publication soit choisie par l'utilisateur dans l'ecran
    « Publier sur TikTok » (visibilite, interactions, contenu commercial) : sans `tiktok_post`, TikTok est saute."""
    cfg = config.load()
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if platforms is None:
        platforms = [p for p in ("youtube", "tiktok", "instagram") if cfg.get(f"publish_{p}")]
    if "tiktok" in platforms and tiktok_post is None:
        platforms = [p for p in platforms if p != "tiktok"]
        log.info("TikTok saute : a publier depuis l'ecran « Publier sur TikTok »")
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
                link = upload(video, caption if caption is not None else meta["caption"], tiktok_post)
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
