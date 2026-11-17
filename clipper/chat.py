"""Recuperation du chat (VOD Twitch / YouTube, et live)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .utils import log

# Mots / emotes qui signalent un moment fort, avec leur poids
HYPE_WORDS = {
    # rire
    "kekw": 2.0, "lul": 1.5, "lulw": 1.8, "omegalul": 2.2, "lmao": 1.8, "lmfao": 2.0, "lol": 1.0,
    "icant": 2.0, "xdd": 1.5, "xd": 1.0, "mdr": 1.5, "ptdr": 1.8, "jpp": 1.6, "😂": 1.5, "🤣": 1.5,
    "💀": 1.8, "dead": 1.0, "kekl": 1.8, "pepelaugh": 1.8,
    # hype / choc
    "pog": 1.6, "pogchamp": 1.6, "poggers": 1.8, "pogu": 1.8, "omg": 1.5, "wtf": 1.8, "holy": 1.2,
    "insane": 1.5, "noway": 1.8, "nooo": 1.6, "noooo": 1.8, "ez": 1.0, "w": 0.8, "ww": 1.0, "www": 1.2,
    "goat": 1.2, "clip": 2.5, "clipped": 2.0, "clipit": 2.5, "monkas": 1.4, "monkaw": 1.6,
    "dinkdonk": 1.0, "aware": 1.0, "sadge": 0.8, "🔥": 1.2, "😱": 1.6, "😭": 1.4, "ratio": 0.8,
    "putain": 1.2, "wesh": 1.0, "incroyable": 1.4, "ouf": 1.4, "chaud": 1.0,
}
_TOKEN = re.compile(r"[\w']+|[\U0001F300-\U0001FAFF]", re.UNICODE)


def hype_weight(text: str) -> float:
    toks = _TOKEN.findall(text.lower())
    w = sum(HYPE_WORDS.get(t, 0.0) for t in toks[:12])
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        w += 0.6  # MAJUSCULES
    return min(w, 5.0)


def _downloader():
    from chat_downloader import ChatDownloader
    return ChatDownloader()


def download_vod_chat(url: str, cache: Path) -> list[tuple[float, str]]:
    """Liste (temps_en_secondes, message) pour une VOD. Mis en cache dans un fichier json."""
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
    out: list[tuple[float, str]] = []
    try:
        chat = _downloader().get_chat(url, message_types=["text_message"])
        for i, msg in enumerate(chat):
            t = msg.get("time_in_seconds")
            if t is None:
                continue
            out.append((float(t), msg.get("message") or ""))
            if i and i % 20000 == 0:
                log.info("Chat : %d messages recuperes...", i)
    except Exception as e:  # chat indisponible (Kick, VOD sans replay...)
        log.warning("Chat indisponible (%s) : detection basee sur l'audio uniquement.", e)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def live_chat(url: str) -> Iterator[str]:
    """Flux de messages d'un live (bloquant)."""
    chat = _downloader().get_chat(url, message_types=["text_message"])
    for msg in chat:
        yield msg.get("message") or ""
