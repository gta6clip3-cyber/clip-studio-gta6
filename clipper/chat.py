"""Recuperation du chat (VOD Twitch / YouTube, et live)."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from .sources import _ytdlp_base, platform_of
from .utils import log, run

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


# Par defaut chat-downloader reessaie 15 fois avec des pauses qui doublent (plus de 2 h au total) et la
# rediffusion reste bloquee pendant ce temps : on abandonne vite, l'app continue avec l'audio seul.
_RETRY = {"max_attempts": 5, "retry_timeout": 3, "interruptible_retry": False}


def _downloader():
    from chat_downloader import ChatDownloader
    return ChatDownloader()


def _youtube_vod_chat(url: str, folder: Path) -> list[tuple[float, str]]:
    """Chat rejoue d'une rediffusion YouTube, via yt-dlp (chat-downloader n'arrive plus a lire YouTube)."""
    run(_ytdlp_base() + ["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                         "-o", str(folder / "ytchat"), url], timeout=3600)
    file = folder / "ytchat.live_chat.json"
    out: list[tuple[float, str]] = []
    for line in file.read_text(encoding="utf-8").splitlines() if file.exists() else []:  # pas de fichier = pas de chat
        replay = json.loads(line).get("replayChatItemAction", {})
        for action in replay.get("actions", []):
            item = action.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer", {})
            text = "".join(r.get("text") or _emoji(r.get("emoji", {})) for r in item.get("message", {}).get("runs", []))
            if text.strip():
                out.append((int(replay.get("videoOffsetTimeMsec", 0)) / 1000, text))
    return out


def _emoji(e: dict) -> str:
    """Emoji standard -> le caractere ; emote de la chaine -> son nom (":kekw:")."""
    name = (e.get("shortcuts") or [""])[0] if e.get("isCustomEmoji") else e.get("emojiId", "")
    return f" {name} "


def _twitch_vod_chat(vod_id: str, duration: float, parts: int = 4) -> list[tuple[float, str]]:
    """Chat d'une rediffusion Twitch, lu en `parts` tranches de temps a la fois (4x plus rapide).

    On appelle directement le lecteur de commentaires de chat-downloader : son etape "infos de la video" est
    refusee par Twitch (requete perimee) et faisait tout echouer, alors que les commentaires repondent tres bien."""
    from chat_downloader.sites.twitch import TwitchChatDownloader
    if duration <= 0:
        parts = 1  # duree inconnue : une seule tranche, du debut a la fin
    step = int(duration // parts) + 1

    def part(k: int) -> list[tuple[float, str]]:
        # les temps sont des secondes entieres : [k*step, (k+1)*step - 1] ne laisse ni trou ni doublon ;
        # la derniere tranche n'a pas de fin (elle prend aussi ce qui depasse la duree annoncee)
        end = None if k == parts - 1 else (k + 1) * step - 1
        start, got = k * step, []
        for _ in range(3):  # une page refusee : on reprend a la derniere seconde lue, 3 essais au plus
            if got:  # cette derniere seconde est peut-etre incomplete : on la relit en entier
                start = int(got[-1][0])
                got = [m for m in got if m[0] < start]
            params = dict(_RETRY, message_types=["text_message"], start_time=start, end_time=end)
            try:
                for m in TwitchChatDownloader()._get_chat_messages_by_vod_id(vod_id, params, duration):
                    if m.get("time_in_seconds") is not None:
                        got.append((float(m["time_in_seconds"]), m.get("message") or ""))
                return got
            except Exception as e:
                log.warning("Chat Twitch : tranche %d/%d interrompue (%s)", k + 1, parts, e)
        return got  # on garde ce qui est lu : un chat incomplet vaut mieux que pas de chat

    with ThreadPoolExecutor(parts) as pool:
        return [m for chunk in pool.map(part, range(parts)) for m in chunk]


def download_vod_chat(url: str, cache: Path, duration: float = 0.0) -> list[tuple[float, str]]:
    """Liste (temps_en_secondes, message) pour une VOD. Mis en cache dans un fichier json."""
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
    out: list[tuple[float, str]] = []
    twitch_vod = re.search(r"twitch\.tv/videos/(\d+)", url)
    try:
        if platform_of(url) == "youtube":
            out = _youtube_vod_chat(url, cache.parent)
        elif twitch_vod:
            out = _twitch_vod_chat(twitch_vod.group(1), duration)
        else:
            chat = _downloader().get_chat(url, message_types=["text_message"], **_RETRY)
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
    channel = re.search(r"twitch\.tv/(\w+)/?(?:[?#].*)?$", url)
    if channel:
        # lecture directe du chat Twitch (IRC) : on saute l'etape "infos du live" de chat-downloader, que Twitch
        # peut refuser du jour au lendemain comme il l'a fait pour les rediffusions
        from chat_downloader.sites.twitch import TwitchChatDownloader
        params = dict(_RETRY, message_types=["text_message"], message_receive_timeout=0.1, buffer_size=4096)
        chat = TwitchChatDownloader()._get_chat_messages_by_stream_id(channel.group(1).lower(), params)
    else:
        chat = _downloader().get_chat(url, message_types=["text_message"], **_RETRY)
    for msg in chat:
        yield msg.get("message") or ""
