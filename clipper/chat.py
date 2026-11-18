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
        out += _chat_line(line)
    return out


def _chat_line(line: str) -> list[tuple[float, str]]:
    """Une ligne du fichier de chat yt-dlp (rediffusion ou live) -> [(temps, message)]."""
    try:
        row = json.loads(line)
    except ValueError:
        return []
    replay = row.get("replayChatItemAction", {})
    # rediffusion : le temps est dans replayChatItemAction ; live : yt-dlp le met a cote
    ms = replay.get("videoOffsetTimeMsec", row.get("videoOffsetTimeMsec", 0))
    out = []
    for action in replay.get("actions", []):
        item = action.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer", {})
        text = "".join(r.get("text") or _emoji(r.get("emoji", {})) for r in item.get("message", {}).get("runs", []))
        if text.strip():
            out.append((int(ms or 0) / 1000, text))
    return out


def _youtube_live_chat(url: str, stop=None, poll: float = 1.0) -> Iterator[str]:
    """Chat d'un live YouTube en direct : yt-dlp ecrit le chat au fur et a mesure dans un fichier, on le lit
    en continu (chat-downloader ne sait plus lire YouTube). yt-dlp est arrete quand `stop` est leve."""
    import shutil
    import subprocess
    import tempfile
    import time

    from . import config
    from .utils import CREATE_NO_WINDOW
    config.WORK.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="ytlivechat_", dir=config.WORK))
    err = open(folder / "erreurs.txt", "w", encoding="utf-8")
    proc = subprocess.Popen(_ytdlp_base() + ["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                                             "-o", str(folder / "chat"), url],
                            stdout=subprocess.DEVNULL, stderr=err, creationflags=CREATE_NO_WINDOW)
    pos, rest, got = 0, b"", 0
    try:
        while not (stop is not None and stop.is_set()):
            files = sorted(folder.glob("chat.live_chat.json*"))
            data = b""
            if files:
                with open(files[0], "rb") as f:
                    f.seek(pos)
                    data = f.read()
                pos += len(data)
            if data:
                lines = (rest + data).split(b"\n")
                rest = lines.pop()  # derniere ligne peut-etre incomplete
                for line in lines:
                    for _, text in _chat_line(line.decode("utf-8", "replace")):
                        got += 1
                        yield text
            elif proc.poll() is not None:
                if not got:
                    err.flush()
                    why = (folder / "erreurs.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    raise RuntimeError(why[-1] if why else "chat YouTube vide ou desactive")
                return
            else:
                time.sleep(poll)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        err.close()
        for _ in range(10):  # Windows libere le fichier un peu apres l'arret de yt-dlp
            shutil.rmtree(folder, ignore_errors=True)
            if not folder.exists():
                break
            time.sleep(0.5)


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


def live_chat(url: str, stop=None) -> Iterator[str]:
    """Flux de messages d'un live (bloquant)."""
    channel = re.search(r"twitch\.tv/(\w+)/?(?:[?#].*)?$", url)
    if platform_of(url) == "youtube":
        yield from _youtube_live_chat(url, stop)
        return
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
