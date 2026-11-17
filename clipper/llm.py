"""IA locale via Ollama (gratuit) : choix des meilleurs moments, traduction, titres, legendes."""
from __future__ import annotations

import json
import re

import requests

from . import config
from .utils import log


def available() -> bool:
    cfg = config.load()
    try:
        r = requests.get(cfg["ollama_url"] + "/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
        want = cfg["ollama_model"]
        return any(n == want or n.startswith(want + ":") or n.split(":")[0] == want.split(":")[0] for n in names)
    except Exception:
        return False


_thinks: dict[str, bool] = {}


def _can_think(url: str, model: str) -> bool:
    """Le modele raisonne-t-il avant de repondre (qwen3, qwen3.5, gemma4...) ? Si oui on coupe ce raisonnement :
    il rend chaque reponse plusieurs fois plus lente. C'est Ollama qui le dit, modele par modele."""
    if model not in _thinks:
        try:
            r = requests.post(url + "/api/show", json={"model": model}, timeout=5)
            _thinks[model] = "thinking" in r.json().get("capabilities", [])
        except Exception:
            return False  # Ollama injoignable : on redemandera au prochain appel
    return _thinks[model]


def chat_json(system: str, user: str, temperature: float = 0.4, keep_alive: str = "5m") -> dict | list | None:
    cfg = config.load()
    try:
        r = requests.post(cfg["ollama_url"] + "/api/chat", json={
            "model": cfg["ollama_model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_ctx": 8192},
            **({"think": False} if _can_think(cfg["ollama_url"], cfg["ollama_model"]) else {}),
        }, timeout=600)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)
    except Exception as e:
        log.warning("Ollama : %s", e)
        return None


def unload() -> None:
    cfg = config.load()
    try:
        requests.post(cfg["ollama_url"] + "/api/generate",
                      json={"model": cfg["ollama_model"], "keep_alive": 0}, timeout=10)
    except Exception:
        pass


JUDGE_SYSTEM = """You are an expert short-form video editor who makes viral TikTok / YouTube Shorts / Reels clips
from gaming livestreams. You receive the stream title, a timestamped transcript of a stream window (any language)
plus chat/audio hype signals. Find the single best self-contained moment for a vertical clip and judge it HARSHLY.
Identify the game being played from the stream title, the video tags/description and the game vocabulary in the
transcript. NEVER name a game that is not actually being played (do not assume it is GTA).
If you are not sure, write "game": "unknown".
IMPORTANT calibration: most windows of a long livestream are ordinary. On average only 1 window in 5 deserves a clip.
People talking a lot, or a story going on, is NOT enough: there must be a payoff a stranger would laugh at or react to.
Rate these criteria from 0 to 10:
- "payoff": a clear punchline, twist, fail, kill, escape, or climax happens in the window (0 = nothing happens);
- "reaction": people laugh hard, scream, rage, are shocked or hyped (0 = calm voices);
- "standalone": a stranger understands and enjoys it in 60 s without knowing the story or the characters;
- "boring": how much of it is logistics, menus, waiting, explaining, small talk, inside jokes or mumbling.
Rules:
- The clip must start right before the setup and end right after the payoff/reaction (no dead air, no cut mid-sentence).
- Duration between {min_s} and {max_s} seconds, ideally about {target_s} seconds.
- Times are in seconds relative to the window start (as in the transcript brackets).
- "best_line": quote the single funniest / strongest line of the window (in the original language), or "" if none.
- The punch_time is the single climax second (the scream, the kill, the punchline).
- All text you write MUST be in English, punchy, native social-media tone. No hashtags in the title.
Return JSON: {{"payoff": number, "reaction": number, "standalone": number, "boring": number, "best_line": string,
"start": number, "end": number, "title": string (max 60 chars, curiosity hook),
"hook": string (max 6 words, ALL CAPS, shown on screen during first 3 s),
"caption": string (1-2 sentences in English, can include 1-2 emojis),
"hashtags": [5-8 lowercase tags without #], "punch_time": number (the exact climax second, relative to window start),
"summary": string (what happens, 1 sentence), "game": string (name of the game actually played)}}"""


# ajoute au message seulement quand la transcription porte des notes (voir highlights.reactions)
NOTES_LEGEND = """
Some transcript lines end with <<notes>> measured on the audio and on the live chat (the text cannot show them):
- LOUD = very loud during or right after the line (shouting, laughter or a loud game sound), far above the rest
  of the stream;
- chat: KEKW x12 = how many viewers typed that reaction right after the line (KEKW, LUL, MDR, PTDR = laughing;
  POG, OMG, WTF = shock or hype; CLIP = viewers asking for a clip).
Use the notes to rate "reaction" and to place punch_time.
"""


def combine_score(res: dict) -> float:
    """Note finale 1-10 calculee a partir des criteres (plus stable que la note globale du modele)."""
    def g(k, d=5.0):
        try:
            return max(0.0, min(10.0, float(res.get(k, d))))
        except (TypeError, ValueError):
            return d
    s = 0.42 * g("payoff") + 0.30 * g("reaction") + 0.28 * g("standalone") - 0.30 * g("boring")
    if not str(res.get("best_line", "")).strip():
        s -= 0.8  # pas de phrase forte = rarement un bon clip
    return round(max(1.0, min(10.0, 1.0 + s * 1.25)), 1)


def judge_candidate(transcript_text: str, window_len: float, streamer: str, signals: str,
                    stream_title: str = "", game_hints: str = "") -> dict | None:
    cfg = config.load()
    system = JUDGE_SYSTEM.format(min_s=cfg["clip_min_seconds"], max_s=cfg["clip_max_seconds"],
                                 target_s=cfg["clip_target_seconds"])
    user = (f"Streamer: {streamer}\nStream title: {stream_title or 'unknown'}\nWindow length: {window_len:.0f} s\n"
            f"Hype signals: {signals}\n")
    if game_hints:
        user += f"\nWhat makes a good clip in this game:\n{game_hints}\n"
    if "<<" in transcript_text:
        user += NOTES_LEGEND
    user += f"\nTranscript:\n{transcript_text or '(no speech detected)'}"
    for temp in (0.2, 0.5):  # 2e essai si la reponse est illisible (comme pour la traduction)
        res = chat_json(system, user, temperature=temp)
        if not isinstance(res, dict):
            continue
        try:
            res["start"] = float(res.get("start", 0))
            res["end"] = float(res.get("end", window_len))
            res["score"] = combine_score(res)
            res["punch_time"] = float(res.get("punch_time", (res["start"] + res["end"]) / 2))
        except (TypeError, ValueError):
            continue
        return res
    return None

TRANSLATE_SYSTEM = """You are a professional subtitle translator for gaming livestream clips.
Translate the TARGET line into natural, casual spoken ENGLISH (gamer slang ok, keep swear words mild).
The source is usually French but can be any language; the speech-to-text may contain small errors: fix them
using context. Keep game terms in their usual English name (e.g. "boule de feu" -> "fireball", "cochon" -> "hog rider"
in Clash Royale, "bélier" -> "battle ram", "flic" -> "cop", "PV" -> "HP"). Keep names of people as-is.
When the streamer says "chat" they are talking to their Twitch/YouTube chat: keep "chat" (never "cat").
Interjections (euh, bah, aïe, wesh, putain) become natural English (uh, well, ouch, yo, damn).
You also get the previous lines (already translated) ONLY as context: translate the target line and nothing else
(never repeat the previous lines).
Return JSON {"en": "..."}: English only, never French."""


def translate_lines(lines: list[str], context: str = "") -> list[str] | None:
    """Traduit en anglais ligne par ligne (avec les lignes voisines en contexte) et verifie chaque ligne.

    Ligne par ligne = jamais de decalage ni de doublon entre les lignes. Retourne None si l'IA ne repond pas."""
    from .textlang import is_english
    if not lines:
        return []
    out: list[str] = []
    answered = False
    for i, src in enumerate(lines):
        payload = {
            "context": context,
            "previous": [{"source": lines[k], "english": out[k]} for k in range(max(0, i - 2), i)],
            "target": src,
        }
        got = ""
        for temp in (0.2, 0.6):
            res = chat_json(TRANSLATE_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=temp)
            if res is not None:
                answered = True
            cand = res.get("en") if isinstance(res, dict) else None
            if isinstance(cand, list):
                cand = " ".join(map(str, cand))
            if isinstance(cand, str) and cand.strip() and (is_english(cand) or is_english(src)):
                # garde-fou : une traduction 2x plus longue que l'original = elle a deborde sur les voisines
                if len(cand.split()) <= 2 * len(src.split()) + 3:
                    got = cand.strip()
                    break
        if not answered and i == 0:
            return None
        if not got:
            log.warning("Ligne non traduite : %s", src)
        out.append(got)  # vide plutot qu'une ligne en francais
    return dedupe_lines(out)


def dedupe_lines(out: list[str], n: int = 4) -> list[str]:
    """Retire d'une ligne un morceau de >= n mots deja present dans la ligne precedente (doublon de traduction)."""
    res = list(out)
    for i in range(1, len(res)):
        prev = re.findall(r"[\w']+", res[i - 1].lower())
        grams = {tuple(prev[k:k + n]) for k in range(len(prev) - n + 1)}
        toks = res[i].split()
        norm = [re.sub(r"[^\w']", "", t.lower()) for t in toks]
        keep = [True] * len(toks)
        for k in range(len(toks) - n + 1):
            if tuple(norm[k:k + n]) in grams:
                for j in range(k, k + n):
                    keep[j] = False
        if not all(keep):
            res[i] = " ".join(t for t, kp in zip(toks, keep) if kp)
    return res

META_SYSTEM = """You write the packaging of a viral vertical gaming clip (TikTok / YouTube Shorts / Reels).
You receive the game, the streamer, a one-line summary and the ENGLISH subtitles of the exact clip.
Everything must describe what REALLY happens in this clip (no invented events, no other game).
Style of top clip channels: the title creates curiosity or tension so people watch to the end; it is NOT a dry
description and does not start with the game name. Mention the concrete thing that happens (the unit, the car,
the cops, the 1v4, the betrayal, the scream...). Vary the structure from clip to clip, for example:
a question ("Did He Really Just...?"), a confession ("I Should Never Have Picked This Deck"), a challenge
("Try Not To Laugh At This Defense"), a twist ("It Was Going Perfectly Until..."), a number ("3 Seconds From Disaster").
Bad (too generic, forbidden): "Epic Moment", "Funny Clip", "Insane Play", "Crazy Moment", "Wait For It".
Return JSON {"titles": [3 different title options, max 60 chars each, English, no hashtags, no emojis],
"hooks": [3 different on-screen hooks, 3 to 7 words, ALL CAPS, English, specific to the clip, and EACH hook
MUST contain the streamer's name followed by the concrete event of THIS clip (the unit, the number, the outcome);
never a generic phrase]
"caption": string (1-2 short English sentences, may include 1-2 emojis),
"hashtags": [5-8 lowercase English tags without #, include the game name as one tag]}"""

_GENERIC = {"epic", "moment", "funny", "clip", "insane", "play", "crazy", "wait", "for", "it", "the", "a", "this",
            "what", "happened", "you", "won't", "believe", "best", "ever", "of", "in", "and", "is", "was"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2 and w not in _GENERIC}


def _first3(t: str) -> tuple:
    return tuple(re.findall(r"[a-z0-9']+", t.lower())[:3])


def _pick(options: list[str], ref: set[str], max_len: int, max_words: int | None = None,
          avoid: list[str] | None = None, name: str = "") -> str:
    from .textlang import is_english
    best, best_s = "", -1e9
    for o in options:
        o = str(o).strip().strip('"')
        if not o or not is_english(o) or len(o) > max_len + 15 or re.search(r"[À-ÿ]", o):
            continue  # pas de mot francais / accentue dans un titre anglais
        words = o.split()
        if max_words and not (2 <= len(words) <= max_words):
            continue
        cw = _content_words(o)
        s = 2.0 * len(cw & ref) + 0.5 * len(cw) - (3.0 if not cw else 0)
        if len(o) > max_len:
            s -= 2
        if " - " in o or ":" in o:
            s -= 1.5  # style "Jeu: description" = titre plat
        if avoid and _first3(o) in {_first3(a) for a in avoid if a}:
            s -= 5  # meme debut qu'un autre clip de la video : on varie
        if name and _has_name(o, name):
            s += 3
        if s > best_s:
            best, best_s = o, s
    return best


def _has_name(text: str, streamer: str) -> bool:
    t = text.upper()
    toks = [x for x in re.findall(r"[A-Z0-9]+", streamer.upper()) if len(x) >= 3]
    return bool(toks) and any(x in t for x in toks)


def hook_with_name(hook: str, streamer: str) -> str:
    """L'accroche du debut contient toujours le nom du streamer (et jamais d'emoji : la police ne les affiche pas)."""
    hook = re.sub(r"[^\w\s'!?.,:&%$#@-]", "", hook, flags=re.ASCII).strip().upper()
    hook = re.sub(r"\s{2,}", " ", hook)
    name = re.sub(r"\s+", " ", streamer).strip().upper()
    if not name or _has_name(hook, streamer):
        return hook
    if not hook:
        return f"WATCH {name}"
    return f"{name}: {hook}"


def write_metadata(english_text: str, game: str, streamer: str, summary: str, stream_title: str = "",
                   avoid: list[str] | None = None, style: list[str] | None = None) -> dict | None:
    """Titre / accroche / legende ecrits a partir des sous-titres ANGLAIS du clip final.

    L'IA propose 3 titres et 3 accroches ; on garde les plus precis (qui parlent de ce qui se passe vraiment)."""
    from .textlang import is_english
    user = (f"Game: {game or 'unknown (deduce from the stream title)'}\nStreamer: {streamer}\n"
            f"Stream title: {stream_title or 'unknown'}\nSummary: {summary}\n\nEnglish subtitles:\n{english_text[:3500]}")
    if style:
        user += ("\n\nTitles that went viral for this game (STYLE reference only, never copy them, write about THIS "
                 "clip): " + " | ".join(style))
    if avoid:
        user += ("\n\nTitles already used for other clips of this stream (use a DIFFERENT structure and different first "
                 "words): " + " | ".join(a for a in avoid if a))
    ref = _content_words(english_text + " " + summary + " " + streamer)
    for temp in (0.7, 0.9):
        res = chat_json(META_SYSTEM, user, temperature=temp)
        if not isinstance(res, dict):
            continue
        titles = res.get("titles") or [res.get("title", "")]
        hooks = res.get("hooks") or [res.get("hook", "")]
        title = _pick(titles if isinstance(titles, list) else [titles], ref, 60, avoid=(avoid or []) + (style or []))
        hook = _pick(hooks if isinstance(hooks, list) else [hooks], ref, 48, max_words=7, name=streamer).upper()
        caption = re.sub(r"\s*\|\s*", " ", str(res.get("caption", "")))
        caption = re.sub(r"\s*#\w+", "", caption).strip()  # les hashtags sont ajoutes a part
        tags = [str(t).strip("# ").lower().replace(" ", "") for t in res.get("hashtags", []) if str(t).strip()]
        if not title or not hook or not caption or not is_english(caption):
            continue
        if len(title) > 70:
            title = title[:67].rsplit(" ", 1)[0] + "..."
        return {"title": title, "hook": hook_with_name(hook, streamer), "caption": caption, "hashtags": tags[:8]}
    return None


def fallback_meta(streamer: str, text: str) -> dict:
    """Si Ollama n'est pas dispo : titre/legende generiques."""
    words = re.findall(r"[A-Za-z']+", text)
    hook = " ".join(words[:4]).upper() or "WAIT FOR IT"
    return {
        "title": f"{streamer} can't believe what just happened",
        "hook": hook[:40],
        "caption": f"{streamer} had the craziest moment on stream 😂",
        "hashtags": ["gaming", "streamer", "twitch", "funny", "clips"],
        "summary": "",
        "game": "",
    }
