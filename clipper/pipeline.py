"""Chaine complete : VOD -> moments forts -> clips montes, avec titre, legende et miniature."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config, games, llm, speakers, transcribe
from .audio import loudness_per_second
from .chat import download_vod_chat
from .highlights import Candidate, pick_candidates, reactions, score_timeline
from .jobs import Job, manager
from .layout import compute_layout
from .render import make_cover, render_clip
from .sfx import pick_music
from .sources import align_offset, download_audio, download_section, get_info
from .subtitles import build_ass, english_words
from .textlang import english_ratio, is_english
from .utils import log, slug

# (debut, fin) dans la source -> (fichier video, decalage dans ce fichier correspondant a `debut`)
VideoProvider = Callable[[float, float, Path], tuple[Path, float]]


@dataclass
class Moment:
    cand: Candidate
    tr: dict                  # transcription de la fenetre (temps source)
    start: float              # bornes finales du clip (temps source)
    end: float
    punch: float | None
    meta: dict
    rank: float


def _clamp_bounds(c: Candidate, j: dict | None, cfg: dict, tr: dict | None = None) -> tuple[float, float, float | None]:
    mn, mx, tgt = cfg["clip_min_seconds"], cfg["clip_max_seconds"], cfg["clip_target_seconds"]
    win = c.end - c.start
    if j and 0 <= j["start"] < j["end"] <= win + 5 and j["end"] - j["start"] >= 8:
        s, e = c.start + j["start"], c.start + j["end"]
        punch = c.start + j.get("punch_time", 0)
    else:  # reponse IA absente ou incoherente -> autour du pic
        s, e, punch = c.peak - tgt * 0.72, c.peak + tgt * 0.28, c.peak
    if e - s < mn:
        pad = (mn - (e - s)) / 2
        s, e = s - pad, e + pad
    if e - s > mx:
        e = s + mx
    if tr:
        s, e = _snap(s, e, tr, mn, mx)
    s = max(0.0, s)
    if punch is not None and not (s < punch < e):
        punch = None
    return s, e, punch


def _snap(s: float, e: float, tr: dict, mn: float, mx: float) -> tuple[float, float]:
    """Cale le debut sur le debut d'une phrase et la fin juste apres une phrase (jamais coupe en plein mot)."""
    segs = tr.get("segments") or []
    if not segs:
        return s, e
    # debut : la phrase en cours a `s` (si elle a commence il y a < 4 s) sinon la prochaine phrase (< 2 s)
    for seg in segs:
        if seg["start"] <= s < seg["end"] and s - seg["start"] < 4:
            s = seg["start"] - 0.25
            break
    else:
        nxt = [seg["start"] for seg in segs if s <= seg["start"] <= s + 2]
        if nxt:
            s = nxt[0] - 0.25
    # fin : si on coupe une phrase, on va jusqu'a sa fin (+ un peu de reaction), dans la limite de mx
    for seg in segs:
        if seg["start"] < e < seg["end"]:
            if seg["end"] + 0.6 - s <= mx:
                e = seg["end"] + 0.6
            elif seg["start"] - s >= mn:
                e = seg["start"] - 0.1
            break
    return s, e


def _title_with_hints(job: Job) -> str:
    title = job.options.get("stream_title", "")
    ctx = job.options.get("stream_context", "")
    return f"{title} (video tags / description: {ctx[:400]})" if ctx else title


def analyse_moments(job: Job, audio: Path, messages: list, streamer: str, cfg: dict,
                    max_clips: int, cands: list[Candidate] | None = None, parts: dict | None = None) -> list[Moment]:
    """Transcrit + fait juger par l'IA chaque candidat, et renvoie les meilleurs moments.

    `parts` : signaux par seconde (voir score_timeline) alignes sur `audio`. Le mode direct les fournit avec ses
    candidats : ils y sont mesures sur 15 min d'historique, pas sur la seule fenetre du clip."""
    if cands is None:
        loud = loudness_per_second(audio, audio.with_suffix(".loud.npy"))
        score, parts = score_timeline(loud, messages)
        # toujours au moins 2x plus de candidats que de clips demandes, sinon l'IA n'a rien a trier
        cands = pick_candidates(score, parts, max(cfg["candidates_to_analyze"], 2 * max_clips + 2))
    job.say(f"{len(cands)} moments candidats trouves, transcription...", 0.35)
    if not cands:
        return []
    trs = []
    lang = None
    votes: dict[str, int] = {}
    with manager.gpu_lock:
        llm.unload()  # l'IA et Whisper ne tiennent pas ensemble dans 8 Go : sans ca, Whisper passe sur le processeur
        for i, c in enumerate(cands):
            if job.stopped and job.kind != "live":  # live arrete : le clip deja commence est fini et publie
                return []
            tr = transcribe.transcribe(audio, c.start, c.end, language=lang)
            trs.append(tr)
            if lang is None:
                # langue du stream = vote des premieres fenetres (une seule fenetre musicale ne decide plus)
                n_words = sum(len(s["words"]) for s in tr["segments"])
                if n_words >= 15:
                    votes[tr["language"]] = votes.get(tr["language"], 0) + n_words
                if sum(votes.values()) >= 120 or len(votes) and i >= 2:
                    lang = max(votes, key=votes.get)
                    job.options["language"] = lang
            job.say(f"Transcription {i + 1}/{len(cands)} ({tr['language']})", 0.35 + 0.25 * (i + 1) / len(cands))
        transcribe.unload()

    use_llm = llm.available()
    if not use_llm:
        job.say("IA locale (Ollama) indisponible : selection basee sur les signaux uniquement.")
    moments: list[Moment] = []
    max_sig = max(c.score for c in cands) or 1
    for i, (c, tr) in enumerate(zip(cands, trs)):
        if job.stopped and job.kind != "live":
            return []
        j = None
        if use_llm:
            with manager.gpu_lock:
                prof = job.options.get("profile")
                # l'IA lit le texte + ce qu'il ne montre pas : phrases criees, reactions du chat
                notes = reactions(tr["segments"], parts, messages) if parts else None
                script = transcribe.as_text(tr, c.start, notes)
                j = llm.judge_candidate(script, c.end - c.start, streamer, c.reasons,
                                        _title_with_hints(job),
                                        game_hints=(games.judge_hints(prof) + "\n\n" + games.judge_examples(prof)).strip())
            job.say(f"Analyse IA {i + 1}/{len(cands)} : note {j['score'] if j else '?'} - "
                    f"{(j or {}).get('summary', '')[:80]}", 0.6 + 0.15 * (i + 1) / len(cands))
        s, e, punch = _clamp_bounds(c, j, cfg, tr)
        text = " ".join(seg["text"] for seg in tr["segments"])
        meta = j if j else llm.fallback_meta(streamer, text)
        ai = llm.combine_score(j, cfg["ollama_model"], clip=False) if j else 5.0  # sans plafond : pas d'ex aequo a 10
        if sum(len(seg["words"]) for seg in tr["segments"]) < 8:
            ai -= 1.5  # presque personne ne parle : rarement un bon clip
        rank = 0.7 * ai + 3.0 * (c.score / max_sig)
        moments.append(Moment(c, tr, s, e, punch, meta, rank))
    # jeu du stream = vote de toutes les analyses (evite qu'un clip parle d'un autre jeu)
    found_games: dict[str, int] = {}
    for m in moments:
        g = str(m.meta.get("game", "")).strip()
        if g and g.lower() not in ("unknown", "n/a", "none"):
            found_games[g] = found_games.get(g, 0) + 1
    if found_games:
        game = max(found_games, key=found_games.get)
        job.options["game"] = game
        job.options["profile"] = job.options.get("profile") or games.detect(game)
        for m in moments:
            m.meta["game"] = game
    moments.sort(key=lambda m: m.rank, reverse=True)
    good = [m for m in moments if (m.meta.get("score", 6) >= 5)] or moments[:1]
    # pas deux clips qui se recouvrent
    picked: list[Moment] = []
    for m in good:
        if all(min(m.end, p.end) - max(m.start, p.start) < 10 for p in picked):
            picked.append(m)
    return picked[:max_clips]


def _caption(meta: dict, streamer: str, platform: str, stream_title: str = "", job_profile: str | None = None) -> str:
    tags = [re.sub(r"[^a-z0-9]", "", str(t).lower()) for t in meta.get("hashtags", []) if t]
    tags = [t for t in dict.fromkeys(tags) if 2 < len(t) <= 30]
    cfg = config.load()
    context = f"{meta.get('game', '')} {stream_title}".lower()
    is_target = any(k in context for k in cfg["game_keywords"])
    if not is_target:
        tags = [t for t in tags if not t.startswith("gta")]
    game = str(meta.get("game", "")).lower()
    game_tag = "" if game in ("unknown", "n/a", "none") else "".join(ch for ch in game if ch.isalnum())
    extra = ["gta6", "gtavi"] if is_target else ([game_tag] if game_tag else [])
    if job_profile == "gta" and not is_target:
        extra = ["gta", "gtarp" if re.search(r"\brp\b|roleplay|fivem", context) else "gta5"]
    for t in (*extra, re.sub(r"[^a-z0-9]", "", streamer.lower())):
        if t and t not in tags:
            tags.append(t)
    plat = {"twitch": "Twitch", "youtube": "YouTube", "kick": "Kick"}.get(platform, "stream")
    return (f"{meta.get('caption', '').strip()}\n\n🎥 {streamer} on {plat}\n\n"
            + " ".join("#" + t for t in tags[:10]))


def finish_clip(job: Job, m: Moment, audio: Path, video: VideoProvider, streamer: str,
                platform: str, source_url: str, index: int) -> Path:
    """Traduit, telecharge/coupe la video, monte, cree miniature + textes."""
    cfg = config.load()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = config.OUTPUT / f"{stamp}_{slug(streamer, 20)}_{index}"
    folder.mkdir(parents=True, exist_ok=True)
    dur = m.end - m.start

    # --- sous-titres anglais (toujours verifies ligne par ligne : jamais de francais a l'ecran)
    segs = transcribe.sentences([s for s in m.tr["segments"] if s["end"] > m.start and s["start"] < m.end])
    # --- qui parle ? (sur les mots d'origine, aux horaires exacts) puis une phrase = une seule voix
    if cfg.get("speaker_colors", True):
        segs = speakers.split_by_speaker(segs, audio, m.start, m.end)
        n_voices = len({s.get("spk", 0) for s in segs})
        if n_voices > 1:
            job.say(f"Clip {index} : {n_voices} voix differentes -> sous-titres en {n_voices} couleurs")
    lines = [s["text"] for s in segs]
    game = m.meta.get("game") or job.options.get("game", "")
    translations = None
    if m.tr["language"] != "en" or english_ratio(lines) < 0.97:
        if llm.available():
            with manager.gpu_lock:
                translations = llm.translate_lines(lines, context=f"Game: {game or 'unknown'}. "
                                                   f"Livestream of {streamer}. "
                                                   + games.glossary_text(job.options.get("profile")))
        if translations is None:
            job.say("IA locale indisponible : traduction via Whisper")
            with manager.gpu_lock:
                tr_en = transcribe.transcribe(audio, m.start, m.end, language=m.tr["language"], task="translate")
                transcribe.unload()
            segs = [s for s in tr_en["segments"] if is_english(s["text"])]
        else:
            bad = sum(1 for t in translations if not t)
            if bad:
                job.say(f"Clip {index} : {bad} ligne(s) intraduisible(s) retiree(s) des sous-titres")
    words = english_words(segs, translations, m.start, m.end)
    english_text = " ".join(translations) if translations is not None else " ".join(s["text"] for s in segs)

    # --- titre / accroche / legende ecrits a partir des sous-titres anglais du clip final
    if llm.available():
        with manager.gpu_lock:
            pm = llm.write_metadata(english_text, game, streamer, m.meta.get("summary", ""),
                                    job.options.get("stream_title", ""),
                                    avoid=job.options.setdefault("titles_done", []),
                                    style=games.title_examples(job.options.get("profile")))
        if pm:
            m.meta.update(pm)
    m.meta["hook"] = llm.hook_with_name(m.meta.get("hook", ""), streamer)
    job.options.setdefault("titles_done", []).append(m.meta.get("title", ""))
    for key in ("title", "hook", "caption"):
        if m.meta.get(key) and not is_english(str(m.meta[key])):
            m.meta[key] = llm.fallback_meta(streamer, english_text)[key]
    with manager.gpu_lock:
        llm.unload()

    # --- video source
    job.say(f"Clip {index} : recuperation de la video ({dur:.0f} s)")
    src, offset = video(m.start, m.end, folder)
    layout = compute_layout(src, streamer)
    cfg_h = cfg["height"]
    if layout.has_cam:
        sub_y = layout.panel_y + int(layout.panel_h * 0.62)
        hook_y = layout.panel_y + 150
        credit_y = layout.panel_y + 26
    else:
        sub_y = layout.fg_y + layout.fg_h + 150
        hook_y = max(140, layout.fg_y - 150)
        credit_y = layout.fg_y + 20
    sub_y = min(sub_y, cfg_h - 420)  # zone sure (interface TikTok en bas)
    # plusieurs voix : la voix principale en bas, les autres plus haut (lisibles en meme temps)
    if layout.has_cam:
        lanes = {0: sub_y, 1: layout.panel_y + int(layout.panel_h * 0.30), 2: layout.panel_y + int(layout.panel_h * 0.46)}
    else:
        lanes = {0: sub_y, 1: layout.fg_y + int(layout.fg_h * 0.22), 2: layout.fg_y + int(layout.fg_h * 0.45)}

    meta = m.meta
    ass = build_ass(folder / "subs.ass", words, dur, meta.get("hook", ""), streamer, sub_y, hook_y, credit_y,
                    lanes=lanes)
    punch_rel = (m.punch - m.start) if m.punch is not None else None
    out = folder / "clip.mp4"
    job.say(f"Clip {index} : montage...")
    with manager.gpu_lock:
        render_clip(src, offset, dur, layout, ass, punch_rel, out, pick_music(index + int(m.start)))
    cover_t = offset + (punch_rel if punch_rel is not None else dur * 0.4)
    make_cover(src, cover_t, layout, meta.get("title", ""), folder / "cover.jpg")

    title = meta.get("title", "").strip() or f"{streamer} can't believe this"
    caption = _caption(meta, streamer, platform, job.options.get("stream_title", ""), job.options.get("profile"))
    (folder / "title.txt").write_text(title, encoding="utf-8")
    (folder / "caption.txt").write_text(caption, encoding="utf-8")
    info = {
        "title": title, "caption": caption, "hook": meta.get("hook", ""), "hashtags": meta.get("hashtags", []),
        "summary": meta.get("summary", ""), "score": meta.get("score"), "streamer": streamer,
        "platform": platform, "source_url": source_url, "source_start": round(m.start, 1),
        "source_end": round(m.end, 1), "duration": round(dur, 1), "created": stamp, "job": job.id,
        "language": m.tr["language"], "published": {},
    }
    (folder / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    job.clips.append(folder.name)
    job.say(f"Clip {index} pret : {title}")

    if cfg["publish_mode"] == "auto":
        from . import ytqueue
        from .publish import publish_clip
        platforms = [p for p in ("youtube", "tiktok", "instagram") if cfg.get(f"publish_{p}")]
        for q in ytqueue.PLATFORMS:
            if q in platforms and ytqueue.enabled(cfg, q):
                # file d'attente : un clip a chaque heure prevue (une rafale fait passer le compte pour un robot)
                platforms.remove(q)
                ytqueue.add(folder, q)
                st = ytqueue.status(q)
                job.say(f"{q} : en file ({len(st['waiting'])} en attente, prochaine publication a {st['next']})")
        try:
            res = publish_clip(folder, platforms)
            job.say(f"Publication : {res}")
        except Exception as e:
            job.say(f"Publication echouee : {e}")
    if cfg.get("tiktok_auto_draft"):
        # TikTok interdit la publication automatique : le clip va dans les brouillons, tu le publies d'un clic
        from .publish import draft_clip
        job.say(f"Brouillon TikTok : {draft_clip(folder)['tiktok_draft']}")
    return folder


def run_vod(job: Job) -> None:
    cfg = config.load()
    max_clips = int(job.options.get("max_clips") or cfg["max_clips_per_vod"])
    job.say("Lecture des informations de la video...", 0.02)
    info = get_info(job.url)
    if info["is_live"]:
        job.say("C'est un live en cours -> bascule en mode live.")
        from .live import run_live
        return run_live(job)
    streamer = info["streamer"]
    job.streamer = streamer
    job.options["stream_title"] = info["title"]
    job.options["stream_context"] = info.get("context", "")
    job.options["profile"] = games.detect(info["title"], info.get("context", ""))
    work =config.WORK / f"{info['platform']}_{slug(info['id'] or job.id)}_{job.id}"
    work.mkdir(parents=True, exist_ok=True)
    job.say(f"{streamer} - {info['title'][:70]} ({info['duration'] / 3600:.1f} h)", 0.04)
    try:
        _process_vod(job, info, work, cfg, max_clips)
    finally:
        # la rediffusion (audio, chat, analyses) est supprimee des que les clips sont faits
        shutil.rmtree(work, ignore_errors=True)
        job.say("Fichiers temporaires de la rediffusion supprimes.")


def _process_vod(job: Job, info: dict, work: Path, cfg: dict, max_clips: int) -> None:
    streamer = info["streamer"]
    messages: list = []

    def get_chat():
        messages.extend(download_vod_chat(job.url, work / "chat.json", info["duration"]))

    t = threading.Thread(target=get_chat, daemon=True)
    t.start()
    job.say("Telechargement de l'audio + du chat...", 0.06)
    audio = download_audio(job.url, work)
    job.say("Analyse du volume sonore...", 0.2)
    loudness_per_second(audio, audio.with_suffix(".loud.npy"))
    t.join()
    job.say(f"{len(messages)} messages de chat recuperes", 0.3)
    if job.stopped:
        return

    moments = analyse_moments(job, audio, messages, streamer, cfg, max_clips)
    if not moments:
        job.say("Aucun moment fort trouve.")
        return

    # les passages video se telechargent en avance (2 a la fois) pendant que le GPU traduit et monte les clips
    pad = 4.0
    pool = ThreadPoolExecutor(max_workers=2)
    srcs = [pool.submit(download_section, job.url, m.start - pad, m.end + 2.0, work / f"src_{i}.mp4")
            for i, m in enumerate(moments, 1)]
    try:
        for i, m in enumerate(moments, 1):
            if job.stopped:
                return

            def provider(s: float, e: float, folder: Path, fut=srcs[i - 1]) -> tuple[Path, float]:
                src = fut.result()
                return src, (align_offset(audio, s, src, search=6.0) if s > pad else s)

            job.progress = 0.75 + 0.25 * (i - 1) / len(moments)
            try:
                finish_clip(job, m, audio, provider, streamer, info["platform"], info["webpage_url"], i)
            except Exception as e:
                for d in config.OUTPUT.glob(f"*_{slug(streamer, 20)}_{i}"):
                    if d.is_dir() and not (d / "clip.mp4").exists():
                        shutil.rmtree(d, ignore_errors=True)
                job.say(f"Clip {i} echoue : {e}")
                log.exception("clip")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # arret demande : on n'attend que les telechargements en cours
    job.say(f"Termine : {len(job.clips)} clip(s) dans le dossier output.", 1.0)
