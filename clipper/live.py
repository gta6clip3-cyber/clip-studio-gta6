"""Mode LIVE : enregistrement tampon + surveillance chat/audio en temps reel + clips automatiques."""
from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import config
from .audio import SR, loudness_of_pcm
from .chat import hype_weight, live_chat
from .highlights import AFTER, CHAT_DELAY, Candidate, score_timeline
from .jobs import Job
from .pipeline import analyse_moments, finish_clip
from .sources import get_info, platform_of
from .utils import CREATE_NO_WINDOW, log, run, slug, tool, venv_bin


class LiveRecorder:
    """streamlink -> ffmpeg -> segments de 6 s sur disque."""

    SEG = 6

    def __init__(self, url: str, folder: Path):
        self.url = url
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.list_file = folder / "segments.csv"
        self.t0 = time.time()
        self.sl = None
        self.ff = None

    def start(self) -> None:
        self.sl = subprocess.Popen(
            [venv_bin("streamlink"), "--stdout", "--hls-live-edge", "2",
             "--retry-streams", "5", self.url, "1080p60,1080p,936p60,720p60,720p,best"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self.ff = subprocess.Popen(
            [tool("ffmpeg"), "-v", "error", "-i", "pipe:0", "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
             "-f", "segment", "-segment_time", str(self.SEG), "-reset_timestamps", "1",
             "-segment_list", str(self.list_file), "-segment_list_type", "csv",
             str(self.folder / "seg_%06d.ts")],
            stdin=self.sl.stdout, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self.t0 = time.time()

    def alive(self) -> bool:
        return self.ff is not None and self.ff.poll() is None

    def stop(self) -> None:
        for p in (self.sl, self.ff):
            if p and p.poll() is None:
                p.terminate()

    def segments(self) -> list[tuple[Path, float, float]]:
        """Segments termines : (fichier, debut, fin) en secondes depuis le debut de l'enregistrement."""
        if not self.list_file.exists():
            return []
        out = []
        with open(self.list_file, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    p = self.folder / row[0]
                    if p.exists():
                        out.append((p, float(row[1]), float(row[2])))
        return out


def _segment_loudness(path: Path) -> list[float]:
    p = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "s16le", "-"], capture_output=True, creationflags=CREATE_NO_WINDOW)
    pcm = np.frombuffer(p.stdout, dtype=np.int16)
    return [loudness_of_pcm(pcm[i:i + SR]) for i in range(0, len(pcm) - SR // 2, SR)]


def _cut_window(segs: list[tuple[Path, float, float]], start: float, end: float, out: Path) -> float:
    """Assemble les segments couvrant [start, end]. Retourne le temps (enregistrement) du debut du fichier."""
    use = [s for s in segs if s[2] > start and s[1] < end]
    if not use:
        raise RuntimeError("Segments manquants pour ce moment")
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p, _, _ in use), encoding="utf-8")
    run([tool("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", str(out)])
    return use[0][1]


def _follow_chat(job: Job, rec: LiveRecorder, chat_msgs: list, live_over: threading.Event, retry_s: float = 30) -> None:
    """Lit le chat tant que le live tourne, et se reconnecte apres une coupure (reseau, chat ferme)."""
    warned = False
    while not live_over.is_set():
        try:
            # shortcut: live_over n'est vu qu'a l'arrivee d'un message (le lecteur de chat-downloader ne rend pas la
            # main entre deux messages) ; a revoir si des fils restent ouverts apres beaucoup de lives a la suite
            for msg in live_chat(job.url):
                if live_over.is_set():
                    return
                chat_msgs.append((time.time() - rec.t0, msg))
        except Exception as e:
            if not warned:
                job.say(f"Chat live indisponible ({e}) : audio uniquement, nouvel essai toutes les {retry_s:.0f} s")
                warned = True
        live_over.wait(retry_s)


def run_live(job: Job) -> None:
    cfg = config.load()
    info = get_info(job.url)
    streamer = info["streamer"]
    job.streamer = streamer
    job.options["stream_title"] = info["title"]
    job.options["stream_context"] = info.get("context", "")
    from . import games
    # un live Twitch n'annonce pas son jeu : le mode automatique, lui, le connait (job.options["game"])
    job.options["profile"] = games.detect(info["title"], info.get("context", ""), job.options.get("game", ""))
    if not info["is_live"]:
        job.say("Ce lien n'est pas en live actuellement -> traitement comme une rediffusion.")
        from .pipeline import run_vod
        return run_vod(job)
    folder = config.WORK / f"live_{slug(streamer)}_{time.strftime('%Y%m%d_%H%M%S')}"
    rec = LiveRecorder(job.url, folder / "buffer")
    rec.start()
    job.say(f"LIVE : enregistrement de {streamer} demarre", 0.0)

    # --- chat en temps reel
    chat_msgs: list[tuple[float, str]] = []
    live_over = threading.Event()
    threading.Thread(target=_follow_chat, args=(job, rec, chat_msgs, live_over), daemon=True).start()

    loud: dict[int, float] = {}
    done_segs: set[str] = set()
    last_clip_peak = -1e9
    pending: list[threading.Thread] = []
    clip_index = 0
    buffer_s = cfg["live_buffer_minutes"] * 60

    try:
        while not job.stopped:
            time.sleep(3)
            if not rec.alive():
                job.say("Le live est termine (ou le flux s'est coupe).")
                break
            segs = rec.segments()
            for p, s, _ in segs:
                if p.name not in done_segs:
                    done_segs.add(p.name)
                    for k, v in enumerate(_segment_loudness(p)):
                        loud[int(s) + k] = v
            if not loud:
                continue
            now = max(loud)
            job.progress = 0.0
            mins = now / 60
            job.step = f"LIVE {streamer} : {mins:.1f} min enregistrees, {len(chat_msgs)} msgs, {clip_index} clip(s)"

            # menage du tampon
            for p, s, e in segs:
                if e < now - buffer_s:
                    try:
                        p.unlink()
                    except OSError:
                        pass

            if now < 150:  # il faut un peu d'historique pour juger un pic
                continue
            lo = max(0, now - 900)
            arr = np.array([loud.get(i, np.nan) for i in range(lo, now + 1)], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)))
            msgs = [(t - lo, m) for t, m in chat_msgs if lo <= t <= now]
            score, parts = score_timeline(arr, msgs)
            # on cherche un pic entre -45 s et -22 s (il faut la suite du moment)
            a, b = len(score) - 45, len(score) - 22
            if a <= 0:
                continue
            i = a + int(np.argmax(score[a:b]))
            peak = lo + i
            if score[i] < cfg["live_spike_threshold"] or peak - last_clip_peak < cfg["live_cooldown_seconds"]:
                continue
            last_clip_peak = peak
            clip_index += 1
            reasons = ", ".join(f"{k}={float(v[i]):.1f}" for k, v in parts.items())
            job.say(f"PIC detecte a {peak / 60:.1f} min (score {score[i]:.1f}) -> creation du clip {clip_index}")
            w_start, w_end = max(0, peak - 55), peak + 22
            win = folder / f"win_{clip_index}.ts"
            try:
                file_t0 = _cut_window(rec.segments(), w_start - 4, w_end + 2, win)
            except Exception as e:
                job.say(f"Clip ignore : {e}")
                continue
            cand = Candidate(peak=peak - file_t0, score=float(score[i]), start=w_start - file_t0,
                             end=w_end - file_t0, reasons=reasons)
            win_msgs = [(t - file_t0, m) for t, m in chat_msgs if w_start - 5 <= t <= w_end + CHAT_DELAY + AFTER]
            # signaux de la fenetre, mesures sur les 15 dernieres minutes du live (index 0 = debut du fichier)
            win_parts = {k: v[max(0, int(file_t0) - lo):] for k, v in parts.items()}

            def worker(cand=cand, win=win, idx=clip_index, win_msgs=win_msgs, win_parts=win_parts):
                try:
                    moments = analyse_moments(job, win, win_msgs, streamer, cfg, 1, cands=[cand], parts=win_parts)
                    moments = [m for m in moments if m.meta.get("score", 6) >= 6]
                    if not moments:
                        job.say(f"Clip {idx} juge pas assez fort par l'IA, ignore.")
                        return
                    finish_clip(job, moments[0], win, lambda s, e, f: (win, s), streamer,
                                platform_of(job.url), info["webpage_url"], idx)
                except Exception as e:
                    job.say(f"Clip {idx} echoue : {e}")
                    log.exception("live clip")
                finally:
                    try:
                        win.unlink()
                    except OSError:
                        pass

            th = threading.Thread(target=worker, daemon=True)
            th.start()
            pending.append(th)
    finally:
        live_over.set()  # arrete aussi le fil du chat (avant, il continuait apres la fin du live)
        rec.stop()
        for th in pending:
            th.join(timeout=900)
        shutil.rmtree(folder, ignore_errors=True)  # tampon du live supprime
        job.say(f"Live arrete : {len(job.clips)} clip(s) crees.")
