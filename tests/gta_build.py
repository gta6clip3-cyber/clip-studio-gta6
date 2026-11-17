"""Jeu de donnees GTA RP (FR) pour entrainer / evaluer l'IA :
- POSITIFS : passages de compilations "best of" (moments choisis par des humains car viraux) ;
- NEGATIFS : passages pris au hasard dans des lives GTA RP complets (moments ordinaires, le plus souvent).

python tests/gta_build.py   -> tests/_fixtures/gta/*.json (audio telecharge par petits morceaux puis supprime)"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import transcribe  # noqa: E402
from clipper.sources import _ytdlp_base  # noqa: E402

OUT = Path(__file__).resolve().parent / "_fixtures" / "gta"
WIN = 75

POSITIVE = {  # compilations de moments forts (id YouTube -> duree)
    "C2V6d99Zewk": 1440,   # Best of Anyme - Grand Line
    "jvUpaBRQcVY": 654,    # plus grosses courses-poursuites
    "f_3KGkU_7qE": 510,    # Walid seme les flics
    "GO1CjxTtVSo": 923,    # moments les plus droles de Flashback
    "fhD05Jnjbls": 499,    # best of JP
    "GQLlivhNkh8": 665,    # pires rages Flashland
    "Pge-8Isvbeg": 1023,   # meilleurs moments de l'annee sur Flashback
    "YOZh-2YI2Nk": 1477,   # best of GrandLine
    "AfLqDeUFElk": 1014,   # top 30 des clips GTA RP les plus vus de la semaine
}
NEGATIVE = {  # lives complets (rediffusions)
    "iE2OzgbYtnw": 17033,  # Anyme live Grand Line
    "6g2xs1zTkGU": 18991,  # Domingo Flashback
    "QitACAyuoic": 18432,  # JL Tomy
    "uRNFKrlcTHs": 19847,  # Noto Flashback
    "PUrupovody8": 14511,  # Medja Flashback
    "3-Oa9ZELW0c": 8205,   # Kameto Flashback
}


def windows(vid: str, dur: int, n: int, positive: bool) -> list[int]:
    rng = random.Random(vid)
    if positive:  # repartis dans la compilation (on saute l'intro et la fin)
        usable = dur - 60 - WIN - 30
        return [60 + int(usable * (k + 0.5) / n) for k in range(n)]
    return sorted(rng.randrange(1200, dur - WIN - 600) for _ in range(n))


def fetch(vid: str, start: int) -> Path | None:
    tmp = OUT / f"tmp_{vid}_{start}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(_ytdlp_base() + ["-q", "-f", "bestaudio", "--download-sections", f"*{start}-{start + WIN}",
                                        "-o", str(tmp / "a.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"],
                       check=True, capture_output=True, timeout=600)
        return next((p for p in tmp.iterdir() if p.suffix not in (".part", ".ytdl")), None)
    except Exception as e:
        print("  echec", vid, start, e)
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # compilations : jusqu'a 8 passages qui ne se recouvrent pas ; lives : 10 passages au hasard
    todo = [(v, s, True) for v, d in POSITIVE.items() for s in windows(v, d, min(8, (d - 165) // WIN), True)]
    todo += [(v, s, False) for v, d in NEGATIVE.items() for s in windows(v, d, 10, False)]
    todo = [(v, s, pos, OUT / f"{'pos' if pos else 'neg'}_{v}_{s}.json") for v, s, pos in todo]
    todo = [t for t in todo if not t[3].exists()]
    with ThreadPoolExecutor(6) as pool:  # les telechargements en meme temps, la transcription ensuite
        audios = list(pool.map(lambda t: fetch(t[0], t[1]), todo))
    for (vid, start, pos, out), audio in zip(todo, audios):
        if not audio:
            continue
        tr = transcribe.transcribe(audio, 0, WIN)
        out.write_text(json.dumps({"id": vid, "start": start, "positive": pos, "language": tr["language"],
                                   "segments": tr["segments"]}, ensure_ascii=False), encoding="utf-8")
        n_words = sum(len(s["words"]) for s in tr["segments"])
        print(f"{'POS' if pos else 'neg'} {vid} {start:6d}s  {n_words:3d} mots ({tr['language']})", flush=True)
        for p in audio.parent.iterdir():
            p.unlink()
        audio.parent.rmdir()
    transcribe.unload()


if __name__ == "__main__":
    main()
