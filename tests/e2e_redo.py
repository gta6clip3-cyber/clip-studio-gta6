"""Test de bout en bout sur un vrai passage de VOD : python tests/e2e_redo.py <url> <debut_s> <fin_s>

Telecharge seulement ce passage (+ marge), le fait analyser par l'IA puis cree le clip complet dans output/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipper import config  # noqa: E402
from clipper.highlights import Candidate  # noqa: E402
from clipper.jobs import Job  # noqa: E402
from clipper.pipeline import analyse_moments, finish_clip  # noqa: E402
from clipper.sources import download_section, get_info  # noqa: E402


def main(url: str, a: float, b: float) -> None:
    # jamais de publication pendant un test
    orig = config.load
    config.load = lambda: dict(orig(), publish_mode="manual")
    info = get_info(url)
    job = Job(kind="vod", url=url)
    job.options["stream_title"] = info["title"]
    job.options["stream_context"] = info.get("context", "")
    from clipper import games
    job.options["profile"] = games.detect(info["title"], info.get("context", ""))
    job.say = lambda msg, p=None: print("  >", msg)  # type: ignore[method-assign]
    work = config.WORK / f"e2e_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)
    pad = 15.0
    src = download_section(url, max(0, a - pad), b + pad, work / "src.mp4")
    # la fenetre du fichier = la fenetre "source" (le fichier sert aussi d'audio)
    cand = Candidate(peak=pad + (b - a) * 0.6, score=3.0, start=0.0, end=(b - a) + 2 * pad, reasons="e2e")
    moments = analyse_moments(job, src, [], info["streamer"], config.load(), 1, cands=[cand])
    m = moments[0]
    print("  > moment", round(m.start, 1), "->", round(m.end, 1), m.meta.get("game"), m.meta.get("score"))
    folder = finish_clip(job, m, src, lambda s, e, f: (src, s), info["streamer"], info["platform"],
                         info["webpage_url"], 99)
    print("DOSSIER", folder)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
