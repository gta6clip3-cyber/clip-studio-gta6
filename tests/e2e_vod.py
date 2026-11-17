"""Test complet d'une VOD comme depuis l'interface : python tests/e2e_vod.py <url> [nb_clips]  (sans publication)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import config  # noqa: E402
from clipper.jobs import Job  # noqa: E402
from clipper.pipeline import run_vod  # noqa: E402

orig = config.load
config.load = lambda: dict(orig(), publish_mode="manual")
job = Job(kind="vod", url=sys.argv[1])
job.options["max_clips"] = int(sys.argv[2]) if len(sys.argv) > 2 else 2
job.say = lambda msg, p=None: print(time.strftime("%H:%M:%S"), ">", msg, flush=True)  # type: ignore
t = time.time()
run_vod(job)
print("CLIPS", job.clips, f"en {time.time() - t:.0f} s")
