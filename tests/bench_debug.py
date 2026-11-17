"""Affiche, pour chaque melange du banc, les indicateurs de separation (pour regler le critere)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import speakers  # noqa: E402
from clipper.transcribe import load_pcm  # noqa: E402

BENCH = Path(__file__).resolve().parent / "_fixtures" / "bench"
models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common"]


def kmeans(E, W, k, iters=30, seeds=12):
    best, best_obj = None, -1e9
    rng = np.random.default_rng(0)
    for s in range(seeds):
        C = E[rng.choice(len(E), k, replace=False)]
        for _ in range(iters):
            lab = np.argmax(E @ C.T, 1)
            newC = []
            for g in range(k):
                m = E[lab == g]
                v = (m * W[lab == g, None]).sum(0) if len(m) else C[g]
                newC.append(v / (np.linalg.norm(v) + 1e-9))
            C = np.array(newC)
        obj = float(sum(W[i] * E[i] @ C[lab[i]] for i in range(len(E))))
        if obj > best_obj:
            best, best_obj = lab.copy(), obj
    return best


rows = []
for wav in sorted(BENCH.glob("mix_*.wav")):
    words = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    pcm = load_pcm(wav, 0, 999)
    runs = speakers.runs_of(words, [None] * len(words))
    E, W, T = [], [], []
    for run in runs:
        a, b = words[run[0]]["start"], words[run[-1]]["end"]
        if b - a >= 0.6:
            E.append(speakers._embed(pcm, a - 0.05, b + 0.05, models))
            W.append(min(b - a, 5.0))
            T.append(words[run[0]]["true"])
    E, W = np.array(E), np.array(W)
    S = E @ E.T
    lab = kmeans(E, W, 2)
    sizes = [W[lab == g].sum() for g in (0, 1)]
    minor = int(np.argmin(sizes))
    major = 1 - minor
    def mw(g):
        idx = np.where(lab == g)[0]
        v = [S[i, j] for i in idx for j in idx if i < j]
        return np.mean(v) if v else float("nan")
    between = np.mean(S[np.ix_(lab == 0, lab == 1)])
    # purete vs verite
    acc = max(np.mean([(l == 0) == (t == T[0]) for l, t in zip(lab, T)]),
              np.mean([(l == 1) == (t == T[0]) for l, t in zip(lab, T)]))
    rows.append((len(set(T)), wav.stem[4:], mw(major), mw(minor), between, sizes[minor], int((lab == minor).sum()), acc))
for n, name, wM, wm, b, sz, nr, acc in sorted(rows):
    print(f"{n} voix {name:14s} intraMaj={wM:.2f} intraMin={wm:.2f} inter={b:.2f} "
          f"cohesion={wm - b:+.2f} min={sz:4.1f}s/{nr} tours  acc={acc:.0%}")
