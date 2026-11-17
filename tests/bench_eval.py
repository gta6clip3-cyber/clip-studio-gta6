"""Evalue la separation des voix sur le banc (vraies voix d'hommes, meme micro) : python tests/bench_eval.py"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import speakers  # noqa: E402

BENCH = Path(__file__).resolve().parent / "_fixtures" / "bench"

_emb_cache: dict = {}
_orig_embed = speakers._embed
_orig_diar = speakers.diarize
_diar_cache: dict = {}


def _cached_embed(pcm, a, b, models=None):
    key = (len(pcm), float(pcm[: 16000].sum()), round(a, 3), round(b, 3), tuple(models or speakers.emb_models()))
    if key not in _emb_cache:
        _emb_cache[key] = _orig_embed(pcm, a, b, models)
    return _emb_cache[key]


def _cached_diar(pcm, threshold=None):
    key = (len(pcm), float(pcm[: 16000].sum()))
    if key not in _diar_cache:
        _diar_cache[key] = _orig_diar(pcm, threshold)
    return _diar_cache[key]


speakers._embed = _cached_embed
speakers.diarize = _cached_diar


def accuracy(true: list[str], pred: list[int]) -> float:
    ts, ps = sorted(set(true)), sorted(set(pred))
    best = 0
    for perm in itertools.permutations(ps + [None] * max(0, len(ts) - len(ps)), len(ts)):
        m = dict(zip(perm, ts))
        best = max(best, sum(1 for t, p in zip(true, pred) if m.get(p) == t))
    return best / len(true)


def evaluate(opts: dict, verbose: bool = False) -> dict:
    pair_acc, solo_pur, found = [], [], 0
    n_pairs = 0
    for wav in sorted(BENCH.glob("mix_*.wav")):
        name = wav.stem[4:]
        words = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        ws = speakers.label_words(wav, 0, 999, [dict(w) for w in words], opts)
        pred = [w["spk"] for w in ws]
        true = [w["true"] for w in ws]
        if name.startswith("solo"):
            solo_pur.append(pred.count(0) / len(pred))
            if verbose:
                print(f"   {name:18s} pureté {solo_pur[-1]:.0%}")
        else:
            n_pairs += 1
            acc = accuracy(true, pred)
            pair_acc.append(acc)
            found += len(set(pred)) > 1
            if verbose:
                print(f"   {name:18s} précision {acc:.0%}  voix trouvées {len(set(pred))}/{len(set(true))}")
    return {"pairs": float(np.mean(pair_acc)), "pairs_min": float(np.min(pair_acc)), "found": f"{found}/{n_pairs}",
            "solo": float(np.mean(solo_pur)), "solo_min": float(np.min(solo_pur))}


if __name__ == "__main__":
    import logging
    logging.getLogger("clipper").setLevel(logging.WARNING)
    variants = []
    for models in (["nemo_en_titanet_large"],):
        for mi in (0.43, 0.46, 0.49, 0.52):
            variants.append({"models": models, "max_inter": mi})
    for v in variants:
        r = evaluate(v)
        print(f"{'+'.join(m[:12] for m in v['models']):28s} max_inter={v['max_inter']:.2f} -> paires {r['pairs']:.0%} "
              f"(min {r['pairs_min']:.0%}, séparées {r['found']}) | solo {r['solo']:.0%} (min {r['solo_min']:.0%})",
              flush=True)
