"""Mesure : l'IA classe-t-elle les vrais moments viraux GTA RP au-dessus des moments ordinaires ?
python tests/gta_eval.py [--test-only]   (AUC : 0.5 = hasard, 1.0 = parfait)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import games, llm  # noqa: E402

DATA = Path(__file__).resolve().parent / "_fixtures" / "gta"
# sources reservees aux exemples donnes a l'IA (jamais utilisees pour la mesure "test")
TRAIN_SOURCES = {"C2V6d99Zewk", "iE2OzgbYtnw"}


def load(test_only: bool = False) -> list[dict]:
    items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DATA.glob("*.json"))]
    items = [it for it in items if sum(len(s["words"]) for s in it["segments"]) >= 10]
    if test_only:
        items = [it for it in items if it["id"] not in TRAIN_SOURCES]
    return items


def text_of(it: dict) -> str:
    return "\n".join(f"[{s['start']:6.1f}-{s['end']:6.1f}] {s['text']}" for s in it["segments"])


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


HINTS = "--no-hints" not in sys.argv
SHOTS = "--shots" in sys.argv


def evaluate(items: list[dict], verbose: bool = False) -> dict:
    pos, neg, found, raw_neg = [], [], [], []
    for it in items:
        j = llm.judge_candidate(text_of(it), 75, "streamer", "chat=unknown, hype=unknown, audio=unknown",
                                "GTA RP live stream (French)",
                                game_hints=(games.judge_hints("gta") + ("\n\n" + games.judge_examples("gta") if SHOTS else ""))
                                if HINTS else "")
        s = j["score"] if j else 5.0
        (pos if it["positive"] else neg).append(s)
        if j and not it["positive"]:
            raw_neg.append(llm.criteria(j))
        found.append(j.get("game", "") if j else "")
        if verbose:
            print(f"  {'POS' if it['positive'] else 'neg'} {it['id']} {it['start']:6d} -> {s:4.1f}  "
                  f"{(j or {}).get('summary', '')[:70]}", flush=True)
    gta = np.mean([("gta" in g.lower() or "grand theft" in g.lower()) for g in found])
    return {"auc": auc(pos, neg), "pos": float(np.mean(pos)), "neg": float(np.mean(neg)),
            "kept_pos": float(np.mean(np.array(pos) >= 6)), "kept_neg": float(np.mean(np.array(neg) >= 6)),
            "game_ok": float(gta), "n": (len(pos), len(neg)),
            # decalage qui met 5/10 la ou 1 moment ordinaire sur 10 arrive (5/10 = criteres reunis a 3.2)
            "severity": round(3.2 - float(np.percentile(raw_neg, 90)), 1) if raw_neg else 0.0}


if __name__ == "__main__":
    r = evaluate(load("--test-only" in sys.argv), verbose=True)
    print(f"AUC {r['auc']:.2f} | note moyenne viral {r['pos']:.1f} vs ordinaire {r['neg']:.1f} | "
          f"gardes (>=6) viral {r['kept_pos']:.0%} / ordinaire {r['kept_neg']:.0%} | jeu reconnu {r['game_ok']:.0%} "
          f"| n={r['n']}")
    from clipper import config  # noqa: E402
    model = config.load()["ollama_model"]
    print(f"Severite de {model} : decalage conseille {r['severity']:+.1f} (regle : {llm.SEVERITY.get(model, 0.0):+.1f}, "
          f"dans SEVERITY de clipper/llm.py). A refaire si on change de modele.")
