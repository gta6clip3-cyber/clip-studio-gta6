"""Compare les modeles d'IA locaux sur le test GTA : python tests/gta_compare_models.py qwen2.5:7b qwen3.5:9b"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gta_eval  # noqa: E402

from clipper import config, llm  # noqa: E402

orig = config.load
gta_eval.SHOTS = True
items = gta_eval.load(test_only=True)
for model in sys.argv[1:]:
    config.load = lambda m=model: dict(orig(), ollama_model=m)
    llm.chat_json("Reply JSON {\"ok\": true}", "ping")  # chargement du modele
    t = time.time()
    r = gta_eval.evaluate(items)
    dt = (time.time() - t) / len(items)
    print(f"{model:12s} AUC {r['auc']:.2f} | viral {r['pos']:.1f} vs ordinaire {r['neg']:.1f} | gardes viral "
          f"{r['kept_pos']:.0%} / ordinaire {r['kept_neg']:.0%} | jeu {r['game_ok']:.0%} | {dt:.1f} s par passage",
          flush=True)
    llm.unload()
