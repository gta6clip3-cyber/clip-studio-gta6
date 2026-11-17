"""Compare les modeles d'empreinte vocale sur le banc (separabilite meme voix / voix differentes, par clip)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sherpa_onnx as so  # noqa: E402

from clipper.transcribe import load_pcm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "_fixtures" / "bench"
MODELS = {"wespeaker_r34": ROOT / "assets/models/wespeaker_en_voxceleb_resnet34_LM.onnx",
          **{p.stem: p for p in sorted((ROOT / "assets/models/bench").glob("*.onnx"))}}

_den = None


def denoise(x: np.ndarray) -> np.ndarray:
    global _den
    if _den is None:
        cfg = so.OfflineSpeechDenoiserConfig(model=so.OfflineSpeechDenoiserModelConfig(
            gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=str(ROOT / "assets/models/gtcrn_simple.onnx"))))
        _den = so.OfflineSpeechDenoiser(cfg)
    r = _den(x.astype(np.float32), 16000)
    return np.asarray(r.samples, dtype=np.float32)


def runs(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w.get("eos"):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    mixes = sorted(p.stem[4:] for p in BENCH.glob("mix_*.wav"))
    pairs = [m for m in mixes if not m.startswith("solo")]
    for den in (False, True):
        for name, path in MODELS.items():
            ext = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=4))
            aucs, same_all, diff_all = [], [], []
            for m in pairs:
                words = json.loads((BENCH / f"mix_{m}.json").read_text(encoding="utf-8"))
                pcm = load_pcm(BENCH / f"mix_{m}.wav", 0, 999)
                if den:
                    pcm = denoise(pcm)
                E, lab = [], []
                for r in runs(words):
                    a, b = r[0]["start"], r[-1]["end"]
                    st = ext.create_stream()
                    st.accept_waveform(16000, pcm[int(a * 16000):int(b * 16000)])
                    st.input_finished()
                    e = np.asarray(ext.compute(st))
                    E.append(e / np.linalg.norm(e))
                    lab.append(r[0]["true"])
                S = np.array(E) @ np.array(E).T
                same = [S[i, j] for i in range(len(E)) for j in range(i + 1, len(E)) if lab[i] == lab[j]]
                diff = [S[i, j] for i in range(len(E)) for j in range(i + 1, len(E)) if lab[i] != lab[j]]
                aucs.append(auc(same, diff))
                same_all += same
                diff_all += diff
            print(f"{'denoise' if den else 'raw    '} {name:55s} AUC/clip={np.nanmean(aucs):.3f} "
                  f"min={np.nanmin(aucs):.3f} | global={auc(same_all, diff_all):.3f} "
                  f"same={np.mean(same_all):.2f} diff={np.mean(diff_all):.2f}", flush=True)


if __name__ == "__main__":
    main()
