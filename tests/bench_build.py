"""Construit un banc d'essai realiste pour la separation des voix : vraies voix d'hommes francais (streams),
alternees comme dans une conversation, passees par le MEME "micro" (filtre + compression) et le meme bruit de jeu.

python tests/bench_build.py     (telecharge ~2 min d'audio par source, une seule fois)"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import speakers, transcribe  # noqa: E402
from clipper.sources import _ytdlp_base  # noqa: E402
from clipper.utils import tool  # noqa: E402

BENCH = Path(__file__).resolve().parent / "_fixtures" / "bench"
SR = 16000
SOURCES = {  # nom -> (url ou recherche, debut en s)
    "ashtax": ("https://www.youtube.com/watch?v=3gAVFuaJz98", 1700),
    "ouah": ("https://www.youtube.com/watch?v=TXKq7JYEptU", 60),
    "s4": ("ytsearch1:squeezie minecraft hardcore", 900),
    "s6": ("ytsearch1:zerator stream fr", 1500),
}


def fetch(name: str, src: str, start: int, dur: int = 150) -> Path:
    out = BENCH / f"src_{name}.wav"
    if out.exists():
        return out
    tmp = BENCH / f"dl_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(_ytdlp_base() + ["-q", "--no-warnings", "-f", "bestaudio", "--download-sections",
                    f"*{start}-{start + dur}", "-o", str(tmp / "a.%(ext)s"), src], check=True)
    f = next(p for p in tmp.iterdir() if p.suffix not in (".part", ".ytdl"))
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(f), "-ac", "1", "-ar", str(SR), str(out)], check=True)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return out


def solo_sentences(name: str, wav: Path) -> list[dict]:
    """Phrases d'une source ou une seule voix domine (verifie par pyannote sur des fenetres de 30 s)."""
    cache = BENCH / f"sent_{name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tr = transcribe.transcribe(wav, 0, 150)
    pcm = transcribe.load_pcm(wav, 0, 150)
    segs = speakers.diarize(pcm)
    tot: dict[int, float] = {}
    for a, b, k in segs:
        tot[k] = tot.get(k, 0) + b - a
    main = max(tot, key=tot.get) if tot else 0
    sents = []
    for s in transcribe.sentences(tr["segments"], max_words=18):
        if len(s["words"]) < 3 or s["end"] - s["start"] < 1.0 or s["end"] - s["start"] > 7:
            continue
        ov = {}
        for a, b, k in segs:
            o = min(b, s["end"]) - max(a, s["start"])
            if o > 0:
                ov[k] = ov.get(k, 0) + o
        if ov and max(ov, key=ov.get) == main and ov[main] / sum(ov.values()) > 0.9:
            sents.append(s)
    cache.write_text(json.dumps(sents, ensure_ascii=False), encoding="utf-8")
    return sents


def _same_mic(x: np.ndarray) -> np.ndarray:
    """Meme chaine audio pour tout le monde : passe-bande 'micro casque' + compression + niveau fixe."""
    p = subprocess.run([tool("ffmpeg"), "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-af", "highpass=f=120,lowpass=f=6500,acompressor=threshold=0.1:ratio=4,dynaudnorm=f=200",
                        "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def build_mix(name: str, voices: list[str], n_turns: int = 14, seed: int = 0, bed_level: float = 0.18) -> None:
    rng = np.random.default_rng(seed)
    pcms = {v: transcribe.load_pcm(BENCH / f"src_{v}.wav", 0, 150) for v in voices}
    pools = {v: list(solo_sentences(v, BENCH / f"src_{v}.wav")) for v in voices}
    for v in voices:
        rng.shuffle(pools[v])
    pieces, words, t = [np.zeros(int(0.3 * SR), np.float32)], [], 0.3
    order = []
    for k in range(n_turns):  # alternance irreguliere, comme une vraie conversation
        v = voices[k % len(voices)] if len(voices) > 1 and rng.random() > 0.25 else (order[-1] if order else voices[0])
        order.append(v)
    for v in order:
        if not pools[v]:
            continue
        s = pools[v].pop()
        a, b = s["start"] - 0.1, s["end"] + 0.1
        x = pcms[v][int(a * SR):int(b * SR)]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.08
        for w in s["words"]:
            words.append({"word": w["word"], "start": round(t + w["start"] - a, 3), "end": round(t + w["end"] - a, 3),
                          "true": v})
        if words:
            words[-1]["eos"] = True
        pieces.append(x.astype(np.float32))
        gap = float(rng.uniform(0.1, 0.5))
        pieces.append(np.zeros(int(gap * SR), np.float32))
        t += len(x) / SR + gap
    audio = np.concatenate(pieces)
    # bruit de jeu commun (bruit rose + "explosions")
    white = rng.standard_normal(len(audio)).astype(np.float32)
    bed = np.convolve(white, np.ones(25) / 25, mode="same")
    bed *= 1 + 3 * (rng.random(len(audio)) > 0.9995).astype(np.float32).cumsum() % 2
    audio = audio + bed_level * 0.08 * bed / (np.sqrt(np.mean(bed ** 2)) + 1e-9)
    audio = np.clip(_same_mic(audio) * 0.5, -1, 1)
    with wave.open(str(BENCH / f"mix_{name}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())
    (BENCH / f"mix_{name}.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    print("mix", name, voices, f"{len(audio) / SR:.0f} s", len(words), "mots")


if __name__ == "__main__":
    BENCH.mkdir(parents=True, exist_ok=True)
    ok = []
    for n, (src, st) in SOURCES.items():
        try:
            fetch(n, src, st)
            print(n, len(solo_sentences(n, BENCH / f"src_{n}.wav")), "phrases solo")
            ok.append(n)
        except Exception as e:  # une source indisponible ne bloque pas le banc
            print(n, "indisponible :", e)
    transcribe.unload()
    seed = 0
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            build_mix(f"{ok[i]}+{ok[j]}", [ok[i], ok[j]], seed=seed)
            seed += 1
    for n in ok:
        build_mix(f"solo_{n}", [n], n_turns=12, seed=seed)
        seed += 1
    if len(ok) >= 3:
        build_mix("trio", ok[:3], n_turns=18, seed=seed)
