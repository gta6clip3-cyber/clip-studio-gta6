import pytest

from clipper import speakers
from fixtures import EN_LINES, FIX, FR_LINES, _load, dialogue

pytestmark = pytest.mark.skipif(not speakers.available(), reason="modeles de diarisation absents")


def _labels(segs, truth):
    out = []
    for a, b, _ in truth:
        ov = {}
        for s, e, spk in segs:
            o = min(b, e) - max(a, s)
            if o > 0:
                ov[spk] = ov.get(spk, 0) + o
        out.append(max(ov, key=ov.get) if ov else None)
    return out


def _consistent(truth, labels):
    """Meme voix -> meme etiquette, voix differentes -> etiquettes differentes."""
    m = {}
    for (_, _, v), lab in zip(truth, labels):
        if lab is None:
            return False
        m.setdefault(v, set()).add(lab)
    if any(len(s) != 1 for s in m.values()):
        return False
    labs = [next(iter(s)) for s in m.values()]
    return len(set(labs)) == len(labs)


@pytest.mark.parametrize("noise", [0.0, 0.02])
def test_three_voices(noise):
    turns = [("A", FR_LINES[0]), ("B", FR_LINES[1]), ("A", FR_LINES[2]), ("C", FR_LINES[3]),
             ("A", FR_LINES[4]), ("B", FR_LINES[5]), ("A", FR_LINES[1])]
    wav, truth = dialogue(f"three_n{noise}", turns, noise=noise)
    speakers._sd = None
    segs = speakers.diarize(_load(wav, None))
    assert _consistent(truth, _labels(segs, truth))


@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_single_voice_stays_one_color(noise):
    wav, truth = dialogue(f"solo_n{noise}", [("A", t) for t in FR_LINES], noise=noise)
    segs = speakers.diarize(_load(wav, None))
    words = []
    for a, b, _ in truth:
        n = 6
        for k in range(n):
            words.append({"word": "x", "start": a + (b - a) * k / n, "end": a + (b - a) * (k + 1) / n})
    speakers.assign(words, segs)
    assert {w["spk"] for w in words} == {0}


def test_english_two_voices():
    turns = [("B", EN_LINES[0]), ("A", EN_LINES[1]), ("B", EN_LINES[2]), ("A", EN_LINES[3])]
    wav, truth = dialogue("en_two", turns, noise=0.01)
    segs = speakers.diarize(_load(wav, None))
    assert _consistent(truth, _labels(segs, truth))


def _turn_words(truth, n=6):
    words = []
    for a, b, v in truth:
        for k in range(n):
            words.append({"word": "x", "start": a + (b - a) * k / n, "end": a + (b - a) * (k + 0.9) / n,
                          "v": v, "eos": k == n - 1})
    return words


def test_label_words_distinct_voices_get_distinct_colors():
    # une vraie 2e personne dit de vraies phrases (>= 2 s) : repliques de 2 phrases
    turns = [("A", FR_LINES[0]), ("B", FR_LINES[1] + " " + FR_LINES[5]), ("A", FR_LINES[2]),
             ("B", FR_LINES[3] + " " + FR_LINES[4]), ("A", FR_LINES[4])]
    wav, truth = dialogue("label_ab2", turns, noise=0.005)
    words = speakers.label_words(wav, 0, truth[-1][1] + 0.5, _turn_words(truth))
    by_voice = {}
    for w in words:
        by_voice.setdefault(w["v"], set()).add(w["spk"])
    assert len(by_voice["A"]) == 1 and len(by_voice["B"]) == 1 and by_voice["A"] != by_voice["B"], by_voice


def test_label_words_single_voice_one_color():
    wav, truth = dialogue("solo_n0.03", [("A", t) for t in FR_LINES], noise=0.03)
    words = speakers.label_words(wav, 0, truth[-1][1] + 0.5, _turn_words(truth))
    assert {w["spk"] for w in words} == {0}


REAL = FIX / "spliced_real.wav"


@pytest.mark.skipif(not REAL.exists(), reason="extrait reel absent")
def test_real_stream_audio_with_inserted_voice():
    """Vrai son du clip Ouah Leouff + Ashtax (deux hommes, meme micro, jeu en fond), avec des voix connues inserees."""
    import json
    words = json.loads((FIX / "spliced_real_words.json").read_text())
    speakers.label_words(REAL, 0, 80, words)
    real_colors = {w["spk"] for w in words if w["real"]}
    assert len(real_colors) == 2  # Ouah et Ashtax sont separes (avant : une seule couleur)
    for v in "AB":  # chaque voix inseree garde UNE couleur, differente des deux streamers
        cols = {w["spk"] for w in words if w.get("v") == v}
        assert len(cols) == 1 and not cols & real_colors, (v, cols, real_colors)
    # l'echange "Tu me fais confiance ?" (Ashtax) / "Oui, je te fais confiance." (Ouah)
    text = [(w["word"], w["spk"]) for w in words if w["real"]]
    i = next(k for k, (t, _) in enumerate(text) if t == "Oui," and text[k + 1][0] == "je")
    assert text[i][1] != text[i - 2][1]  # "confiance ?" (Ashtax) vs "Oui," (Ouah)


def test_assign_main_speaker_is_zero_and_ghosts_merged():
    segs = [(0, 10, 5), (10, 10.4, 7), (10.4, 14, 3), (14, 20, 5)]
    words = [{"word": "w", "start": t / 2, "end": t / 2 + 0.4} for t in range(40)]
    speakers.assign(words, segs)
    assert words[0]["spk"] == 0  # le locuteur 5 parle le plus -> 0
    assert {w["spk"] for w in words} == {0, 1}  # le "7" (0.4 s) est fusionne
    assert all(w["spk"] == 1 for w in words if 11 <= w["start"] < 14)


def test_assign_without_segments():
    words = [{"word": "w", "start": 0, "end": 1}]
    assert speakers.assign(words, [])[0]["spk"] == 0


BENCH = FIX / "bench"


@pytest.mark.skipif(not (BENCH / "mix_trio.wav").exists(), reason="banc d'essai non construit (tests/bench_build.py)")
def test_real_male_voices_benchmark():
    """Vraies voix d'hommes francais (streamers), meme micro, meme bruit de jeu."""
    import bench_eval
    r = bench_eval.evaluate({})
    assert r["solo"] == 1.0, r          # jamais de fausse couleur sur une voix seule
    assert r["pairs"] >= 0.78, r        # ~80 % des mots attribues a la bonne voix


def test_split_by_speaker_cuts_sentences(monkeypatch):
    seg = {"start": 10, "end": 13, "text": "Tu me fais confiance ? Oui je te fais confiance.",
           "words": [{"word": w, "start": 10 + k * 0.3, "end": 10 + k * 0.3 + 0.25}
                     for k, w in enumerate("Tu me fais confiance ? Oui je te fais confiance.".split())]}

    def fake(audio, t0, t1, words, opts=None):
        for k, w in enumerate(words):
            w["spk"] = 0 if k < 5 else 1
        return words
    monkeypatch.setattr(speakers, "label_words", fake)
    out = speakers.split_by_speaker([seg], "x.wav", 10, 20)
    assert [o["spk"] for o in out] == [0, 1]
    assert out[0]["text"] == "Tu me fais confiance ?" and out[1]["text"].startswith("Oui")
    assert out[1]["start"] == seg["words"][5]["start"]


@pytest.mark.skipif(not (FIX / "ashtax_shout.wav").exists(), reason="extrait absent")
def test_streamer_shouting_is_not_a_second_voice():
    """Ashtax seul qui s'emballe ("non, oui, non, les mousquetaires !") : une seule couleur."""
    import json
    words = json.loads((FIX / "ashtax_shout_words.json").read_text(encoding="utf-8"))
    speakers.label_words(FIX / "ashtax_shout.wav", 0, 70, words)
    assert {w["spk"] for w in words} == {0}


@pytest.mark.skipif(not (FIX / "ashtax_excited.wav").exists(), reason="extrait absent")
def test_streamer_getting_excited_is_not_a_second_voice():
    """Ashtax seul, qui s'emballe plusieurs fois pendant la partie : une seule couleur."""
    import json
    words = json.loads((FIX / "ashtax_excited_words.json").read_text(encoding="utf-8"))
    speakers.label_words(FIX / "ashtax_excited.wav", 0, 62, words)
    assert {w["spk"] for w in words} == {0}
