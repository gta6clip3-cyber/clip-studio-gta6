import re
import subprocess

from clipper.subtitles import (SPEAKER_COLORS, build_ass, chunk_times, clean_words, english_words,
                               make_chunks)
from clipper.utils import tool


def W(word, s, e, spk=0, **kw):
    return {"word": word, "start": s, "end": e, "spk": spk, **kw}


def _events(path):
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("Dialogue: 1,"):
            f = line.split(",", 9)
            out.append((_sec(f[1]), _sec(f[2]), f[9]))
    return out


def _sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fast_speech():
    # parole tres rapide + chevauchements comme dans les vrais clips (Whisper)
    t, words = 0.26, []
    for k, w in enumerate("oh it's beautiful let's go let's go man it's the moment ok let's push".split()):
        words.append(W(w, t, t + 0.09))
        t += 0.11 if k % 4 else 0.5
    words.append(W("does", 9.03, 9.10))
    words.append(W("he", 9.02, 9.18))  # desordre + chevauchement
    return words


def test_clean_words_no_overlap_sorted():
    ws = clean_words(fast_speech() + [W("...", 3, 3.1), W("#", 4, 4.1)])
    assert all(a["start"] < b["start"] for a, b in zip(ws, ws[1:]))
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(ws, ws[1:]))
    assert not any(w["word"] in ("...", "#") for w in ws)


def test_events_never_overlap_and_never_flash(tmp_path):
    p = build_ass(tmp_path / "s.ass", fast_speech(), 12.0, "HOOK", "Streamer", 1400, 800, 700)
    ev = _events(p)
    assert ev
    for (s1, e1, _), (s2, e2, _) in zip(ev, ev[1:]):
        assert e1 <= s2 + 1e-6, "deux sous-titres affiches en meme temps"
    # duree mini (sauf si le suivant arrive vraiment tres vite)
    short = [e - s for s, e, _ in ev if e - s < 0.3]
    assert len(short) <= 1, short


def test_one_event_per_chunk_with_smooth_highlight(tmp_path):
    words = [W("we", 0.0, 0.3), W("did", 0.3, 0.6), W("it", 0.6, 0.9, eos=True), W("let's", 1.0, 1.3), W("go", 1.3, 1.8)]
    p = build_ass(tmp_path / "s.ass", words, 3.0, "", "", 1400, 800, 700)
    ev = _events(p)
    assert len(ev) == 2  # 1 evenement par groupe (avant : 1 par mot -> clignotement)
    assert "\\t(" in ev[0][2] and "\\fad(" in ev[0][2]
    # enchainement sans trou entre les 2 groupes (pause de 0.1 s)
    assert abs(ev[0][1] - ev[1][0]) < 0.02


def test_chunks_split_on_speaker_change():
    words = [W("hey", 0, 0.2, 0), W("look", 0.2, 0.4, 0), W("what", 0.45, 0.6, 1), W("bro", 0.6, 0.8, 1)]
    ch = make_chunks(words)
    assert [[w["word"] for w in c] for c in ch] == [["hey", "look"], ["what", "bro"]]


def test_chunks_do_not_end_on_weak_word():
    words = [W(w, i * 0.2, i * 0.2 + 0.18) for i, w in enumerate("i saw the dragon coming".split())]
    ch = make_chunks(words)
    assert all(c[-1]["word"] not in ("the", "a") for c in ch[:-1]), [[w["word"] for w in c] for c in ch]


def test_speaker_colors_used(tmp_path):
    words = [W("hey", 0, 0.3, 0), W("man", 0.3, 0.6, 0, eos=True), W("what", 0.8, 1.0, 1), W("up", 1.0, 1.3, 1)]
    ev = _events(build_ass(tmp_path / "s.ass", words, 2.0, "", "", 1400, 800, 700))
    assert SPEAKER_COLORS[0][0] in ev[0][2]
    assert SPEAKER_COLORS[1][0] in ev[1][2]


def test_smaller_font(tmp_path):
    txt = build_ass(tmp_path / "s.ass", [W("hi", 0, 1)], 1.0, "", "", 1400, 800, 700).read_text(encoding="utf-8-sig")
    size = int(re.search(r"Style: Sub,[^,]+,(\d+),", txt).group(1))
    assert size <= 72


def test_translation_follows_speech_rhythm():
    seg = {"start": 0, "end": 4, "text": "x", "words": [
        {"word": "a", "start": 0.0, "end": 0.5}, {"word": "b", "start": 3.0, "end": 3.5}]}
    ws = english_words([seg], ["hello there"], 0, 10)
    # le 2e mot anglais doit tomber sur la 2e partie de la parole (apres le silence), pas au milieu du silence
    assert ws[1]["start"] >= 2.9


def test_libass_renders(tmp_path):
    words = [W(w, i * 0.3, i * 0.3 + 0.28, spk=i // 4) for i, w in enumerate(
        "this is a test of the subtitles with two people talking here".split())]
    p = build_ass(tmp_path / "s.ass", words, 4.0, "WAIT FOR IT", "Tester", 1400, 800, 700)
    out = tmp_path / "f.png"
    r = subprocess.run([tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:d=4",
                        "-vf", "ass=s.ass", "-ss", "1.0", "-frames:v", "1", str(out)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 1000


def test_chunk_times_bridge_small_gaps():
    ch = make_chunks([W("one", 0, 0.3, eos=True), W("two", 0.5, 0.8, eos=True), W("three", 3.0, 3.3)])
    t = chunk_times(ch, 5.0)
    assert t[0][1] == t[1][0]  # petite pause -> pas de trou
    assert t[1][1] < 3.0      # longue pause -> le texte disparait


def _events_all(path):
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("Dialogue:") and ",Sub," in line:
            f = line.split(",", 9)
            y = int(re.search(r"\\move\(\d+,\d+,\d+,(\d+)", f[9]).group(1))
            out.append((_sec(f[1]), _sec(f[2]), y, f[9]))
    return out


def test_two_voices_on_two_lines(tmp_path):
    words = [W("you", 0.0, 0.3, 0), W("trust", 0.3, 0.6, 0), W("me?", 0.6, 0.9, 0, eos=True),
             W("yes", 1.0, 1.3, 1), W("i", 1.3, 1.4, 1), W("do", 1.4, 1.7, 1, eos=True),
             W("okay", 2.0, 2.4, 0)]
    p = build_ass(tmp_path / "s.ass", words, 4.0, "", "", 1400, 800, 700, lanes={0: 1400, 1: 1000})
    ev = _events_all(p)
    ys = {y for _, _, y, _ in ev}
    assert ys == {1400, 1000}  # chaque voix a sa ligne
    a = [e for e in ev if e[2] == 1400]
    b = [e for e in ev if e[2] == 1000]
    assert SPEAKER_COLORS[1][0] in b[0][3]
    # la question reste visible pendant la reponse (les deux en meme temps)
    assert a[0][1] > b[0][0]
    # sur une meme ligne, jamais deux textes superposes
    for lane in (a, b):
        for (s1, e1, _, _), (s2, e2, _, _) in zip(lane, lane[1:]):
            assert e1 <= s2 + 1e-6


def test_single_voice_ignores_lanes(tmp_path):
    words = [W("hello", 0, 0.4), W("there", 0.4, 0.8)]
    ev = _events_all(build_ass(tmp_path / "s.ass", words, 2.0, "", "", 1400, 800, 700, lanes={0: 1400, 1: 1000}))
    assert {y for _, _, y, _ in ev} == {1400}


def test_hook_always_has_streamer_name():
    from clipper.llm import hook_with_name
    assert hook_with_name("NOBODY EXPECTED THIS", "Ashtax") == "ASHTAX: NOBODY EXPECTED THIS"
    assert hook_with_name("ASHTAX HAD 1 HP LEFT", "Ashtax") == "ASHTAX HAD 1 HP LEFT"
    assert hook_with_name("OUAH LOST HIS TOWER", "Ouah Leouff") == "OUAH LOST HIS TOWER"
    assert hook_with_name("", "Ashtax").startswith("WATCH ASHTAX")


def test_no_progress_bar_in_render():
    import inspect
    from clipper import render
    assert "0xFFD400" not in inspect.getsource(render.render_clip)


def test_hook_has_no_emoji():
    from clipper.llm import hook_with_name
    assert hook_with_name("JL TOMY'S NAMING DISASTER 🚔", "JL Tomy") == "JL TOMY'S NAMING DISASTER"
