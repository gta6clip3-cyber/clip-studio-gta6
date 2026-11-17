from clipper.highlights import Candidate
from clipper.pipeline import _clamp_bounds, _snap

CFG = {"clip_min_seconds": 40, "clip_max_seconds": 75, "clip_target_seconds": 60}


def tr(segs):
    return {"language": "fr", "segments": [{"start": a, "end": b, "text": "x", "words": []} for a, b in segs]}


def test_snap_start_to_sentence_start():
    s, e = _snap(102.0, 160.0, tr([(100.5, 104), (150, 158), (158.5, 163)]), 40, 75)
    assert abs(s - 100.25) < 1e-6
    assert abs(e - 163.6) < 1e-6  # fin de la phrase coupee + reaction


def test_snap_respects_max():
    s, e = _snap(100.0, 174.0, tr([(99.0, 101), (170, 190)]), 40, 75)
    assert e - s <= 75
    assert e <= 170


def test_bad_llm_bounds_fall_back_to_peak():
    c = Candidate(peak=500, score=3, start=445, end=530, reasons="")
    s, e, p = _clamp_bounds(c, {"start": 70, "end": 20, "punch_time": 10}, CFG)
    assert s < 500 < e and 40 <= e - s <= 75


def test_good_llm_bounds_used():
    c = Candidate(peak=500, score=3, start=445, end=530, reasons="")
    s, e, p = _clamp_bounds(c, {"start": 10, "end": 70, "punch_time": 50}, CFG)
    assert (s, e, p) == (455, 515, 495)
