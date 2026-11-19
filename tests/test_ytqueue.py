"""File YouTube : un Short a chaque heure prevue, jamais de rafale."""
import datetime as dt
import json

import pytest

from clipper import publish, ytqueue


@pytest.fixture
def env(tmp_path, monkeypatch):
    out, data = tmp_path / "output", tmp_path / "data"
    out.mkdir(), data.mkdir()
    monkeypatch.setattr(ytqueue.config, "OUTPUT", out)
    monkeypatch.setattr(ytqueue.config, "DATA", data)
    monkeypatch.setattr(ytqueue.config, "load", lambda: {"youtube_queue": True, "youtube_slots": "12:00, 18h,21:00"})
    sent = []

    def fake_publish(folder, platforms=None, **k):
        sent.append(folder.name)
        m = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        m["published"][(platforms or ["youtube"])[0]] = {"ok": True, "link": "https://example.com/x"}
        (folder / "meta.json").write_text(json.dumps(m), encoding="utf-8")
        return {"youtube": "ok"}

    monkeypatch.setattr(publish, "publish_clip", fake_publish)

    def clip(name, score):
        d = out / name
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"x")
        (d / "meta.json").write_text(json.dumps({"score": score, "published": {}}), encoding="utf-8")
        ytqueue.add(d)
        return d
    return clip, sent


def at(h, m=0):
    return dt.datetime(2026, 11, 19, h, m)


def test_slots_are_read_flexibly():
    assert [s.strftime("%H:%M") for s in ytqueue.slots({"youtube_slots": "21:00, 12h,18:30,bad"})] == \
        ["12:00", "18:30", "21:00"]


def test_one_short_per_slot_best_first_and_no_burst(env):
    clip, sent = env
    clip("a_ok", 8), clip("b_top", 10), clip("c_meh", 8)
    assert ytqueue.tick(at(11, 59)) is None              # avant la 1re heure : rien
    assert ytqueue.tick(at(12, 1)) == "b_top"            # le meilleur d'abord
    assert ytqueue.tick(at(12, 2)) is None               # meme heure : pas un 2e
    assert ytqueue.tick(at(17, 0)) is None
    assert ytqueue.tick(at(21, 30)) == "a_ok"            # 18:00 et 21:00 manquees (app fermee) : un seul rattrapage
    assert ytqueue.tick(at(21, 31)) is None
    assert sent == ["b_top", "a_ok"]
    assert [d.name for d in ytqueue.queued()] == ["c_meh"]


def test_published_clip_leaves_the_queue(env):
    clip, sent = env
    d = clip("a", 9)
    ytqueue.tick(at(12, 5))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert "youtube_queue" not in meta["published"] and meta["published"]["youtube"]["ok"]
    assert ytqueue.queued() == []


def test_instagram_has_its_own_queue_at_the_same_hours(env):
    clip, sent = env
    d = clip("a", 9)  # en file YouTube
    ytqueue.add(d, "instagram")
    assert ytqueue.tick(at(12, 1), "youtube") == "a"
    assert ytqueue.tick(at(12, 1), "instagram") == "a"   # meme heure, mais sa propre file
    assert ytqueue.tick(at(12, 2), "instagram") is None  # un seul Reel par heure prevue
    assert ytqueue.queued("instagram") == [] and ytqueue.queued("youtube") == []
