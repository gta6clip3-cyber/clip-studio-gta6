"""Statistiques des comptes : releve, deltas 24 h, regroupement des videos par clip, plateformes sans permission."""
import time

import pytest

from clipper import stats


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "DB", tmp_path / "stats.db")
    monkeypatch.setattr(stats, "_clips_index", lambda: {"youtube": {"yt1": "clipA"}, "instagram": {}, "tiktok": []})
    return tmp_path


def fake_reader(followers, views):
    def r():
        return {"account": "moi", "followers": followers, "views": views, "videos": [
            {"id": "yt1", "title": "Clip A", "url": "https://youtube.com/shorts/yt1", "thumb": "", "published_at": 1,
             "views": views, "likes": views // 10, "comments": 3, "shares": None}]}
    return r


def test_collect_records_snapshots_and_24h_deltas(db, monkeypatch):
    clock = [time.time() - 2 * 86400]
    monkeypatch.setattr(stats.time, "time", lambda: clock[0])
    monkeypatch.setitem(stats.READERS, "youtube", fake_reader(100, 1000))
    stats.collect(("youtube",))
    clock[0] += 86400 + 60
    monkeypatch.setitem(stats.READERS, "youtube", fake_reader(150, 4000))
    stats.collect(("youtube",))
    o = stats.overview(30)
    yt = o["platforms"]["youtube"]
    assert yt["followers"] == 150 and yt["views"] == 4000
    assert yt["followers_24h"] == 50 and yt["views_24h"] == 3000
    assert yt["status"]["ok"] and yt["status"]["account"] == "moi"
    assert len(yt["series"]) == 2 and o["totals"]["followers"] == 150


def test_missing_permission_is_reported_not_raised(db, monkeypatch):
    def nope():
        raise stats.Missing("Reconnecte YouTube")
    monkeypatch.setitem(stats.READERS, "youtube", nope)
    r = stats.collect(("youtube",))
    assert r["youtube"] == {"ok": False, "message": "Reconnecte YouTube"}
    st = stats.overview()["platforms"]["youtube"]["status"]
    assert st["ok"] is False and "Reconnecte" in st["message"]


def test_videos_are_linked_to_their_clip(db, monkeypatch):
    monkeypatch.setitem(stats.READERS, "youtube", fake_reader(10, 500))
    stats.collect(("youtube",))
    v = stats.videos()
    assert len(v) == 1 and v[0]["key"] == "clipA" and v[0]["folder"] == "clipA"
    assert v[0]["views"] == 500 and v[0]["platforms"]["youtube"]["likes"] == 50
    d = stats.video_detail("clipA")
    assert d["series"]["youtube"][0]["views"] == 500


def test_tiktok_scope_error_explains_what_to_do(monkeypatch):
    class R:
        def json(self):
            return {"data": {}, "error": {"code": "scope_not_authorized"}}
    monkeypatch.setattr(stats.config, "load", lambda: {"tiktok_access_token": "x", "tiktok_refresh_token": ""})
    monkeypatch.setattr(stats.requests, "get", lambda *a, **k: R())
    with pytest.raises(stats.Missing, match="video.list"):
        stats._tiktok()
