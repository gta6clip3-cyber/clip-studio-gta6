"""Regles TikTok « Direct Post » : choix de l'utilisateur obligatoires, contenu commercial, pas de publication auto."""
import json

import pytest

from clipper import publish
from clipper.publish import tiktok

INFO = {"privacy_level_options": ["FOLLOWER_OF_CREATOR", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"],
        "comment_disabled": False, "duet_disabled": True, "stitch_disabled": True, "max_video_post_duration_sec": 600}


def test_privacy_must_be_chosen():
    with pytest.raises(ValueError, match="qui peut voir"):
        tiktok.check_options({}, INFO, 60)
    with pytest.raises(ValueError):
        tiktok.check_options({"privacy_level": "PUBLIC_TO_EVERYONE"}, INFO, 60)  # non propose pour ce compte


def test_interactions_off_by_default_and_respect_account_settings():
    p = tiktok.check_options({"privacy_level": "SELF_ONLY", "allow_comment": True, "allow_duet": True}, INFO, 60)
    assert p["disable_comment"] is False          # coche par l'utilisateur
    assert p["disable_duet"] is True              # desactive dans le compte : impossible de l'autoriser
    assert p["disable_stitch"] is True            # pas coche


def test_commercial_content_rules():
    with pytest.raises(ValueError, match="marque"):
        tiktok.check_options({"privacy_level": "SELF_ONLY", "commercial": True}, INFO, 60)
    with pytest.raises(ValueError, match="priv"):
        tiktok.check_options({"privacy_level": "SELF_ONLY", "commercial": True, "brand_content_toggle": True}, INFO, 60)
    p = tiktok.check_options({"privacy_level": "MUTUAL_FOLLOW_FRIENDS", "commercial": True,
                              "brand_content_toggle": True}, INFO, 60)
    assert p["brand_content_toggle"] is True and p["brand_organic_toggle"] is False


def test_too_long_video_refused():
    with pytest.raises(ValueError, match="longue"):
        tiktok.check_options({"privacy_level": "SELF_ONLY"}, INFO, 700)


def test_tiktok_never_posted_without_the_publish_screen(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"title": "t", "caption": "c", "published": {}}), encoding="utf-8")
    monkeypatch.setattr(publish.config, "load", lambda: {"publish_youtube": False, "publish_tiktok": True,
                                                         "publish_instagram": False})
    called = []
    monkeypatch.setattr(tiktok, "upload", lambda *a, **k: called.append(a) or "ok")
    assert publish.publish_clip(tmp_path) == {}        # publication auto / bouton general : TikTok saute
    assert not called
    publish.publish_clip(tmp_path, ["tiktok"], tiktok_post={"privacy_level": "SELF_ONLY"}, caption="ma legende")
    assert called and called[0][1] == "ma legende" and called[0][2] == {"privacy_level": "SELF_ONLY"}


def test_draft_goes_to_inbox_and_is_recorded(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"published": {}}), encoding="utf-8")
    sent = {}

    class R:
        def __init__(self, d, code=200):
            self.d, self.status_code, self.text = d, code, ""

        def json(self):
            return self.d

    def post(url, headers=None, json=None, timeout=None):
        if url.endswith("/inbox/video/init/"):
            sent["init"] = json
            return R({"data": {"upload_url": "https://up", "publish_id": "p1"}, "error": {"code": "ok"}})
        return R({"data": {"status": "SEND_TO_USER_INBOX"}})

    (tmp_path / "clip.mp4").write_bytes(b"x" * 1000)
    monkeypatch.setattr(tiktok, "_refresh", lambda cfg: "tok")
    monkeypatch.setattr(tiktok.config, "load", lambda: {"tiktok_privacy": "SELF_ONLY"})
    monkeypatch.setattr(tiktok.requests, "post", post)
    monkeypatch.setattr(tiktok.requests, "put", lambda *a, **k: R({}, 201))
    res = publish.draft_clip(tmp_path)
    assert res["tiktok_draft"].startswith("dans tes brouillons")
    assert "post_info" not in sent["init"]  # brouillon : aucun choix de publication envoye, rien n'est publie
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["published"]["tiktok_draft"]["ok"] is True
