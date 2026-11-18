"""Ce que l'IA recoit en plus du texte (phrases criees, reactions du chat) et le telechargement en avance."""
import json
import time

import numpy as np

from clipper import chat, llm, pipeline
from clipper.highlights import CHAT_DELAY, Candidate, reactions, score_timeline
from clipper.jobs import Job
from clipper.transcribe import as_text


def seg(a, b, text="x"):
    return {"start": a, "end": b, "text": text, "words": []}


def test_reactions_mark_the_shouted_line_and_the_chat_burst():
    n = 120
    audio, hype = np.zeros(n), np.zeros(n)
    audio[40:43] = 5.0   # cri pendant la 2e phrase
    hype[50:56] = 6.0    # le chat explose apres la 3e phrase (deja recale sur l'evenement)
    parts = {"audio": audio, "chat": np.zeros(n), "hype": hype}
    msgs = ([(20.0, "salut")] + [(52 + CHAT_DELAY + k * 0.1, "KEKW KEKW") for k in range(12)]
            + [(53.0 + CHAT_DELAY, "mdr"), (54.0 + CHAT_DELAY, "MDR c'est trop")])
    segs = [seg(30, 35), seg(39, 44), seg(50, 53), seg(70, 75)]
    assert reactions(segs, parts, msgs) == ["", "LOUD", "chat: KEKW x12, MDR x2", ""]


def test_reactions_without_chat_only_hear_the_volume():
    parts = {"audio": np.array([0, 0, 4.0, 0, 0])}
    assert reactions([seg(0, 1), seg(2, 3)], parts, [(8.0, "KEKW"), (8.5, "KEKW")]) == ["", "LOUD"]
    assert reactions([], parts, []) == []


def test_quiet_chat_one_message_is_not_a_peak_but_a_burst_is():
    loud = np.random.default_rng(0).normal(-30, 2, 3600).astype(np.float32)
    msgs = [(60.0 * k, "salut") for k in range(1, 56)]        # chat calme : 1 message par minute
    msgs += [(1800 + 0.5 * k, "KEKW") for k in range(8)]      # 8 rires en 4 s
    msgs += [(2.0, "KEKW")] * 8                               # rafale tout au debut du live
    score, parts = score_timeline(loud, sorted(msgs))
    assert parts["chat"][590:600].max() <= 1.5                # avant : un message isole = pic maximal (8)
    assert parts["chat"][1790:1800].max() >= 5
    assert 1785 <= 30 + int(np.argmax(score[30:-30])) <= 1805
    assert parts["chat"][-CHAT_DELAY:].max() == 0             # avant : le debut du live etait recopie a la fin


def test_youtube_chat_replay_is_read_with_ytdlp(monkeypatch, tmp_path):
    def item(ms, kind, runs):
        return json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(ms), "actions": [
            {"addChatItemAction": {"item": {kind: {"message": {"runs": runs}}}}}]}})

    lines = [item(0, "liveChatViewerEngagementMessageRenderer", [{"text": "Live chat replay is on."}]),
             item(12500, "liveChatTextMessageRenderer", [{"text": "no way"}, {"emoji": {"emojiId": "😂"}}, {"emoji": {
                 "emojiId": "UC1/x", "isCustomEmoji": True, "shortcuts": [":kekw:"]}}])]
    monkeypatch.setattr(chat, "_ytdlp_base", lambda: ["yt-dlp"])
    monkeypatch.setattr(chat, "run", lambda cmd, **kw: (tmp_path / "ytchat.live_chat.json").write_text(
        "\n".join(lines), encoding="utf-8"))
    out = chat.download_vod_chat("https://www.youtube.com/watch?v=x", tmp_path / "chat.json")
    assert out == [(12.5, "no way 😂  :kekw: ")]
    assert chat.hype_weight(out[0][1]) == 3.5                 # l'emoji et l'emote de la chaine comptent


def test_twitch_vod_chat_slices_leave_no_gap_and_no_duplicate(monkeypatch, tmp_path):
    import chat_downloader.sites.twitch as tw
    asked = []

    class FakeTwitch:  # un message par seconde de 0 a 1003 s (la duree annoncee, 1000 s, est un peu courte)
        def _get_chat_messages_by_vod_id(self, vod_id, params, max_duration):
            asked.append((vod_id, params["start_time"], params["end_time"], params["max_attempts"]))
            end = 1003 if params["end_time"] is None else params["end_time"]
            return ({"time_in_seconds": t, "message": f"m{t}"} for t in range(params["start_time"], end + 1))

    monkeypatch.setattr(tw, "TwitchChatDownloader", FakeTwitch)
    out = chat.download_vod_chat("https://www.twitch.tv/videos/123?t=1h", tmp_path / "chat.json", 1000.0)
    assert out == [(float(t), f"m{t}") for t in range(1004)]  # tout, une seule fois, dans l'ordre
    assert sorted(asked) == [("123", 0, 250, 5), ("123", 251, 501, 5), ("123", 502, 752, 5), ("123", 753, None, 5)]


def test_twitch_vod_chat_resumes_a_slice_and_keeps_what_was_read(monkeypatch, tmp_path):
    import chat_downloader.sites.twitch as tw
    fails = []

    class FakeTwitch:  # un message par seconde ; Twitch refuse 2 fois une page a 255 s, et toujours a 900 s
        def _get_chat_messages_by_vod_id(self, vod_id, params, max_duration):
            end = 1003 if params["end_time"] is None else params["end_time"]
            for t in range(params["start_time"], end + 1):
                if t == 900 or (t == 255 and fails.count(255) < 2):
                    fails.append(t)
                    raise KeyError("data")
                yield {"time_in_seconds": t, "message": "m"}

    monkeypatch.setattr(tw, "TwitchChatDownloader", FakeTwitch)
    out = chat.download_vod_chat("https://www.twitch.tv/videos/123", tmp_path / "chat.json", 1000.0)
    assert [t for t, _ in out] == [float(t) for t in range(900)]  # reprise sans trou ni doublon, puis abandon a 900 s
    assert sorted(fails) == [255, 255, 900, 900, 900]             # 3 essais au plus par tranche


def test_twitch_live_chat_reads_irc_directly(monkeypatch):
    import chat_downloader.sites.twitch as tw
    asked = []

    class FakeTwitch:
        def _get_chat_messages_by_stream_id(self, channel, params):
            asked.append((channel, params["max_attempts"], params["message_receive_timeout"], params["buffer_size"]))
            return iter([{"message": "KEKW"}, {"message": None}])

    monkeypatch.setattr(tw, "TwitchChatDownloader", FakeTwitch)
    assert list(chat.live_chat("https://www.twitch.tv/Kamet0/")) == ["KEKW", ""]
    assert asked == [("kamet0", 5, 0.1, 4096)]


def test_chat_downloader_still_has_the_readers_we_call():
    """Notre reparation appelle deux fonctions internes de chat-downloader : ce test previent si elles changent."""
    import inspect

    from chat_downloader.sites.twitch import TwitchChatDownloader as T
    assert list(inspect.signature(T._get_chat_messages_by_vod_id).parameters)[:4] == ["self", "vod_id", "params", "max_duration"]
    assert list(inspect.signature(T._get_chat_messages_by_stream_id).parameters) == ["self", "stream_id", "params"]


def test_live_chat_reconnects_after_a_cut_and_stops_with_the_live(monkeypatch):
    import threading
    from types import SimpleNamespace

    from clipper import live
    calls, msgs, over = [], [], threading.Event()

    def fake_chat(url, stop=None):
        assert stop is over  # le lecteur recoit le signal de fin du live (pour arreter yt-dlp sur YouTube)
        calls.append(url)
        if len(calls) == 1:
            raise ConnectionError("coupure")
        if len(calls) == 3:
            over.set()  # le live se termine
        yield from ("a", "b")

    monkeypatch.setattr(live, "live_chat", fake_chat)
    job = Job("live", "u")
    live._follow_chat(job, SimpleNamespace(t0=0.0), msgs, over, retry_s=0.01)
    assert [m for _, m in msgs] == ["a", "b"] and len(calls) == 3  # reconnecte apres la coupure, arrete a la fin
    assert sum("Chat live indisponible" in line for line in job.logs) == 1


def test_live_notes_use_the_signals_given_with_the_candidate(monkeypatch, tmp_path):
    tr = {"language": "fr", "segments": [dict(seg(10, 14, "NON !"), words=[{"word": "NON", "start": 10, "end": 11}] * 9)]}
    seen = []
    monkeypatch.setattr(pipeline.transcribe, "transcribe", lambda *a, **k: tr)
    monkeypatch.setattr(pipeline.transcribe, "unload", lambda: None)
    monkeypatch.setattr(pipeline.llm, "available", lambda: True)
    monkeypatch.setattr(pipeline.llm, "unload", lambda: None)
    monkeypatch.setattr(pipeline.llm, "judge_candidate", lambda text, *a, **k: seen.append(text))
    monkeypatch.setattr(pipeline, "loudness_per_second", lambda *a: 1 / 0)  # ne doit pas etre recalcule en direct
    cfg = {"clip_min_seconds": 40, "clip_max_seconds": 75, "clip_target_seconds": 60}
    loud = np.zeros(80)
    loud[11] = 4.0
    for parts in ({"audio": loud}, None):
        pipeline.analyse_moments(Job("live", "u"), tmp_path / "win.ts", [], "s", cfg, 1,
                                 cands=[Candidate(40, 3, 0, 77)], parts=parts)
    assert seen[0].endswith("NON !  <<LOUD>>") and "<<" not in seen[1]


def test_twitch_live_title_is_the_real_stream_title(monkeypatch):
    from clipper import games, sources
    live = {"id": "1", "title": "Kamet0 (live) 2026-11-17 14:35", "description": "GTA RP avec les potes",
            "uploader": "Kamet0", "is_live": True}
    monkeypatch.setattr(sources, "_ytdlp_base", lambda: ["yt-dlp"])
    monkeypatch.setattr(sources, "run", lambda cmd, **kw: type("R", (), {"stdout": json.dumps(live)}))
    info = sources.get_info("https://www.twitch.tv/kamet0")
    assert info["title"] == "GTA RP avec les potes" and info["is_live"]
    assert games.detect(info["title"], info["context"]) == "gta"  # le jeu est reconnu des le debut du live


def test_a_scream_long_after_a_line_is_not_attached_to_it():
    audio = np.zeros(60)
    audio[38:40] = 5.0   # cri non transcrit, 26 s apres la 1re phrase
    audio[14] = 5.0      # cri 2 s apres la 1re phrase : c'est sa reaction
    parts = {"audio": audio}
    assert reactions([seg(10, 12), seg(40, 42)], parts, []) == ["LOUD", ""]
    audio[14] = 0
    assert reactions([seg(10, 12), seg(40, 42)], parts, []) == ["", ""]


def test_normal_speech_after_a_long_silence_is_not_loud():
    loud = np.random.default_rng(1).normal(-30, 3, 7200).astype(np.float32)
    loud[3000:3500] = -90.0  # micro coupe / ecran d'attente pendant 8 minutes
    score, parts = score_timeline(loud, [])
    assert parts["audio"][3500:3560].mean() < 1 and score[3500:3560].max() < 3  # avant : 6.0 partout (pic maximal)
    assert parts["audio"][3100:3400].max() < 0                                   # le silence lui-meme n'est pas un pic


def test_notes_are_appended_to_transcript_lines():
    tr = {"segments": [seg(100, 102, "Salut"), seg(103, 105, "NON !")]}
    assert as_text(tr, 100) == "[   0.0-   2.0] Salut\n[   3.0-   5.0] NON !"  # sans note : rien ne change
    assert as_text(tr, 100, ["", "LOUD"]) == "[   0.0-   2.0] Salut\n[   3.0-   5.0] NON !  <<LOUD>>"


def test_judge_explains_notes_only_when_present_and_retries(monkeypatch):
    prompts = []
    good = {"payoff": 8, "reaction": 8, "standalone": 8, "boring": 1, "best_line": "x", "start": 5, "end": 60}

    def fake_chat(system, user, temperature=0.4, **kw):
        prompts.append(user)
        return None if len(prompts) == 1 else dict(good)  # 1re reponse illisible

    monkeypatch.setattr(llm, "chat_json", fake_chat)
    j = llm.judge_candidate("[   0.0-   2.0] NON !  <<LOUD>>", 85, "S", "audio=3.0")
    assert j and j["score"] > 7 and len(prompts) == 2  # 2e essai apres la reponse illisible
    assert "<<notes>>" in prompts[0]
    llm.judge_candidate("[   0.0-   2.0] NON !", 85, "S", "audio=3.0")
    assert "<<notes>>" not in prompts[-1]  # sans note : le message envoye a l'IA est le meme qu'avant


def test_score_is_shifted_only_for_models_measured_as_harsher():
    res = {"payoff": 4, "reaction": 3, "standalone": 4, "boring": 7, "best_line": "x"}
    base = llm.combine_score(res)
    assert base == llm.combine_score(res, "qwen2.5:7b") == llm.combine_score(res, "modele-inconnu") == 3.0
    shift = llm.SEVERITY["qwen3.5:9b"]
    assert llm.combine_score(res, "qwen3.5:9b") == round(base + 1.25 * shift, 1)  # meme formule, decalee
    top = {"payoff": 9, "reaction": 9, "standalone": 9, "boring": 1, "best_line": "x"}
    assert llm.combine_score(top, "qwen3.5:9b") == 10.0 < llm.combine_score(top, "qwen3.5:9b", clip=False)


def test_unload_waits_until_ollama_has_freed_the_card(monkeypatch):
    from types import SimpleNamespace
    loaded = [[{"name": "m"}], [{"name": "m"}], []]  # Ollama garde encore le modele 2 fois, puis plus rien
    asked = []
    monkeypatch.setattr(llm.requests, "post", lambda url, json=None, timeout=None: asked.append(json["keep_alive"]))
    monkeypatch.setattr(llm.requests, "get", lambda url, timeout=None: SimpleNamespace(json=lambda: {"models": loaded.pop(0)}))
    monkeypatch.setattr(llm.time, "sleep", lambda s: asked.append("attente"))
    llm.unload()
    assert asked == [0, "attente", "attente"] and not loaded


def test_thinking_is_switched_off_only_for_models_that_think(monkeypatch):
    from types import SimpleNamespace
    asked = []

    def fake_post(url, json=None, timeout=None):
        asked.append(json["model"])
        if json["model"] == "panne":
            raise OSError("Ollama arrete")
        caps = ["completion", "thinking"] if json["model"] == "qwen3.5:9b" else ["completion"]
        return SimpleNamespace(json=lambda: {"capabilities": caps})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm, "_thinks", {})
    assert llm._can_think("u", "qwen3.5:9b") and llm._can_think("u", "qwen3.5:9b")
    assert not llm._can_think("u", "qwen2.5:7b")
    assert not llm._can_think("u", "panne") and not llm._can_think("u", "panne")
    assert asked == ["qwen3.5:9b", "qwen2.5:7b", "panne", "panne"]  # reponse gardee en memoire, sauf en cas de panne


def test_vod_sections_are_downloaded_ahead(monkeypatch, tmp_path):
    moments = [pipeline.Moment(Candidate(0, 1, 0, 0), {}, 100.0 * k, 100.0 * k + 60, None, {}, 1.0) for k in (1, 2, 3)]
    asked, got = [], []
    monkeypatch.setattr(pipeline, "download_vod_chat", lambda url, cache, duration: [])
    monkeypatch.setattr(pipeline, "download_audio", lambda url, work: tmp_path / "audio.m4a")
    monkeypatch.setattr(pipeline, "loudness_per_second", lambda *a: None)
    monkeypatch.setattr(pipeline, "analyse_moments", lambda *a, **k: moments)
    monkeypatch.setattr(pipeline, "download_section", lambda url, s, e, out: asked.append((s, e)) or out)
    monkeypatch.setattr(pipeline, "align_offset", lambda audio, s, src, search: 4.0)

    def fake_finish(job, m, audio, video, *rest):
        if not got:  # pendant le 1er clip, les passages des clips suivants arrivent deja
            t = time.time()
            while len(asked) < 3 and time.time() - t < 2:
                time.sleep(0.01)
            assert len(asked) == 3
        got.append(video(m.start, m.end, tmp_path))

    monkeypatch.setattr(pipeline, "finish_clip", fake_finish)
    info = {"streamer": "s", "platform": "twitch", "webpage_url": "u", "duration": 400.0}
    pipeline._process_vod(Job("vod", "u"), info, tmp_path, {}, 3)
    assert sorted(asked) == [(96.0, 162.0), (196.0, 262.0), (296.0, 362.0)]
    assert got == [(tmp_path / f"src_{i}.mp4", 4.0) for i in (1, 2, 3)]
