"""Chat des lives YouTube : lu en continu dans le fichier que yt-dlp remplit, yt-dlp arrete avec le live."""
import json
import subprocess
import threading

from clipper import chat


def line(text, ms=1000):
    return json.dumps({"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]}}}}}]},
        "videoOffsetTimeMsec": str(ms), "isLive": True})


def test_chat_line_parses_live_format():
    assert chat._chat_line(line("KEKW no way", 2500)) == [(2.5, "KEKW no way")]
    assert chat._chat_line("pas du json") == []


def test_youtube_live_chat_streams_messages_and_stops_ytdlp(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "_ytdlp_base", lambda: ["yt-dlp"])
    monkeypatch.setattr("clipper.config.WORK", tmp_path)
    procs = []

    class FakeYtdlp:
        def __init__(self, args, **kw):
            out = args[args.index("-o") + 1]
            self.file = open(out + ".live_chat.json.part", "w", encoding="utf-8")
            self.file.write(line("first") + "\n" + line("POG")[:20])  # 2e ligne pas encore finie
            self.file.flush()
            self.code = None
            procs.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.file.close()
            self.code = 1

        def wait(self, t=None):
            return self.code

    monkeypatch.setattr(subprocess, "Popen", FakeYtdlp)
    stop = threading.Event()
    gen = chat.live_chat("https://www.youtube.com/watch?v=abc", stop)
    assert next(gen) == "first"
    procs[0].file.write(line("POG")[20:] + "\n")  # la ligne se termine : le message arrive en entier
    procs[0].file.flush()
    assert next(gen) == "POG"
    stop.set()
    assert list(gen) == []
    assert procs[0].code == 1                     # yt-dlp arrete avec le live
    assert not list(tmp_path.glob("ytlivechat_*"))  # dossier temporaire supprime
