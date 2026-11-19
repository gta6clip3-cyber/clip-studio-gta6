from clipper.transcribe import clean_segments, sentences


def S(a, b, text, n=None):
    n = n or len(text.split())
    return {"start": a, "end": b, "text": text,
            "words": [{"word": w, "start": a + (b - a) * k / n, "end": a + (b - a) * (k + 1) / n}
                      for k, w in enumerate(text.split())]}


def test_hallucinated_repeat_removed():
    segs = [S(18.33, 21.11, "Il y a tellement de combos"), S(21.11, 21.91, "Il y a tellement de trucs à faire"),
            S(21.91, 22.67, "Il y a tellement de trucs à faire")]
    out = clean_segments(segs)
    assert len(out) == 2


def test_real_repetition_kept():
    segs = [S(30.39, 30.89, "Ça arrive"), S(30.89, 31.37, "Ça arrive")]
    assert len(clean_segments(segs)) == 2  # c'est vraiment dit deux fois


def test_junk_removed():
    assert clean_segments([S(0, 2, "Sous-titres réalisés par la communauté d'Amara.org")]) == []


def test_sentences_merge_cut_phrases():
    segs = [S(46.7, 51.1, "on a mis des bons dommages ça ne suffira pas ok on est derrière on"),
            S(51.2, 60.5, "est un peu derrière attention à la valkyrie"),
            S(62.0, 63.0, "Ok.")]
    out = sentences(segs)
    assert len(out) == 2
    assert "derrière on est un peu derrière" in out[0]["text"]
    assert len(out[0]["words"]) == len(segs[0]["words"]) + len(segs[1]["words"])


def test_sentences_respect_punctuation():
    out = sentences([S(0, 1, "Regardez-moi ça."), S(1.1, 2, "Je vais bûcher")])
    assert len(out) == 2


def test_dedupe_translated_lines():
    from clipper.llm import dedupe_lines
    out = dedupe_lines(["The baby dragon is coming. He shouldn't have much elixir, the earthquake too",
                        "he shouldn't have much elixir either, so we do this", "It's coming, it's coming"])
    assert "shouldn't have much elixir" not in out[1].lower()
    assert out[2] == "It's coming, it's coming"


def test_streamer_name_without_channel_suffix():
    from clipper.sources import clean_name
    assert clean_name("JL Tomy - Live") == "JL Tomy"
    assert clean_name("Kameto - Replays et VODs") == "Kameto"
    assert clean_name("Anyme Live Rediff") == "Anyme"
    assert clean_name("Ouah Leouff") == "Ouah Leouff"
    assert clean_name("Live") == "Live"


def test_youtube_cookies_go_to_ytdlp_and_streamlink(tmp_path, monkeypatch):
    """Les cookies d'un 2e compte YouTube levent le blocage « confirm you're not a bot »."""
    from clipper import config, sources
    cookies = tmp_path / "youtube_cookies.txt"
    monkeypatch.setattr(config, "load", lambda: {"youtube_cookies_file": str(cookies)})
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(sources, "tool", lambda name: str(tmp_path / "ffmpeg.exe"))
    assert sources.youtube_cookies() is None and "--cookies" not in sources._ytdlp_base()  # pas de fichier
    cookies.write_text("# Netscape HTTP Cookie File\n"
                       ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tabc\n"
                       "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1999999999\t__Secure-3PSID\tdef\n"
                       ".google.com\tTRUE\t/\tTRUE\t1999999999\tNID\tzzz\n", encoding="utf-8")
    base = sources._ytdlp_base()
    copy = base[base.index("--cookies") + 1]
    assert copy != str(cookies) and open(copy, encoding="utf-8").read() == cookies.read_text(encoding="utf-8")
    assert sources.streamlink_cookie_args("https://www.youtube.com/watch?v=x") == [
        "--http-cookie", "SID=abc", "--http-cookie", "__Secure-3PSID=def"]
    assert sources.streamlink_cookie_args("https://www.twitch.tv/kaicenat") == []  # jamais envoyes a Twitch
