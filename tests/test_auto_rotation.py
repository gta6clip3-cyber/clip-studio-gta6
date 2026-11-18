"""Mode automatique : un live termine est remplace, un live nettement plus gros prend la place du plus petit."""
import time

from clipper import trending
from clipper.jobs import Job

T = 10_000.0


def L(name, viewers):
    return {"url": f"https://twitch.tv/{name}", "streamer": name, "viewers": viewers}


def W(*names, since=T - 3600):
    return {f"https://twitch.tv/{n}": {"since": since} for n in names}


def test_free_slots_filled_with_biggest():
    stop, start = trending.plan_rotation({}, [L("a", 900), L("b", 500), L("c", 300), L("d", 100)], 3, now=T)
    assert stop == [] and [s["streamer"] for s in start] == ["a", "b", "c"]


def test_bigger_live_replaces_smallest():
    lives = [L("new", 5000), L("a", 900), L("b", 500), L("c", 300)]
    stop, start = trending.plan_rotation(W("a", "b", "c"), lives, 3, now=T)
    assert stop == ["https://twitch.tv/c"] and [s["streamer"] for s in start] == ["new"]


def test_two_bigger_lives_replace_two_smallest():
    lives = [L("x", 8000), L("y", 6000), L("a", 900), L("b", 500), L("c", 300)]
    stop, start = trending.plan_rotation(W("a", "b", "c"), lives, 3, now=T)
    assert set(stop) == {"https://twitch.tv/b", "https://twitch.tv/c"}
    assert [s["streamer"] for s in start] == ["x", "y"]


def test_small_gap_does_not_swap():
    lives = [L("a", 900), L("b", 500), L("new", 330), L("c", 300)]  # +10 % seulement
    assert trending.plan_rotation(W("a", "b", "c"), lives, 3, now=T) == ([], [])


def test_recent_live_is_not_swapped_out():
    lives = [L("new", 5000), L("a", 900), L("b", 500), L("c", 300)]
    assert trending.plan_rotation(W("a", "b", "c", since=T - 60), lives, 3, now=T) == ([], [])


def test_watched_live_out_of_list_counts_as_zero():
    lives = [L("new", 50), L("a", 900), L("b", 500)]
    stop, start = trending.plan_rotation(W("a", "b", "gone"), lives, 3, now=T)
    assert stop == ["https://twitch.tv/gone"] and start[0]["streamer"] == "new"


def test_skipped_lives_never_started():
    stop, start = trending.plan_rotation({}, [L("a", 900), L("b", 500)], 3, now=T, skip={"https://twitch.tv/a"})
    assert [s["streamer"] for s in start] == ["b"]


def test_run_auto_replaces_ended_and_weaker(monkeypatch):
    """Simulation complete : 3 lives suivis, l'un se termine, puis un gros live arrive."""
    lists = [[L("a", 900), L("b", 500), L("c", 300), L("d", 200)]]
    monkeypatch.setattr(trending, "auto_lives", lambda: lists[0])
    monkeypatch.setattr(trending.config, "load", lambda: {"game_name": "Grand Theft Auto V", "auto_check_minutes": 0,
                                                          "auto_min_watch_minutes": 0})
    started, stopped = {}, []

    def submit(j, fn, queued=True):
        j.status = "en cours"
        started[j.url.rsplit("/", 1)[1]] = j

    def stop(jid):
        for j in started.values():
            if j.id == jid:
                stopped.append(j.url.rsplit("/", 1)[1])
                j.stop_event.set()
                j.status = "arrete"

    monkeypatch.setattr(trending.manager, "submit", submit)
    monkeypatch.setattr(trending.manager, "stop", stop)
    ticks = iter(range(1000))

    def fake_sleep(_):
        n = next(ticks)
        if n == 6:      # le live de c se termine
            started["c"].status = "termine"
        if n == 18:     # un gros live apparait
            lists[0] = [L("big", 9000), L("a", 900), L("b", 500), L("d", 200)]
        if n == 30:
            job.stop_event.set()

    monkeypatch.setattr(trending.time, "sleep", fake_sleep)
    job = Job(kind="auto", url="top", options={"top_n": 3})
    trending.run_auto(job)
    assert list(started)[:3] == ["a", "b", "c"]
    assert "d" in started          # c termine -> remplace par d
    assert "big" in started        # gros live -> prend la place
    assert stopped[0] == "d"       # ... du plus petit (d, 200 viewers)
    assert "c" not in stopped      # c etait deja fini, pas arrete par nous
