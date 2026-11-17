"""Tests de l'IA locale (Ollama). Lents : ~2-4 min."""
import pytest

from clipper import llm
from clipper.textlang import english_ratio, is_english

pytestmark = pytest.mark.skipif(not llm.available(), reason="Ollama indisponible")

# vraies lignes Whisper d'un clip Clash Royale francais qui etait reste en francais
CLASH_FR = [
    "Ça va être compliqué.", "Kava bélier, cochon, euh boule de feu.", "La boule de feu superbe, incroyable.",
    "Ah, mais j'ai plus rien pour défendre !", "C'est trop dur aïe aïe aïe", "Oh le kava quand même !",
    "Oh il va connecter.", "Ah c'est d'fommage franchement t'étais vraiment pas loin.",
    "J'ai essayé de faire des choses, j'ai essayé, j'ai essayé mais ton combo faut avouer que du coup il est pas top.",
    "Il manque une défense.", "Ouais c'est ça.",
    "Ouais parce que du coup bah ouais la mousquetaire, j'ai essayé pourtant j'ai jamais réussi à la protéger",
    "Bon ouais.", "Écoute je vais te demander, on va te demander de refaire un nouveau combo ?",
    "On va faire un autre deck, un autre truc.", "Ok, vas-y dis-moi.",
    "Bah du coup, # ce propose-moi un troisième combo, parce que là le premier malheureusement n'a pas",
]
GTA_FR = [
    "Attends attends, les flics arrivent derrière, fonce fonce !", "Non non non pas le mur, pas le mur !",
    "Putain il m'a rentré dedans ce bâtard", "Mais t'es sérieux là, on avait le braquage !",
    "Wesh c'est quoi cette voiture, elle va à deux à l'heure", "Je suis mort, je suis mort, c'est fini",
    "Hahaha regarde le mec il est tombé du pont", "Il me reste 20 secondes pour livrer le colis",
    "Chat, vous avez vu ça ? C'est le meilleur saut de ma vie",
]
SPANISH = ["¡No puede ser! ¡Me mató otra vez!", "Vamos vamos, ya casi llegamos", "Mira lo que hizo este tío, increíble"]


@pytest.mark.parametrize("lines,ctx", [(CLASH_FR, "Game: Clash Royale."), (GTA_FR, "Game: Grand Theft Auto VI."),
                                       (SPANISH, "Game: unknown.")])
def test_translation_is_english_and_complete(lines, ctx):
    out = llm.translate_lines(lines, context=ctx)
    assert out is not None and len(out) == len(lines)
    non_empty = [t for t in out if t]
    assert len(non_empty) >= len(lines) - 1, list(zip(lines, out))
    assert english_ratio(non_empty) == 1.0, [t for t in non_empty if not is_english(t)]


def test_translation_keeps_game_terms():
    out = llm.translate_lines(["La boule de feu superbe, incroyable."], context="Game: Clash Royale.")
    assert "fireball" in out[0].lower(), out


def test_stream_chat_is_not_a_cat():
    out = llm.translate_lines(["Chat, vous avez vu ça ? C'est le meilleur saut de ma vie"], context="Game: GTA VI.")
    assert "chat" in out[0].lower() and "cat" not in out[0].lower().replace("chat", ""), out


def test_title_is_not_a_dry_description():
    eng = "Wait wait he's coming. I have one HP left. I see him, I snipe him. NO WAY! WHAT A SHOT! One versus four!"
    meta = llm.write_metadata(eng, "Fortnite", "Streamer", "The streamer wins a 1v4 with one HP left.")
    assert meta and not meta["title"].lower().startswith("fortnite"), meta
    assert meta["hook"] not in ("EPIC MOMENT", "FUNNY CLIP"), meta


def test_long_batch_keeps_alignment():
    lines = [f"Ligne numéro {i}, il a tué {i} joueurs" for i in range(1, 41)]
    out = llm.translate_lines(lines, context="Game: Fortnite.")
    assert len(out) == 40
    for i, t in enumerate(out, 1):
        assert str(i) in t, (i, t)


def test_metadata_is_english_and_about_the_right_game():
    eng = ("It's going to be complicated. Hog rider, fireball. The fireball, amazing! I've got nothing left to defend! "
           "Oh he's going to connect. Honestly you weren't far. Your combo is missing a defense.")
    meta = llm.write_metadata(eng, "Clash Royale", "Ouah Leouff", "A fireball combo almost wins the game.")
    assert meta
    for k in ("title", "hook", "caption"):
        assert is_english(meta[k]), meta
    assert 2 <= len(meta["hook"].split()) <= 9
    assert "OUAH" in meta["hook"], meta  # le nom du streamer est toujours dans l'accroche
    blob = (meta["title"] + meta["hook"] + meta["caption"] + " ".join(meta["hashtags"])).lower()
    assert "gta" not in blob and "grand theft" not in blob, meta


def _judge(text, title):
    return llm.judge_candidate(text, 85, "Streamer", "chat=1.0, hype=1.0, audio=1.0", title)


def test_judge_prefers_hype_over_boring():
    boring = "\n".join(f"[{i * 5:6.1f}-{i * 5 + 4:6.1f}] {t}" for i, t in enumerate([
        "Bon alors je vais regarder les paramètres", "On va changer la sensibilité de la souris",
        "Merci pour le follow", "Je vais boire un peu d'eau", "Alors, on en était où", "Je lis le chat deux secondes",
        "Ouais c'est ça les gars", "Bon on relance une partie"]))
    hype = "\n".join(f"[{i * 5:6.1f}-{i * 5 + 4:6.1f}] {t}" for i, t in enumerate([
        "Attends il arrive, il arrive", "Il me reste un PV, un seul PV", "Je le vois, je le vois, je le snipe",
        "NOOON ! MAIS QUEL TIR ! QUEL TIR !", "Un contre quatre ! J'ai gagné un contre quatre !",
        "Le chat devient fou, regardez le chat", "Je tremble, regardez mes mains", "C'est le top 1 les gars !"]))
    jb, jh = _judge(boring, "Fortnite ranked"), _judge(hype, "Fortnite ranked")
    assert jb and jh
    assert jh["score"] >= jb["score"] + 2, (jb["score"], jh["score"])
    assert jh["score"] >= 7


def test_judge_identifies_game_without_game_in_title():
    t = "\n".join(f"[{i * 5:6.1f}-{i * 5 + 4:6.1f}] {x}" for i, x in enumerate(CLASH_FR[:10]))
    j = _judge(t, "Ashtax choisit mes decks... (c'est une catastrophe) (video tags / description: Gaming | clash, "
                  "cr fr, clash royale, deck trop fort)")
    assert j and "clash" in j.get("game", "").lower(), j


def test_judge_identifies_game_from_title():
    t = "[   0.0-   5.0] Oh le cochon arrive, je pose la boule de feu\n[   5.0-  10.0] Trois couronnes ! Incroyable !"
    j = _judge(t, "TOP 1 MONDIAL SUR CLASH ROYALE")
    assert j and "clash" in j.get("game", "").lower(), j
    assert "gta" not in (j.get("title", "") + j.get("hook", "")).lower()


ASHTAX_FR = [
    "Messieurs dames On a un combo", "Et je veux essayer ce combo",
    "Le bébé dragon Messieurs dames Qui arrive Et qui va accélérer tout le monde Avec son pouvoir Le bûcheron évolue en dessous Les chevaucheurs de cochons",
    "qui arrive en plus de ça il doit pas non plus avoir masse d'élixir le séisme en plus on fait",
    "passer tout ça c'est magnifique le cochon qui est en train de faire le boulot le fantôme qui",
    "a un petit peu accéléré aussi on a mis des bons dommages ça ne suffira pas ok on est derrière on",
    "est un peu derrière attention attention attention à la valkyrie il est mis un petit coup on est",
]


def test_no_duplicated_content_between_lines():
    import re
    out = llm.translate_lines(ASHTAX_FR, context="Game: Clash Royale. Livestream of Ashtax.")
    for a, b in zip(out, out[1:]):
        ta, tb = re.findall(r"[\w']+", a.lower()), re.findall(r"[\w']+", b.lower())
        ga = {tuple(ta[k:k + 4]) for k in range(len(ta) - 3)}
        assert not any(tuple(tb[k:k + 4]) in ga for k in range(len(tb) - 3)), (a, b)
    # pas de debordement : chaque traduction reste proche de la longueur de l'original
    for s, t in zip(ASHTAX_FR, out):
        assert len(t.split()) <= 2 * len(s.split()) + 3, (s, t)


def test_titles_vary_between_clips():
    eng = "The giant is coming. I have no elixir left. He's going to take my tower. No no no! Saved it at the last second!"
    used = ["He Was One Card Away From Losing...", "He was one Giant away from losing..."]
    meta = llm.write_metadata(eng, "Clash Royale", "Ashtax", "Ashtax saves his tower at the last second.", avoid=used)
    assert meta
    first = " ".join(meta["title"].lower().split()[:3])
    assert first != "he was one", meta
    assert "ASHTAX" in meta["hook"], meta


GTA_RP_FR = [
    "Monsieur l'agent, on discutait avec ma voiture sur le parking, on s'est fait contrôler.",
    "Mais vous avez bu là, tu pues là !",
    "Les keufs arrivent, fonce, fonce, prends la caisse !",
    "On a raté le braquage à cause de toi, t'as parlé au flic !",
    "Il est parti en garde à vue, on va le chercher au comico.",
    "Chat, regardez ce qu'il fait, il est complètement 300 QI là.",
]


def test_gta_rp_translation_uses_gta_vocabulary():
    from clipper import games
    out = llm.translate_lines(GTA_RP_FR, context="Game: Grand Theft Auto V. " + games.glossary_text("gta"))
    assert english_ratio([t for t in out if t]) == 1.0, out
    blob = " ".join(out).lower()
    assert "cop" in blob or "police" in blob, out
    assert "heist" in blob or "robbery" in blob, out
    assert "chat" in blob and " cat" not in blob, out


def test_gta_judge_separates_viral_from_ordinary():
    """Vrais passages GTA RP : best-of (viraux) contre passages au hasard de lives (ordinaires)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import gta_eval
    items = gta_eval.load(test_only=True)
    if len(items) < 20:
        pytest.skip("jeu de donnees GTA absent (tests/gta_build.py)")
    gta_eval.SHOTS = True
    r = gta_eval.evaluate(items)
    # mesure avec qwen3.5:9b sur 51 viraux / 50 ordinaires : AUC 0.82, gardes (>= 6) 43 % / 4 %, jeu reconnu 99 %
    assert r["auc"] >= 0.72, r
    assert r["kept_neg"] <= 0.15, r     # presque tous les moments ordinaires sont rejetes
    assert r["kept_pos"] >= 0.33, r
    assert r["game_ok"] >= 0.8, r
