from clipper.textlang import english_ratio, is_english

FRENCH = [
    "Ça va être compliqué.",
    "Mais j'ai plus rien pour défendre !",
    "C'est trop dur aïe aïe aïe",
    "Oh il va connecter.",
    "Ah c'est dommage franchement t'étais vraiment pas loin.",
    "Ouais parce que du coup bah ouais la mousquetaire",
    "Écoute je vais te demander de refaire un nouveau combo ?",
    "On va faire un autre deck, un autre truc.",
    "Vas-y dis-moi.",
    "Non mais il est fou lui, regarde ce qu'il fait",
    "Attends attends attends",
    "Putain le mec il m'a tué",
    "je suis mort",
    "les gars vous avez vu ça",
]
ENGLISH = [
    "Oh it's beautiful",
    "Let's go man",
    "Does he have the zap?",
    "He's really pissing people off with that ice",
    "This is going straight to Twitter!",
    "I don't have lightning, that's the problem",
    "We're a little behind.",
    "The baby dragon, ladies and gentlemen",
    "OK",
    "Oh!",
    "Bro what",
    "No way, this guy killed me again",
    "Elixir advantage",
]
SPANISH = ["No puedo creer lo que está pasando", "Vamos vamos, mira eso", "¿Qué hace este tío?"]


def test_french_lines_detected():
    bad = [ln for ln in FRENCH if is_english(ln)]
    assert not bad, bad


def test_english_lines_accepted():
    bad = [ln for ln in ENGLISH if not is_english(ln)]
    assert not bad, bad


def test_spanish_lines_detected():
    assert not any(is_english(ln) for ln in SPANISH)


def test_ratio():
    assert english_ratio(ENGLISH) == 1.0
    assert english_ratio(FRENCH) == 0.0
    assert english_ratio([]) == 1.0
