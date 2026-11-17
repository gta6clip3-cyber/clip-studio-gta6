"""Detection rapide de la langue d'une ligne (sans modele) pour verifier que les sous-titres sont bien en anglais."""
from __future__ import annotations

import re

_SW = {
    "en": """the a an and or but is are was were be been to of in on at for with that this it its it's i you he she
        we they me my your his her our their what why how who where when not no yes yeah do does did don't can can't
        will would gonna wanna just so very really oh okay ok go going get got have has had there here look let's man bro
        guys im i'm you're we're that's what's dude all like now come out up down one two right wait damn god holy shit""",
    "fr": """le la les un une des de du et ou mais est sont était être à au aux en dans sur pour avec que qui ce cette
        ça il elle on nous vous ils elles je tu me te se mon ma mes ton ta tes son sa ses notre votre leur pas ne non
        oui ouais quoi pourquoi comment où quand c'est j'ai t'es y'a il y a fait faire va vas aller trop très bien bah
        euh bon voilà alors donc putain mec frère regarde attends vas-y chui suis es avez avoir peu plus rien tout tous
        là ici encore déjà aussi même comme parce qu'il qu'on j'suis vraiment truc gars wesh""",
    "es": """el la los las un una de del y o pero es son era ser en con por para que qué este esta eso yo tú él
        nosotros ellos mi tu su no sí muy bien vamos mira hay está estoy tengo hacer como porque cuando dónde todo""",
    "de": """der die das ein eine und oder aber ist sind war sein zu von in auf für mit dass dies ich du er sie wir
        ihr mein dein nicht nein ja was warum wie wer wo wann sehr gut los schau hier jetzt alles auch noch""",
    "it": """il lo la gli le un una di del e o ma è sono era essere in con per che questo questa io tu lui lei noi
        loro mio tuo suo non sì molto bene andiamo guarda c'è sto ho fare come perché quando dove tutto""",
    "pt": """o a os as um uma de do da e ou mas é são era ser em com por para que este esta isso eu você ele ela
        nós eles meu seu não sim muito bem vamos olha tem estou tenho fazer como porque quando onde tudo""",
}
STOP = {k: set(v.split()) for k, v in _SW.items()}
_ACCENTS = re.compile(r"[àâçéèêëîïôûùüÿœæñáíóúãõäöß]")
_TOKEN = re.compile(r"[a-zA-Zàâçéèêëîïôûùüÿœæñáíóúãõäöß']+")


def scores(text: str) -> dict[str, float]:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    if not toks:
        return {}
    out = {k: sum(t in sw for t in toks) / len(toks) for k, sw in STOP.items()}
    acc = len(_ACCENTS.findall(text.lower())) / max(1, len(text))
    out["en"] -= acc * 6  # l'anglais n'a quasiment pas d'accents
    return out


def is_english(text: str) -> bool:
    """True si la ligne est (probablement) en anglais. Les lignes trop courtes / ambigues comptent comme anglais."""
    text = text.strip()
    if not text:
        return True
    if _ACCENTS.search(text.lower()) and len(_TOKEN.findall(text)) >= 1:
        # un mot accentue -> quasi jamais de l'anglais (sauf noms propres : on tolere 1 accent dans une longue ligne)
        toks = _TOKEN.findall(text)
        if len(toks) < 6 or len(_ACCENTS.findall(text.lower())) >= 2:
            return False
    s = scores(text)
    if not s:
        return True
    best = max(s, key=s.get)
    if best == "en":
        return True
    # langue etrangere nettement devant l'anglais
    return not (s[best] >= 0.2 and s[best] > s["en"] + 0.08)


def english_ratio(lines: list[str]) -> float:
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return 1.0
    return sum(is_english(ln) for ln in lines) / len(lines)
