"""Connaissances par jeu : ce qui fait un bon clip, le vocabulaire (FR -> EN) et le style de titres qui marchent.

Le profil est choisi d'apres le titre / les tags du stream (puis confirme par l'IA)."""
from __future__ import annotations

PROFILES: dict[str, dict] = {
    "gta": {
        "name": "Grand Theft Auto (GTA RP / GTA Online / story)",
        "keywords": ["gta", "grand theft", "fivem", " rp ", "gtarp", "gta rp", "los santos", "vice city", "flashback",
                     "grand line", "grandline", "nopixel", "flashland", "lspd", "bcso", "roleplay", "paris rp",
                     "school rp", "leonida"],
        "judge": (
            "GTA (mostly GTA RP = roleplay on FiveM servers like Flashback, Grand Line, NoPixel). "
            "Viral GTA moments: police chases and wild escapes, getting arrested or searched in absurd ways, "
            "lying to cops with a ridiculous excuse, heists or robberies that go wrong, trolling other players, "
            "insane driving / crashes / explosions, gunfights, betrayals inside a gang, rage and screaming, "
            "a roleplay punchline that makes everyone laugh, court / jail / hospital scenes that turn absurd, "
            "a twist nobody expected. "
            "Ordinary (low score) GTA RP: calm in-character conversations without a punchline, planning or "
            "logistics (where is my car, meet me at...), driving quietly, buying / crafting / inventory, phone menus, "
            "waiting, explaining server rules, long serious drama monologues, out-of-character (HRP) talk with chat."),
        "glossary": {
            "flic / keuf / condé / les bleus": "cop / the cops", "LSPD / BCSO": "LSPD / BCSO (police)",
            "braquage": "heist / robbery", "go-fast": "drug run", "caisse / gova": "car", "bécane": "bike",
            "gun fight": "shootout", "drive-by": "drive-by", "orga": "crew", "gang": "gang",
            "HRP / hors RP": "out of character", "comico": "police station", "menottes": "cuffs",
            "fourrière": "impound", "amende": "fine", "taule / zonzon / prison": "jail", "garde à vue": "custody",
            "300 QI": "300 IQ (galaxy brain)", "PLS": "knocked out on the floor", "le maire": "the mayor",
            "le directeur": "the director", "tu pues": "you stink", "la ville": "the city",
            "fdp / ntm": "(insult -> 'jerk')"},
        "examples": (
            "Example of an ORDINARY GTA RP window (payoff 1, reaction 2, standalone 1, boring 8 -> no clip):\n"
            "  \"Appuie sur confirmer.\" / \"Ah, attends, bouge pas.\" / \"Je te donne un petit quelque chose.\" / "
            "\"Je comprends rien !\" / \"Regarde dans tes poches si tu as bien le petit.\" / \"Ah je l'ai.\" / "
            "\"Vas-y, refais ce que tu viens de faire.\"  (logistics, menus, nothing happens)\n"
            "Example of a VIRAL GTA RP window (payoff 8, reaction 8, standalone 8, boring 1 -> great clip):\n"
            "  \"On va sûrement aller au bar. Et au club de striptease.\" / \"On va avoir plein de fesses qui ne sont "
            "pas les vôtres.\" / \"Vous avez entendu ce qu'il a dit ?\" / \"TA GUEULE !\" / \"Pardon, monsieur le "
            "directeur.\"  (a punchline in front of the girlfriends, instant outraged reaction, anyone gets it)"),
        "titles": [
            "He Lied To The Cops With The Worst Excuse Ever", "This Chase Lasted Way Too Long",
            "The Heist Was Perfect Until He Opened His Mouth", "Nobody Expected The Mayor To Do This",
            "He Escaped The Cops In The Dumbest Way Possible", "The Whole Server Was Crying Laughing"],
    },
    "clash royale": {
        "name": "Clash Royale",
        "keywords": ["clash royale", "clash", "cr fr", "deck", "elixir", "élixir"],
        "judge": (
            "Clash Royale. Viral moments: last-second tower takes, 1 HP saves, insane comebacks, a crazy combo or "
            "a huge misplay, rage at the opponent, a funny random deck that works. Ordinary: deck building menus, "
            "explaining cards, calm gameplay without a turning point."),
        "glossary": {"boule de feu": "fireball", "cochon / chevaucheur de cochon": "hog rider",
                     "bélier": "battle ram", "mousquetaire": "musketeer", "bûcheron": "lumberjack",
                     "méga chevalier": "mega knight", "fantôme": "royal ghost", "séisme": "earthquake",
                     "élixir": "elixir", "la tour du roi": "the king tower"},
        "titles": ["He Was At 1 HP And Still Won", "This Random Deck Should NOT Work"],
    },
}


def detect(*texts: str) -> str | None:
    """Profil du jeu d'apres le titre, les tags, la description ou le nom du jeu donne par l'IA."""
    blob = " " + " ".join(t for t in texts if t).lower() + " "
    best, best_n = None, 0
    for key, p in PROFILES.items():
        n = sum(blob.count(k) for k in p["keywords"])
        if n > best_n:
            best, best_n = key, n
    return best


def judge_hints(key: str | None) -> str:
    return PROFILES[key]["judge"] if key in PROFILES else ""


def judge_examples(key: str | None) -> str:
    return PROFILES[key].get("examples", "") if key in PROFILES else ""


def glossary_text(key: str | None) -> str:
    if key not in PROFILES:
        return ""
    return "Game vocabulary (French -> English): " + "; ".join(f"{a} = {b}" for a, b in PROFILES[key]["glossary"].items())


def title_examples(key: str | None) -> list[str]:
    return PROFILES[key]["titles"] if key in PROFILES else []
