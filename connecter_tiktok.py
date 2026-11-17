"""Connexion TikTok : obtient les jetons (access + refresh) et les enregistre dans data/config.json.

Lancement : double-clic sur « Connecter TikTok.bat ».
"""
from __future__ import annotations

import getpass
import secrets
import sys
import webbrowser
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from clipper import config

SCOPES = "user.info.basic,video.publish"
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"


def main() -> int:
    cfg = config.load()
    key = cfg.get("tiktok_client_key", "").strip()
    secret = cfg.get("tiktok_client_secret", "").strip()

    print("=== Connexion TikTok ===\n")
    if key:
        print(f"Client key trouvee dans la config ({key[:4]}...).")
        if input("La garder ? (Entree = oui, n = en saisir une autre) : ").strip().lower() == "n":
            key = ""
    if not key:
        key = input("Colle le Client key de la SANDBOX : ").strip()
    if not secret or input("Garder le Client secret enregistre ? (Entree = oui, n = non) : ").strip().lower() == "n":
        secret = getpass.getpass("Colle le Client secret de la SANDBOX (il ne s'affiche pas) : ").strip()
    redirect = cfg.get("tiktok_redirect_uri", "").strip()
    if not redirect:
        print("\nTikTok exige une page web (https) ou il te renvoie apres la connexion.")
        print("C'est l'adresse que tu as mise dans 'Redirect URI' de ton app TikTok (ex. un site Netlify gratuit).")
        redirect = input("Colle cette adresse (Redirect URI) : ").strip()
    if not key or not secret or not redirect:
        print("Client key, secret ou Redirect URI manquant.")
        return 1
    config.save({"tiktok_client_key": key, "tiktok_client_secret": secret, "tiktok_redirect_uri": redirect})

    state = secrets.token_urlsafe(16)
    url = AUTH_URL + "?" + urlencode({
        "client_key": key, "scope": SCOPES, "response_type": "code",
        "redirect_uri": redirect, "state": state,
    })
    print("\n1) Une page TikTok va s'ouvrir. Connecte-toi avec le compte ajoute dans Target Users et autorise l'app.")
    print("2) Tu arriveras sur ta page de redirection. Copie TOUTE l'adresse de la barre du navigateur.\n")
    print("Si la page ne s'ouvre pas, copie ce lien dans ton navigateur :\n" + url + "\n")
    webbrowser.open(url)

    back = input("Colle ici l'adresse complete de la page de redirection : ").strip()
    q = parse_qs(urlparse(back).query)
    if "error" in q:
        print(f"TikTok a refuse : {q.get('error')} {q.get('error_description', '')}")
        return 1
    if q.get("state", [""])[0] != state:
        print("L'adresse collee ne correspond pas a cette connexion (state different). Relance le script.")
        return 1
    code = q.get("code", [""])[0]
    if not code:
        print("Pas de code dans l'adresse collee. Verifie que tu as copie l'adresse en entier.")
        return 1

    r = requests.post(TOKEN_URL, data={
        "client_key": key, "client_secret": secret, "code": code,
        "grant_type": "authorization_code", "redirect_uri": redirect,
    }, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    d = r.json()
    if "access_token" not in d:
        print(f"Echec de l'echange du code : {d}")
        return 1

    config.save({
        "tiktok_access_token": d["access_token"],
        "tiktok_refresh_token": d.get("refresh_token", ""),
        # Une app non validee par TikTok ne peut publier qu'en prive.
        "tiktok_privacy": "SELF_ONLY",
    })
    days = int(d.get("refresh_expires_in", 0)) // 86400
    print("\nTikTok connecte ! Jetons enregistres dans data/config.json.")
    print("Visibilite TikTok reglee sur SELF_ONLY (prive) tant que l'app n'est pas validee.")
    if days:
        print(f"Le refresh token est valable environ {days} jours ; l'app renouvelle l'acces toute seule.")
    print(f"Scopes accordes : {d.get('scope', '?')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
