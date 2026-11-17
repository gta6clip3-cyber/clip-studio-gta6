"""Connexion Instagram : transforme un jeton court (Graph API Explorer) en jeton de 60 jours,
trouve l'ID du compte Instagram professionnel et enregistre le tout dans data/config.json.

Lancement : double-clic sur « Connecter Instagram.bat ».
"""
from __future__ import annotations

import getpass
import sys
import time

import requests

from clipper import config

GRAPH = "https://graph.facebook.com/v27.0"


def main() -> int:
    print("=== Connexion Instagram ===\n")
    app_id = input("Colle l'App ID de ton app Meta : ").strip()
    app_secret = getpass.getpass("Colle l'App Secret (il ne s'affiche pas) : ").strip()
    short = getpass.getpass("Colle le jeton genere dans Graph API Explorer (il ne s'affiche pas) : ").strip()
    if not (app_id and app_secret and short):
        print("Il manque une valeur.")
        return 1

    r = requests.get(f"{GRAPH}/oauth/access_token", params={
        "grant_type": "fb_exchange_token", "client_id": app_id,
        "client_secret": app_secret, "fb_exchange_token": short}, timeout=30).json()
    if "access_token" not in r:
        print(f"Echec de l'echange du jeton : {r.get('error', r)}")
        return 1
    token = r["access_token"]
    expires = int(r.get("expires_in") or 60 * 86400)

    pages = requests.get(f"{GRAPH}/me/accounts", params={
        "fields": "name,instagram_business_account{id,username}", "access_token": token},
        timeout=30).json()
    if "error" in pages:
        print(f"Impossible de lire tes pages Facebook : {pages['error']}")
        return 1
    accounts = [(p["name"], p["instagram_business_account"]) for p in pages.get("data", [])
                if p.get("instagram_business_account")]
    if not accounts:
        print("Aucun compte Instagram professionnel relie a tes pages Facebook.")
        print("Verifie que le compte est en mode professionnel, relie a une page, et que tu as")
        print("coche la page et le compte Instagram lors de la generation du jeton.")
        return 1

    if len(accounts) == 1:
        page, ig = accounts[0]
    else:
        for i, (page, ig) in enumerate(accounts, 1):
            print(f"  {i}. @{ig.get('username', '?')} (page {page})")
        n = input("Numero du compte a utiliser : ").strip()
        try:
            page, ig = accounts[int(n) - 1]
        except (ValueError, IndexError):
            print("Choix invalide.")
            return 1

    config.save({"instagram_user_id": ig["id"], "instagram_access_token": token})
    end = time.strftime("%d/%m/%Y", time.localtime(time.time() + expires))
    print(f"\nInstagram connecte : @{ig.get('username', '?')} (page {page}).")
    print("Identifiants enregistres dans data/config.json.")
    print(f"Le jeton expire vers le {end} : relance ce script avant cette date.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
