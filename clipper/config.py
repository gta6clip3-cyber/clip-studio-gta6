"""Configuration de l'application (stockee dans data/config.json)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
WORK = ROOT / "work"
ASSETS = ROOT / "assets"
CONFIG_FILE = DATA / "config.json"

for _d in (DATA, OUTPUT, WORK, ASSETS):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULTS: dict = {
    # --- Jeu cible ---
    "game_name": "Grand Theft Auto VI",
    "game_keywords": ["gta 6", "gta vi", "gta6", "grand theft auto vi"],

    # --- Clips ---
    "clip_target_seconds": 60,
    "clip_min_seconds": 40,
    "clip_max_seconds": 75,
    "max_clips_per_vod": 6,
    "candidates_to_analyze": 14,

    # --- IA locale (gratuite) ---
    "whisper_model": "large-v3",          # traduit vers l'anglais
    "whisper_device": "cuda",             # "cuda" ou "cpu"
    "whisper_compute_type": "int8_float16",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen3.5:9b",         # le mieux classe des 6 modeles mesures ; il lui faut 8 Go de memoire video

    # --- Montage ---
    "width": 1080,
    "height": 1920,
    "facecam_height": 640,                # haut = facecam, bas = jeu
    "game_widen": 1.3,                    # 1.0 = jeu plein cadre ; plus grand = on voit plus de la scene
    "subtitle_font": "Arial Black",
    "subtitle_size": 68,                  # taille des sous-titres (avant : 92)
    "hook_seconds": 10,                   # duree d'affichage de la phrase d'accroche en haut du clip
    "speaker_colors": True,               # une couleur par personne qui parle
    "speaker_threshold": 0.4,             # detecteur de tours de parole (pyannote)
    "speaker_max_inter_similarity": 0.45, # plus haut = separe plus facilement les voix (risque de fausses couleurs)
    "speaker_models": ["nemo_en_titanet_large"],  # empreinte vocale (TitaNet, le meilleur sur voix FR au banc)
    "subtitle_color": "&H00FFFFFF",       # blanc (ASS BGR)
    "subtitle_highlight": "&H0000E5FF",   # jaune
    "title_font": "Impact",
    "music_volume": 0.10,                 # musique de fond (dossier assets/music)
    "sfx_volume": 0.55,
    "use_nvenc": True,

    # --- Detection de facecam par streamer (optionnel, sinon auto) ---
    # "nom_streamer": [x, y, w, h] en fraction de l'image (0-1)
    "facecam_overrides": {},

    # --- Live ---
    "live_buffer_minutes": 8,
    "live_cooldown_seconds": 150,
    "live_spike_threshold": 2.6,
    "live_min_ai_score": 6,               # note IA minimale (sur 10) pour garder un clip en live

    # --- Decouverte ---
    "twitch_client_id": "",
    "twitch_client_secret": "",
    "youtube_api_key": "",

    # --- Publication ---
    "publish_mode": "manual",             # "manual" | "auto"
    "stats_interval_minutes": 15,         # releve des statistiques des comptes (tableau de bord)
    "auto_check_minutes": 5,              # mode auto : recherche de plus gros lives toutes les N minutes
    "auto_swap_margin": 0.2,              # un live remplace le plus petit suivi s'il a 20 % de spectateurs en plus
    "auto_min_watch_minutes": 15,         # ... et si ce dernier est suivi depuis au moins 15 min
    "tiktok_app_approved": False,         # True une fois l'app validee par TikTok (publication publique)
    "tiktok_auto_draft": False,          # deposer chaque nouveau clip dans les brouillons TikTok (rien n'est publie)
    "publish_youtube": False,
    "publish_tiktok": False,
    "publish_instagram": False,
    "tiktok_access_token": "",
    "tiktok_refresh_token": "",
    "tiktok_client_key": "",
    "tiktok_client_secret": "",
    "tiktok_redirect_uri": "",            # page https declaree dans ton app TikTok
    "tiktok_privacy": "SELF_ONLY",        # PUBLIC_TO_EVERYONE apres validation de l'app TikTok
    "instagram_user_id": "",
    "instagram_access_token": "",
    "youtube_privacy": "public",
    "credit_streamer": True,
}

_lock = threading.Lock()


def load() -> dict:
    with _lock:
        cfg = dict(DEFAULTS)
        if CONFIG_FILE.exists():
            try:
                cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        return cfg


def save(values: dict) -> dict:
    with _lock:
        cfg = dict(DEFAULTS)
        if CONFIG_FILE.exists():
            try:
                cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        for k, v in values.items():
            if k in DEFAULTS:
                cfg[k] = v
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return cfg
