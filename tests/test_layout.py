from pathlib import Path

import pytest

from clipper.layout import compute_layout

FIX = Path(__file__).resolve().parent / "_fixtures"


@pytest.mark.skipif(not (FIX / "ashtax_cam.mp4").exists(), reason="extrait absent")
def test_real_webcam_not_decor_character():
    """Stream d'Ashtax : vraie webcam au centre droit, personnages Clash Royale dessines dans le decor."""
    lay = compute_layout(FIX / "ashtax_cam.mp4", "Ashtax")
    assert lay.has_cam
    x, y, w, h = lay.cam
    sw, sh = lay.src_w, lay.src_h
    cx, cy = (x + w / 2) / sw, (y + h / 2) / sh
    # la webcam est dans le cadre orange (environ x 50-95 %, y 27-72 %), pas en bas sur le decor
    assert 0.5 < cx < 0.95 and 0.27 < cy < 0.72, lay.cam
    gx, gy, gw, gh = lay.game
    assert (gx + gw) / sw < 0.52  # le jeu est a gauche


@pytest.mark.skipif(not (FIX / "gta_rp_cam.mp4").exists(), reason="extrait absent")
def test_gta_rp_small_webcam_overlay():
    """Stream GTA RP : petite webcam en haut a gauche posee sur le jeu en plein ecran."""
    lay = compute_layout(FIX / "gta_rp_cam.mp4", "streamer")
    assert lay.has_cam
    x, y, w, h = lay.cam
    sw, sh = lay.src_w, lay.src_h
    assert (x + w / 2) / sw < 0.2 and (y + h / 2) / sh < 0.35, lay.cam   # en haut a gauche
    assert w / sw >= 0.18, lay.cam                                       # toute la webcam, pas juste le visage
    gx, gy, gw, gh = lay.game
    assert gx >= x + w * 0.8 and gh / sh > 0.9, lay.game                 # le jeu, a cote de la webcam
