# Clip Studio : clips GTA 6 automatiques

Crée des clips verticaux (TikTok, Shorts, Reels) à partir de lives et rediffusions Twitch, YouTube et Kick :
détection des moments forts, sous-titres, montage et publication. Tout tourne en local, gratuitement.

## Installation (une seule fois, Windows)
1. Installe **Python 3.13**, **[FFmpeg](https://www.gyan.dev/ffmpeg/builds/)** (ajouté au PATH) et
   **[Ollama](https://ollama.com)**, puis télécharge le modèle d'IA : `ollama pull qwen2.5:7b`.
   Une carte graphique NVIDIA est conseillée (sinon mets `"whisper_device": "cpu"` dans `data/config.json`).
2. Dans le dossier du projet :
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Télécharge les modèles dans `assets/models/` (ils sont trop lourds pour GitHub) :
   - [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) (détection de la facecam)
   - [nemo_en_titanet_large.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx) et
     [wespeaker_en_voxceleb_resnet34_LM.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx) (reconnaissance des voix)
   - [sherpa-onnx-pyannote-segmentation-3-0.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2), à décompresser dans `assets/models/`
4. **Tes propres comptes** : rien n'est fourni, chacun met ses identifiants. Lance l'app, puis va dans **Paramètres**
   (Twitch, YouTube, TikTok, Instagram). Tout est enregistré dans `data/config.json`, qui reste sur ton PC et n'est jamais envoyé sur GitHub.
   - YouTube : mets ton fichier OAuth Google dans `data/youtube_client_secret.json`.
   - TikTok : double-clic sur `Connecter TikTok.bat` (il demande ton Client key, ton Client secret et ta Redirect URI).
   - Instagram : double-clic sur `Connecter Instagram.bat`.

## Lancer l'application
Double-clique sur **`Lancer.bat`**. L'interface s'ouvre dans ton navigateur (http://127.0.0.1:8765).
Laisse la fenêtre noire ouverte tant que tu utilises l'app.

## Utilisation
- **Créer** : colle le lien d'un live ou d'une rediffusion (Twitch, YouTube, Kick), puis clique sur « Créer les clips ».
  - *Rediffusion* : l'app télécharge l'audio et le chat, repère les moments forts, ne télécharge en 1080p que les passages retenus et sort les clips.
  - *Live* : l'app enregistre le live dans un tampon de 8 min, surveille le chat et l'audio, et crée un clip à chaque pic, jusqu'à ce que tu cliques sur « Arrêter ».
- **Mode automatique** : surveille en continu les N plus gros lives GTA 6 (il faut les identifiants Twitch).
- **Streamers du moment** : classement des lives et rediffusions GTA 6 les plus regardés, avec un bouton pour clipper.
- **Mes clips** : aperçu, titre, légende, cover, téléchargement, publication.

Chaque clip est rangé dans `output/<date>_<streamer>_<n>/` :
`clip.mp4` (1080×1920), `cover.jpg`, `title.txt`, `caption.txt` (avec hashtags), `meta.json`.

## Comment les moments forts sont détectés (100 % local, gratuit)
1. **Chat** : explosion du nombre de messages, emotes de rire ou de choc (KEKW, LUL, OMEGALUL, Pog, « clip », mdr…), messages en MAJUSCULES.
2. **Audio** : pics de volume (cris, rires, explosions).
3. **IA locale (Ollama + Qwen 2.5)** : lit la transcription de chaque candidat, note son potentiel viral de 1 à 10, choisit le début et la fin exacts, puis écrit le titre, l'accroche, la légende et les hashtags en anglais.
4. **Whisper large-v3 (GPU)** : transcrit la parole ; les sous-titres sont traduits en anglais.

## Montage
Facecam détectée automatiquement en haut, gameplay en bas (ou gameplay centré sur fond flou s'il n'y a pas de facecam),
sous-titres animés mot par mot façon TikTok, accroche affichée les 3 premières secondes, zoom sur le moment clé,
effets sonores (whoosh et impact), barre de progression, @streamer, son normalisé à -14 LUFS.
Musique de fond : dépose des musiques **libres de droits** dans `assets/music`. Elle est baissée automatiquement quand le streamer parle.

Si la facecam d'un streamer est mal détectée, ajoute dans `data/config.json` :
`"facecam_overrides": {"nomdustreamer": [x, y, largeur, hauteur]}` (fractions de l'image entre 0 et 1, ex. `[0.75, 0.0, 0.25, 0.3]`).

**Sous-titres** : toujours en anglais (chaque phrase est vérifiée ; une phrase impossible à traduire est retirée plutôt
qu'affichée en français). Une couleur par personne qui parle : blanc/jaune pour la voix principale, puis cyan, rose,
vert… La reconnaissance utilise TitaNet (NVIDIA), qui distingue aussi deux voix d'hommes sur le même micro
(banc d'essai sur 5 vraies voix de streamers : 87 % des mots bien attribués, aucune fausse couleur sur une voix seule).
Les petites répliques très courtes (« ouais », « ok ») restent les plus difficiles.

Réglages utiles dans `data/config.json` :
- `"subtitle_size": 68` : taille des sous-titres ;
- `"speaker_colors": true` : couleurs par voix (mettre `false` pour tout en blanc/jaune) ;
- `"speaker_max_inter_similarity": 0.45` : plus haut = sépare plus facilement les voix (risque de fausses couleurs) ;
- `"game_widen": 1.3` : plus grand = on voit plus de la scène de jeu.

## Entraînement GTA
L'IA a été entraînée sur de vrais extraits GTA RP français : 24 moments de compilations « best of » (moments
viraux choisis par des humains : Anyme, Walid, Flashback, Flashland…) contre 20 passages pris au hasard dans des lives
complets. Elle connaît maintenant ce qui fait un bon clip GTA (courses-poursuites, mensonges aux flics, braquages
ratés, punchlines RP…), ce qui est banal (logistique, menus, conduite tranquille), et le vocabulaire GTA RP
(keufs, braquage, comico, garde à vue…). Ces connaissances sont dans `clipper/games.py`.
Mesure sur des vidéos jamais vues : elle sépare les moments viraux des ordinaires avec une AUC de 0,81 (0,65 avant)
et rejette 87 à 93 % des moments ordinaires (10 % avant).
Pour refaire la mesure : `tests\gta_build.py` puis `tests\gta_eval.py --test-only --shots`.

## Tests
`.venv\Scripts\python.exe -m pytest tests` lance la cinquantaine de tests (sous-titres, langue, voix, IA locale).
`.venv\Scripts\python.exe tests\e2e_vod.py <lien> 2` fait un vrai essai complet sur une vidéo (sans publier).
`.venv\Scripts\python.exe tests\bench_build.py` puis `tests\bench_eval.py` : banc d'essai des voix (vraies voix de streamers).

## Publication automatique
Dans **Paramètres** : connecte YouTube, TikTok et Instagram (guide détaillé dans l'onglet), puis choisis
« Mode de publication : auto ». Par défaut le mode est **manuel** : tu vérifies chaque clip et tu cliques sur « Publier ».

Limites des plateformes :
- **YouTube** : environ 6 vidéos par jour avec le quota gratuit de l'API.
- **TikTok** : tant que TikTok n'a pas validé ton app (audit), les vidéos sont publiées en **privé**.
- **Instagram** : il faut un compte professionnel relié à une page Facebook.

## Droits
Le contenu des streams appartient aux streamers. Avant de publier, vérifie qu'ils autorisent les clippeurs
(beaucoup ont des programmes de clipping officiels) et respecte les règles de chaque plateforme.
