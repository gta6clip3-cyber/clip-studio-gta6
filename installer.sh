#!/bin/sh
# Linux : prepare tout ce qu'il faut pour Clip Studio. A lancer une fois ; on peut le relancer sans risque.
# Ne demande jamais de mot de passe : s'il manque un programme, il le dit et s'arrete.
set -e
cd "$(dirname "$0")"

manque=0
for outil in python3 ffmpeg curl tar bzip2 ollama; do
    command -v "$outil" >/dev/null 2>&1 || { echo "Il manque : $outil"; manque=1; }
done
if [ "$manque" = 1 ]; then
    echo "Installe-les avec les paquets de ta distribution (Ollama : https://ollama.com/download), puis relance."
    exit 1
fi
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 ||
    echo "Attention : ce ffmpeg n'a pas libx264 (Fedora : RPM Fusion ; openSUSE : Packman). Sans carte NVIDIA, le montage echouera."

echo "1/4 Environnement Python..."
[ -x .venv/bin/python ] || python3 -m venv .venv || {
    echo "Creation de .venv impossible. Sur Debian : sudo apt install python3-venv"
    exit 1
}
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
if command -v nvidia-smi >/dev/null 2>&1; then # carte NVIDIA : bibliotheques CUDA pour Whisper
    .venv/bin/python -m pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12
fi

echo "2/4 Modeles de detection (visage, voix)..."
mkdir -p assets/models
prendre() { # prendre <url> <fichier> : ne retelecharge pas ce qui est deja la
    [ -e "assets/models/$2" ] && return
    curl -fL --retry 3 -o "assets/models/$2.part" "$1"
    mv "assets/models/$2.part" "assets/models/$2"
}
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download
prendre https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx face_detection_yunet_2023mar.onnx
prendre $SHERPA/speaker-recongition-models/nemo_en_titanet_large.onnx nemo_en_titanet_large.onnx
prendre $SHERPA/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx wespeaker_en_voxceleb_resnet34_LM.onnx
if [ ! -d assets/models/sherpa-onnx-pyannote-segmentation-3-0 ]; then
    curl -fL --retry 3 $SHERPA/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 |
        tar -xj -C assets/models
fi

echo "3/4 Whisper (transcription, environ 3 Go la premiere fois)..."
.venv/bin/python -c "
from faster_whisper import download_model
from clipper import config
download_model(config.load()['whisper_model'])"

echo "4/4 IA locale (Ollama)..."
ollama list >/dev/null 2>&1 || {
    ollama serve >/dev/null 2>&1 &
    sleep 3
}
ollama pull "$(.venv/bin/python -c "from clipper import config; print(config.load()['ollama_model'])")"

echo "Pret. Lance l'application avec : ./lancer.sh"
