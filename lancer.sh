#!/bin/sh
# Linux : lance Clip Studio (meme role que Lancer.bat sous Windows)
cd "$(dirname "$0")" && exec .venv/bin/python run.py
