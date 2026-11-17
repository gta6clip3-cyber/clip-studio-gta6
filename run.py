"""Lance l'application : python run.py  (ou double-clic sur Lancer.bat)."""
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8765


def ensure_ollama() -> None:
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    try:
        import requests
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return
    except Exception:
        pass
    if exe.exists():
        subprocess.Popen([str(exe), "serve"], creationflags=0x08000000)


if __name__ == "__main__":
    import uvicorn

    ensure_ollama()
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(f"http://127.0.0.1:{PORT}")), daemon=True).start()
    uvicorn.run("clipper.server:app", host="127.0.0.1", port=PORT, log_level="warning")
