@echo off
cd /d "%~dp0"
echo Clip Studio demarre... (ne ferme pas cette fenetre)
".venv\Scripts\python.exe" run.py
pause
