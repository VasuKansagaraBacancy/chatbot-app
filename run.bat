@echo off
cd /d "%~dp0"
echo Starting chatbot on http://127.0.0.1:8000 ...
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
