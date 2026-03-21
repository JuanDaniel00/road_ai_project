@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run src/app.py --server.maxUploadSize 10240
pause