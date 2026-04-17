@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://localhost:8501"
set "EXIT_CODE=0"

echo [INFO] Iniciando Road AI...

if not exist "venv\Scripts\python.exe" (
	echo [INFO] No se encontro el entorno virtual. Creando venv...
	where py >nul 2>nul
	if %errorlevel%==0 (
		py -m venv venv
	) else (
		where python >nul 2>nul
		if %errorlevel%==0 (
			python -m venv venv
		) else (
			echo [ERROR] No se encontro Python en el sistema.
			echo [ERROR] Instala Python 3.10+ y vuelve a ejecutar este archivo.
			set "EXIT_CODE=1"
			goto :END
		)
	)
)

call venv\Scripts\activate.bat
if errorlevel 1 (
	echo [ERROR] No se pudo activar el entorno virtual.
	set "EXIT_CODE=1"
	goto :END
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
	if exist requirements.txt (
		echo [INFO] Instalando dependencias desde requirements.txt...
		python -m pip install -r requirements.txt
		if errorlevel 1 (
			echo [ERROR] Fallo la instalacion de dependencias.
			set "EXIT_CODE=1"
			goto :END
		)
	) else (
		echo [ERROR] Streamlit no esta instalado y no existe requirements.txt.
		set "EXIT_CODE=1"
		goto :END
	)
)

echo [INFO] Abriendo navegador en %APP_URL%
start "" "%APP_URL%"

echo [INFO] Lanzando aplicacion...
streamlit run src/app.py --server.maxUploadSize 10240
set "EXIT_CODE=%errorlevel%"

:END
echo.
if not "%EXIT_CODE%"=="0" (
	echo [INFO] El lanzador termino con errores. Codigo: %EXIT_CODE%
)
pause
endlocal & exit /b %EXIT_CODE%