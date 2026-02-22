# Road AI Project

Pipeline en Python para:

1. Parsear metadatos GPS desde un archivo `.srt` del dron.
2. Calcular distancia recorrida entre frames.
3. Seleccionar frames cada **N metros** (por defecto 7 m).
4. Extraer del video esos frames seleccionados como imágenes `.jpg`.

---

## Tabla de contenido

- [Descripción general](#descripción-general)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Flujo completo paso a paso](#flujo-completo-paso-a-paso)
- [Detalle de funcionalidades por script](#detalle-de-funcionalidades-por-script)
- [Entradas y salidas](#entradas-y-salidas)
- [Personalización](#personalización)
- [Solución de problemas](#solución-de-problemas)
- [Comandos rápidos](#comandos-rápidos)

---

## Descripción general

El proyecto toma la telemetría GPS embebida en `data/gps_logs/drone_data.srt`, la convierte en un CSV limpio (`output/drone_metadata.csv`), calcula distancias geográficas entre frames consecutivos y genera una lista de frames espaciados por distancia (`output/frames_every_7m_real.csv`).

Luego, usa esa lista para extraer imágenes del video `data/raw_video/video.MP4` y guardarlas en `output/frames/frames_7m`.

---

## Estructura del proyecto

```text
road_ai_project/
├── README.md
├── requirements.txt
├── data/
│   ├── gps_logs/
│   │   └── drone_data.srt
│   └── raw_video/
│       └── video.MP4
├── output/
│   ├── drone_metadata.csv
│   ├── frames_every_7m_real.csv
│   ├── frames/
│   │   └── frames_7m/
│   └── processed/
└── src/
		├── parser.py
		├── sync_frames_with_gps.py
		└── extract_selected_frames.py
```

---

## Requisitos

- Python 3.10+ (recomendado 3.11 o superior)
- `pip`
- Archivo SRT con metadatos GPS en `data/gps_logs/drone_data.srt`
- Video fuente en `data/raw_video/video.MP4`

Dependencias principales (incluidas en `requirements.txt`):

- `pandas`
- `haversine`
- `opencv-python`
- `numpy`
- `matplotlib` (disponible si luego quieres graficar)

---

## Instalación

### 1) Clonar / abrir el proyecto

Abre esta carpeta como workspace en VS Code.

### 2) Crear entorno virtual

En PowerShell (desde la raíz del proyecto):

```powershell
python -m venv venv
```

### 3) Activar entorno virtual

```powershell
venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts, ejecuta (solo para usuario actual):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4) Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## Flujo completo paso a paso

> Ejecuta los comandos desde la raíz del proyecto.

### Paso 1: Parsear metadatos GPS desde el SRT

```powershell
python src/parser.py
```

**Qué hace:**

- Lee `data/gps_logs/drone_data.srt`.
- Extrae `frame_number`, `lat`, `lon` con regex.
- Genera `output/drone_metadata.csv`.

### Paso 2: Seleccionar frames cada 7 metros (distancia geográfica)

```powershell
python src/sync_frames_with_gps.py
```

**Qué hace:**

- Lee `output/drone_metadata.csv`.
- Calcula distancia entre cada par de puntos consecutivos con fórmula haversine.
- Acumula distancia y selecciona un frame cuando llega a 7 m.
- Genera `output/frames_every_7m_real.csv`.

### Paso 3: Extraer imágenes de los frames seleccionados

```powershell
python src/extract_selected_frames.py
```

**Qué hace:**

- Lee `output/frames_every_7m_real.csv` (si no existe, intenta `output/processed/frames_every_7m_real.csv`).
- Abre `data/raw_video/video.MP4`.
- Extrae los frames indicados y guarda JPEGs en `output/frames/frames_7m`.

---

## Detalle de funcionalidades por script

### `src/parser.py`

- Parseo del archivo SRT por bloques.
- Extracción robusta por expresiones regulares:
  - `FrameCnt`
  - `latitude`
  - `longitude`
- Conversión a DataFrame y exportación CSV.

### `src/sync_frames_with_gps.py`

- Carga de metadata por frame.
- Cálculo de distancia frame a frame con `haversine` en metros.
- Acumulación de distancia y selección por intervalo fijo (`DISTANCE_INTERVAL = 7`).
- Exportación de frames seleccionados.

### `src/extract_selected_frames.py`

- Resolución de rutas basada en la ubicación del script (`BASE_DIR`).
- Lectura de CSV principal y fallback a ruta legacy.
- Apertura y validación del video con OpenCV.
- Posicionamiento directo por número de frame (`CAP_PROP_POS_FRAMES`).
- Exportación de imágenes JPG y progreso en consola.

---

## Entradas y salidas

### Entradas

- `data/gps_logs/drone_data.srt`
- `data/raw_video/video.MP4`

### Salidas

- `output/drone_metadata.csv` → metadatos por frame (`frame_number, lat, lon`)
- `output/frames_every_7m_real.csv` → frames seleccionados por distancia
- `output/frames/frames_7m/frame_<n>.jpg` → imágenes extraídas

---

## Personalización

### Cambiar distancia de muestreo

En `src/sync_frames_with_gps.py`, modifica:

```python
DISTANCE_INTERVAL = 7  # metros
```

Ejemplos:

- `5` para extraer más denso (cada 5 m)
- `10` para extraer más espaciado (cada 10 m)

### Cambiar nombre/ruta del video

En `src/extract_selected_frames.py`, ajusta:

```python
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_video", "video.MP4")
```

---

## Solución de problemas

### 1) `No se pudo abrir el video`

- Verifica que exista `data/raw_video/video.MP4`.
- Revisa mayúsculas/minúsculas del nombre (`video.MP4`).
- Prueba abrir ese archivo manualmente para validar que no esté corrupto.

### 2) `No se encontró el CSV...`

- Ejecuta antes:
  1.  `python src/parser.py`
  2.  `python src/sync_frames_with_gps.py`

### 3) Se extraen pocos frames

- Es normal si el dron recorrió poca distancia o si `DISTANCE_INTERVAL` es alto.
- Reduce `DISTANCE_INTERVAL` (por ejemplo 7 → 5).

### 4) Error al activar entorno virtual en PowerShell

- Ejecuta:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Comandos rápidos

Con el entorno virtual activo:

```powershell
python src/parser.py
python src/sync_frames_with_gps.py
python src/extract_selected_frames.py
```

---

## Resultado esperado

Al finalizar, tendrás:

- CSV de metadata GPS por frame.
- CSV de frames seleccionados por distancia.
- Carpeta de imágenes (`.jpg`) listas para análisis, etiquetado o entrenamiento de modelos de visión por computador.
