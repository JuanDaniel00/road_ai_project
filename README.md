# Road AI Inspector

Pipeline de inspección vial con dron. Procesa video aéreo y telemetría GPS para extraer y mejorar frames espaciados por distancia, listos para análisis con modelos de visión artificial.

---

## Qué hace

1. Parsea metadatos GPS desde el archivo `.srt` del dron.
2. Calcula la distancia recorrida entre frames con fórmula haversine.
3. Selecciona un frame cada **N metros** (por defecto 7 m).
4. Extrae esos frames del video como imágenes `.jpg`.
5. Aplica un pipeline de mejora de imagen (Bilateral → CLAHE → Unsharp Mask → Normalización de color) en paralelo usando todos los núcleos disponibles.
6. Empaca todo en un `.zip` descargable desde la interfaz.

---

## Estructura del proyecto

```
road_ai_project/
├── run.bat               ← doble clic para lanzar la app
├── requirements.txt
├── src/
│   └── app.py            ← aplicación Streamlit (pipeline completo)
└── venv/                 ← entorno virtual Python
```

Los resultados se guardan automáticamente en `road_ai_output/` dentro de la misma carpeta donde está el video.

---

## Requisitos

- Python 3.10 o superior (recomendado 3.11+)
- Windows (el `.bat` usa rutas de Windows)
- Video del dron en cualquier ruta local del equipo
- Archivo `.srt` con telemetría GPS embebida (generado por el dron)

---

## Instalación

### 1. Crear entorno virtual

Abre PowerShell en la raíz del proyecto:

```powershell
python -m venv venv
```

### 2. Activar entorno virtual

```powershell
venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución de scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## Uso

Haz doble clic en `run.bat`. Se abre el navegador con la interfaz automáticamente.

En la interfaz:

1. **Escribe la ruta completa al video** (ej. `C:\Vuelos\vuelo1.MP4`). No se sube — se lee directo del disco para evitar problemas de memoria con archivos grandes.
2. **Sube el archivo `.srt`** con la telemetría GPS.
3. Ajusta el **intervalo de muestreo** en metros si lo necesitas.
4. Abre el panel de **parámetros de mejora de imagen** si quieres afinar los filtros.
5. Presiona **Ejecutar pipeline** y espera a que los 4 pasos terminen.
6. Descarga el `.zip` con todos los resultados.

---

## Modelo usado en pruebas

Para las pruebas y validaciones del proyecto se está usando un **modelo/agente entrenado propiamente para detección de baches en vista aérea**.

El modelo por defecto/base (por ejemplo uno genérico como `yolov8n.pt`) se mantiene como referencia rápida, pero **no ofrece el mismo rendimiento** en este caso de uso: suele presentar menor precisión y más falsos positivos/negativos frente al modelo entrenado específicamente para este dominio.

Si vas a evaluar resultados o hacer demos, se recomienda usar siempre el modelo entrenado propio.

---

## Parámetros de mejora de imagen

| Parámetro        | Valor por defecto | Efecto                                                             |
| ---------------- | ----------------- | ------------------------------------------------------------------ |
| CLAHE clip limit | 3.0               | Contraste local. Más alto = más realce. >4 puede amplificar ruido. |
| CLAHE tile size  | 8 px              | Tamaño de región local. 8 es el estándar.                          |
| Unsharp Mask     | 1.2               | Realce de bordes finos. >1.5 introduce halos.                      |

El denoising usa `bilateralFilter` con parámetros fijos optimizados para imágenes de dron — no requiere ajuste manual.

---

## Resultados

Al terminar el pipeline, el `.zip` descargado contiene:

```
road_ai_output/
├── frames_7m/            ← imágenes originales extraídas del video
├── frames_7m_enhanced/   ← imágenes con pipeline de mejora aplicado
└── metadata/
    ├── drone_metadata.csv     ← frame_number, lat, lon por cada frame GPS
    └── frames_selected.csv    ← frames seleccionados por distancia
```

---

## Dependencias

```
pandas
haversine
opencv-python
numpy
matplotlib
pillow
streamlit
```

---

## Solución de problemas

**La app no abre al hacer doble clic en `run.bat`**
Verifica que el entorno virtual se llame `venv` y esté en la raíz del proyecto. Si lo nombraste diferente, edita la línea `call venv\Scripts\activate.bat` en el `.bat`.

**"Ruta no válida o archivo no encontrado"**
Copia la ruta directamente desde el Explorador de Windows con Shift + clic derecho → "Copiar como ruta". Asegúrate de que el archivo sea `.mp4`, `.mov` o `.avi`.

**Se extraen pocos frames**
Normal si el dron recorrió poca distancia o el intervalo está muy alto. Reduce el intervalo de muestreo (ej. de 7 m a 5 m).

**El paso 4 (mejora) tarda mucho**
El pipeline usa todos los núcleos disponibles menos uno. En equipos con pocos núcleos el procesamiento es más lento. Puedes reducir el `Unsharp Mask` o el `CLAHE clip limit` para acortar el tiempo sin impacto significativo en la calidad.

**No aparece el botón de descarga**
Espera a que los 4 pasos muestren "COMPLETADO". El botón aparece solo cuando el pipeline termina correctamente.
