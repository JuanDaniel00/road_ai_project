# Guía de etiquetado — dataset de baches (Road AI Inspector)

Este documento cubre el flujo para pasar de las fotografías mejoradas (`output/train|val|test/frames_enhanced/`) a un dataset etiquetado listo para reentrenar YOLOv8.

## 0. Contexto y volumen esperado

El batch (`src/batch_process.py`) genera entre 3500 y 4500 fotografías en total repartidas en train/val/test (ver conteo real en `output/<split>/frames_enhanced/` cuando termine). Etiquetar todo eso a mano una por una es poco realista para una sola persona. La estrategia recomendada es **etiquetado asistido por modelo** (secciones 3-4), no etiquetado manual completo.

## 1. Herramienta de etiquetado

Recomendada: **LabelImg** (offline, gratuita, exporta directo en formato YOLO `.txt`, no sube nada a internet — relevante porque son fotos de vía pública/infraestructura).

```powershell
pip install labelImg
labelImg
```

Alternativas si prefieres algo en navegador:
- **makesense.ai** — corre en el navegador, no requiere cuenta, exporta YOLO.
- **CVAT** (self-hosted) — mejor si en algún momento etiquetan entre varias personas; instalación más pesada (Docker).

Evita Roboflow salvo que te parezca bien subir las imágenes a su nube — no es necesario para este volumen.

## 2. Definición de clases (fijarla ANTES de etiquetar)

Decide esto primero y no lo cambies a mitad de camino, o tendrás que re-revisar lo ya hecho:

| id | clase | criterio |
|----|-------|----------|
| 0 | `pothole` | bache real: hueco con profundidad visible en el pavimento |
| 1 | `crack` *(opcional)* | grieta/fisura sin hundimiento — sólo si te interesa distinguirla del bache |

Recomendación: si es tu primer reentrenamiento, empieza **solo con `pothole`** (una clase). Añadir `crack` como segunda clase se puede hacer después, cuando ya tengas el pipeline funcionando — mezclar clases desde el día uno es una fuente común de retrabajo.

Reglas de consistencia a mantener mientras etiquetas:
- Bounding box ajustado al hueco visible, no a la sombra que proyecta.
- Si el bache está cortado por el borde del frame, etiquétalo igual (recorta el box al borde).
- Ante duda razonable (mancha vs. bache leve), sé consistente con el mismo criterio en todo el dataset — mejor una regla simple y pareja que una "correcta" pero inconsistente.

## 3. Fase 1 — etiquetar un subconjunto semilla

No arranques por `output/train` completo. Selecciona una muestra representativa:

- ~15-20 imágenes por video (de los 22), priorizando variedad de iluminación/textura de pavimento → esto da un set semilla de ~350-450 imágenes de `train`, más las de `val` que ya tienes.
- Usa `val/` y `test/` completos igual (son más chicos: ~630 y ~420 imágenes) porque los necesitas para medir el modelo desde ya.

Guarda los `.txt` de LabelImg junto a cada imagen o en una carpeta paralela `labels/` (ver estructura en la sección 5).

## 4. Fase 2 — entrenamiento inicial + pre-anotación asistida

Con el set semilla etiquetado:

1. Entrena un YOLOv8n rápido (pocas épocas, ej. 50-80) sólo con ese subconjunto.
2. Corre inferencia de ese modelo sobre el resto de `train/` sin etiquetar.
3. Carga esas predicciones como punto de partida en LabelImg (abre el `.txt` que generó el modelo) y **corrige** en vez de dibujar desde cero: borra falsos positivos, ajusta boxes mal puestos, agrega baches que el modelo no vio.
4. Reentrena con el dataset ampliado. Repite 1-2 veces si el tiempo lo permite (active learning simple).

Esto reduce el trabajo de "dibujar cada box" a "revisar y corregir", que es varias veces más rápido una vez el modelo ya detecta razonablemente bien.

## 5. Estructura de carpetas para entrenamiento YOLO

Ultralytics espera imágenes y labels en árboles paralelos:

```
dataset/
  images/
    train/   <- copias o symlinks de output/train/frames_enhanced/*.jpg
    val/     <- output/val/frames_enhanced/*.jpg
    test/    <- output/test/frames_enhanced/*.jpg
  labels/
    train/   <- un .txt por imagen, mismo nombre base
    val/
    test/
```

Cada línea de un `.txt` de label (formato YOLO, valores normalizados 0-1):
```
0 0.512 0.633 0.084 0.061
```
`class_id x_center y_center width height`

Y el archivo de configuración `dataset.yaml`:
```yaml
path: D:/Estudio/proyectos/road_ai_project/dataset
train: images/train
val: images/val
test: images/test
names:
  0: pothole
```

## 6. Comando de entrenamiento (referencia)

```powershell
venv\Scripts\python.exe -m ultralytics train `
  data=dataset/dataset.yaml `
  model=yolov8n.pt `
  epochs=80 `
  imgsz=640 `
  project=runs/detect `
  name=road_ai_training_v14
```

Ajusta `epochs`/`imgsz` según cuánto dataset tengas en cada fase.

## 7. Buenas prácticas mientras etiquetas

- Commitea los `.txt` de labels a git periódicamente (son texto, pesan nada) — así no dependes de que las imágenes (gitignored) sigan intactas para no perder el trabajo de etiquetado.
- Anota mentalmente (o en un CSV aparte) cuántas imágenes de cada video llevas etiquetadas, para no saltarte ninguno al hacer la muestra representativa de la Fase 1.
- Revisa `val/` y `test/` con el mismo criterio que `train/` — un sesgo entre splits (por ejemplo, boxes más generosos en val) infla o hunde las métricas artificialmente.
