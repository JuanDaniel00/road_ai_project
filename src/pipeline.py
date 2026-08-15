"""
pipeline.py — Lógica de procesamiento de Road AI Inspector, sin dependencias de UI.

Usado tanto por app.py (Streamlit) como por batch_process.py (procesamiento
masivo por línea de comandos). Las funciones que reportan progreso aceptan
progress_bar / status_text opcionales (objetos de Streamlit); si se omiten
(None), simplemente no reportan progreso — así sirven igual desde un script.
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from haversine import haversine

# Torch es opcional en entornos donde el analizador no resuelve la dependencia
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# YOLOv8 — se importa aquí para que el error sea claro si no está instalado
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ultralyticsplus — permite cargar modelos fine-tuned desde Hugging Face
try:
    from ultralyticsplus import YOLO as YOLO_HF  # type: ignore
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Clases COCO que NO son defectos de pavimento — filtro anti-ruido
COCO_ROAD_NOISE = {
    "motorcycle", "bicycle", "car", "truck", "bus", "person",
    "dog", "cat", "traffic light", "stop sign", "parking meter",
}

# Nombre del modelo HF y cache local
POTHOLE_HF_MODEL   = "keremberke/yolov8n-pothole-segmentation"
POTHOLE_MODEL_NAME = "yolov8_pothole_hf.pt"
PROJECT_ROOT        = Path(__file__).resolve().parent.parent


def _patch_torch_safe_globals():
    """
    PyTorch 2.6 rompió la carga de modelos legacy al cambiar weights_only=True.
    Solución: monkey-patch torch.load para forzar weights_only=False globalmente.
    """
    try:
        if not TORCH_AVAILABLE:
            return
        _orig = torch.load
        def _patched(f, *args, **kwargs):
            kwargs["weights_only"] = False
            return _orig(f, *args, **kwargs)
        torch.load = _patched
    except Exception:
        pass

_patch_torch_safe_globals()


def load_pothole_model(model_path_resolved: str, status_slot=None):
    if model_path_resolved == "AUTO_POTHOLE":
        if not HF_AVAILABLE:
            raise RuntimeError(
                "ultralyticsplus no instalado. "
                "Ejecuta: pip install ultralyticsplus"
            )
        if status_slot is not None:
            status_slot.markdown(
                "<span style='color:#b45309;font-size:0.85rem'>"
                "⬇ Preparando el sistema de detección de baches…</span>",
                unsafe_allow_html=True,
            )
        model = YOLO_HF(POTHOLE_HF_MODEL)
        model.overrides["conf"]         = 0.25
        model.overrides["iou"]          = 0.45
        model.overrides["agnostic_nms"] = False
        model.overrides["max_det"]      = 1000
        if status_slot is not None:
            status_slot.empty()
        return model, True
    else:
        if status_slot is not None:
            status_slot.markdown(
                "<span style='color:#78716c;font-size:0.85rem'>"
                "Cargando modelo de detección…</span>",
                unsafe_allow_html=True,
            )
        model = YOLO(model_path_resolved)
        if status_slot is not None:
            status_slot.empty()
        labels = set(model.names.values()) if model.names else set()
        is_pothole = any(
            "pothole" in l.lower() or "hole" in l.lower() or "crack" in l.lower()
            for l in labels
        )
        return model, is_pothole


def parse_srt(srt_content: str) -> pd.DataFrame:
    data = []
    for block in srt_content.split("\n\n"):
        fm = re.search(r"FrameCnt:\s*(\d+)", block)
        la = re.search(r"\[latitude:\s*([-\d\.]+)\]", block)
        lo = re.search(r"\[longitude:\s*([-\d\.]+)\]", block)
        if fm and la and lo:
            data.append({
                "frame_number": int(fm.group(1)),
                "lat": float(la.group(1)),
                "lon": float(lo.group(1)),
            })
    return pd.DataFrame(data)


def select_frames_by_distance(df: pd.DataFrame, interval_m: float) -> pd.DataFrame:
    selected, accumulated = [], 0.0
    for i in range(1, len(df)):
        p1 = (df.iloc[i-1]["lat"], df.iloc[i-1]["lon"])
        p2 = (df.iloc[i]["lat"],   df.iloc[i]["lon"])
        accumulated += haversine(p1, p2) * 1000
        if accumulated >= interval_m:
            selected.append(df.iloc[i])
            accumulated = 0.0
    return pd.DataFrame(selected)


def extract_frames(video_path: str, frame_numbers: list, out_dir: Path,
                   progress_bar=None, status_text=None, filename_prefix: str = "") -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    for idx, fn in enumerate(sorted(frame_numbers)):
        if fn >= total_video:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual_pos != fn:
            while actual_pos < fn:
                cap.grab()
                actual_pos += 1
        ret, frame = cap.read()
        if not ret:
            continue
        path = out_dir / f"{filename_prefix}frame_{fn}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(str(path))
        if progress_bar is not None:
            progress_bar.progress((idx + 1) / len(frame_numbers))
        if status_text is not None:
            status_text.markdown(
                f"<span style='color:#78716c;font-size:0.85rem'>Extrayendo fotografía {idx+1} de {len(frame_numbers)}</span>",
                unsafe_allow_html=True,
            )
    cap.release()
    return saved


def apply_clahe(img, clip, tile):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def apply_unsharp(img, strength):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def apply_bilateral(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def normalize_color(img):
    result = img.copy()
    for i in range(3):
        ch = img[:, :, i]
        mn, mx = int(ch.min()), int(ch.max())
        rng = mx - mn
        if 0 < rng < 51:
            result[:, :, i] = ((ch.astype(np.float32) - mn) / rng * 255).astype(np.uint8)
    return result

def enhance_frame_fn(img, clahe_clip, clahe_tile, unsharp_str):
    img = apply_bilateral(img)
    img = apply_clahe(img, clahe_clip, clahe_tile)
    img = apply_unsharp(img, unsharp_str)
    img = normalize_color(img)
    return img

# ── FIX: procesamiento secuencial, sin multiprocessing ────────────────────────
def enhance_all(frame_paths, out_dir, clahe_clip, clahe_tile,
                unsharp_str, progress_bar=None, status_text=None):
    enhanced = []
    total = len(frame_paths)
    for idx, fp in enumerate(frame_paths):
        img = cv2.imread(fp)
        if img is None:
            continue
        enh = enhance_frame_fn(img, clahe_clip, clahe_tile, unsharp_str)
        out_path = Path(out_dir) / Path(fp).name
        cv2.imwrite(str(out_path), enh, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced.append(str(out_path))
        if progress_bar is not None:
            progress_bar.progress((idx + 1) / total)
        if status_text is not None:
            status_text.markdown(
                f"<span style='color:#78716c;font-size:0.85rem'>"
                f"Mejorando fotografía {idx+1} de {total}</span>",
                unsafe_allow_html=True,
            )
    return enhanced


def reset_output_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def run_yolo_detection(model, is_pothole_model: bool,
                       enhanced_paths: list, selected_df: pd.DataFrame,
                       det_dir: Path, conf_threshold: float,
                       max_box_ratio: float, road_margin_pct: float,
                       progress_bar=None, status_text=None) -> tuple:
    annotated_paths   = []
    detection_records = []

    gps_index = {}
    if not selected_df.empty and "frame_number" in selected_df.columns:
        for _, row in selected_df.iterrows():
            gps_index[int(row["frame_number"])] = (row["lat"], row["lon"])

    for idx, fp in enumerate(enhanced_paths):
        img = cv2.imread(fp)
        if img is None:
            annotated_paths.append(None)
            continue

        h, w = img.shape[:2]
        frame_area = h * w

        margin_px = int(w * road_margin_pct / 100)
        roi_x_min = margin_px
        roi_x_max = w - margin_px

        results   = model(img, conf=conf_threshold, verbose=False)[0]
        annotated = img.copy()

        if road_margin_pct > 0:
            cv2.line(annotated, (roi_x_min, 0), (roi_x_min, h), (50, 80, 120), 1)
            cv2.line(annotated, (roi_x_max, 0), (roi_x_max, h), (50, 80, 120), 1)

        frame_num = int(Path(fp).stem.split("_")[-1])
        lat, lon  = gps_index.get(frame_num, (None, None))

        valid_boxes = []
        rejected = 0
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = model.names.get(cls, str(cls))

            if not is_pothole_model and label.lower() in COCO_ROAD_NOISE:
                rejected += 1
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_area = (x2 - x1) * (y2 - y1)

            if box_area / frame_area * 100 > max_box_ratio:
                rejected += 1
                continue

            cx = (x1 + x2) // 2
            if cx < roi_x_min or cx > roi_x_max:
                rejected += 1
                continue

            valid_boxes.append((box, label, x1, y1, x2, y2))

        for box, label, x1, y1, x2, y2 in valid_boxes:
            conf_val = float(box.conf[0])
            color = (0, 50, 255) if is_pothole_model else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detection_records.append({
                "frame_number": frame_num,
                "lat": lat,
                "lon": lon,
                "label": label,
                "confidence": round(conf_val, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width_px":  x2 - x1,
                "height_px": y2 - y1,
            })

        if valid_boxes:
            out_path = det_dir / Path(fp).name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            annotated_paths.append(str(out_path))
        else:
            annotated_paths.append(None)

        if progress_bar is not None:
            progress_bar.progress((idx + 1) / len(enhanced_paths))
        n_det = len(valid_boxes)
        if status_text is not None:
            color_hex = "b91c1c" if n_det else "57534e"
            rej_txt = f" · <span style='color:#a8a29e'>{rejected} descartadas</span>" if rejected else ""
            status_text.markdown(
                f"<span style='color:#78716c;font-size:0.85rem'>"
                f"Analizando fotografía {idx+1} de {len(enhanced_paths)} · "
                f"<span style='color:#{color_hex}'>"
                f"{n_det} bache{'s' if n_det != 1 else ''} detectado{'s' if n_det != 1 else ''}"
                f"</span>{rej_txt}</span>",
                unsafe_allow_html=True,
            )

    detections_df = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()
    return annotated_paths, detections_df
