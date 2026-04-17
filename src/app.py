"""
app.py — Road AI Inspector
Pipeline completo de inspección vial con dron.
Ejecutar: streamlit run app.py --server.maxUploadSize 10240

Dependencias:
    pip install streamlit opencv-python-headless numpy pandas haversine ultralytics
"""

import streamlit as st
import shutil
import re
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from haversine import haversine

# YOLOv8 — se importa aquí para que el error sea claro si no está instalado
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ultralyticsplus — permite cargar modelos fine-tuned desde Hugging Face
try:
    from ultralyticsplus import YOLO as YOLO_HF
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
PROJECT_ROOT       = Path(__file__).resolve().parent.parent


def _patch_torch_safe_globals():
    """
    PyTorch 2.6 rompió la carga de modelos legacy al cambiar weights_only=True.
    Solución: monkey-patch torch.load para forzar weights_only=False globalmente.
    """
    try:
        import torch
        _orig = torch.load
        def _patched(f, *args, **kwargs):
            kwargs["weights_only"] = False
            return _orig(f, *args, **kwargs)
        torch.load = _patched
    except Exception:
        pass

_patch_torch_safe_globals()


def load_pothole_model(model_path_resolved: str, status_slot):
    if model_path_resolved == "AUTO_POTHOLE":
        if not HF_AVAILABLE:
            raise RuntimeError(
                "ultralyticsplus no instalado. "
                "Ejecuta: pip install ultralyticsplus"
            )
        status_slot.markdown(
            "<span style='color:#fbbf24;font-size:0.82rem'>"
            "⬇ Cargando modelo fine-tuned de baches desde Hugging Face…</span>",
            unsafe_allow_html=True,
        )
        model = YOLO_HF(POTHOLE_HF_MODEL)
        model.overrides["conf"]         = 0.25
        model.overrides["iou"]          = 0.45
        model.overrides["agnostic_nms"] = False
        model.overrides["max_det"]      = 1000
        status_slot.empty()
        return model, True
    else:
        status_slot.markdown(
            "<span style='color:#64748b;font-size:0.82rem'>"
            "Cargando modelo local…</span>",
            unsafe_allow_html=True,
        )
        model = YOLO(model_path_resolved)
        status_slot.empty()
        labels = set(model.names.values()) if model.names else set()
        is_pothole = any(
            "pothole" in l.lower() or "hole" in l.lower() or "crack" in l.lower()
            for l in labels
        )
        return model, is_pothole


# ── Página ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road AI Inspector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
.stApp { background-color: #0d0f14; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero .subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.accent { color: #38bdf8; }

.step-card {
    background: #131720;
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.step-title  { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; }
.step-desc   { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }

.metric-row  { display: flex; gap: 1rem; margin: 1.5rem 0; }
.metric-box  {
    flex: 1;
    background: #131720;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.stButton > button {
    background: #38bdf8 !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stFileUploader"] {
    background: #131720;
    border: 1px dashed #2d3748;
    border-radius: 10px;
    padding: 0.5rem;
}

.stProgress > div > div { background: #38bdf8 !important; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-ok   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-err  { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.badge-warn { background: #2d1b00; color: #fbbf24; border: 1px solid #78350f; }

hr { border-color: #1e2530; }

[data-testid="stExpander"] {
    background: #131720;
    border: 1px solid #1e2530;
    border-radius: 10px;
}

.img-caption {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-align: center;
    margin-top: 0.3rem;
}
.path-hint      { font-size: 0.78rem; margin-top: 0.3rem; font-family: 'Space Mono', monospace; color: #475569; }
.path-hint.ok   { color: #4ade80; }
.path-hint.err  { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🛣️ Road <span class="accent">AI</span> Inspector</h1>
    <div class="subtitle">Pipeline de inspección vial con dron · Detección de baches y defectos</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

TOTAL_STEPS = 5
if "step_status" not in st.session_state:
    st.session_state.step_status = ["pending"] * TOTAL_STEPS
if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None

# ── Funciones pipeline ─────────────────────────────────────────────────────────

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
                   progress_bar, status_text) -> list:
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
        path = out_dir / f"frame_{fn}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(str(path))
        progress_bar.progress((idx + 1) / len(frame_numbers))
        status_text.markdown(
            f"<span style='color:#64748b;font-size:0.82rem'>Extrayendo frame {idx+1}/{len(frame_numbers)}</span>",
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
                unsharp_str, progress_bar, status_text):
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
        progress_bar.progress((idx + 1) / total)
        status_text.markdown(
            f"<span style='color:#64748b;font-size:0.82rem'>"
            f"Mejorando frame {idx+1}/{total}</span>",
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
                       progress_bar, status_text) -> tuple[list, pd.DataFrame]:
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

        progress_bar.progress((idx + 1) / len(enhanced_paths))
        n_det = len(valid_boxes)
        color_hex = "f87171" if n_det else "38bdf8"
        rej_txt = f" · <span style='color:#475569'>{rejected} descartadas</span>" if rejected else ""
        status_text.markdown(
            f"<span style='color:#64748b;font-size:0.82rem'>"
            f"Analizando frame {idx+1}/{len(enhanced_paths)} · "
            f"<span style='color:#{color_hex}'>"
            f"{n_det} bache{'s' if n_det != 1 else ''} detectado{'s' if n_det != 1 else ''}"
            f"</span>{rej_txt}</span>",
            unsafe_allow_html=True,
        )

    detections_df = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()
    return annotated_paths, detections_df


# ── Layout ─────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.6], gap="large")

# ── COLUMNA IZQUIERDA ──────────────────────────────────────────────────────────

with col_left:
    st.markdown("### Archivos de entrada")

    st.markdown("**Video del dron (.MP4)**")
    video_path_input = st.text_input(
        label="video_path",
        placeholder=r"C:\Users\TuUsuario\Videos\vuelo1.MP4",
        label_visibility="collapsed",
    )

    video_ok = False
    if video_path_input:
        vp = Path(video_path_input.strip())
        if vp.exists() and vp.suffix.lower() in [".mp4", ".mov", ".avi"]:
            st.markdown("<div class='path-hint ok'>✓ Archivo encontrado</div>", unsafe_allow_html=True)
            video_ok = True
        else:
            st.markdown("<div class='path-hint err'>✗ Ruta no válida o archivo no encontrado</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='path-hint'>Escribe la ruta completa al archivo de video</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Telemetría GPS (.SRT)**")
    srt_file = st.file_uploader(
        label="srt_upload",
        type=["srt", "SRT"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Modelo YOLOv8")

    if not YOLO_AVAILABLE:
        st.markdown(
            "<div style='color:#f87171;font-size:0.83rem'>⚠️ ultralytics no instalado. "
            "Ejecuta: <code>pip install ultralytics ultralyticsplus</code></div>",
            unsafe_allow_html=True,
        )

    use_custom_model = st.toggle("Usar modelo propio (.pt)", value=False)

    if use_custom_model:
        model_path_input = st.text_input(
            label="model_path_custom",
            placeholder=r"C:\modelos\mi_modelo_baches.pt",
            label_visibility="collapsed",
            help="Ruta a los weights .pt de tu modelo fine-tuned.",
        )
        if model_path_input.strip():
            mp = Path(model_path_input.strip())
            if mp.exists():
                st.markdown("<div class='path-hint ok'>✓ Modelo encontrado</div>", unsafe_allow_html=True)
                model_path_resolved = model_path_input.strip()
                model_ok = True
            else:
                st.markdown("<div class='path-hint err'>✗ Archivo no encontrado</div>", unsafe_allow_html=True)
                model_path_resolved = None
                model_ok = False
        else:
            model_path_resolved = None
            model_ok = False
    else:
        local_pothole = Path(POTHOLE_MODEL_NAME)
        if local_pothole.exists():
            st.markdown(
                "<div class='path-hint ok'>✓ Modelo de baches listo (keremberke/yolov8-pothole)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='path-hint' style='color:#fbbf24'>"
                "⬇ Se descargará el modelo fine-tuned en baches (~22 MB) al ejecutar"
                "</div>",
                unsafe_allow_html=True,
            )
        model_path_resolved = "AUTO_POTHOLE"
        model_ok = True

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Parámetros del pipeline")

    distance_interval = st.slider("Intervalo de muestreo (metros)", 3, 20, 7, step=1)

    with st.expander("⚙️ Parámetros de mejora de imagen"):
        clahe_clip  = st.slider("CLAHE — clip limit",        1.0, 5.0, 3.0, step=0.5)
        clahe_tile  = st.slider("CLAHE — tile size (px)",    4,   16,  8,   step=4)
        unsharp_str = st.slider("Unsharp Mask — intensidad", 0.3, 2.0, 1.2, step=0.1)

    with st.expander("⚙️ Parámetros de detección YOLOv8"):
        conf_threshold = st.slider("Confianza mínima (confidence)", 0.1, 0.9, 0.35, step=0.05)
        max_box_ratio  = st.slider(
            "Tamaño máximo del box (% del frame)",
            5, 60, 20, step=5,
            help="Descarta detecciones cuya área supere este % del frame total."
        )
        road_margin_pct = st.slider(
            "Margen lateral a ignorar (% de cada lado)",
            0, 40, 20, step=5,
            help="Ignora detecciones fuera del corredor central de la imagen."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button("▶  Ejecutar pipeline completo", use_container_width=True)

    yolo_ready = YOLO_AVAILABLE and (model_ok) and (
        (model_path_resolved != "AUTO_POTHOLE") or HF_AVAILABLE
    )
    ready = video_ok and (srt_file is not None) and yolo_ready

    if not ready:
        msgs = []
        if not video_path_input:
            msgs.append("Escribe la ruta del video")
        elif not video_ok:
            msgs.append("Ruta de video no válida")
        if srt_file is None:
            msgs.append("Sube el archivo .SRT")
        if not YOLO_AVAILABLE:
            msgs.append("Instala ultralytics ultralyticsplus")
        elif model_path_resolved == "AUTO_POTHOLE" and not HF_AVAILABLE:
            msgs.append("Instala ultralyticsplus")
        elif not model_ok:
            msgs.append("Ruta de modelo no válida")
        st.markdown(
            f"<div style='text-align:center;color:#f87171;font-size:0.82rem;margin-top:0.5rem'>"
            f"{'  ·  '.join(msgs)}</div>",
            unsafe_allow_html=True,
        )


# ── COLUMNA DERECHA ────────────────────────────────────────────────────────────

with col_right:
    st.markdown("### Estado del pipeline")

    steps = [
        ("01", "Parseo GPS",              "Extrae coordenadas y frame_number del .SRT"),
        ("02", "Selección por distancia", f"1 frame cada {distance_interval} m recorrido"),
        ("03", "Extracción de frames",    "Lee el video y exporta los JPEGs seleccionados"),
        ("04", "Mejora de imagen",        "Denoise → CLAHE → Unsharp Mask → Normalización"),
        ("05", "Detección YOLOv8",        "Inferencia de baches · Bounding boxes + GPS"),
    ]

    STATUS_BADGE = {"pending": "badge-warn", "running": "badge-warn", "done": "badge-ok", "error": "badge-err"}
    STATUS_LABEL = {"pending": "PENDIENTE",  "running": "PROCESANDO…","done": "COMPLETADO","error": "ERROR"}
    BORDER_COLOR = {"pending": "#1e2530",    "running": "#1e3a8a",    "done": "#166534",   "error": "#7f1d1d"}

    def render_step(ph, i, state):
        num, title, desc = steps[i]
        ph.markdown(f"""
        <div class="step-card" style="border-color:{BORDER_COLOR[state]}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div class="step-label">Paso {num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                <span class="badge {STATUS_BADGE[state]}">{STATUS_LABEL[state]}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    step_ph = []
    for i, _ in enumerate(steps):
        ph = st.empty()
        step_ph.append(ph)
        render_step(ph, i, st.session_state.step_status[i])

    progress_bar = st.progress(0)
    status_text  = st.empty()
    results_slot = st.empty()

    def update_step(i, state):
        st.session_state.step_status[i] = state
        render_step(step_ph[i], i, state)

    # ── Ejecución ──────────────────────────────────────────────────────────────

    if run_btn:
        if not ready:
            st.warning("⚠️ Completa los archivos de entrada antes de ejecutar.")
        else:
            st.session_state.step_status = ["pending"] * TOTAL_STEPS
            st.session_state["zip_ready"] = False
            st.session_state["zip_bytes"] = None

            out_root = PROJECT_ROOT / "output"
            out_root.mkdir(parents=True, exist_ok=True)

            raw_dir = Path(tempfile.mkdtemp(prefix="road_ai_raw_"))
            enh_dir = Path(tempfile.mkdtemp(prefix="road_ai_enh_"))
            det_dir = out_root / "frames_detected"
            csv_dir = out_root / "metadata"
            for d in [det_dir, csv_dir]:
                reset_output_dir(d)

            # PASO 1
            update_step(0, "running")
            try:
                srt_content = srt_file.read().decode("utf-8")
                metadata_df = parse_srt(srt_content)
                if metadata_df.empty:
                    update_step(0, "error")
                    st.error("❌ No se encontraron datos GPS en el SRT.")
                    st.stop()
                update_step(0, "done")
            except Exception as e:
                update_step(0, "error"); st.error(f"❌ Parseo SRT: {e}"); st.stop()

            # PASO 2
            update_step(1, "running")
            try:
                selected_df = select_frames_by_distance(metadata_df, distance_interval)
                if selected_df.empty:
                    update_step(1, "error")
                    st.error("❌ Sin frames seleccionados. Reduce el intervalo.")
                    st.stop()
                frame_numbers = selected_df["frame_number"].astype(int).tolist()
                update_step(1, "done")
            except Exception as e:
                update_step(1, "error"); st.error(f"❌ Selección: {e}"); st.stop()

            # PASO 3
            update_step(2, "running")
            progress_bar.progress(0)
            try:
                saved_paths = extract_frames(
                    video_path_input.strip(), frame_numbers,
                    raw_dir, progress_bar, status_text,
                )
                if not saved_paths:
                    update_step(2, "error")
                    st.error("❌ No se pudieron extraer frames del video.")
                    st.stop()
                update_step(2, "done")
            except Exception as e:
                update_step(2, "error"); st.error(f"❌ Extracción: {e}"); st.stop()

            # PASO 4
            update_step(3, "running")
            progress_bar.progress(0)
            try:
                enhanced_paths = enhance_all(
                    saved_paths, enh_dir,
                    clahe_clip, clahe_tile, unsharp_str,
                    progress_bar, status_text,
                )
                update_step(3, "done")
            except Exception as e:
                update_step(3, "error"); st.error(f"❌ Mejora: {e}"); st.stop()

            # PASO 5 — YOLOv8
            update_step(4, "running")
            progress_bar.progress(0)
            try:
                yolo_model, is_pothole_model = load_pothole_model(
                    model_path_resolved, status_text
                )
                annotated_paths, detections_df = run_yolo_detection(
                    yolo_model, is_pothole_model,
                    enhanced_paths, selected_df,
                    det_dir, conf_threshold,
                    max_box_ratio, road_margin_pct,
                    progress_bar, status_text,
                )
                update_step(4, "done")
            except Exception as e:
                update_step(4, "error"); st.error(f"❌ Detección YOLO: {e}"); st.stop()

            progress_bar.progress(1.0)
            status_text.empty()

            # ── Métricas finales ───────────────────────────────────────────────
            total_baches = len(detections_df) if not detections_df.empty else 0
            frames_con_baches = (
                detections_df["frame_number"].nunique()
                if not detections_df.empty else 0
            )

            results_slot.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{len(metadata_df):,}</div>
                    <div class="metric-lbl">Frames GPS</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{len(frame_numbers)}</div>
                    <div class="metric-lbl">Analizados</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color:{'#f87171' if total_baches > 0 else '#4ade80'}">{total_baches}</div>
                    <div class="metric-lbl">Baches detectados</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{frames_con_baches}</div>
                    <div class="metric-lbl">Frames afectados</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Preview ────────────────────────────────────────────────────────
            preview_paths = [p for p in annotated_paths if p is not None]
            if not detections_df.empty:
                frames_con_det = set(detections_df["frame_number"].tolist())
                priority  = [p for p in preview_paths
                             if int(Path(p).stem.split("_")[-1]) in frames_con_det]
                sin_det   = [p for p in preview_paths
                             if int(Path(p).stem.split("_")[-1]) not in frames_con_det]
                preview_paths = (priority + sin_det)[:3]
            else:
                preview_paths = preview_paths[:3]

            if preview_paths:
                st.markdown("#### Preview — frames detectados")
                for i, ap in enumerate(preview_paths):
                    frame_num = int(Path(ap).stem.split("_")[-1])
                    enh_match = [p for p in enhanced_paths
                                 if Path(p).stem == Path(ap).stem]
                    enh_p = enh_match[0] if enh_match else None

                    n_det_frame = (
                        len(detections_df[detections_df["frame_number"] == frame_num])
                        if not detections_df.empty else 0
                    )
                    det_label = (
                        f"🔴 {n_det_frame} bache{'s' if n_det_frame != 1 else ''}"
                        if n_det_frame > 0 else "✅ Sin detecciones"
                    )

                    if enh_p:
                        c1, c2 = st.columns(2)
                        enh_rgb = cv2.cvtColor(cv2.imread(enh_p), cv2.COLOR_BGR2RGB)
                        det_rgb = cv2.cvtColor(cv2.imread(ap),    cv2.COLOR_BGR2RGB)
                        with c1:
                            st.image(enh_rgb, use_container_width=True)
                            st.markdown(f"<div class='img-caption'>MEJORADA · frame_{frame_num}</div>",
                                        unsafe_allow_html=True)
                        with c2:
                            st.image(det_rgb, use_container_width=True)
                            st.markdown(f"<div class='img-caption'>DETECTADA · {det_label}</div>",
                                        unsafe_allow_html=True)
                    else:
                        det_rgb = cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB)
                        st.image(det_rgb, use_container_width=True)
                        st.markdown(f"<div class='img-caption'>frame_{frame_num} · {det_label}</div>",
                                    unsafe_allow_html=True)

                    if i < len(preview_paths) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

            # ── Tabla de detecciones ───────────────────────────────────────────
            if not detections_df.empty:
                with st.expander(f"📋 Tabla de detecciones ({total_baches} registros)"):
                    display_cols = ["frame_number", "lat", "lon", "label", "confidence",
                                    "x1", "y1", "x2", "y2"]
                    st.dataframe(
                        detections_df[display_cols].style.format({
                            "confidence": "{:.3f}",
                            "lat": "{:.6f}",
                            "lon": "{:.6f}",
                        }),
                        use_container_width=True,
                    )
            else:
                st.info("ℹ️ No se detectaron baches con el umbral de confianza actual. "
                        "Prueba reducir el slider de confianza mínima.")

            # ── Guardar CSVs ───────────────────────────────────────────────────
            metadata_df.to_csv(csv_dir / "drone_metadata.csv", index=False)
            selected_df.to_csv(csv_dir / "frames_selected.csv", index=False)
            if not detections_df.empty:
                detections_df.to_csv(csv_dir / "detections.csv", index=False)

            # ── ZIP ────────────────────────────────────────────────────────────
            master_zip_path = out_root.parent / "road_ai_results"
            shutil.make_archive(str(master_zip_path), "zip", str(out_root))

            zip_file_path = str(master_zip_path) + ".zip"
            with open(zip_file_path, "rb") as zf:
                st.session_state["zip_bytes"] = zf.read()
            st.session_state["zip_ready"] = True

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"📁 Archivos guardados en: `{out_root}`")

# ── Descarga persistente ───────────────────────────────────────────────────────

if st.session_state.get("zip_ready") and st.session_state.get("zip_bytes"):
    st.markdown("<br>", unsafe_allow_html=True)
    with col_right:
        st.download_button(
            label="⬇  Descargar todo (imágenes + detecciones + CSVs)",
            data=st.session_state["zip_bytes"],
            file_name="road_ai_results.zip",
            mime="application/zip",
            use_container_width=True,
        )