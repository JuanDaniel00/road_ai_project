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
import io
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd

from pipeline import (
    TORCH_AVAILABLE, YOLO_AVAILABLE, HF_AVAILABLE,
    COCO_ROAD_NOISE, POTHOLE_HF_MODEL, POTHOLE_MODEL_NAME, PROJECT_ROOT,
    load_pothole_model, parse_srt, select_frames_by_distance, extract_frames,
    apply_clahe, apply_unsharp, apply_bilateral, normalize_color,
    enhance_frame_fn, enhance_all, reset_output_dir, run_yolo_detection,
)


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
html, body, [class*="css"] {
    font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: #f5f5f4;
    color: #292524;
}
.stApp { background-color: #f5f5f4; }

.hero {
    text-align: center;
    padding: 2.5rem 0 2rem 0;
    border-bottom: 3px solid #d97706;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1c1917;
    margin: 0;
}
.hero .subtitle {
    font-size: 1rem;
    color: #57534e;
    margin-top: 0.5rem;
    font-weight: 400;
}
.accent { color: #d97706; }

.step-card {
    background: #ffffff;
    border: 1px solid #e7e5e4;
    border-left: 5px solid #d6d3d1;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 2px rgba(41, 37, 36, 0.06);
}
.step-label {
    font-size: 0.72rem;
    color: #78716c;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.step-title  { font-size: 1.05rem; font-weight: 600; color: #1c1917; }
.step-desc   { font-size: 0.87rem; color: #78716c; margin-top: 0.2rem; }

.metric-row  { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-box  {
    flex: 1;
    min-width: 130px;
    background: #ffffff;
    border: 1px solid #e7e5e4;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #44403c;
}
.metric-lbl {
    font-size: 0.8rem;
    color: #78716c;
    margin-top: 0.3rem;
}

.stButton > button {
    background: #d97706 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
}
.stButton > button:hover { background: #b45309 !important; }

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #d6d3d1;
    border-radius: 10px;
    padding: 0.5rem;
}

.stProgress > div > div { background: #d97706 !important; }

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-pending { background: #f5f5f4; color: #78716c; border: 1px solid #d6d3d1; }
.badge-running { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
.badge-ok      { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.badge-err     { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }

hr { border-color: #e7e5e4; }

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e7e5e4;
    border-radius: 10px;
}

.img-caption {
    font-size: 0.78rem;
    color: #78716c;
    text-align: center;
    margin-top: 0.3rem;
}
.path-hint      { font-size: 0.82rem; margin-top: 0.3rem; color: #78716c; }
.path-hint.ok   { color: #15803d; }
.path-hint.err  { color: #b91c1c; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🛣️ Road <span class="accent">AI</span> Inspector</h1>
    <div class="subtitle">Inspección de pavimento a partir de video de dron</div>
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
if "zip_filename" not in st.session_state:
    st.session_state.zip_filename = None
if "input_generation" not in st.session_state:
    st.session_state.input_generation = 0

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
        key=f"video_path_{st.session_state.input_generation}",
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

    st.markdown("**Registro de vuelo GPS (.SRT)**")
    srt_file = st.file_uploader(
        label="srt_upload",
        type=["srt", "SRT"],
        label_visibility="collapsed",
        help="Archivo de telemetría que el dron genera junto con el video, con la ubicación GPS de cada instante del vuelo.",
        key=f"srt_upload_{st.session_state.input_generation}",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Sistema de detección de baches")

    if not YOLO_AVAILABLE:
        st.markdown(
            "<div style='color:#b91c1c;font-size:0.85rem'>⚠️ El sistema de detección no está instalado en este equipo.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Instrucciones de instalación"):
            st.code("pip install ultralytics ultralyticsplus")

    use_custom_model = st.toggle(
        "Usar un modelo propio entrenado",
        value=False,
        help="Actívalo solo si cuentas con tu propio modelo de detección entrenado. "
             "Si lo dejas apagado se usa el modelo de baches incluido por defecto.",
    )

    if use_custom_model:
        model_path_input = st.text_input(
            label="model_path_custom",
            placeholder=r"C:\modelos\mi_modelo_baches.pt",
            label_visibility="collapsed",
            help="Ruta al archivo .pt de tu modelo entrenado.",
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
                "<div class='path-hint ok'>✓ Sistema de detección de baches listo para usar</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='path-hint' style='color:#b45309'>"
                "⬇ El sistema de detección (~22 MB) se descargará automáticamente al iniciar el análisis"
                "</div>",
                unsafe_allow_html=True,
            )
        model_path_resolved = "AUTO_POTHOLE"
        model_ok = True

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Parámetros del análisis")

    distance_interval = st.slider(
        "Distancia entre fotografías analizadas (metros)", 3, 20, 7, step=1,
        help="Cada cuántos metros recorridos se toma una fotografía para analizar. "
             "Un valor menor da más detalle, pero genera más imágenes para procesar.",
    )

    with st.expander("⚙️ Ajustes de calidad de imagen"):
        clahe_clip  = st.slider(
            "Nivel de contraste local", 1.0, 5.0, 3.0, step=0.5,
            help="Resalta el contraste en zonas con sombra o poca luz. Valores muy altos pueden generar ruido.",
        )
        clahe_tile  = st.slider(
            "Tamaño de zona de contraste (px)", 4, 16, 8, step=4,
            help="Tamaño de la región usada para calcular el contraste local. 8 px es el valor estándar.",
        )
        unsharp_str = st.slider(
            "Nitidez de bordes", 0.3, 2.0, 1.2, step=0.1,
            help="Resalta los bordes y contornos de la vía. Valores mayores a 1.5 pueden generar halos.",
        )

    with st.expander("⚙️ Ajustes de detección de baches"):
        conf_threshold = st.slider(
            "Sensibilidad de detección", 0.1, 0.9, 0.35, step=0.05,
            help="Qué tan segura debe estar la IA antes de marcar un bache. "
                 "Un valor más bajo detecta más baches, pero también más falsos positivos.",
        )
        max_box_ratio  = st.slider(
            "Tamaño máximo de un bache detectado (% de la imagen)",
            5, 60, 20, step=5,
            help="Descarta detecciones cuya área supere este porcentaje de la fotografía completa."
        )
        road_margin_pct = st.slider(
            "Margen lateral ignorado (%)",
            0, 40, 20, step=5,
            help="Ignora detecciones cerca de los bordes laterales de la imagen (berma, vegetación, etc.)."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button("▶  Iniciar análisis de la vía", use_container_width=True)

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
            msgs.append("Instala el sistema de detección (ver instrucciones arriba)")
        elif model_path_resolved == "AUTO_POTHOLE" and not HF_AVAILABLE:
            msgs.append("Falta un componente del sistema de detección")
        elif not model_ok:
            msgs.append("Ruta de modelo no válida")
        st.markdown(
            f"<div style='text-align:center;color:#b91c1c;font-size:0.85rem;margin-top:0.5rem'>"
            f"{'  ·  '.join(msgs)}</div>",
            unsafe_allow_html=True,
        )


# ── COLUMNA DERECHA ────────────────────────────────────────────────────────────

with col_right:
    st.markdown("### Estado del análisis")

    steps = [
        ("1", "Ubicación del recorrido",  "Lee la posición GPS registrada por el dron en cada punto del vuelo."),
        ("2", "Selección de fotografías", f"Elige una imagen cada {distance_interval} m recorridos, para cubrir toda la vía sin repetir tomas."),
        ("3", "Extracción de imágenes",   "Obtiene del video las fotografías de los puntos seleccionados."),
        ("4", "Mejora de calidad",        "Ajusta contraste, nitidez y color para que los defectos se distingan mejor."),
        ("5", "Detección de baches",      "Analiza cada fotografía y marca los baches encontrados junto con su ubicación GPS."),
    ]

    STATUS_BADGE = {"pending": "badge-pending", "running": "badge-running", "done": "badge-ok", "error": "badge-err"}
    STATUS_ICON  = {"pending": "○", "running": "⏳", "done": "✓", "error": "⚠"}
    STATUS_LABEL = {"pending": "En espera", "running": "Procesando…", "done": "Completado", "error": "Necesita atención"}
    BORDER_COLOR = {"pending": "#d6d3d1", "running": "#fcd34d", "done": "#86efac", "error": "#fca5a5"}

    def render_step(ph, i, state):
        num, title, desc = steps[i]
        ph.markdown(f"""
        <div class="step-card" style="border-left-color:{BORDER_COLOR[state]}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div class="step-label">Paso {num} de {TOTAL_STEPS}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                <span class="badge {STATUS_BADGE[state]}">{STATUS_ICON[state]} {STATUS_LABEL[state]}</span>
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
            st.warning("⚠️ Completa los datos de entrada antes de iniciar el análisis.")
        else:
            st.session_state.step_status = ["pending"] * TOTAL_STEPS
            st.session_state["zip_ready"] = False
            st.session_state["zip_bytes"] = None
            st.session_state["zip_filename"] = None

            video_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video_path_input.strip()).stem)
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_stem}"

            out_root = PROJECT_ROOT / "output" / run_id
            out_root.mkdir(parents=True, exist_ok=True)

            raw_dir = Path(tempfile.mkdtemp(prefix="road_ai_raw_"))
            enh_dir = Path(tempfile.mkdtemp(prefix="road_ai_enh_"))
            det_dir = out_root / "frames_detected"
            csv_dir = out_root / "metadata"
            for d in [det_dir, csv_dir]:
                reset_output_dir(d)

            with st.spinner("Analizando la vía — esto puede tardar varios minutos según la duración del video. No cierres esta ventana."):
                # PASO 1
                update_step(0, "running")
                try:
                    srt_content = srt_file.read().decode("utf-8")
                    metadata_df = parse_srt(srt_content)
                    if metadata_df.empty:
                        update_step(0, "error")
                        st.error(
                            "No se encontraron datos de ubicación GPS en el archivo .SRT. "
                            "Verifica que sea el archivo de telemetría generado por el dron, no el video."
                        )
                        st.stop()
                    update_step(0, "done")
                except Exception as e:
                    update_step(0, "error")
                    st.error("No se pudo leer el archivo de registro de vuelo GPS.")
                    with st.expander("Detalle técnico"):
                        st.code(str(e))
                    st.stop()

                # PASO 2
                update_step(1, "running")
                try:
                    selected_df = select_frames_by_distance(metadata_df, distance_interval)
                    if selected_df.empty:
                        update_step(1, "error")
                        st.error(
                            f"No se seleccionó ninguna fotografía con un intervalo de {distance_interval} m. "
                            "El recorrido puede ser muy corto — prueba con una distancia menor."
                        )
                        st.stop()
                    frame_numbers = selected_df["frame_number"].astype(int).tolist()
                    update_step(1, "done")
                except Exception as e:
                    update_step(1, "error")
                    st.error("No se pudieron seleccionar los puntos de muestreo.")
                    with st.expander("Detalle técnico"):
                        st.code(str(e))
                    st.stop()

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
                        st.error(
                            "No se pudieron extraer fotografías del video. "
                            "Verifica que la ruta sea correcta y que el archivo no esté dañado."
                        )
                        st.stop()
                    update_step(2, "done")
                except Exception as e:
                    update_step(2, "error")
                    st.error("Ocurrió un problema al leer el video.")
                    with st.expander("Detalle técnico"):
                        st.code(str(e))
                    st.stop()

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
                    update_step(3, "error")
                    st.error("Ocurrió un problema al mejorar la calidad de las imágenes.")
                    with st.expander("Detalle técnico"):
                        st.code(str(e))
                    st.stop()

                # PASO 5 — Detección
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
                    update_step(4, "error")
                    st.error("Ocurrió un problema durante la detección de baches.")
                    with st.expander("Detalle técnico"):
                        st.code(str(e))
                    st.stop()

                progress_bar.progress(1.0)
                status_text.empty()

            st.success("✓ Análisis completado. Resultados listos más abajo.")

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
                    <div class="metric-lbl">Puntos GPS registrados</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{len(frame_numbers)}</div>
                    <div class="metric-lbl">Fotografías analizadas</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color:{'#b91c1c' if total_baches > 0 else '#15803d'}">{total_baches}</div>
                    <div class="metric-lbl">Baches detectados</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{frames_con_baches}</div>
                    <div class="metric-lbl">Fotografías con baches</div>
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
                st.markdown("#### Vista previa — antes y después")
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
                            st.markdown(f"<div class='img-caption'>MEJORADA · foto {frame_num}</div>",
                                        unsafe_allow_html=True)
                        with c2:
                            st.image(det_rgb, use_container_width=True)
                            st.markdown(f"<div class='img-caption'>DETECTADA · {det_label}</div>",
                                        unsafe_allow_html=True)
                    else:
                        det_rgb = cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB)
                        st.image(det_rgb, use_container_width=True)
                        st.markdown(f"<div class='img-caption'>foto {frame_num} · {det_label}</div>",
                                    unsafe_allow_html=True)

                    if i < len(preview_paths) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

            # ── Tabla de detecciones ───────────────────────────────────────────
            if not detections_df.empty:
                with st.expander(f"📋 Detalle de baches detectados ({total_baches} registros)"):
                    display_cols = ["frame_number", "lat", "lon", "label", "confidence",
                                    "x1", "y1", "x2", "y2"]
                    tabla = detections_df[display_cols].rename(columns={
                        "frame_number": "Fotografía",
                        "lat": "Latitud",
                        "lon": "Longitud",
                        "label": "Tipo",
                        "confidence": "Confianza",
                        "x1": "x1 (px)", "y1": "y1 (px)", "x2": "x2 (px)", "y2": "y2 (px)",
                    })
                    st.dataframe(
                        tabla.style.format({
                            "Confianza": "{:.3f}",
                            "Latitud": "{:.6f}",
                            "Longitud": "{:.6f}",
                        }),
                        use_container_width=True,
                    )
            else:
                st.info("ℹ️ No se detectaron baches con la sensibilidad actual. "
                        "Prueba reducir el valor de sensibilidad de detección.")

            # ── Guardar CSVs ───────────────────────────────────────────────────
            metadata_df.to_csv(csv_dir / "drone_metadata.csv", index=False)
            selected_df.to_csv(csv_dir / "frames_selected.csv", index=False)
            if not detections_df.empty:
                detections_df.to_csv(csv_dir / "detections.csv", index=False)

            # ── ZIP (armado en memoria, sin duplicar archivos en disco) ─────────
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in out_root.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, arcname=(Path(run_id) / file_path.relative_to(out_root)).as_posix())
            st.session_state["zip_bytes"] = zip_buffer.getvalue()
            st.session_state["zip_filename"] = f"road_ai_results_{run_id}.zip"
            st.session_state["zip_ready"] = True

            # Las carpetas temporales de extracción/mejora ya no se necesitan
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(enh_dir, ignore_errors=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"📁 Resultados guardados en: `output/{run_id}` — cada video que proceses "
                "queda en su propia carpeta con fecha y nombre, para poder revisarlos o compararlos después."
            )

            # Fuerza a adjuntar de nuevo el video y el .SRT en la próxima corrida:
            # evita reutilizar por error la telemetría de este video con el siguiente.
            st.session_state.input_generation += 1
            st.caption(
                "Para procesar otro video, escribe su ruta y sube su propio archivo .SRT — "
                "los campos se limpiaron para evitar mezclar datos de vuelos distintos."
            )

# ── Descarga persistente ───────────────────────────────────────────────────────

if st.session_state.get("zip_ready") and st.session_state.get("zip_bytes"):
    st.markdown("<br>", unsafe_allow_html=True)
    with col_right:
        st.download_button(
            label="⬇  Descargar resultados completos (fotografías y datos)",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state.get("zip_filename") or "road_ai_results.zip",
            mime="application/zip",
            use_container_width=True,
        )