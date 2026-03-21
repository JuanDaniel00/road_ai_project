"""
app.py — Road AI Inspector
Pipeline completo de inspección vial con dron.
Ejecutar: streamlit run src/app.py --server.maxUploadSize 10240
"""

import streamlit as st
import tempfile
import shutil
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from haversine import haversine

# ── Página ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road AI Inspector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

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

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🛣️ Road <span class="accent">AI</span> Inspector</h1>
    <div class="subtitle">Pipeline de inspección vial con dron · Detección de baches y defectos</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "step_status" not in st.session_state:
    st.session_state.step_status = ["pending"] * 4
if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None

# ── Funciones pipeline ────────────────────────────────────────────────────────

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
    for idx, fn in enumerate(frame_numbers):
        if fn >= total_video:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
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
    # Reemplaza fastNlMeans: preserva bordes, 15-20x mas rapido
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def normalize_color(img):
    result = np.zeros_like(img)
    for i in range(3):
        ch = img[:, :, i]
        mn, mx = ch.min(), ch.max()
        result[:, :, i] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else ch
    return result

def enhance_frame_fn(img, clahe_clip, clahe_tile, unsharp_str):
    img = apply_bilateral(img)
    img = apply_clahe(img, clahe_clip, clahe_tile)
    img = apply_unsharp(img, unsharp_str)
    img = normalize_color(img)
    return img

# Worker top-level para multiprocessing (no puede ser lambda ni nested)
def _enhance_worker(args):
    fp, out_dir, clahe_clip, clahe_tile, unsharp_str = args
    img = cv2.imread(fp)
    if img is None:
        return None
    enh = enhance_frame_fn(img, clahe_clip, clahe_tile, unsharp_str)
    out_path = Path(out_dir) / Path(fp).name
    cv2.imwrite(str(out_path), enh, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(out_path)

def enhance_all(frame_paths, out_dir, clahe_clip, clahe_tile,
                unsharp_str, progress_bar, status_text):
    workers = max(1, cpu_count() - 1)
    args = [(fp, str(out_dir), clahe_clip, clahe_tile, unsharp_str) for fp in frame_paths]
    enhanced = []
    with Pool(processes=workers) as pool:
        for idx, result in enumerate(pool.imap(_enhance_worker, args)):
            if result:
                enhanced.append(result)
            progress_bar.progress((idx + 1) / len(frame_paths))
            status_text.markdown(
                f"<span style='color:#64748b;font-size:0.82rem'>"
                f"Mejorando frame {idx+1}/{len(frame_paths)} "
                f"<span style='color:#38bdf8'>({workers} nucleos)</span></span>",
                unsafe_allow_html=True,
            )
    return enhanced


# ── Layout ────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.6], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNA IZQUIERDA
# ─────────────────────────────────────────────────────────────────────────────

with col_left:
    st.markdown("### Archivos de entrada")

    # Video — ruta local, sin uploader, sin límite de tamaño
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

    # SRT — uploader normal (archivo pequeño, sin problema)
    st.markdown("**Telemetría GPS (.SRT)**")
    srt_file = st.file_uploader(
        label="srt_upload",
        type=["srt", "SRT"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Parámetros del pipeline")

    distance_interval = st.slider("Intervalo de muestreo (metros)", 3, 20, 7, step=1)

    with st.expander("⚙️ Parámetros de mejora de imagen"):
        clahe_clip  = st.slider("CLAHE — clip limit",         1.0, 5.0, 3.0, step=0.5)
        clahe_tile  = st.slider("CLAHE — tile size (px)",     4,   16,  8,   step=4)
        unsharp_str = st.slider("Unsharp Mask — intensidad",  0.3, 2.0, 1.2, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Botón SIN disabled — la validación ocurre dentro del bloque run
    run_btn = st.button("▶  Ejecutar pipeline", use_container_width=True)

    ready = video_ok and (srt_file is not None)

    if not ready:
        msgs = []
        if not video_path_input:
            msgs.append("Escribe la ruta del video")
        elif not video_ok:
            msgs.append("Ruta de video no válida")
        if srt_file is None:
            msgs.append("Sube el archivo .SRT")
        st.markdown(
            f"<div style='text-align:center;color:#f87171;font-size:0.82rem;margin-top:0.5rem'>"
            f"{'  ·  '.join(msgs)}</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COLUMNA DERECHA
# ─────────────────────────────────────────────────────────────────────────────

with col_right:
    st.markdown("### Estado del pipeline")

    steps = [
        ("01", "Parseo GPS",              "Extrae coordenadas y frame_number del .SRT"),
        ("02", "Selección por distancia", f"1 frame cada {distance_interval} m recorrido"),
        ("03", "Extracción de frames",    "Lee el video y exporta los JPEGs seleccionados"),
        ("04", "Mejora de imagen",        "Denoise → CLAHE → Unsharp Mask → Normalización"),
    ]

    STATUS_BADGE  = {"pending": "badge-warn", "running": "badge-warn", "done": "badge-ok",  "error": "badge-err"}
    STATUS_LABEL  = {"pending": "PENDIENTE",  "running": "PROCESANDO…","done": "COMPLETADO","error": "ERROR"}
    BORDER_COLOR  = {"pending": "#1e2530",    "running": "#1e3a8a",    "done": "#166534",   "error": "#7f1d1d"}

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

    # ── Ejecución ─────────────────────────────────────────────────────────────

    if run_btn:
        if not ready:
            st.warning("⚠️ Completa los archivos de entrada antes de ejecutar.")
        else:
            st.session_state.step_status = ["pending"] * 4

            # Carpeta persistente en disco — sobrevive al re-render del download_button
            out_root = Path(video_path_input.strip()).parent / "road_ai_output"
            raw_dir = out_root / "frames_7m"
            enh_dir = out_root / "frames_7m_enhanced"
            csv_dir = out_root / "metadata"
            for d in [raw_dir, enh_dir, csv_dir]:
                d.mkdir(parents=True, exist_ok=True)

            if True:  # bloque de ejecución (reemplaza with tempfile)

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

                progress_bar.progress(1.0)
                status_text.empty()

                # Métricas
                results_slot.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-val">{len(metadata_df):,}</div>
                        <div class="metric-lbl">Frames GPS</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{len(frame_numbers)}</div>
                        <div class="metric-lbl">Seleccionados</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{len(frame_numbers) * distance_interval}m</div>
                        <div class="metric-lbl">Distancia aprox.</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{len(enhanced_paths)}</div>
                        <div class="metric-lbl">Mejoradas</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Preview
                st.markdown("#### Preview — original vs mejorada")
                for i in range(min(3, len(saved_paths))):
                    raw_rgb = cv2.cvtColor(cv2.imread(saved_paths[i]),  cv2.COLOR_BGR2RGB)
                    enh_rgb = cv2.cvtColor(cv2.imread(enhanced_paths[i]), cv2.COLOR_BGR2RGB)
                    c1, c2 = st.columns(2)
                    fn = Path(saved_paths[i]).stem
                    with c1:
                        st.image(raw_rgb, use_container_width=True)
                        st.markdown(f"<div class='img-caption'>ORIGINAL · {fn}</div>", unsafe_allow_html=True)
                    with c2:
                        st.image(enh_rgb, use_container_width=True)
                        st.markdown(f"<div class='img-caption'>MEJORADA · {fn}</div>", unsafe_allow_html=True)
                    if i < min(3, len(saved_paths)) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

                # CSVs a disco
                metadata_df.to_csv(csv_dir / "drone_metadata.csv", index=False)
                selected_df.to_csv(csv_dir / "frames_selected.csv", index=False)

                # ZIP único con todo
                master_zip_path = out_root.parent / "road_ai_results"
                shutil.make_archive(str(master_zip_path), "zip", str(out_root))
                with open(str(master_zip_path) + ".zip", "rb") as zf:
                    master_zip_bytes = zf.read()
                st.session_state["zip_bytes"] = master_zip_bytes
                st.session_state["zip_ready"] = True

                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"📁 Archivos guardados en: `{out_root}`")

# ── Descarga persistente — fuera del bloque run, sobrevive re-renders ─────────

if st.session_state.get("zip_ready") and st.session_state.get("zip_bytes"):
    st.markdown("<br>", unsafe_allow_html=True)
    with col_right:
        st.download_button(
            label="⬇  Descargar todo (imágenes + CSVs)",
            data=st.session_state["zip_bytes"],
            file_name="road_ai_results.zip",
            mime="application/zip",
            use_container_width=True,
        )