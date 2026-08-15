# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Road AI Inspector — a single-page Streamlit app that turns drone flight footage + GPS telemetry into an enhanced, defect-annotated image set. There is no backend/API layer; `src/app.py` is the entire application (UI, pipeline logic, and orchestration in one file, ~890 lines).

## Commands

There is no test suite, linter, or build step in this repo — don't invent one.

**Run the app:**
```powershell
run.bat
```
This creates `venv/` if missing, installs `requirements.txt` if Streamlit isn't importable, and launches:
```powershell
streamlit run src/app.py --server.maxUploadSize 10240
```
Opens at `http://localhost:8501`.

**Manual setup** (if not using `run.bat`):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Important:** `requirements.txt` is UTF-16LE encoded (not UTF-8) — an artifact of how it was generated on Windows. Tools that assume UTF-8 will mangle it. When editing it, preserve the encoding or re-save explicitly as UTF-16LE.

Note the app itself additionally requires `ultralytics` and `ultralyticsplus` for YOLOv8 detection (step 5); these are imported defensively (`YOLO_AVAILABLE` / `HF_AVAILABLE` flags) and are not currently listed in `requirements.txt`, so the app degrades gracefully but detection won't run without them installed.

## Architecture

### The pipeline (`src/app.py`)

Everything happens in one Streamlit run, driven by a 5-step pipeline triggered by the "Ejecutar pipeline completo" button. Each step updates `st.session_state.step_status` and renders a status card:

1. **Parse GPS** (`parse_srt`) — regex-extracts `FrameCnt`, `latitude`, `longitude` from the drone's `.srt` telemetry file into a DataFrame.
2. **Select frames by distance** (`select_frames_by_distance`) — walks the GPS trace accumulating haversine distance, picks one frame every N meters (`distance_interval` slider, default 7m).
3. **Extract frames** (`extract_frames`) — seeks into the source video with OpenCV (`cv2.VideoCapture`) and dumps the selected frame numbers as `.jpg`. The video is read directly from the path the user types in (never uploaded, to avoid memory blowups with large files).
4. **Enhance frames** (`enhance_all` → `enhance_frame_fn`) — sequential per-frame pipeline: bilateral filter (denoise) → CLAHE (local contrast, on LAB L-channel) → unsharp mask → per-channel color normalization (only when channel range is narrow, <51).
5. **YOLOv8 detection** (`run_yolo_detection`) — runs inference per enhanced frame, then filters raw detections through: a COCO noise blocklist (`COCO_ROAD_NOISE` — cars, people, etc., only applied when *not* using the dedicated pothole model), a max bounding-box-area-ratio cutoff, and a lateral ROI margin (ignores detections near the frame edges, since those are usually road shoulder/off-road). Detections are cross-referenced back to GPS coordinates via `frame_number`.

Outputs land in `<project_root>/output/` (`frames_detected/`, `metadata/*.csv`) and get zipped to `<project_root>/road_ai_results.zip`, which is also kept in `st.session_state["zip_bytes"]` so the download button survives Streamlit reruns.

### Model loading (`load_pothole_model`)

Two mutually exclusive modes, toggled by the "Usar modelo propio (.pt)" switch in the UI:
- **Default**: downloads/loads `keremberke/yolov8n-pothole-segmentation` from Hugging Face via `ultralyticsplus`, cached locally as `yolov8_pothole_hf.pt`.
- **Custom**: loads a local `.pt` path via `ultralytics.YOLO`; the code sniffs `model.names` for "pothole"/"hole"/"crack" to decide whether to apply the COCO noise filter.

A module-level `_patch_torch_safe_globals()` monkey-patches `torch.load` to force `weights_only=False`, working around PyTorch 2.6's default change that breaks loading legacy YOLO checkpoints. This runs unconditionally at import time — keep it if touching model-loading code, since removing it will break loading older `.pt` files.

### UI structure

Two-column Streamlit layout: left column is all inputs (video path, `.srt` uploader, model selection, pipeline parameter sliders); right column is pipeline status cards + live progress + results (metrics, before/after preview images, detections table, download button). All styling is inline CSS injected via a single `st.markdown(..., unsafe_allow_html=True)` block near the top — there's no separate stylesheet.

### Directories

- `data/gps_logs/` — sample `.srt` telemetry.
- `runs/detect/road_ai_training/` — YOLO training run artifacts (multiple versions: v1, v12, v13 — see README for v13's metrics).
- `docs/assets/` — training curves and example detection images referenced from `README.md`.
- `data/raw_video/`, `output/`, `venv/`, `*.pt`/`*.pth`/`*.onnx` weights, and `*.zip` results are all gitignored (large/generated artifacts, kept local-only).
