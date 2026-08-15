"""
batch_process.py — Procesamiento masivo de videos de dron para preparar dataset.

Recorre una carpeta de videos, empareja cada .MP4 con su .SRT de telemetría
(por el número base DJI_XXXX, ignorando sufijos como "-011"), y para cada uno
ejecuta los pasos 1-4 del pipeline (GPS -> selección por distancia -> extracción
-> mejora de calidad). NO corre detección YOLO: el objetivo es generar
fotografías mejoradas + metadata GPS para etiquetar manualmente como dataset
de entrenamiento.

Las fotografías resultantes se depositan directamente en output/train,
output/val u output/test según el split fijo definido en SPLIT_ASSIGNMENT,
con el nombre del video como prefijo para evitar colisiones entre videos
distintos que reutilizan los mismos números de frame.

Es reanudable: si se interrumpe, al volver a correrlo salta los videos que
ya tengan su marcador .done y sigue con los pendientes.

Uso:
    venv\\Scripts\\python.exe src\\batch_process.py
    venv\\Scripts\\python.exe src\\batch_process.py --distance-interval 5
    venv\\Scripts\\python.exe src\\batch_process.py --only DJI_0628
"""

import argparse
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

from pipeline import parse_srt, select_frames_by_distance, extract_frames, enhance_all

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR   = PROJECT_ROOT / "videos"
OUTPUT_ROOT  = PROJECT_ROOT / "output"

# Reparto fijo train/val/test — semilla 42, acordado con el usuario.
SPLIT_ASSIGNMENT = {
    "DJI_0627": "train", "DJI_0628": "train", "DJI_0630": "train", "DJI_0631": "train",
    "DJI_0632": "train", "DJI_0635": "train", "DJI_0636": "train", "DJI_0637": "train",
    "DJI_0638": "train", "DJI_0639": "train", "DJI_0640": "train", "DJI_0641": "train",
    "DJI_0642": "train", "DJI_0644": "train", "DJI_0645": "train", "DJI_0647": "train",
    "DJI_0633": "val",   "DJI_0634": "val",   "DJI_0643": "val",
    "DJI_0626": "test",  "DJI_0629": "test",  "DJI_0646": "test",
}

VIDEO_BASE_RE = re.compile(r"^(DJI_\d+)")


def find_video_srt_pairs(videos_dir: Path):
    """Empareja cada .MP4 con su .SRT por el número base DJI_XXXX."""
    srt_by_base = {}
    for srt_path in videos_dir.glob("*.[Ss][Rr][Tt]"):
        m = VIDEO_BASE_RE.match(srt_path.stem)
        if m:
            srt_by_base[m.group(1)] = srt_path

    pairs = []
    for video_path in sorted(videos_dir.glob("*.[Mm][Pp]4")):
        m = VIDEO_BASE_RE.match(video_path.stem)
        if not m:
            print(f"  [SKIP] {video_path.name}: no coincide con el patrón DJI_XXXX")
            continue
        base = m.group(1)
        srt_path = srt_by_base.get(base)
        if srt_path is None:
            print(f"  [SKIP] {video_path.name}: no se encontró .SRT para {base}")
            continue
        pairs.append((base, video_path, srt_path))
    return pairs


def process_video(base: str, video_path: Path, srt_path: Path,
                   split: str, distance_interval: float,
                   clahe_clip: float, clahe_tile: int, unsharp_str: float):
    split_dir = OUTPUT_ROOT / split
    frames_dir = split_dir / "frames_enhanced"
    meta_dir = split_dir / "metadata"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    done_marker = meta_dir / f"{base}.done"
    if done_marker.exists():
        print(f"  [SKIP] {base}: ya procesado (marcador .done presente)")
        return "skipped"

    t0 = time.time()

    srt_content = srt_path.read_text(encoding="utf-8", errors="replace")
    metadata_df = parse_srt(srt_content)
    if metadata_df.empty:
        raise RuntimeError(f"{srt_path.name}: no se encontraron puntos GPS")

    selected_df = select_frames_by_distance(metadata_df, distance_interval)
    if selected_df.empty:
        raise RuntimeError(f"{base}: 0 fotografías seleccionadas (recorrido muy corto para {distance_interval} m)")

    frame_numbers = selected_df["frame_number"].astype(int).tolist()

    raw_dir = Path(tempfile.mkdtemp(prefix=f"road_ai_raw_{base}_"))
    try:
        saved_paths = extract_frames(
            str(video_path), frame_numbers, raw_dir,
            filename_prefix=f"{base}_",
        )
        if not saved_paths:
            raise RuntimeError(f"{base}: no se pudo extraer ninguna fotografía del video")

        enhanced_paths = enhance_all(
            saved_paths, frames_dir, clahe_clip, clahe_tile, unsharp_str,
        )
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)

    metadata_df.to_csv(meta_dir / f"{base}_gps_full.csv", index=False)
    selected_df.to_csv(meta_dir / f"{base}_frames_selected.csv", index=False)

    done_marker.write_text(
        f"video={video_path.name}\nsrt={srt_path.name}\n"
        f"frames_gps_total={len(metadata_df)}\nframes_selected={len(frame_numbers)}\n"
        f"frames_enhanced={len(enhanced_paths)}\ndistance_interval_m={distance_interval}\n",
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    print(f"  [OK] {base} -> {split}: {len(enhanced_paths)}/{len(frame_numbers)} fotos "
          f"({len(metadata_df)} puntos GPS) en {elapsed:.0f}s")
    return "done"


def main():
    ap = argparse.ArgumentParser(description="Procesamiento masivo de videos de dron")
    ap.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    ap.add_argument("--distance-interval", type=float, default=7.0,
                     help="Metros entre fotografías seleccionadas (default: 7)")
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--unsharp", type=float, default=1.2)
    ap.add_argument("--only", default=None,
                     help="Procesa solo el video cuyo base coincide, ej. DJI_0628")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"ERROR: no existe la carpeta de videos: {videos_dir}")
        sys.exit(1)

    pairs = find_video_srt_pairs(videos_dir)
    if args.only:
        pairs = [p for p in pairs if p[0] == args.only]

    unassigned = [base for base, _, _ in pairs if base not in SPLIT_ASSIGNMENT]
    if unassigned:
        print(f"ERROR: sin split asignado para: {unassigned}")
        sys.exit(1)

    print(f"Videos a procesar: {len(pairs)}")
    print(f"Parámetros: intervalo={args.distance_interval}m, clahe_clip={args.clahe_clip}, "
          f"clahe_tile={args.clahe_tile}, unsharp={args.unsharp}")
    print("-" * 70)

    results = {"done": 0, "skipped": 0, "error": 0}
    errors = []
    t_start = time.time()

    for i, (base, video_path, srt_path) in enumerate(pairs, 1):
        split = SPLIT_ASSIGNMENT[base]
        print(f"[{i}/{len(pairs)}] {base} ({video_path.name}) -> {split}")
        try:
            status = process_video(
                base, video_path, srt_path, split,
                args.distance_interval, args.clahe_clip, args.clahe_tile, args.unsharp,
            )
            results[status] += 1
        except Exception as e:
            results["error"] += 1
            errors.append((base, str(e)))
            print(f"  [ERROR] {base}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print("-" * 70)
    print(f"Terminado en {elapsed/60:.1f} min — "
          f"{results['done']} procesados, {results['skipped']} saltados, {results['error']} con error")
    if errors:
        print("\nVideos con error:")
        for base, msg in errors:
            print(f"  - {base}: {msg}")


if __name__ == "__main__":
    main()
