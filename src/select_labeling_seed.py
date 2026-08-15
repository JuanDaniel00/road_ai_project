"""
select_labeling_seed.py — Arma el subconjunto semilla para la Fase 1 de etiquetado.

Por cada video, toma N fotos de output/<split>/frames_enhanced/ espaciadas
uniformemente a lo largo del recorrido (ordenadas por frame_number), y las
copia (no mueve) a dataset/images/<split>/, listas para abrir en LabelImg.

No toca output/ ni el split "test" — test se deja intacto para evaluación final.

Uso:
    venv\\Scripts\\python.exe src\\select_labeling_seed.py
    venv\\Scripts\\python.exe src\\select_labeling_seed.py --per-video-train 18 --per-video-val 8
"""

import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
DATASET_ROOT = PROJECT_ROOT / "dataset"

FRAME_RE = re.compile(r"^(DJI_\d+)_frame_(\d+)\.jpg$", re.IGNORECASE)


def group_by_video(frames_dir: Path):
    groups = {}
    for img_path in frames_dir.glob("*.jpg"):
        m = FRAME_RE.match(img_path.name)
        if not m:
            continue
        base, frame_num = m.group(1), int(m.group(2))
        groups.setdefault(base, []).append((frame_num, img_path))
    for base in groups:
        groups[base].sort(key=lambda t: t[0])
    return groups


def evenly_spaced_indices(n_available: int, n_wanted: int):
    if n_wanted >= n_available:
        return list(range(n_available))
    step = n_available / n_wanted
    return sorted({int(i * step) for i in range(n_wanted)})


def select_seed(split: str, per_video: int, dest_root: Path):
    src_dir = OUTPUT_ROOT / split / "frames_enhanced"
    dest_dir = dest_root / "images" / split
    dest_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_video(src_dir)
    total_copied = 0
    total_skipped = 0
    print(f"\n[{split}] {len(groups)} videos encontrados en {src_dir}")

    for base in sorted(groups):
        frames = groups[base]
        idxs = evenly_spaced_indices(len(frames), per_video)
        selected = [frames[i][1] for i in idxs]

        copied = 0
        for img_path in selected:
            dest_path = dest_dir / img_path.name
            if dest_path.exists():
                total_skipped += 1
                continue
            shutil.copy2(img_path, dest_path)
            copied += 1
            total_copied += 1

        print(f"  {base}: {len(frames)} disponibles -> {len(selected)} seleccionadas ({copied} copiadas)")

    print(f"[{split}] total copiadas: {total_copied}, ya existentes: {total_skipped}")
    return total_copied


def write_dataset_yaml(dest_root: Path):
    yaml_path = dest_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dest_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pothole\n",
        encoding="utf-8",
    )
    print(f"\ndataset.yaml escrito en {yaml_path}")


def main():
    ap = argparse.ArgumentParser(description="Arma el subconjunto semilla para etiquetado")
    ap.add_argument("--per-video-train", type=int, default=18,
                     help="Fotos por video a incluir en el seed de train (default: 18)")
    ap.add_argument("--per-video-val", type=int, default=8,
                     help="Fotos por video a incluir en el seed de val (default: 8)")
    ap.add_argument("--dataset-dir", default=str(DATASET_ROOT))
    args = ap.parse_args()

    dest_root = Path(args.dataset_dir)
    n_train = select_seed("train", args.per_video_train, dest_root)
    n_val = select_seed("val", args.per_video_val, dest_root)

    (dest_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dest_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    write_dataset_yaml(dest_root)

    print(f"\nListo: {n_train} fotos nuevas en dataset/images/train, {n_val} en dataset/images/val")
    print("Carpetas dataset/labels/train y dataset/labels/val creadas (vacías) para los .txt de LabelImg.")
    print("test/ no fue tocado — se mantiene reservado para evaluación final.")


if __name__ == "__main__":
    main()
