import cv2
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_video", "video.MP4")
CSV_PATH = os.path.join(BASE_DIR, "output", "frames_every_7m_real.csv")
LEGACY_CSV_PATH = os.path.join(BASE_DIR, "output", "processed", "frames_every_7m_real.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output", "frames", "frames_7m")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Leer CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
elif os.path.exists(LEGACY_CSV_PATH):
    df = pd.read_csv(LEGACY_CSV_PATH)
else:
    raise FileNotFoundError(
        f"No se encontró el CSV en '{CSV_PATH}' ni en '{LEGACY_CSV_PATH}'"
    )

# Convertir a enteros (muy importante)
frame_numbers = df["frame_number"].astype(int).tolist()

# Abrir video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ No se pudo abrir el video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames en video: {total_frames}")
print(f"Frames a extraer: {len(frame_numbers)}")

total_to_extract = len(frame_numbers)
progress_step = max(1, total_to_extract // 10)

for index, frame_number in enumerate(frame_numbers, start=1):

    if frame_number >= total_frames:
        print(f"⚠ Frame {frame_number} fuera de rango")
        continue

    # Mover cursor directamente al frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print(f"❌ Error leyendo frame {frame_number}")
        continue

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"frame_{frame_number}.jpg"
    )

    cv2.imwrite(output_path, frame)

    if index % progress_step == 0 or index == total_to_extract:
        progress_pct = (index / total_to_extract) * 100
        print(f"Progreso: {index}/{total_to_extract} ({progress_pct:.1f}%)")

cap.release()

print("✅ Extracción completada")