import cv2
import os
import pandas as pd

video_path = "data/raw_video/test.mp4"
output_folder = "output/frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ No se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS del video: {fps}")

frame_interval = int(fps)  # 1 imagen por segundo (temporal por ahora)

frame_count = 0
saved_count = 0

records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tiempo exacto del frame en segundos
    timestamp_sec = frame_count / fps

    if frame_count % frame_interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        full_path = os.path.join(output_folder, filename)

        cv2.imwrite(full_path, frame)

        records.append({
            "image_name": filename,
            "frame_number": frame_count,
            "timestamp_sec": timestamp_sec
        })

        print(f"Guardado {filename} | frame {frame_count} | t={timestamp_sec:.3f}s")
        saved_count += 1

    frame_count += 1

cap.release()

df = pd.DataFrame(records)
df.to_csv("output/frame_timestamps.csv", index=False)

print("✅ Frames y tiempos guardados correctamente")
