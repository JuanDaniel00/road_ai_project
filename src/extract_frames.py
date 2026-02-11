import cv2
import os

video_path = "data/raw_video/test.mp4"
output_folder = "output/frames"

# Crear carpeta si no existe
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ No se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS del video: {fps}")

# Guardar 1 imagen por segundo
frame_interval = int(fps)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"{output_folder}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Guardado: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print("✅ Extracción terminada")
