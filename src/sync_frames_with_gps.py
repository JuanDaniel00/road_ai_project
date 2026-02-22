import pandas as pd
from haversine import haversine

df = pd.read_csv("output/drone_metadata.csv")

DISTANCE_INTERVAL = 7  # metros

selected_frames = []
accumulated_distance = 0

for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    current = df.iloc[i]

    p1 = (prev["lat"], prev["lon"])
    p2 = (current["lat"], current["lon"])

    distance = haversine(p1, p2) * 1000  # metros

    accumulated_distance += distance
    if distance > 0:
        print(f"Frame {i} → distancia: {distance:.4f} m")

    if accumulated_distance >= DISTANCE_INTERVAL:
        selected_frames.append(current)
        print(f"✅ Frame seleccionado: {int(current['frame_number'])} (acumulado: {accumulated_distance:.2f} m)")
        accumulated_distance = 0

result_df = pd.DataFrame(selected_frames)
result_df.to_csv("output/frames_every_7m_real.csv", index=False)

print("Frames seleccionados:")
print(result_df.head())
print(f"Total seleccionados: {len(result_df)}")