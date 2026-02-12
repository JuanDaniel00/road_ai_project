import pandas as pd
from haversine import haversine

# Cargar frames sincronizados con GPS
df = pd.read_csv("output/frames_with_gps.csv")

DISTANCE_INTERVAL = 7  # metros

selected = []
accumulated_distance = 0

for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    current = df.iloc[i]

    p1 = (prev["lat"], prev["lon"])
    p2 = (current["lat"], current["lon"])

    # distancia en metros
    distance = haversine(p1, p2) * 1000

    accumulated_distance += distance

    if accumulated_distance >= DISTANCE_INTERVAL:
        selected.append(current)
        accumulated_distance = 0

result_df = pd.DataFrame(selected)
result_df.to_csv("output/frames_every_7m.csv", index=False)

print("✅ Frames seleccionados cada 7 metros")
print(result_df)
