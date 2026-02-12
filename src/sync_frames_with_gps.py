import pandas as pd

# Cargar datos
frames = pd.read_csv("output/frame_timestamps.csv")
gps = pd.read_csv("data/gps_logs/fake_gps.csv")

# Función para encontrar GPS más cercano por tiempo
def find_nearest_gps(timestamp):
    if timestamp < gps["timestamp_sec"].min() or timestamp > gps["timestamp_sec"].max():
        return None

    gps["time_diff"] = abs(gps["timestamp_sec"] - timestamp)
    nearest = gps.loc[gps["time_diff"].idxmin()]
    return nearest

results = []

for _, frame in frames.iterrows():
    nearest_gps = find_nearest_gps(frame["timestamp_sec"])

    if nearest_gps is None:
        continue  # saltar frames fuera del rango GPS

    results.append({
        "image_name": frame["image_name"],
        "frame_time": frame["timestamp_sec"],
        "gps_time": nearest_gps["timestamp_sec"],
        "lat": nearest_gps["lat"],
        "lon": nearest_gps["lon"]
    })


final_df = pd.DataFrame(results)
final_df.to_csv("output/frames_with_gps.csv", index=False)

print("✅ Sincronización completada")
