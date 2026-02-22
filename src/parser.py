import re
import pandas as pd

srt_path = "data/gps_logs/drone_data.srt"

data = []

with open(srt_path, "r", encoding="utf-8") as file:
    content = file.read()

blocks = content.split("\n\n")

for block in blocks:
    frame_match = re.search(r"FrameCnt:\s*(\d+)", block)
    lat_match = re.search(r"\[latitude:\s*([-\d\.]+)\]", block)
    lon_match = re.search(r"\[longitude:\s*([-\d\.]+)\]", block)

    if frame_match and lat_match and lon_match:
        data.append({
            "frame_number": int(frame_match.group(1)),
            "lat": float(lat_match.group(1)),
            "lon": float(lon_match.group(1)),
        })

df = pd.DataFrame(data)
df.to_csv("output/drone_metadata.csv", index=False)

print("✅ Metadata extraída")
print(df.head())