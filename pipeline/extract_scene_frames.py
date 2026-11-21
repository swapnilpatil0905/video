import csv
import os
import cv2

VIDEO = "input/source.mp4"
SCENE_CSV = "scenes/scene_timeline.csv"
OUTPUT_DIR = "scenes/frames"
OUTPUT_CSV = "scenes/frame_manifest.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_size = (480, int(height * 480 / width))

with open(SCENE_CSV, newline="", encoding="utf-8") as f:
    scenes = list(csv.DictReader(f))

manifest = []

for i, scene in enumerate(scenes, 1):
    start = float(scene["start_seconds"])
    end = float(scene["end_seconds"])
    midpoint = (start + end) / 2

    cap.set(cv2.CAP_PROP_POS_MSEC, midpoint * 1000)
    success, frame = cap.read()

    if not success:
        print(f"Skipping scene {i}: frame read failed")
        continue

    frame = cv2.resize(frame, new_size)

    filename = f"scene_{i:05d}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    manifest.append({
        "scene_id": i,
        "start_seconds": start,
        "end_seconds": end,
        "midpoint_seconds": round(midpoint, 3),
        "frame_path": path
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(scenes)} scenes")

cap.release()

if manifest:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

print("\nFrame extraction completed.")
print("Scenes:", len(scenes))
print("Frames extracted:", len(manifest))
print("Output:", OUTPUT_DIR)
print("Manifest:", OUTPUT_CSV)
