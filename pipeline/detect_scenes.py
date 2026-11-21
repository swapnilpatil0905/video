import csv
import os

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

VIDEO = "input/source.mp4"
OUTPUT = "scenes"

os.makedirs(OUTPUT, exist_ok=True)

video = open_video(VIDEO)

scene_manager = SceneManager()
scene_manager.add_detector(
    ContentDetector(
        threshold=27.0,
        min_scene_len=15
    )
)

print("Detecting scenes...")

scene_manager.detect_scenes(video, show_progress=True)

scenes = scene_manager.get_scene_list()

with open(
    f"{OUTPUT}/scene_timeline.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "scene_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds"
    ])

    for i, (start, end) in enumerate(scenes, 1):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()

        writer.writerow([
            i,
            round(start_sec, 3),
            round(end_sec, 3),
            round(end_sec - start_sec, 3)
        ])

print("Scene detection completed.")
print("Total scenes:", len(scenes))
print("Saved: scenes/scene_timeline.csv")
