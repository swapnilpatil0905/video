import csv
import json
import os
from collections import defaultdict

AUDIT_FILE = "faces/cluster_audit.csv"
FACES_FILE = "faces/clustered_faces.csv"
REPRESENTATIVES_FILE = "faces/cluster_representatives.csv"

OUTPUT_JSON = "characters/character_database.json"
OUTPUT_CSV = "characters/character_scene_mapping.csv"

os.makedirs("characters", exist_ok=True)

# Load cluster audit.
with open(AUDIT_FILE, newline="", encoding="utf-8") as f:
    audit = {
        row["character_id"]: row
        for row in csv.DictReader(f)
    }

# Load face-to-cluster assignments.
with open(FACES_FILE, newline="", encoding="utf-8") as f:
    face_records = list(csv.DictReader(f))

# Load representative faces.
representatives = defaultdict(list)

with open(REPRESENTATIVES_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        representatives[row["character_id"]].append({
            "face_id": int(row["face_id"]),
            "scene": row["scene"],
            "image": row["face_crop"],
            "similarity": float(row["centroid_similarity"])
        })

characters = {}
scene_mapping = defaultdict(list)

for record in face_records:

    character_id = record["cluster_id"]

    if character_id == "UNASSIGNED":
        continue

    scene = record["scene"]
    face_id = int(record["face_id"])

    if character_id not in characters:
        characters[character_id] = {
            "character_id": character_id,
            "face_count": 0,
            "scene_ids": set(),
            "face_ids": []
        }

    character = characters[character_id]

    character["face_count"] += 1
    character["face_ids"].append(face_id)

    scene_id = int(
        scene.replace("scene_", "").replace(".jpg", "")
    )

    character["scene_ids"].add(scene_id)

    scene_mapping[scene_id].append({
        "character_id": character_id,
        "face_id": face_id
    })

# Build final JSON.
output_characters = []

for character_id, character in characters.items():

    metrics = audit[character_id]

    output_characters.append({
        "character_id": character_id,
        "status": "provisional",
        "face_count": character["face_count"],
        "scene_count": len(character["scene_ids"]),
        "first_scene": min(character["scene_ids"]),
        "last_scene": max(character["scene_ids"]),
        "mean_similarity": float(metrics["mean_similarity"]),
        "p10_similarity": float(metrics["p10_similarity"]),
        "scene_ids": sorted(character["scene_ids"]),
        "representative_faces": representatives[character_id]
    })

output_characters.sort(
    key=lambda x: x["face_count"],
    reverse=True
)

database = {
    "source_video": "input/source.mp4",
    "total_detected_faces": len(face_records),
    "identified_face_count": sum(
        c["face_count"] for c in output_characters
    ),
    "character_cluster_count": len(output_characters),
    "characters": output_characters
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(database, f, ensure_ascii=False, indent=2)

# Create scene-to-character mapping.
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "scene_id",
        "character_id",
        "face_id"
    ])

    for scene_id in sorted(scene_mapping):
        for item in scene_mapping[scene_id]:
            writer.writerow([
                scene_id,
                item["character_id"],
                item["face_id"]
            ])

print("\nCharacter database created.")
print("Characters:", len(output_characters))
print("Assigned faces:", database["identified_face_count"])
print("Scene mappings:", sum(map(len, scene_mapping.values())))
print("JSON:", OUTPUT_JSON)
print("CSV:", OUTPUT_CSV)

print("\nTop 10 characters:")

for character in output_characters[:10]:
    print(
        character["character_id"],
        "| Faces:", character["face_count"],
        "| Scenes:", character["scene_count"]
    )
