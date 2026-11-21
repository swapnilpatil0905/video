import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont

EMBEDDINGS_FILE = "faces/face_embeddings.npz"
MANIFEST_FILE = "faces/embedding_manifest.csv"

OUTPUT_CSV = "faces/clustered_faces.csv"
SUMMARY_CSV = "faces/cluster_summary.csv"
CONTACT_DIR = "faces/cluster_contact_sheets"

# Initial clustering parameters.
EPS = 0.35
MIN_SAMPLES = 2

os.makedirs(CONTACT_DIR, exist_ok=True)

# Load embeddings.
data = np.load(EMBEDDINGS_FILE)
face_ids = data["face_ids"]
embeddings = data["embeddings"]

with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))

if len(face_ids) != len(manifest):
    raise RuntimeError("Embedding and manifest counts do not match.")

print("Loaded embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])
print("Running DBSCAN...")

# Normalized embeddings allow cosine distance comparison.
model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="cosine",
    n_jobs=-1
)

labels = model.fit_predict(embeddings)

# Build cluster membership.
clusters = defaultdict(list)

for index, label in enumerate(labels):
    clusters[int(label)].append(index)

# Create face-to-cluster mapping.
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "face_id",
        "scene",
        "face_crop",
        "cluster_id"
    ])

    for index, record in enumerate(manifest):
        label = int(labels[index])

        cluster_id = (
            "UNASSIGNED"
            if label == -1
            else f"CHAR_{label + 1:04d}"
        )

        writer.writerow([
            record["face_id"],
            record["scene"],
            record["face_crop"],
            cluster_id
        ])

# Create cluster summary.
summary = []

for label, indices in sorted(clusters.items()):
    cluster_id = (
        "UNASSIGNED"
        if label == -1
        else f"CHAR_{label + 1:04d}"
    )

    scene_ids = set()

    for index in indices:
        scene = manifest[index]["scene"]
        scene_id = scene.replace("scene_", "").replace(".jpg", "")
        scene_ids.add(int(scene_id))

    summary.append({
        "cluster_id": cluster_id,
        "face_count": len(indices),
        "scene_count": len(scene_ids),
        "first_scene": min(scene_ids),
        "last_scene": max(scene_ids)
    })

with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "cluster_id",
            "face_count",
            "scene_count",
            "first_scene",
            "last_scene"
        ]
    )
    writer.writeheader()
    writer.writerows(summary)

# Generate contact sheets, grouped by cluster.
font = ImageFont.load_default()

THUMB_SIZE = 112
LABEL_HEIGHT = 28
COLUMNS = 8
PER_SHEET = 80

for label, indices in sorted(clusters.items()):

    cluster_id = (
        "UNASSIGNED"
        if label == -1
        else f"CHAR_{label + 1:04d}"
    )

    cluster_dir = os.path.join(CONTACT_DIR, cluster_id)
    os.makedirs(cluster_dir, exist_ok=True)

    for sheet_number in range(
        (len(indices) + PER_SHEET - 1) // PER_SHEET
    ):

        batch = indices[
            sheet_number * PER_SHEET:
            (sheet_number + 1) * PER_SHEET
        ]

        rows = (len(batch) + COLUMNS - 1) // COLUMNS

        sheet = Image.new(
            "RGB",
            (
                COLUMNS * THUMB_SIZE,
                rows * (THUMB_SIZE + LABEL_HEIGHT)
            ),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for position, index in enumerate(batch):

            record = manifest[index]

            image = Image.open(
                record["face_crop"]
            ).convert("RGB")

            image = image.resize((THUMB_SIZE, THUMB_SIZE))

            x = (position % COLUMNS) * THUMB_SIZE
            y = (position // COLUMNS) * (
                THUMB_SIZE + LABEL_HEIGHT
            )

            sheet.paste(image, (x, y))

            face_id = record["face_id"]
            scene = record["scene"].replace(
                "scene_", ""
            ).replace(".jpg", "")

            draw.text(
                (x + 2, y + THUMB_SIZE + 3),
                f"F{face_id} S{scene}",
                fill="black",
                font=font
            )

        output = os.path.join(
            cluster_dir,
            f"sheet_{sheet_number + 1:03d}.jpg"
        )

        sheet.save(output, quality=85)

# Statistics.
valid_clusters = [
    label for label in clusters if label != -1
]

noise_count = len(clusters.get(-1, []))

print("\nClustering completed.")
print("Total faces:", len(embeddings))
print("Character clusters:", len(valid_clusters))
print("Unassigned faces:", noise_count)
print("Largest clusters:")

for item in sorted(
    [x for x in summary if x["cluster_id"] != "UNASSIGNED"],
    key=lambda x: x["face_count"],
    reverse=True
)[:20]:
    print(
        item["cluster_id"],
        "| Faces:", item["face_count"],
        "| Scenes:", item["scene_count"]
    )

print("\nSaved:")
print(OUTPUT_CSV)
print(SUMMARY_CSV)
print(CONTACT_DIR)
