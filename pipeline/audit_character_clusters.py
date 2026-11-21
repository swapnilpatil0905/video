import csv
import os
import numpy as np

from sklearn.cluster import DBSCAN

EMBEDDINGS_FILE = "faces/face_embeddings.npz"
MANIFEST_FILE = "faces/embedding_manifest.csv"

OUTPUT_CSV = "faces/cluster_audit.csv"
REPRESENTATIVES_CSV = "faces/cluster_representatives.csv"

EPS = 0.35
MIN_SAMPLES = 2
REPRESENTATIVES_PER_CLUSTER = 5

data = np.load(EMBEDDINGS_FILE)

embeddings = data["embeddings"]
face_ids = data["face_ids"]

with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))

print("Loaded embeddings:", len(embeddings))

model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="cosine",
    n_jobs=-1
)

labels = model.fit_predict(embeddings)

clusters = {}

for index, label in enumerate(labels):
    if label == -1:
        continue

    clusters.setdefault(int(label), []).append(index)

audit = []
representatives = []

for label, indices in clusters.items():

    vectors = embeddings[indices]

    centroid = vectors.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    similarities = vectors @ centroid

    scene_ids = set()

    for index in indices:
        scene = manifest[index]["scene"]
        scene_id = int(
            scene.replace("scene_", "").replace(".jpg", "")
        )
        scene_ids.add(scene_id)

    sorted_indices = np.argsort(similarities)[::-1]

    character_id = f"CHAR_{label + 1:04d}"

    audit.append({
        "character_id": character_id,
        "face_count": len(indices),
        "scene_count": len(scene_ids),
        "first_scene": min(scene_ids),
        "last_scene": max(scene_ids),
        "mean_similarity": round(float(np.mean(similarities)), 4),
        "p10_similarity": round(float(np.percentile(similarities, 10)), 4),
        "min_similarity": round(float(np.min(similarities)), 4),
        "max_similarity": round(float(np.max(similarities)), 4)
    })

    for rank, local_index in enumerate(
        sorted_indices[:REPRESENTATIVES_PER_CLUSTER],
        1
    ):
        index = indices[local_index]
        record = manifest[index]

        representatives.append({
            "character_id": character_id,
            "rank": rank,
            "face_id": record["face_id"],
            "scene": record["scene"],
            "face_crop": record["face_crop"],
            "centroid_similarity": round(
                float(similarities[local_index]), 4
            )
        })

audit.sort(
    key=lambda x: x["face_count"],
    reverse=True
)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=audit[0].keys())
    writer.writeheader()
    writer.writerows(audit)

with open(REPRESENTATIVES_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=representatives[0].keys()
    )
    writer.writeheader()
    writer.writerows(representatives)

print("\nCluster audit completed.")
print("Clusters:", len(clusters))
print("Unassigned faces:", int(np.sum(labels == -1)))

print("\nLargest clusters:")

for item in audit[:20]:
    print(
        item["character_id"],
        "| Faces:", item["face_count"],
        "| Mean similarity:", item["mean_similarity"],
        "| P10 similarity:", item["p10_similarity"],
        "| Scenes:", item["scene_count"]
    )

print("\nSaved:", OUTPUT_CSV)
print("Saved:", REPRESENTATIVES_CSV)
