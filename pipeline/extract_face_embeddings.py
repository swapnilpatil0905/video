import csv
import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis

INPUT_CSV = "faces/face_manifest.csv"
OUTPUT_NPZ = "faces/face_embeddings.npz"
OUTPUT_CSV = "faces/embedding_manifest.csv"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

face_ids = []
embeddings = []
manifest = []

for index, record in enumerate(records, 1):

    image = cv2.imread(record["face_crop"])

    if image is None:
        continue

    detected = app.get(image)

    if not detected:
        continue

    # Select the largest detected face.
    face = max(
        detected,
        key=lambda x: (
            x.bbox[2] - x.bbox[0]
        ) * (
            x.bbox[3] - x.bbox[1]
        )
    )

    embedding = face.normed_embedding.astype(np.float32)

    face_ids.append(int(record["face_id"]))
    embeddings.append(embedding)

    manifest.append({
        "face_id": record["face_id"],
        "scene": record["scene"],
        "face_crop": record["face_crop"],
        "embedding_index": len(embeddings) - 1
    })

    if index % 100 == 0:
        print(
            f"Processed {index}/{len(records)} "
            f"| Embeddings: {len(embeddings)}"
        )

if not embeddings:
    raise RuntimeError("No face embeddings were generated.")

np.savez_compressed(
    OUTPUT_NPZ,
    face_ids=np.array(face_ids),
    embeddings=np.stack(embeddings)
)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "face_id",
            "scene",
            "face_crop",
            "embedding_index"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nEmbedding extraction completed.")
print("Input face crops:", len(records))
print("Embeddings generated:", len(embeddings))
print("Embedding dimensions:", embeddings[0].shape)
print("Saved:", OUTPUT_NPZ)
print("Manifest:", OUTPUT_CSV)
