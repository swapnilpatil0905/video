import csv
import numpy as np

from sklearn.cluster import DBSCAN
from collections import Counter

INPUT = "faces/face_embeddings.npz"
OUTPUT = "faces/clustering_evaluation.csv"

data = np.load(INPUT)
embeddings = data["embeddings"]

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

results = []

print("Evaluating clustering thresholds...\n")

for eps in thresholds:

    model = DBSCAN(
        eps=eps,
        min_samples=2,
        metric="cosine",
        n_jobs=-1
    )

    labels = model.fit_predict(embeddings)

    counts = Counter(labels)

    noise = counts.pop(-1, 0)

    cluster_sizes = sorted(
        counts.values(),
        reverse=True
    )

    result = {
        "eps": eps,
        "clusters": len(cluster_sizes),
        "noise": noise,
        "assigned": len(embeddings) - noise,
        "largest_cluster": cluster_sizes[0] if cluster_sizes else 0,
        "second_largest": cluster_sizes[1] if len(cluster_sizes) > 1 else 0,
        "clusters_with_10_plus_faces": sum(
            size >= 10 for size in cluster_sizes
        ),
        "clusters_with_5_or_fewer_faces": sum(
            size <= 5 for size in cluster_sizes
        )
    }

    results.append(result)

    print(
        f"EPS={eps:.2f} | "
        f"Clusters={result['clusters']} | "
        f"Noise={noise} | "
        f"Assigned={result['assigned']} | "
        f"Largest={result['largest_cluster']} | "
        f"10+ faces={result['clusters_with_10_plus_faces']}"
    )

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", OUTPUT)
