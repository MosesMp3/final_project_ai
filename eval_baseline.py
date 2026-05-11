import json
import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_eval_data():
    data = np.load("cache/features.npz", allow_pickle=True)
    raw_vectors = data["feature_matrix"].astype("float32")
    game_ids = data["game_ids"]
    game_names = data["game_names"]

    with open("cache/similar_truth.json") as f:
        truth_raw = json.load(f)
    truth = {int(k): set(v) for k, v in truth_raw.items()}

    return raw_vectors, game_ids, game_names, truth


def recall_at_k(vectors, game_ids, truth, k=10):
    catalog_ids = set(int(gid) for gid in game_ids)
    id_to_idx = {int(gid): i for i, gid in enumerate(game_ids)}

    nn_index = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_index.fit(vectors)

    recalls = []
    skipped = 0
    for gid in game_ids:
        gid = int(gid)
        relevant = truth.get(gid, set()) & catalog_ids
        if not relevant:
            skipped += 1
            continue

        idx = id_to_idx[gid]
        _, indices = nn_index.kneighbors(vectors[idx : idx + 1], n_neighbors=k + 1)
        recommended = {int(game_ids[i]) for i in indices[0][1:]}

        hits = len(recommended & relevant)
        recalls.append(hits / len(relevant))

    return {
        "recall_at_k": float(np.mean(recalls)),
        "k": k,
        "n_evaluated": len(recalls),
        "n_skipped": skipped,
    }


vectors, game_ids, game_names, truth = load_eval_data()

print("=" * 60)
print("BASELINE: Recall@K on raw one-hot vectors (IGDB ground truth)")
print("=" * 60)
for k in [5, 10, 20]:
    result = recall_at_k(vectors, game_ids, truth, k=k)
    print(
        f"  Recall@{k:2d}: {result['recall_at_k']:.4f}  "
        f"({result['n_evaluated']} evaluated, {result['n_skipped']} skipped)"
    )
