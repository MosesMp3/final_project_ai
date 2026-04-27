import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_index(embeddings, n_neighbors=11):
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_index.fit(embeddings)
    return nn_index


def recommend(game_id, nn_index, embeddings, game_ids, game_names, id_to_idx, k=10):
    if game_id not in id_to_idx:
        raise ValueError(f"Game ID {game_id} not in catalog")

    idx = id_to_idx[game_id]
    distances, indices = nn_index.kneighbors(
        embeddings[idx : idx + 1],
        n_neighbors=k + 1,
    )

    results = []
    for dist, i in zip(distances[0][1:], indices[0][1:]):
        results.append(
            {
                "id": int(game_ids[i]),
                "name": str(game_names[i]),
                "similarity": float(1 - dist),
            }
        )
    return results


data = np.load("cache/embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
game_ids = data["game_ids"]
game_names = data["game_names"]

id_to_idx = {int(gid): i for i, gid in enumerate(game_ids)}
nn_index = build_index(embeddings)

test_ids = [
    125764,  # Guilty Gear Strive
    119133,  # Elden Ring
    14593,  # Hollow Knight
    17000,  # Stardew Valley
    125174,  # Overwatch
    115,  # League of Legends
    331608,  # Knightica (roguelike)
    135915,  # Overcooked
    132181,  # Resident Evil 4
    119277,  # Genshin Impact
    339698,  # dead by daylight
]

for tid in test_ids:
    if tid not in id_to_idx:
        print(f"\n{tid}: not in catalog")
        continue
    name = game_names[id_to_idx[tid]]
    print(f"\nSimilar to {name}:")
    for rec in recommend(
        tid, nn_index, embeddings, game_ids, game_names, id_to_idx, k=5
    ):
        print(f"  {rec['similarity']:.3f}  {rec['name']}")
