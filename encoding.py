import json

with open("cache/games_raw.json") as f:
    games_raw = json.load(f)

with open("cache/lookups.json") as d:
    lookups = json.load(d)
print(len(games_raw))

# have todo for each, will make into functions
unique_mode_ids = set()
for game in games_raw:
    for mode_id in game.get("game_modes") or []:
        unique_mode_ids.add(mode_id)
game_modes_vocab = sorted(unique_mode_ids)
game_modes_positions = {id_: i for i, id_ in enumerate(game_modes_vocab)}


def encode_field(game_field_values, vocab_positions):
    vec = [0] * len(vocab_positions)
    for val in game_field_values or []:
        if val in vocab_positions:
            vec[vocab_positions[val]] = 1
    return vec


test_game = games_raw[0]
print(f"Testing game: {test_game['name']}")
print(f"Its game_modes: {test_game.get('game_modes')}")

encoded = encode_field(test_game.get("game_modes"), game_modes_positions)
print(f"Encoded: {encoded}")
