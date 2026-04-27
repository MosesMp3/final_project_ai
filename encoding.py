import json

def load_games():
    with open("cache/games_raw.json") as f:
        return json.load(f)

def load_lookups():
    with open("cache/lookups.json") as f:
        return json.load(f)

def build_positions(games, field_name):
    unique_ids = set()
    for game in games:
        for id_ in game.get(field_name) or []:
            unique_ids.add(id_)
    sorted_ids = sorted(unique_ids)
    return {id_: i for i, id_ in enumerate(sorted_ids)}

def build_all_positions(games):
    fields = ["genres", "themes", "game_modes", "player_perspectives", "platforms"]
    return {field: build_positions(games, field) for field in fields}

def encode_field(game_field_values, vocab_positions):
    vec = [0] * len(vocab_positions)
    for val in game_field_values or []:
        if val in vocab_positions:
            vec[vocab_positions[val]] = 1
    return vec

def encode_game(game, positions):
    parts = []
    for field in ["genres", "themes", "game_modes", "player_perspectives", "platforms"]:
        parts.append(encode_field(game.get(field), positions[field]))
    return sum(parts, [])

def find_game_by_id(game_id, games):
    for game in games:
        if game["id"] == game_id:
            return game
    return None