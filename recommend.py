import numpy as np
from encoding import load_games, encode_game, build_all_positions, find_game_by_id

games_raw = load_games()
positions = build_all_positions(games_raw)

def recommend(game_id, games_raw, positions, top_n=5):

    input_game = find_game_by_id(game_id, games_raw)
    a = encode_game(input_game, positions)  

    scores = []
    for current_game in games_raw:
        b = encode_game(current_game, positions)  
        
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = sum(x**2 for x in a) ** 0.5
        mag_b = sum(x**2 for x in b) ** 0.5
        similarity = dot / (mag_a * mag_b)
        
        scores.append((current_game, similarity))  

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0]["id"] != game_id]  
    
    return [(s[0]["name"], s[1]) for s in scores[:top_n]]
