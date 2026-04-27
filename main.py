from encoding import load_games, build_all_positions, find_game_by_id
from recommend import recommend

def main():
    # load data
    games_raw = load_games()
    positions = build_all_positions(games_raw)

    # build a searchable list of game names
    game_names = [game["name"] for game in games_raw]

    print("=== Game Recommender ===")
    print(f"Database: {len(games_raw)} games loaded\n")

    while True:
        user_input = input("Enter a game name (or 'quit' to exit): ").strip()
        
        if user_input.lower() == "quit":
            break

        # find the game by name
        match = None
        for game in games_raw:
            if game["name"].lower() == user_input.lower():
                match = game
                break

        if match is None:
            print(f"Game '{user_input}' not found. Try another name.\n")
            continue

        # get recommendations
        results = recommend(match["id"], games_raw, positions)

        print(f"\nBecause you liked {match['name']}, you might also enjoy:")
        for i, (name, score) in enumerate(results, 1):
            print(f"  {i}. {name} (similarity: {score:.3f})")
        print()

if __name__ == "__main__":
    main()