import json
import csv
import pandas as pd
from pathlib import Path

# Configuration
JSON_FILE = "penalty_data/all_penalty_history.json"
CSV_PLAYER_IDS = "sofascore_player_ids.csv"
OUTPUT_CSV = "combined_penalty_data.csv"

def main():
    # Load player lookup from CSV
    print(f"Loading player info from {CSV_PLAYER_IDS}...")
    try:
        # We use pandas to handle potential issues with IDs as floats/strings
        df_players = pd.read_csv(CSV_PLAYER_IDS)
        # Convert Sofascore_ID to string for consistent lookup
        df_players['Sofascore_ID'] = df_players['Sofascore_ID'].astype(str).str.replace('.0', '', regex=False)
        # Handle duplicates by keeping the first entry for each ID
        df_players = df_players.drop_duplicates(subset=['Sofascore_ID'])
        player_lookup = df_players.set_index('Sofascore_ID').to_dict('index')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Load penalty history from JSON
    print(f"Loading penalty history from {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            penalty_history = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # Process and flatten data
    rows = []
    for sofascore_id, player_data in penalty_history.items():
        # Get extra info from CSV lookup if available
        # Strip .0 if it exists from JSON key just in case
        clean_id = str(sofascore_id).replace('.0', '')
        info = player_lookup.get(clean_id, {})
        
        name = info.get('Original_Name', player_data.get('original_name', player_data.get('player_name')))
        club = info.get('Club', player_data.get('club'))
        nationality = info.get('Nationality', 'Unknown')
        
        scored = player_data.get('scored', 0)
        attempts = player_data.get('attempts', 0)
        
        # Create a row for each penalty
        penalties = player_data.get('penalties', [])
        
        if not penalties:
            # Optional: Add a row for players with no penalty details but have totals
            # rows.append({ ... })
            continue
            
        for p in penalties:
            rows.append({
                'Name': name,
                'Club': club,
                'Nationality': nationality,
                'Sofascore_ID': clean_id,
                'event_id': p.get('event_id'),
                'penalty_id': p.get('id'),
                'outcome': p.get('outcome'),
                'zone': p.get('zone'),
                'x': p.get('x'),
                'y': p.get('y'),
                'player_total_scored': scored,
                'player_total_attempts': attempts
            })

    # Write to CSV
    if rows:
        print(f"Saving {len(rows)} penalties to {OUTPUT_CSV}...")
        fieldnames = ['Name', 'Club', 'Nationality', 'Sofascore_ID', 'event_id', 'penalty_id', 'outcome', 'zone', 'x', 'y', 'player_total_scored', 'player_total_attempts']
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("✓ Done!")
    else:
        print("No penalty data found to save.")

if __name__ == "__main__":
    main()
