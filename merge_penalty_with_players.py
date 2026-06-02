import csv
from pathlib import Path

PENALTY_CSV = "./FinalData/penalty_data/all_penalty_historyNew.csv"
PLAYER_CSV = "./FinalData/sofascore_player_idsNew.csv"
OUTPUT_CSV = "./FinalData/penalty_data/combined_penalty_with_players.csv"

# Load player lookup from IDs CSV
player_lookup = {}
with open(PLAYER_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row.get("Sofascore_ID", "").strip()
        if sid:
            player_lookup[sid] = row

# Read penalty CSV and merge
merged_rows = []
with open(PENALTY_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    penalty_fieldnames = [f for f in fieldnames if f != 'sofascore_id']

    # Extra columns to add from player lookup
    extra_cols = ["Nationality", "Caps", "Goals", "Group", "Is_Captain", "Match_Score"]

    output_fieldnames = fieldnames + extra_cols

    for row in reader:
        sid = row.get("sofascore_id", "").strip()
        info = player_lookup.get(sid, {})
        for col in extra_cols:
            row[col] = info.get(col, "")
        merged_rows.append(row)

# Write merged CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=output_fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)

print(f"Merged {len(merged_rows)} rows → {OUTPUT_CSV}")
