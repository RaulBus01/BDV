import csv
import json

CSV_PATH  = "FinalData/penalty_data/combined_penalty_with_players_enriched.csv"
JSON_PATH = "FinalData/penalty_data/all_penalty_history_shotmap.json"
OUT_PATH  = "FinalData/penalty_data/combined_full_with_shotmap.csv"

with open(JSON_PATH, encoding="utf-8") as f:
    enriched_data = json.load(f)

# Build lookup: (player_name, event_id, penalty_id) -> shotmap-enriched dict
lookup = {}
for sid, info in enriched_data.items():
    pname = info.get("player_name", "")
    for pen in info.get("penalties", []):
        key = (pname, pen.get("event_id"), pen.get("id"))
        if "enriched" in pen:
            lookup[key] = pen["enriched"]

print(f"Built lookup with {len(lookup)} shotmap-enriched entries")

# Read CSV and merge
rows_in = 0
rows_out = 0
matched = 0

# Discover shotmap fields, prefix with "shotmap_", exclude internal _strength
sample = next(iter(lookup.values()), {})
shotmap_fields = sorted(f for f in sample if not f.startswith("_"))
shotmap_cols = [f"shotmap_{f}" for f in shotmap_fields]
print(f"Shotmap columns to add ({len(shotmap_cols)}): {shotmap_cols[:5]}...")

with open(CSV_PATH, encoding="utf-8", newline="") as fin, \
     open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    original_header = reader.fieldnames
    new_header = list(original_header) + shotmap_cols
    writer.writerow(new_header)

    for row in reader:
        rows_in += 1
        key = (row.get("player_name", ""),
               int(row["event_id"]),
               int(row["penalty_id"]))

        enriched = lookup.get(key, {})
        if enriched:
            matched += 1

        new_row = [row.get(h, "") for h in original_header]
        for field in shotmap_fields:
            val = enriched.get(field, "")
            if val is None:
                val = ""
            new_row.append(val)
        writer.writerow(new_row)
        rows_out += 1

print(f"Rows read     : {rows_in}")
print(f"Rows written  : {rows_out}")
print(f"Matched       : {matched}")
print(f"Saved to      : {OUT_PATH}")
