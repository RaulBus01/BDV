import csv
import json
import sys

CSV_PATH  = "FinalData/penalty_data/combined_penalty_with_players.csv"
JSON_PATH = "FinalData/penalty_data/all_penalty_history_enriched.json"
OUT_PATH  = "FinalData/penalty_data/combined_penalty_with_players_enriched.csv"

# Load enriched JSON
with open(JSON_PATH, encoding="utf-8") as f:
    enriched_data = json.load(f)

# Build lookup dict: (sofascore_id, event_id, penalty_id) -> enriched dict
lookup = {}
for sid, info in enriched_data.items():
    for pen in info.get("penalties", []):
        key = (int(sid), pen.get("event_id"), pen.get("id"))
        if "enriched" in pen:
            lookup[key] = pen["enriched"]

print(f"Built lookup with {len(lookup)} enriched entries")

# Read CSV and merge
rows_in = 0
rows_out = 0
matched = 0

# First pass: discover all enriched field names
enriched_field_names = set()
for v in lookup.values():
    enriched_field_names.update(v.keys())
# Sort and remove internal _strength
enriched_fields = sorted(f for f in enriched_field_names if not f.startswith("_"))
print(f"Enriched columns to add: {enriched_fields}")

with open(CSV_PATH, encoding="utf-8", newline="") as fin, \
     open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    # Header
    original_header = reader.fieldnames
    # Only add enriched fields that don't already exist in CSV
    enriched_fields = [f for f in enriched_fields if f not in original_header]
    new_header = list(original_header) + enriched_fields
    writer.writerow(new_header)

    for row in reader:
        rows_in += 1
        sid = int(row["sofascore_id"])
        eid = int(row["event_id"])
        pid = int(row["penalty_id"])
        key = (sid, eid, pid)

        enriched = lookup.get(key, {})
        if enriched:
            matched += 1

        new_row = [row.get(h, "") for h in original_header]
        for field in enriched_fields:
            val = enriched.get(field, "")
            # Convert None to empty string
            if val is None:
                val = ""
            new_row.append(val)
        writer.writerow(new_row)
        rows_out += 1

print(f"Rows read   : {rows_in}")
print(f"Rows written: {rows_out}")
print(f"Matched     : {matched}")
print(f"Saved to    : {OUT_PATH}")
