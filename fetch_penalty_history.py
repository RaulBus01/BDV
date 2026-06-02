import asyncio
import json
import csv
from pathlib import Path
from curl_cffi.requests import AsyncSession

# Configuration
API_BASE_URL = "https://www.sofascore.com/api/v1/player"
CSV_FILE = "./FinalData/sofascore_player_idsNew.csv"
OUTPUT_DIR = "./FinalData/penalty_data"
DELAY_BETWEEN_REQUESTS = 1  # seconds, to avoid rate limiting

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)
OUTPUT_CSV = Path(OUTPUT_DIR) / "all_penalty_historyNew.csv"

# Define flat headers for our structured CSV output
CSV_HEADERS = [
    "sofascore_id", "player_name", "original_name", "position", "club",
    "total_scored", "total_attempts", "event_id", "outcome", "zone", 
    "penalty_id", "x", "y"
]

async def fetch_penalty_history(sofascore_id, player_name, session):
    """Fetch penalty history for a single player using curl_cffi"""
    try:
        url = f"{API_BASE_URL}/{int(sofascore_id)}/penalty-history/"
        print(f"Fetching penalty history for {player_name} (ID: {sofascore_id})...")
        
        response = await session.get(url, impersonate="chrome")
        
        if response.status_code == 403:
            print(f"  🛑 403 Forbidden on ID {sofascore_id}. WAF challenge triggered.")
            return None
            
        data = response.json()
        
        # Check if there's penalty data
        if data and ('penalties' in data or 'scored' in data or 'attempts' in data):
            return data
        else:
            print(f"  No penalty data found")
            return None
            
    except Exception as e:
        print(f"  Error fetching data for {player_name}: {e}")
        return None

def extract_penalty_info(penalty):
    """Extract relevant info from a penalty record"""
    try:
        return {
            "event_id": penalty.get("event", {}).get("id"),
            "outcome": penalty.get("outcome"),
            "zone": penalty.get("zone"),
            "id": penalty.get("id"),
            "x": penalty.get("x"),
            "y": penalty.get("y")
        }
    except Exception as e:
        print(f"  Error extracting penalty info: {e}")
        return None

def main():
    """Main function to fetch all penalty histories"""
    asyncio.run(fetch_all_players())

async def fetch_all_players():
    """Scrape penalty history for all players using curl_cffi and save to CSV"""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            players = list(reader)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found")
        return
    
    print(f"Found {len(players)} players in {CSV_FILE}\n")
    
    # Initialize the CSV file and write headers
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        
    successful = 0
    failed = 0
    total = len(players)
    
    async with AsyncSession() as session:
        for idx, player in enumerate(players, 1):
            sofascore_id = player.get("Sofascore_ID")
            player_name = player.get("Sofascore_Name", player.get("Original_Name", "Unknown"))
            
            if not sofascore_id:
                print(f"[{idx}/{total}] Skipping player with no ID: {player_name}")
                continue
            
            print(f"[{idx}/{total}] Processing {player_name}...")
            
            # Fetch penalty history
            penalty_data = await fetch_penalty_history(sofascore_id, player_name, session)
            
            if penalty_data:
                scored = penalty_data.get("scored", 0)
                attempts = penalty_data.get("attempts", 0)
                penalties = penalty_data.get("penalties", [])
                
                # Base structural row metadata
                base_row = {
                    "sofascore_id": sofascore_id,
                    "player_name": player_name,
                    "original_name": player.get("Original_Name"),
                    "position": player.get("Position"),
                    "club": player.get("Sofascore_Club"),
                    "total_scored": scored,
                    "total_attempts": attempts
                }
                
                # Re-open our CSV in append mode ('a') to drop data straight onto disk
                with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    
                    if penalties:
                        # If individual shot event details exist, break them out into separate explicit rows
                        for penalty in penalties:
                            p_info = extract_penalty_info(penalty)
                            if p_info:
                                row = {**base_row, **{
                                    "event_id": p_info["event_id"],
                                    "outcome": p_info["outcome"],
                                    "zone": p_info["zone"],
                                    "penalty_id": p_info["id"],
                                    "x": p_info["x"],
                                    "y": p_info["y"]
                                }}
                                writer.writerow(row)
                    else:
                        # Write a fallback row containing total stats even if specific match coordinates are empty
                        writer.writerow(base_row)
                
                print(f"  ✓ Processed metrics (Scored: {scored}, Attempts: {attempts}) Saved to disk.")
                successful += 1
            else:
                failed += 1
            
            # Delay to avoid rate limiting / 403 triggers
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total players processed: {total}")
    print(f"  Successful updates: {successful}")
    print(f"  Failed/No data: {failed}")
    print(f"  Flat dataset output saved directly to: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()