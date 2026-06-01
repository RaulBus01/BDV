import asyncio
import json
import csv
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
API_BASE_URL = "https://www.sofascore.com/api/v1/player"
CSV_FILE = "sofascore_player_ids.csv"
OUTPUT_DIR = "penalty_data"
DELAY_BETWEEN_REQUESTS = 0.5  # seconds, to avoid rate limiting

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

async def fetch_penalty_history(sofascore_id, player_name, page):
    """Fetch penalty history for a single player using Playwright"""
    try:
        url = f"{API_BASE_URL}/{int(sofascore_id)}/penalty-history/"
        print(f"Fetching penalty history for {player_name} (ID: {sofascore_id})...")
        
        response = await page.goto(url, wait_until='domcontentloaded', timeout=10000)
        
        # Get the response text
        text = await response.text()
        
        # Parse JSON from the response
        data = json.loads(text)
        
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
    """Scrape penalty history for all players using Playwright"""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            players = list(reader)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found")
        return
    
    print(f"Found {len(players)} players in {CSV_FILE}\n")
    
    all_results = {}
    successful = 0
    failed = 0
    
    async with async_playwright() as p:
        # Launch browser with Chromium
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        total = len(players)
        for idx, player in enumerate(players, 1):
            sofascore_id = player.get("Sofascore_ID")
            player_name = player.get("Sofascore_Name", player.get("Original_Name", "Unknown"))
            
            if not sofascore_id:
                print(f"[{idx}/{total}] Skipping player with no ID: {player_name}")
                continue
            
            print(f"[{idx}/{total}] Processing {player_name}...")
            
            # Fetch penalty history
            penalty_data = await fetch_penalty_history(sofascore_id, player_name, page)
            
            if penalty_data:
                # Extract penalties info if available
                penalties_list = []
                if "penalties" in penalty_data:
                    for penalty in penalty_data.get("penalties", []):
                        penalty_info = extract_penalty_info(penalty)
                        if penalty_info:
                            penalties_list.append(penalty_info)
                
                # Store the result
                all_results[str(sofascore_id)] = {
                    "player_name": player_name,
                    "original_name": player.get("Original_Name"),
                    "position": player.get("Position"),
                    "club": player.get("Sofascore_Club"),
                    "scored": penalty_data.get("scored", 0),
                    "attempts": penalty_data.get("attempts", 0),
                    "penalties": penalties_list
                }
                
                print(f"  ✓ Found {len(penalties_list)} penalties (Scored: {penalty_data.get('scored', 0)}, Attempts: {penalty_data.get('attempts', 0)})")
                successful += 1
            else:
                failed += 1
            
            # Delay to avoid rate limiting
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        
        await browser.close()
    
    # Save all results to a single JSON file
    output_file = Path(OUTPUT_DIR) / "all_penalty_history.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total players processed: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed/No data: {failed}")
    print(f"  Output saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
