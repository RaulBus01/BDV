import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import json
import time

# Read the CSV file
csv_path = "world_cup_squads.csv"
df = pd.read_csv(csv_path)

# Keep only relevant columns
df = df[['Player', 'Club', 'Team']].drop_duplicates().dropna()

print(f"Loaded {len(df)} unique players to search")

async def fetch_player_sofascore_id(player_name, club, nationality, page):
    """
    Fetch SofaScore ID for a player using Playwright.
    Matches by player name, club, and nationality.
    """
    try:
        # Construct the search URL
        search_url = f"https://api.sofascore.com/api/v1/search/all?q={player_name}"
        
        # Navigate to the API endpoint
        response = await page.goto(search_url, wait_until='domcontentloaded', timeout=10000)
        
        # Get the response text directly
        text = await response.text()
        
        # Parse JSON from the response
        data = json.loads(text)
        search_results = data.get('results', [])
        
        # Filter results: match player name, club, and team/nationality
        for item in search_results:
            if item.get('type') == 'player':
                entity = item.get('entity', {})
                sofascore_name = entity.get('name', '').lower()
                sofascore_club = entity.get('team', {}).get('name', '').lower()
                
                # Fuzzy matching criteria
                name_match = player_name.lower() in sofascore_name or sofascore_name in player_name.lower()
                club_match = club.lower() in sofascore_club or sofascore_club in club.lower()
                
                if name_match and club_match:
                    print(f"Exact match for {player_name}: Found {entity.get('name')} at {entity.get('team', {}).get('name')}")
                    return {
                        'Original_Name': player_name,
                        'Club': club,
                        'Nationality': nationality,
                        'Sofascore_ID': entity.get('id'),
                        'Sofascore_Name': entity.get('name'),
                        'Sofascore_Club': entity.get('team', {}).get('name'),
                        'Position': entity.get('position'),
                        'Match_Score': 'High' if player_name.lower() == sofascore_name else 'Fuzzy'
                    }
        
        # If no exact match found, try to get the first player result as fallback
        for item in search_results:
            if item.get('type') == 'player':
                entity = item.get('entity', {})
                print(f"Fallback match for {player_name}: Found {entity.get('name')} at {entity.get('team', {}).get('name')}")
                return {
                    'Original_Name': player_name,
                    'Club': club,
                    'Nationality': nationality,
                    'Sofascore_ID': entity.get('id'),
                    'Sofascore_Name': entity.get('name'),
                    'Sofascore_Club': entity.get('team', {}).get('name'),
                    'Position': entity.get('position'),
                    'Match_Score': 'Low - Fallback'
                }
        
        print(f"No match found for {player_name} ({club}, {nationality})")
        return {
            'Original_Name': player_name,
            'Club': club,
            'Nationality': nationality,
            'Sofascore_ID': None,
            'Sofascore_Name': None,
            'Sofascore_Club': None,
            'Position': None,
            'Match_Score': 'No Match'
        }
        
    except Exception as e:
        print(f"Error searching for {player_name}: {e}")
        return {
            'Original_Name': player_name,
            'Club': club,
            'Nationality': nationality,
            'Sofascore_ID': None,
            'Sofascore_Name': None,
            'Sofascore_Club': None,
            'Position': None,
            'Match_Score': f'Error: {str(e)}'
        }

async def scrape_all_players():
    """
    Scrape SofaScore data for all players using Playwright.
    """
    results = []
    
    async with async_playwright() as p:
        # Launch browser with Chromium
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        total = len(df)
        for idx, row in df.iterrows():
            player_name = row['Player']
            club = row['Club']
            nationality = row['Team']
            
            print(f"[{idx+1}/{total}] Searching for {player_name} ({club}, {nationality})...")
            
            result = await fetch_player_sofascore_id(player_name, club, nationality, page)
            results.append(result)
            
            # Add delay between requests to avoid rate limiting
            await asyncio.sleep(0.5)
        
        await browser.close()
    
    return results

async def main():
    """
    Main function to run the scraper and save results.
    """
    print("Starting SofaScore player scraper with Playwright...\n")
    
    results = await scrape_all_players()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = "sofascore_player_ids.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Scraping complete!")
    print(f"✓ Found SofaScore IDs for {results_df['Sofascore_ID'].notna().sum()} players")
    print(f"✓ Results saved to {output_path}")
    
    # Show summary
    print("\nMatch Score Summary:")
    print(results_df['Match_Score'].value_counts())
    
    return results_df

if __name__ == "__main__":
    results_df = asyncio.run(main())
    print("\nFirst 5 results:")
    print(results_df.head())
