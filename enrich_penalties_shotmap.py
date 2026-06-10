"""
Enrich all_penalty_history.json using Sofascore shotmap endpoint.

Matching strategy (pen["id"] does NOT equal the shotmap shot id):
  1. player_id   = sofascore_id (int key of penalty_data)
  2. zone        = goalMouthLocation  (e.g. "low-left", "high-right")
  3. outcome     = shotType           ("goal"/"save"/"miss"/"post")

The combination of these three fields is unique for >99% of cases.
For ties we additionally use shootoutOrder (if available) or just pick
the first remaining candidate.
"""

import asyncio
import json
import os
from curl_cffi.requests import AsyncSession

JSON_FILE   = "./FinalData/penalty_data/all_penalty_history.json"
OUTPUT_FILE = "./FinalData/penalty_data/all_penalty_history_enriched.json"
CACHE_DIR   = "./FinalData/penalty_data/shotmap_cache"
API_BASE    = "https://api.sofascore.com/api/v1/event"
DELAY       = 0.5

os.makedirs(CACHE_DIR, exist_ok=True)

# pen["outcome"] → shotmap shotType
OUTCOME_TO_SHOTTYPE = {
    "goal": "goal",
    "save": "save",
    "miss": "miss",
    "post": "post",
}

# ---------------------------------------------------------------------------
# Fetch (with disk cache)
# ---------------------------------------------------------------------------

async def fetch_shotmap(event_id, session):
    cache_path = os.path.join(CACHE_DIR, f"{event_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    url = f"{API_BASE}/{event_id}/shotmap"
    for attempt in range(3):
        try:
            r = await session.get(url, impersonate="chrome")
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                return data
            elif r.status_code == 429:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                return None
        except Exception:
            await asyncio.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Parse — keyed by player_id (list), only penalty shots
# ---------------------------------------------------------------------------

def extract_penalty_shots(raw):
    """
    Returns dict:  player_id (int) -> list of shot-entry dicts.

    Includes only shots that are penalties:
      - situation == "shootout"
      - goalType  == "penalty"
      - situation == "penalty"   (some regular-match penalties)
    """
    by_player = {}
    if not raw or "shotmap" not in raw:
        return by_player

    for shot in raw["shotmap"]:
        situation = shot.get("situation", "")
        goal_type = shot.get("goalType", "")
        is_penalty = (
            situation == "shootout"
            or goal_type == "penalty"
            or situation == "penalty"
        )
        if not is_penalty:
            continue

        player = shot.get("player") or {}
        pid    = player.get("id")
        if not pid:
            continue

        gk    = shot.get("goalkeeper") or {}
        gmc   = shot.get("goalMouthCoordinates") or {}
        pc    = shot.get("playerCoordinates") or {}
        bc    = shot.get("blockCoordinates") or {}
        draw  = shot.get("draw") or {}

        entry = {
            # classification
            "shot_type":           shot.get("shotType"),        # goal/save/miss/post/block
            "goal_type":           goal_type,                   # penalty/regular/""
            "situation":           situation,                   # shootout/penalty/...
            "body_part":           shot.get("bodyPart"),        # left-foot/right-foot/head
            "shootout_order":      shot.get("shootoutOrder"),   # 1-5 (shootout only)
            # goal mouth
            "goal_mouth_location": shot.get("goalMouthLocation"),# low-right / high-left / ...
            "goal_mouth_x":        gmc.get("x"),
            "goal_mouth_y":        gmc.get("y"),
            "goal_mouth_z":        gmc.get("z"),
            # player start
            "player_coord_x":      pc.get("x"),
            "player_coord_y":      pc.get("y"),
            # draw path
            "draw_start_x":        (draw.get("start") or {}).get("x"),
            "draw_start_y":        (draw.get("start") or {}).get("y"),
            "draw_end_x":          (draw.get("end") or {}).get("x"),
            "draw_end_y":          (draw.get("end") or {}).get("y"),
            "draw_goal_x":         (draw.get("goal") or {}).get("x"),
            "draw_goal_y":         (draw.get("goal") or {}).get("y"),
            "draw_block_x":        (draw.get("block") or {}).get("x"),
            "draw_block_y":        (draw.get("block") or {}).get("y"),
            # block coords (saves/blocks)
            "block_coord_x":       bc.get("x"),
            "block_coord_y":       bc.get("y"),
            "block_coord_z":       bc.get("z"),
            # goalkeeper
            "gk_id":               gk.get("id"),
            "gk_name":             gk.get("name"),
            # identity / timing
            "player_id":           pid,
            "player_name":         player.get("name"),
            "is_home":             shot.get("isHome"),
            "time":                shot.get("time"),
            "time_seconds":        shot.get("timeSeconds"),
        }

        by_player.setdefault(pid, []).append(entry)

    return by_player


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def match_shot(pen, sofascore_id, shots_by_player):
    """
    Find the best-matching shotmap entry for a penalty record.

    Priority:
      1. player_id filter (hard requirement)
      2. zone == goalMouthLocation  +  outcome == shotType  (primary match)
      3. outcome-only fallback
      4. single-candidate fallback
    """
    try:
        pid = int(sofascore_id)
    except (TypeError, ValueError):
        return None

    candidates = shots_by_player.get(pid)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    pen_zone    = pen.get("zone", "")          # e.g. "low-right"
    pen_outcome = pen.get("outcome", "")       # e.g. "goal" / "save"
    shot_type   = OUTCOME_TO_SHOTTYPE.get(pen_outcome, pen_outcome)

    # ── zone + outcome ──────────────────────────────────────────────────────
    zone_outcome = [
        c for c in candidates
        if c.get("goal_mouth_location") == pen_zone
        and c.get("shot_type") == shot_type
    ]
    if len(zone_outcome) == 1:
        return zone_outcome[0]
    if len(zone_outcome) > 1:
        # Rare tie (same player, same zone, same outcome twice in one game).
        # Use shootout_order as tiebreaker if present.
        with_order = [z for z in zone_outcome if z.get("shootout_order") is not None]
        if with_order:
            return with_order[0]          # any order is better than nothing
        return zone_outcome[0]

    # ── outcome-only fallback ───────────────────────────────────────────────
    by_outcome = [c for c in candidates if c.get("shot_type") == shot_type]
    if len(by_outcome) == 1:
        return by_outcome[0]

    # ── zone-only fallback ──────────────────────────────────────────────────
    by_zone = [c for c in candidates if c.get("goal_mouth_location") == pen_zone]
    if len(by_zone) == 1:
        return by_zone[0]

    return None   # genuinely ambiguous — skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    with open(JSON_FILE, encoding="utf-8") as f:
        penalty_data = json.load(f)

    event_ids = set()
    for player_info in penalty_data.values():
        for pen in player_info.get("penalties", []):
            eid = pen.get("event_id")
            if eid:
                event_ids.add(eid)

    cached = sum(
        1 for e in event_ids
        if os.path.exists(os.path.join(CACHE_DIR, f"{e}.json"))
    )
    print(f"Total players    : {len(penalty_data)}")
    print(f"Unique events    : {len(event_ids)}")
    print(f"Cached           : {cached}  |  To fetch live: {len(event_ids) - cached}")

    event_shots = {}
    fetched = errors = 0

    async with AsyncSession() as session:
        for eid in sorted(event_ids):
            was_cached = os.path.exists(os.path.join(CACHE_DIR, f"{eid}.json"))
            label = "(cached)" if was_cached else "..."
            print(f"  [{fetched+errors+1}/{len(event_ids)}] event {eid} {label}", end=" ", flush=True)

            raw = await fetch_shotmap(eid, session)
            if raw:
                shots = extract_penalty_shots(raw)
                event_shots[eid] = shots
                fetched += 1
                total = sum(len(v) for v in shots.values())
                print(f"{total} penalty shots across {len(shots)} players")
            else:
                errors += 1
                print("failed to load shotmap")

            if not was_cached:
                await asyncio.sleep(DELAY)

    print(f"\nEvents loaded : {fetched}  |  Failed: {errors}")

    # ── Enrich ──────────────────────────────────────────────────────────────
    enriched = unmatched = 0
    for sofascore_id, player_info in penalty_data.items():
        for pen in player_info.get("penalties", []):
            eid = pen.get("event_id")
            if eid not in event_shots:
                continue
            match = match_shot(pen, sofascore_id, event_shots[eid])
            if match:
                pen["enriched"] = match
                enriched += 1
            else:
                unmatched += 1

    print(f"Enriched     : {enriched}")
    print(f"Unmatched    : {unmatched}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(penalty_data, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
