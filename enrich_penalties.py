import asyncio
import json
import os
from curl_cffi.requests import AsyncSession

JSON_FILE   = "./FinalData/penalty_data/all_penalty_history.json"
OUTPUT_FILE = "./FinalData/penalty_data/all_penalty_history_enriched.json"
CACHE_DIR   = "./FinalData/penalty_data/incidents_cache"
API_BASE    = "https://api.sofascore.com/api/v1/event"
DELAY       = 0.5
COORD_TOL   = 0.6

os.makedirs(CACHE_DIR, exist_ok=True)

async def fetch_incidents(event_id, session):
    cache_path = os.path.join(CACHE_DIR, f"{event_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    url = f"{API_BASE}/{event_id}/incidents"
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

def _shot_fields(inc):
    out = {}
    actions = inc.get("footballPassingNetworkAction") or []
    if not actions:
        return out
    a = actions[0]
    out["event_type"]     = a.get("eventType")
    out["body_part"]      = a.get("bodyPart")
    out["gk_name"]        = (a.get("goalkeeper") or {}).get("name")
    out["gk_id"]          = (a.get("goalkeeper") or {}).get("id")
    out["shot_x"]         = (a.get("goalShotCoordinates") or {}).get("x")
    out["shot_y"]         = (a.get("goalShotCoordinates") or {}).get("y")
    out["player_coord_x"] = (a.get("playerCoordinates") or {}).get("x")
    out["player_coord_y"] = (a.get("playerCoordinates") or {}).get("y")
    out["gk_coord_x"]     = (a.get("gkCoordinates") or {}).get("x")
    out["gk_coord_y"]     = (a.get("gkCoordinates") or {}).get("y")
    out["goal_mouth_x"]   = (a.get("goalMouthCoordinates") or {}).get("x")
    out["goal_mouth_y"]   = (a.get("goalMouthCoordinates") or {}).get("y")
    return out

def extract_penalty_events(raw):
    by_player = {}
    unowned   = []
    if not raw or "incidents" not in raw:
        return by_player, unowned
    goals_at = {}
    for inc in raw["incidents"]:
        if inc.get("incidentType") == "goal" and inc.get("player"):
            goals_at.setdefault((inc.get("time"), inc.get("isHome")), []).append(inc)
    for inc in raw["incidents"]:
        inc_type = inc.get("incidentType", "")
        inc_cls  = inc.get("incidentClass", "")
        player   = inc.get("player") or {}
        pid      = player.get("id")
        if inc_type == "inGamePenalty" and not pid:
            key = (inc.get("time"), inc.get("isHome"))
            goals = goals_at.get(key, [])
            if goals:
                g = goals[0]
                pid = g["player"]["id"]
                player = g["player"]
        if not pid:
            if inc_type in ("inGamePenalty", "penaltyShootout"):
                entry = {
                    "player_id": None, "player_name": None,
                    "time": inc.get("time"), "added_time": inc.get("addedTime"),
                    "home_score": inc.get("homeScore"), "away_score": inc.get("awayScore"),
                    "is_home": inc.get("isHome"),
                    "type": "match_penalty" if inc_type == "inGamePenalty" else "shootout",
                    "outcome": inc_cls, "reason": inc.get("reason"),
                }
                entry.update(_shot_fields(inc))
                unowned.append(entry)
            continue
        base = {
            "player_id": pid, "player_name": player.get("name"),
            "time": inc.get("time"), "added_time": inc.get("addedTime"),
            "home_score": inc.get("homeScore"), "away_score": inc.get("awayScore"),
            "is_home": inc.get("isHome"),
        }
        if inc_type == "goal" and inc_cls == "penalty":
            entry = {**base, "type": "match_penalty", "outcome": "scored", "_strength": "strong"}
            entry.update(_shot_fields(inc))
        elif inc_type == "goal" and inc_cls == "regular":
            entry = {**base, "type": "match_penalty", "outcome": "scored", "_strength": "weak"}
            entry.update(_shot_fields(inc))
        elif inc_type == "inGamePenalty":
            entry = {**base, "type": "match_penalty", "outcome": inc_cls, "reason": inc.get("reason"), "_strength": "strong"}
            entry.update(_shot_fields(inc))
        elif inc_type == "penaltyShootout":
            entry = {
                **base, "type": "shootout", "outcome": inc_cls,
                "reason": inc.get("reason"), "description": inc.get("description"),
                "sequence": inc.get("sequence"), "_strength": "strong",
            }
            entry.update(_shot_fields(inc))
        else:
            continue
        by_player.setdefault(pid, []).append(entry)
    return by_player, unowned

_OUTCOME_MAP = {"goal": "scored", "save": "missed", "miss": "missed", "post": "missed"}

def _match_group(pen, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    pen_x = pen.get("x")
    pen_y = pen.get("y")
    if pen_x is not None and pen_y is not None:
        coord_matches = [
            c for c in candidates
            if c.get("goal_mouth_x") is not None
            and abs(c["goal_mouth_x"] - pen_x) <= COORD_TOL
            and abs(c["goal_mouth_y"] - pen_y) <= COORD_TOL
        ]
        if len(coord_matches) == 1:
            return coord_matches[0]
        if len(coord_matches) > 1:
            return min(coord_matches, key=lambda c: (c["goal_mouth_x"] - pen_x)**2 + (c["goal_mouth_y"] - pen_y)**2)
    expected = _OUTCOME_MAP.get(pen.get("outcome", ""), "")
    if expected:
        outcome_matches = [c for c in candidates if c.get("outcome") == expected]
        if len(outcome_matches) == 1:
            return outcome_matches[0]
    no_coord = [c for c in candidates if c.get("goal_mouth_x") is None]
    if len(no_coord) == 1:
        return no_coord[0]
    return None

def _match_unowned(pen, unowned):
    if not unowned:
        return None
    if len(unowned) == 1:
        return unowned[0]
    expected = _OUTCOME_MAP.get(pen.get("outcome", ""), "")
    if expected:
        om = [c for c in unowned if c.get("outcome") == expected]
        if len(om) == 1:
            return om[0]
    pen_x = pen.get("x")
    pen_y = pen.get("y")
    if pen_x is not None and pen_y is not None:
        coord_matches = [
            c for c in unowned
            if c.get("goal_mouth_x") is not None
            and abs(c["goal_mouth_x"] - pen_x) <= COORD_TOL
            and abs(c["goal_mouth_y"] - pen_y) <= COORD_TOL
        ]
        if len(coord_matches) == 1:
            return coord_matches[0]
        if len(coord_matches) > 1:
            return min(coord_matches, key=lambda c: (c["goal_mouth_x"] - pen_x)**2 + (c["goal_mouth_y"] - pen_y)**2)
    return None

def _match_candidates(pen, candidates):
    if not candidates:
        return None
    strong = [c for c in candidates if c.get("_strength") != "weak"]
    weak   = [c for c in candidates if c.get("_strength") == "weak"]
    result = _match_group(pen, strong)
    if result:
        return result
    if pen.get("outcome") == "goal":
        result = _match_group(pen, weak)
        if result:
            return result
    return None

def match_incident(pen, sofascore_id, incidents_by_player, unowned=None):
    try:
        pid = int(sofascore_id)
    except (TypeError, ValueError):
        return _match_unowned(pen, unowned)
    candidates = incidents_by_player.get(pid)
    result = _match_candidates(pen, candidates)
    if result:
        return result
    if unowned:
        result = _match_unowned(pen, unowned)
        if result:
            return result
    return None

async def main():
    with open(JSON_FILE, encoding="utf-8") as f:
        penalty_data = json.load(f)
    event_ids = set()
    for player_info in penalty_data.values():
        for pen in player_info.get("penalties", []):
            eid = pen.get("event_id")
            if eid:
                event_ids.add(eid)
    print(f"Total players        : {len(penalty_data)}")
    print(f"Unique event_ids     : {len(event_ids)}")
    cached = sum(1 for e in event_ids if os.path.exists(os.path.join(CACHE_DIR, f"{e}.json")))
    print(f"Already cached       : {cached}  (will fetch {len(event_ids)-cached} live)")
    event_incidents = {}
    event_unowned   = {}
    fetched = errors = 0
    async with AsyncSession() as session:
        for eid in sorted(event_ids):
            was_cached = os.path.exists(os.path.join(CACHE_DIR, f"{eid}.json"))
            label = "(cached)" if was_cached else "..."
            print(f"  [{fetched+errors+1}/{len(event_ids)}] event {eid} {label}", end=" ", flush=True)
            raw = await fetch_incidents(eid, session)
            if raw:
                parsed, unowned = extract_penalty_events(raw)
                total = sum(len(v) for v in parsed.values())
                if total or unowned:
                    event_incidents[eid] = parsed
                    event_unowned[eid]   = unowned
                    fetched += 1
                    print(f"{total} incidents, {len(unowned)} unowned")
                else:
                    errors += 1
                    print("no penalty incidents")
            else:
                errors += 1
                print("failed")
            if not was_cached:
                await asyncio.sleep(DELAY)
    print(f"\nEvents with incidents: {fetched}  |  Empty/failed: {errors}")
    enriched = unmatched = 0
    for sofascore_id, player_info in penalty_data.items():
        for pen in player_info.get("penalties", []):
            eid = pen.get("event_id")
            if eid not in event_incidents:
                continue
            match = match_incident(pen, sofascore_id, event_incidents[eid], event_unowned.get(eid))
            if match:
                pen["enriched"] = match
                enriched += 1
            else:
                unmatched += 1
    print(f"Enriched             : {enriched}")
    print(f"Unmatched            : {unmatched}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(penalty_data, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
