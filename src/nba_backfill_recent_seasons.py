"""
backfill_recent_seasons.py
Our static archive (NocturneBear/NBA-Data-2010-2024) stops at 2023-24.
This pulls everything since -- 2024-25, 2025-26, and the in-progress
2026-27 season once it starts -- via nba_api, and caches it locally so
we don't hit the live API on every single prediction.

Run this once now to backfill the gap, and re-run it periodically during
the season (e.g. weekly) to pick up newly played games.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from nba_live_data import fetch_season_player_logs, normalize_player_logs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "recent_seasons.csv"

SEASONS_TO_FETCH = ["2024-25", "2025-26", "2026-27"]


def main():
    frames = []
    for season in SEASONS_TO_FETCH:
        try:
            raw = fetch_season_player_logs(season)
            if raw.empty:
                print(f"{season}: no games yet (season hasn't started), skipping.")
                continue
            normalized = normalize_player_logs(raw)
            frames.append(normalized)
        except Exception as e:
            print(f"Could not fetch {season}: {e}. Skipping.")

    if not frames:
        print("No data fetched -- nothing to cache.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["PlayerID", "Date"]).reset_index(drop=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CACHE_PATH, index=False)
    print(f"\nCached {len(combined)} player-game rows across {combined['Season'].nunique()} season(s) "
          f"to {CACHE_PATH}")
    print(f"Date range: {combined['Date'].min().date()} -> {combined['Date'].max().date()}")


if __name__ == "__main__":
    main()
