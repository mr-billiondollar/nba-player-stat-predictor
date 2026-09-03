"""
score_predictions.py
Fills in actual point totals for logged predictions once games are
played, and reports each model's running real-world accuracy (MAE) --
the regression equivalent of the Premier League project's win-rate
tracker.

Run this periodically, a day or two after games you've predicted.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import nba_live_data as live

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_LOG = PROJECT_ROOT / "data" / "predictions" / "nba_predictions_log.csv"


def season_for_date(date: pd.Timestamp) -> str:
    """NBA seasons span two calendar years, starting in October."""
    year = date.year
    if date.month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def main():
    if not PREDICTIONS_LOG.exists():
        print(f"No predictions log found at {PREDICTIONS_LOG}. Run predict_next_game.py first.")
        return

    log = pd.read_csv(PREDICTIONS_LOG)
    log["game_date"] = pd.to_datetime(log["game_date"])
    pending = log[log["actual_pts"].isna() | (log["actual_pts"] == "")]

    if pending.empty:
        print("No pending predictions to score.")
    else:
        print(f"{len(pending)} pending prediction(s) to check...")
        seasons_needed = pending["game_date"].apply(season_for_date).unique()
        season_logs = {}
        for season in seasons_needed:
            try:
                raw = live.fetch_season_player_logs(season)
                season_logs[season] = live.normalize_player_logs(raw)
            except Exception as e:
                print(f"Could not fetch {season} results ({e}); skipping.")

        for idx, row in pending.iterrows():
            season = season_for_date(row["game_date"])
            if season not in season_logs:
                continue
            match = season_logs[season][
                (season_logs[season]["PlayerID"] == row["player_id"])
                & (season_logs[season]["Date"] == row["game_date"])
            ]
            if match.empty:
                continue  # game hasn't been played yet, or player didn't play (DNP)
            actual = match.iloc[0]["PTS"]
            log.loc[idx, "actual_pts"] = actual

        log.to_csv(PREDICTIONS_LOG, index=False)

    # --- Running scoreboard ---
    scored = log[log["actual_pts"].notna() & (log["actual_pts"] != "")].copy()
    if scored.empty:
        print("\nNo scored predictions yet.")
        return

    scored["actual_pts"] = scored["actual_pts"].astype(float)
    scored["error_rf"] = (scored["predicted_rf"] - scored["actual_pts"]).abs()
    scored["error_xgb"] = (scored["predicted_xgb"] - scored["actual_pts"]).abs()

    print(f"\n{'=' * 60}")
    print(f"TRACK RECORD ({len(scored)} scored prediction(s))")
    print(f"{'=' * 60}")
    print(f"Random Forest MAE: {scored['error_rf'].mean():.2f} points")
    print(f"XGBoost MAE:       {scored['error_xgb'].mean():.2f} points")
    print(f"\nMost recent:")
    print(scored[["player_name", "game_date", "opponent", "predicted_rf",
                   "predicted_xgb", "actual_pts"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
