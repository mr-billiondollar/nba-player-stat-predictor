"""
predict_next_game.py
The "dashboard" experience: search for a player by name, and get a
prediction for their next scheduled game, from both trained models.

Usage:
    python src/predict_next_game.py "luka doncic"
    python src/predict_next_game.py            (prompts interactively)

Run backfill_recent_seasons.py first (and periodically thereafter) so
this has up-to-date form to work from.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import joblib
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from nba_data_loader import load_all_seasons
from nba_feature_engineering import (
    build_team_game_log, add_pace_and_points_allowed,
    get_latest_player_form, get_latest_team_form, OPPONENT_WINDOW,
)
import nba_live_data as live

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RECENT_CACHE = PROJECT_ROOT / "data" / "processed" / "recent_seasons.csv"
PREDICTIONS_LOG = PROJECT_ROOT / "data" / "predictions" / "nba_predictions_log.csv"
CURRENT_SEASON = "2026-27"


def load_combined_data() -> pd.DataFrame:
    historical = load_all_seasons()
    if RECENT_CACHE.exists():
        recent = pd.read_csv(RECENT_CACHE, parse_dates=["Date"])
        recent["IsStarter"] = recent["IsStarter"].astype("object")  # was written as all-null
        combined = pd.concat([historical, recent], ignore_index=True)
        print(f"Combined historical + cached recent seasons: {len(combined)} total player-games")
    else:
        print(f"WARNING: {RECENT_CACHE} not found. Run backfill_recent_seasons.py first "
              f"for up-to-date form. Using historical archive only (through 2023-24) for now.")
        combined = historical
    return combined


def resolve_player(name_query: str) -> dict:
    matches = live.search_player(name_query)
    if not matches:
        raise ValueError(f"No player found matching '{name_query}'.")
    if len(matches) == 1:
        return matches[0]

    print(f"\nMultiple players match '{name_query}':")
    for i, p in enumerate(matches, 1):
        print(f"  {i}. {p['full_name']}")
    choice = input("Pick a number: ").strip()
    return matches[int(choice) - 1]


def build_feature_row(player_id: int, next_game: dict, player_form: pd.DataFrame, team_form: pd.DataFrame) -> dict:
    if player_id not in player_form.index:
        raise ValueError("No game history found for this player -- can't build features yet.")
    p = player_form.loc[player_id]

    opponent = next_game["opponent"]
    if opponent in team_form.index:
        opp = team_form.loc[opponent]
        opp_pts_allowed = opp["points_allowed_last"]
        opp_pace = opp["pace_last"]
    else:
        print(f"NOTE: no game history found for opponent '{opponent}' -- using league-wide defaults.")
        opp_pts_allowed = team_form["points_allowed_last"].mean()
        opp_pace = team_form["pace_last"].mean()

    days_rest = (next_game["date"] - p["last_game_date"]).days

    return {
        "avg_points_last5": p["avg_points_last5"],
        "avg_points_last10": p["avg_points_last10"],
        "avg_minutes_last5": p["avg_minutes_last5"],
        "avg_minutes_last10": p["avg_minutes_last10"],
        "avg_rebounds_last5": p["avg_rebounds_last5"],
        "avg_assists_last5": p["avg_assists_last5"],
        "avg_fga_last5": p["avg_fga_last5"],
        "std_points_last10": p["std_points_last10"],
        "start_rate_last5": p["start_rate_last5"],
        "season_avg_points_to_date": p["season_avg_points_to_date"],
        "days_rest": days_rest,
        "is_back_to_back": int(days_rest <= 1),
        "IsHome": int(next_game["is_home"]),
        f"opp_points_allowed_last{OPPONENT_WINDOW}": opp_pts_allowed,
        f"opp_pace_last{OPPONENT_WINDOW}": opp_pace,
    }


def main():
    name_query = sys.argv[1] if len(sys.argv) > 1 else input("Search for a player: ")

    player = resolve_player(name_query)
    print(f"\nFound: {player['full_name']} (ID {player['id']})")

    combined = load_combined_data()

    print("\nComputing current player form and opponent defensive form...")
    player_form = get_latest_player_form(combined)
    team_game = build_team_game_log(combined)
    team_game = add_pace_and_points_allowed(team_game)
    team_form = get_latest_team_form(team_game)

    if player["id"] not in player_form.index:
        print(f"No game history found for {player['full_name']} in our data. Can't predict yet.")
        return
    player_team = player_form.loc[player["id"], "Team"]

    print(f"\nFetching schedule for {CURRENT_SEASON}...")
    schedule = live.fetch_schedule(CURRENT_SEASON)
    last_game_date = player_form.loc[player["id"], "last_game_date"]
    next_game = live.get_team_next_game(schedule, player_team, after_date=last_game_date)

    if next_game is None:
        print(f"No upcoming scheduled game found for {player_team}.")
        return

    print(f"\nNext game: {player_team} vs/at {next_game['opponent']} on {next_game['date'].date()} "
          f"({'Home' if next_game['is_home'] else 'Away'})")

    feat_dict = build_feature_row(player["id"], next_game, player_form, team_form)
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    X = pd.DataFrame([feat_dict])[feature_cols]

    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = XGBRegressor()
    xgb.load_model(str(MODELS_DIR / "xgboost_model.json"))

    rf_pred = rf.predict(X)[0]
    xgb_pred = xgb.predict(X)[0]

    print(f"\n{'=' * 60}")
    print(f"PREDICTION: {player['full_name']} vs {next_game['opponent']} ({next_game['date'].date()})")
    print(f"{'=' * 60}")
    print(f"Random Forest predicts: {rf_pred:.1f} points")
    print(f"XGBoost predicts:       {xgb_pred:.1f} points")
    print(f"(Player's own rolling 10-game average: {feat_dict['avg_points_last10']:.1f} -- "
          f"the baseline these models are trying to beat)")

    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_row = pd.DataFrame([{
        "prediction_made_on": datetime.now().date().isoformat(),
        "player_id": player["id"],
        "player_name": player["full_name"],
        "game_date": next_game["date"].date().isoformat(),
        "opponent": next_game["opponent"],
        "predicted_rf": round(rf_pred, 1),
        "predicted_xgb": round(xgb_pred, 1),
        "actual_pts": "",
    }])
    if PREDICTIONS_LOG.exists():
        existing = pd.read_csv(PREDICTIONS_LOG)
        combined_log = pd.concat([existing, log_row], ignore_index=True)
        combined_log = combined_log.drop_duplicates(subset=["player_id", "game_date"], keep="last")
    else:
        combined_log = log_row
    combined_log.to_csv(PREDICTIONS_LOG, index=False)
    print(f"\nLogged to {PREDICTIONS_LOG}")


if __name__ == "__main__":
    main()
