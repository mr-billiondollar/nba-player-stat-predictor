"""
dashboard.py
The actual "dashboard" experience: pick any player from a dropdown, see
their recent scoring trend, and get a next-game points prediction from
both trained models -- with an automatic off-season backtest fallback
so this is genuinely usable before the 2026-27 season starts (Oct 20).

Run with:
    streamlit run src/dashboard.py

First run: `python src/backfill_recent_seasons.py` so predictions use
up-to-date form, not just the archive through 2023-24.
"""

import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import joblib
import streamlit as st
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
CURRENT_SEASON = "2026-27"

st.set_page_config(page_title="NBA Player Points Predictor", layout="centered")


@st.cache_data(show_spinner="Loading historical + recent season data...")
def load_combined_data():
    historical = load_all_seasons()
    if RECENT_CACHE.exists():
        recent = pd.read_csv(RECENT_CACHE, parse_dates=["Date"])
        recent["IsStarter"] = recent["IsStarter"].astype("object")
        return pd.concat([historical, recent], ignore_index=True)
    return historical


@st.cache_data(show_spinner="Computing current form for every player and team...")
def compute_form(_combined):
    player_form = get_latest_player_form(_combined)
    team_game = build_team_game_log(_combined)
    team_game = add_pace_and_points_allowed(team_game)
    team_form = get_latest_team_form(team_game)
    return player_form, team_form


@st.cache_resource(show_spinner=False)
def load_models():
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = XGBRegressor()
    xgb.load_model(str(MODELS_DIR / "xgboost_model.json"))
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return rf, xgb, feature_cols


@st.cache_data(ttl=3600, show_spinner="Fetching this season's schedule...")
def get_schedule():
    try:
        return live.fetch_schedule(CURRENT_SEASON), None
    except Exception as e:
        return None, str(e)


def build_feature_row(player_id, opponent, is_home, days_rest, player_form, team_form):
    p = player_form.loc[player_id]
    if opponent in team_form.index:
        opp = team_form.loc[opponent]
        opp_pts_allowed, opp_pace = opp["points_allowed_last"], opp["pace_last"]
    else:
        opp_pts_allowed = team_form["points_allowed_last"].mean()
        opp_pace = team_form["pace_last"].mean()

    return {
        "avg_points_last5": p["avg_points_last5"], "avg_points_last10": p["avg_points_last10"],
        "avg_minutes_last5": p["avg_minutes_last5"], "avg_minutes_last10": p["avg_minutes_last10"],
        "avg_rebounds_last5": p["avg_rebounds_last5"], "avg_assists_last5": p["avg_assists_last5"],
        "avg_fga_last5": p["avg_fga_last5"], "std_points_last10": p["std_points_last10"],
        "start_rate_last5": float(p["start_rate_last5"]) if pd.notna(p["start_rate_last5"]) else 0.5,
        "season_avg_points_to_date": p["season_avg_points_to_date"],
        "days_rest": days_rest, "is_back_to_back": int(days_rest <= 1), "IsHome": int(is_home),
        f"opp_points_allowed_last{OPPONENT_WINDOW}": opp_pts_allowed,
        f"opp_pace_last{OPPONENT_WINDOW}": opp_pace,
    }


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("🏀 NBA Player Points Predictor")
st.caption("Random Forest vs. XGBoost, predicting next-game points. "
           "Same leakage-safe pipeline as the Premier League match predictor project.")

combined = load_combined_data()
player_form, team_form = compute_form(combined)

active_players = sorted(
    [p for p in player_form.index.map(lambda pid: player_form.loc[pid, "Player"])],
)
player_name = st.selectbox("Choose a player", options=sorted(set(active_players)))

if player_name:
    player_row = player_form[player_form["Player"] == player_name].iloc[0]
    player_id = player_form[player_form["Player"] == player_name].index[0]
    player_team = player_row["Team"]
    last_game_date = player_row["last_game_date"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg last 5", f"{player_row['avg_points_last5']:.1f} pts")
    col2.metric("Avg last 10", f"{player_row['avg_points_last10']:.1f} pts")
    col3.metric("Season avg", f"{player_row['season_avg_points_to_date']:.1f} pts")

    # Recent trend chart
    recent_games = combined[combined["PlayerID"] == player_id].sort_values("Date").tail(10)
    st.line_chart(recent_games.set_index("Date")["PTS"], height=200)

    schedule, schedule_error = get_schedule()
    next_game = None
    if schedule is not None:
        next_game = live.get_team_next_game(schedule, player_team, after_date=last_game_date)

    backtest_mode = next_game is None
    if backtest_mode:
        st.info(
            f"No upcoming scheduled game found for {player_team} yet "
            f"(normal pre-season -- 2026-27 starts Oct 20, 2026). "
            f"Showing a **backtest** on their last played game instead, "
            f"so you can see the model work right now."
            + (f" [{schedule_error}]" if schedule_error else "")
        )
        last_row = combined[combined["PlayerID"] == player_id].sort_values("Date").iloc[-1]
        opponent, is_home = last_row["Opponent"], bool(last_row["IsHome"])
        days_rest, actual_points, game_date = 3, last_row["PTS"], last_row["Date"]
    else:
        opponent, is_home = next_game["opponent"], next_game["is_home"]
        days_rest = (next_game["date"] - last_game_date).days
        actual_points, game_date = None, next_game["date"]

    feat_dict = build_feature_row(player_id, opponent, is_home, days_rest, player_form, team_form)
    rf, xgb, feature_cols = load_models()
    X = pd.DataFrame([feat_dict])[feature_cols]
    rf_pred, xgb_pred = rf.predict(X)[0], xgb.predict(X)[0]

    venue = "vs." if is_home else "@"
    st.subheader(f"{player_name} {venue} {opponent} — {game_date.date()}")

    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Random Forest predicts", f"{rf_pred:.1f} pts")
    pcol2.metric("XGBoost predicts", f"{xgb_pred:.1f} pts")

    if backtest_mode:
        st.write(f"**Actually scored: {actual_points} points**")
        ecol1, ecol2 = st.columns(2)
        ecol1.metric("RF error", f"{abs(rf_pred - actual_points):.1f} pts")
        ecol2.metric("XGBoost error", f"{abs(xgb_pred - actual_points):.1f} pts")

    st.caption(
        "Note: opponent points-allowed/pace has very low feature importance for individual "
        "scoring (see Phase 4) -- most of the signal here is the player's own recent form."
    )
