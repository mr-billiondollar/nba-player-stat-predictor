"""
feature_engineering.py
Builds the model-ready dataset: one row per player-game, with pre-game
features only (no leakage) covering both the player's own recent form
AND their upcoming opponent's defensive strength/pace.

Two rolling windows are used for player form (5 and 10 games) since NBA's
82-game season gives room to distinguish a short hot/cold streak (5) from
a more stable current role (10) -- there's no equivalent distinction in
the 38-game Premier League project.

Output: data/processed/model_ready.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from nba_data_loader import load_all_seasons

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PLAYER_WINDOWS = [5, 10]
OPPONENT_WINDOW = 10


# ---------------------------------------------------------------------
# Player-level rolling form (shifted -- a game's features never include
# that game's own result, same discipline as the PL project)
# ---------------------------------------------------------------------

def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PlayerID", "Date"]).copy()
    grouped = df.groupby("PlayerID", group_keys=False)

    def shifted_roll(col, window, agg="mean"):
        shifted = grouped[col].shift(1)
        return shifted.groupby(df["PlayerID"]).rolling(window, min_periods=window) \
                       .agg(agg).reset_index(level=0, drop=True)

    for w in PLAYER_WINDOWS:
        df[f"avg_points_last{w}"] = shifted_roll("PTS", w)
        df[f"avg_minutes_last{w}"] = shifted_roll("Minutes", w)

    df["avg_rebounds_last5"] = shifted_roll("REB", 5)
    df["avg_assists_last5"] = shifted_roll("AST", 5)
    df["avg_fga_last5"] = shifted_roll("FGA", 5)
    df["std_points_last10"] = shifted_roll("PTS", 10, agg="std")
    df["start_rate_last5"] = shifted_roll("IsStarter", 5)

    # Season-to-date average, reset each season (a player's established
    # role THIS season is a different signal than their trailing-N-games
    # form, which deliberately carries over across the season boundary).
    season_grouped = df.groupby(["PlayerID", "Season"], group_keys=False)
    df["season_avg_points_to_date"] = season_grouped["PTS"].apply(
        lambda s: s.shift(1).expanding().mean()
    )

    # Rest days: purely a function of the previous game's date, not its
    # result -- no leakage risk, but still uses only prior information.
    df["days_rest"] = grouped["Date"].diff().dt.days
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    return df


# ---------------------------------------------------------------------
# Team-level rolling defensive form + pace (same reshape-and-roll
# technique as the Premier League project's team form, applied to
# points-allowed and an estimated pace instead of goals/shots)
# ---------------------------------------------------------------------

def build_team_game_log(player_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: that team's full box score line."""
    agg = player_df.groupby(["GameID", "Team"]).agg(
        Date=("Date", "first"),
        Season=("Season", "first"),
        Opponent=("Opponent", "first"),
        FGM=("FGM", "sum"), FGA=("FGA", "sum"),
        FTA=("FTA", "sum"), OREB=("OREB", "sum"), DREB=("DREB", "sum"),
        TOV=("TOV", "sum"), PTS=("PTS", "sum"),
    ).reset_index()
    return agg


def add_pace_and_points_allowed(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Self-join each team-game row to its opponent's row (same GameID) to
    get points allowed and the inputs needed for a real pace estimate --
    the standard Basketball-Reference-style single-team possession
    formula: FGA + 0.4*FTA - 1.07*(OREB/(OREB+OppDREB))*(FGA-FGM) + TOV
    """
    opp = team_game.rename(columns={
        "Team": "OppTeam", "FGM": "OppFGM", "FGA": "OppFGA",
        "FTA": "OppFTA", "OREB": "OppOREB", "DREB": "OppDREB",
        "TOV": "OppTOV", "PTS": "OppPTS",
    })[["GameID", "OppTeam", "OppFGM", "OppFGA", "OppFTA", "OppOREB", "OppDREB", "OppTOV", "OppPTS"]]

    merged = team_game.merge(opp, on="GameID")
    merged = merged[merged["Team"] != merged["OppTeam"]].copy()

    merged["PointsAllowed"] = merged["OppPTS"]
    merged["Possessions"] = (
        merged["FGA"] + 0.4 * merged["FTA"]
        - 1.07 * (merged["OREB"] / (merged["OREB"] + merged["OppDREB"]).replace(0, np.nan))
        * (merged["FGA"] - merged["FGM"])
        + merged["TOV"]
    )
    return merged


def add_team_rolling_features(team_game: pd.DataFrame, window: int = OPPONENT_WINDOW) -> pd.DataFrame:
    team_game = team_game.sort_values(["Team", "Date"]).copy()
    grouped = team_game.groupby("Team", group_keys=False)

    def shifted_roll(col):
        shifted = grouped[col].shift(1)
        return shifted.groupby(team_game["Team"]).rolling(window, min_periods=window) \
                       .mean().reset_index(level=0, drop=True)

    team_game[f"opp_points_allowed_last{window}"] = shifted_roll("PointsAllowed")
    team_game[f"opp_pace_last{window}"] = shifted_roll("Possessions")
    return team_game


# ---------------------------------------------------------------------
# Assemble the final dataset
# ---------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    player_df = load_all_seasons()
    player_df = add_player_rolling_features(player_df)

    team_game = build_team_game_log(player_df)
    team_game = add_pace_and_points_allowed(team_game)
    team_game = add_team_rolling_features(team_game)

    opp_cols = ["GameID", "Team", f"opp_points_allowed_last{OPPONENT_WINDOW}", f"opp_pace_last{OPPONENT_WINDOW}"]
    opp_lookup = team_game[opp_cols].rename(columns={"Team": "Opponent"})

    dataset = player_df.merge(opp_lookup, on=["GameID", "Opponent"], how="left")

    feature_cols = (
        [f"avg_points_last{w}" for w in PLAYER_WINDOWS]
        + [f"avg_minutes_last{w}" for w in PLAYER_WINDOWS]
        + ["avg_rebounds_last5", "avg_assists_last5", "avg_fga_last5",
           "std_points_last10", "start_rate_last5", "season_avg_points_to_date",
           "days_rest", "is_back_to_back", "IsHome",
           f"opp_points_allowed_last{OPPONENT_WINDOW}", f"opp_pace_last{OPPONENT_WINDOW}"]
    )
    keep = ["GameID", "Date", "Season", "PlayerID", "Player", "Team", "Opponent"] + feature_cols + ["PTS"]
    dataset = dataset[keep]

    before = len(dataset)
    dataset = dataset.dropna().reset_index(drop=True)
    dropped = before - len(dataset)

    print(f"Player-games before dropping cold-start rows: {before}")
    print(f"Dropped {dropped} rows (players without 10 prior games of history yet, "
          f"e.g. early in a rookie season)")
    print(f"Final model-ready dataset: {len(dataset)} player-games")
    print(f"\nTarget (PTS) distribution:\n{dataset['PTS'].describe().round(1)}")

    return dataset


def get_latest_player_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each player's CURRENT rolling form, as of their most recent played
    game -- for live prediction (not training). Same idea as the PL
    project's compute_latest_team_form, applied per player.

    start_rate_last5 uses forward-filled last-known value rather than a
    fresh rolling calc, since nba_api's bulk endpoint doesn't carry
    starter/position data (see nba_live_data.py docstring) -- this is a
    documented simplification, justified by that feature's <2% importance
    in Phase 4.
    """
    df = df.sort_values(["PlayerID", "Date"]).copy()
    grouped = df.groupby("PlayerID", group_keys=False)

    def roll(col, window, agg="mean"):
        return grouped[col].rolling(window, min_periods=1).agg(agg).reset_index(level=0, drop=True)

    df["_avg_points_last5"] = roll("PTS", 5)
    df["_avg_points_last10"] = roll("PTS", 10)
    df["_avg_minutes_last5"] = roll("Minutes", 5)
    df["_avg_minutes_last10"] = roll("Minutes", 10)
    df["_avg_rebounds_last5"] = roll("REB", 5)
    df["_avg_assists_last5"] = roll("AST", 5)
    df["_avg_fga_last5"] = roll("FGA", 5)
    df["_std_points_last10"] = roll("PTS", 10, agg="std")

    df["_start_rate_ffill"] = grouped["IsStarter"].apply(lambda s: s.ffill())

    # Season-to-date average for whichever season is each player's most
    # recent (mirrors the per-season reset used in training) -- computed
    # separately since it needs to reset at season boundaries, unlike the
    # trailing 5/10-game windows above which deliberately don't.
    current_season_avg = (
        df.groupby(["PlayerID", "Season"])["PTS"].transform("mean")
    )
    df["_season_avg_current"] = current_season_avg

    latest = (
        df.groupby("PlayerID")
        .agg(
            Player=("Player", "last"),
            Team=("Team", "last"),
            last_game_date=("Date", "max"),
            avg_points_last5=("_avg_points_last5", "last"),
            avg_points_last10=("_avg_points_last10", "last"),
            avg_minutes_last5=("_avg_minutes_last5", "last"),
            avg_minutes_last10=("_avg_minutes_last10", "last"),
            avg_rebounds_last5=("_avg_rebounds_last5", "last"),
            avg_assists_last5=("_avg_assists_last5", "last"),
            avg_fga_last5=("_avg_fga_last5", "last"),
            std_points_last10=("_std_points_last10", "last"),
            start_rate_last5=("_start_rate_ffill", "last"),
            season_avg_points_to_date=("_season_avg_current", "last"),
        )
    )
    # Players with NO real starter data anywhere in their history (e.g.
    # rookies only seen via nba_api) fall back to the league-wide average.
    latest["start_rate_last5"] = latest["start_rate_last5"].fillna(latest["start_rate_last5"].mean())
    return latest


def get_latest_team_form(team_game: pd.DataFrame, window: int = OPPONENT_WINDOW) -> pd.DataFrame:
    """Each team's current rolling defensive/pace form, as of their most
    recent played game -- for live prediction."""
    team_game = team_game.sort_values(["Team", "Date"]).copy()
    grouped = team_game.groupby("Team", group_keys=False)
    team_game["_pts_allowed_roll"] = grouped["PointsAllowed"].rolling(window, min_periods=1) \
        .mean().reset_index(level=0, drop=True)
    team_game["_pace_roll"] = grouped["Possessions"].rolling(window, min_periods=1) \
        .mean().reset_index(level=0, drop=True)

    return (
        team_game.groupby("Team")
        .agg(
            last_game_date=("Date", "max"),
            points_allowed_last=("_pts_allowed_roll", "last"),
            pace_last=("_pace_roll", "last"),
        )
    )


if __name__ == "__main__":
    dataset = build_dataset()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "model_ready.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\nSample rows:")
    print(dataset.head(3).to_string(index=False))
