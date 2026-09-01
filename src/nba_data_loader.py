"""
data_loader.py
Loads every regular_season_box_scores_*.csv in data/raw/, cleans it, and
returns one player-game-level DataFrame ready for feature engineering.

Source: player-level box scores, 2010-11 through 2023-24 seasons
(https://github.com/NocturneBear/NBA-Data-2010-2024).

Key cleaning steps (found by inspecting the raw data first, same
discipline as the Premier League project):
  - ~17% of rows are DNP/DND/rest-day entries (flagged via the `comment`
    column) with zeroed-out stats. These must be dropped, not treated as
    "the player scored 0" -- including them would corrupt rolling
    averages by mixing in games the player never actually played.
  - `minutes` is a string "MM:SS" (e.g. "32:40"), not a number -- convert
    to a float.
  - `position` is only populated for starters (F/G/C) and blank for bench
    players -- that's a free "is_starter" flag hiding in the data.
  - `matchup` is "TEAM vs. OPP" for home games, "TEAM @ OPP" for away --
    same convention idea as the Premier League project's H/A, just
    encoded differently.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

KEEP_COLS = {
    "season_year": "Season",
    "game_date": "Date",
    "gameId": "GameID",
    "personId": "PlayerID",
    "personName": "Player",
    "teamTricode": "Team",
    "matchup": "Matchup",
    "position": "Position",
    "minutes": "MinutesRaw",
    "fieldGoalsMade": "FGM",
    "fieldGoalsAttempted": "FGA",
    "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A",
    "freeThrowsMade": "FTM",
    "freeThrowsAttempted": "FTA",
    "reboundsOffensive": "OREB",
    "reboundsDefensive": "DREB",
    "reboundsTotal": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    "foulsPersonal": "PF",
    "points": "PTS",
    "plusMinusPoints": "PlusMinus",
}


def minutes_to_float(m: str) -> float:
    if pd.isna(m) or m == "":
        return None
    mins, secs = m.split(":")
    return int(mins) + int(secs) / 60


def load_box_score_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["comment"] + list(KEEP_COLS.keys()))

    # Drop DNP/DND/rest/inactive rows -- these are not "0-point games"
    df = df[df["comment"].isna()].drop(columns=["comment"])

    df = df.rename(columns=KEEP_COLS)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Minutes"] = df["MinutesRaw"].apply(minutes_to_float)
    df = df.drop(columns=["MinutesRaw"])

    # Drop essentially-garbage-time entries (0 minutes played)
    df = df[df["Minutes"] > 0]

    df["IsHome"] = df["Matchup"].str.contains("vs.", regex=False)
    df["Opponent"] = df["Matchup"].str.split(r"\s(?:vs\.|@)\s", regex=True).str[1]
    df["IsStarter"] = df["Position"].notna()
    df = df.drop(columns=["Matchup", "Position"])

    return df


def load_all_seasons() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("regular_season_box_scores_*.csv"))
    if not files:
        raise FileNotFoundError(f"No regular_season_box_scores_*.csv files found in {RAW_DIR}")

    frames = [load_box_score_file(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["PlayerID", "Date"]).reset_index(drop=True)

    print(f"Loaded {len(files)} box score file(s)")
    print(f"Total player-game rows (played games only): {len(data)}")
    print(f"Unique players: {data['PlayerID'].nunique()}")
    print(f"Date range: {data['Date'].min().date()} -> {data['Date'].max().date()}")
    print(f"Seasons: {sorted(data['Season'].unique())}")

    return data


if __name__ == "__main__":
    df = load_all_seasons()
    print("\nColumns:", list(df.columns))
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print("\nPoints distribution:")
    print(df["PTS"].describe().round(1))
