"""
nba_live_data.py
Thin client for nba_api (the standard free, unofficial-but-excellent
Python wrapper around stats.nba.com -- no API key needed at all).

Two real gaps versus our historical archive, both documented honestly:
  1. The bulk player-game-log endpoint used here does NOT include starter/
     position data. Phase 4 showed start_rate_last5 has <2% feature
     importance, so rather than burn hundreds of extra per-game API calls
     to recover it, we freeze it at each player's last known value in
     predict_next_game.py. This is a documented simplification, not a
     silent one.
  2. MIN (minutes) format has been observed to vary across nba_api
     endpoints/versions (sometimes "MM:SS", sometimes a plain decimal).
     minutes_to_float() below handles both defensively, since this
     couldn't be verified by actually hitting the live API from this
     sandbox (stats.nba.com isn't reachable here).
"""

import time
import pandas as pd
from nba_api.stats.endpoints import playergamelogs, scheduleleaguev2
from nba_api.stats.static import players as static_players


def minutes_to_float(m) -> float:
    """Handles both 'MM:SS' strings and plain decimal minutes -- see
    module docstring for why this needs to be defensive."""
    if pd.isna(m):
        return None
    m = str(m)
    if ":" in m:
        mins, secs = m.split(":")
        return int(mins) + int(secs) / 60
    return float(m)


def fetch_season_player_logs(season: str, season_type: str = "Regular Season", max_retries: int = 3) -> pd.DataFrame:
    """
    All players' game logs for one season, in a single API call.
    `season` format: '2024-25'
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = playergamelogs.PlayerGameLogs(
                season_nullable=season, season_type_nullable=season_type
            ).get_data_frames()[0]
            print(f"Fetched {len(raw)} player-game rows for {season} {season_type}")
            return raw
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} failed ({e}), retrying...")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {season} player logs after {max_retries} attempts: {last_error}")


def normalize_player_logs(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape nba_api's bulk player-log schema into our archive's schema
    (see data_loader.py's KEEP_COLS) so the two sources concatenate cleanly."""
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["GAME_DATE"])
    df["Minutes"] = df["MIN"].apply(minutes_to_float)
    df["IsHome"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["Opponent"] = df["MATCHUP"].str.split(r"\s(?:vs\.|@)\s", regex=True).str[1]

    out = pd.DataFrame({
        "Season": df["SEASON_YEAR"],
        "Date": df["Date"],
        "GameID": df["GAME_ID"],
        "PlayerID": df["PLAYER_ID"],
        "Player": df["PLAYER_NAME"],
        "Team": df["TEAM_ABBREVIATION"],
        "Opponent": df["Opponent"],
        "IsHome": df["IsHome"],
        "IsStarter": None,  # not available from this endpoint -- see module docstring
        "Minutes": df["Minutes"],
        "FGM": df["FGM"], "FGA": df["FGA"],
        "FG3M": df["FG3M"], "FG3A": df["FG3A"],
        "FTM": df["FTM"], "FTA": df["FTA"],
        "OREB": df["OREB"], "DREB": df["DREB"], "REB": df["REB"],
        "AST": df["AST"], "STL": df["STL"], "BLK": df["BLK"],
        "TOV": df["TOV"], "PF": df["PF"], "PTS": df["PTS"],
        "PlusMinus": df["PLUS_MINUS"],
    })
    out = out[out["Minutes"] > 0]  # drop DNPs (0 minutes, if any slip through)
    return out


def fetch_schedule(season: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Full season schedule (past AND future games) in one call, via the
    proper nba_api ScheduleLeagueV2 endpoint (not an ad-hoc CDN URL).
    `season` format: '2026-27'
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = scheduleleaguev2.ScheduleLeagueV2(season=season).get_data_frames()[0]
            raw["gameDateEst"] = pd.to_datetime(raw["gameDateEst"])
            print(f"Fetched {len(raw)} scheduled games for {season}")
            return raw
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} failed ({e}), retrying...")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {season} schedule after {max_retries} attempts: {last_error}")


def get_team_next_game(schedule: pd.DataFrame, team_abbr: str, after_date: pd.Timestamp) -> dict | None:
    """First scheduled game for a team strictly after `after_date`."""
    team_games = schedule[
        ((schedule["homeTeam_teamTricode"] == team_abbr) | (schedule["awayTeam_teamTricode"] == team_abbr))
        & (schedule["gameDateEst"] > after_date)
    ].sort_values("gameDateEst")

    if team_games.empty:
        return None

    game = team_games.iloc[0]
    is_home = game["homeTeam_teamTricode"] == team_abbr
    opponent = game["awayTeam_teamTricode"] if is_home else game["homeTeam_teamTricode"]
    return {
        "game_id": game["gameId"],
        "date": game["gameDateEst"],
        "opponent": opponent,
        "is_home": is_home,
    }


def search_player(name_query: str) -> list[dict]:
    """
    Offline, static lookup (no network call) -- nba_api ships a bundled
    players table. Returns partial matches for a name search, so a
    'dashboard' can offer a picker rather than requiring an exact name.
    """
    all_players = static_players.get_players()
    query = name_query.lower().strip()
    matches = [p for p in all_players if query in p["full_name"].lower()]
    return matches
