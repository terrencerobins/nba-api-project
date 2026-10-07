import requests
import pandas as pd
import time
import os
import logging

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = os.getenv("BALLDONTLIE_API_KEY")

SEASONS = [2023, 2024, 2025]
BASE_URL = "https://api.balldontlie.io/v1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if API_KEY is None:
    raise ValueError("BALLDONTLIE_API_KEY environment variable is not set.")


# function for extracting games via api
def extract_games_raw(seasons):
    url = f"{BASE_URL}/games"

    headers = {
        "Authorization": API_KEY
    }
    params = {
        "seasons[]": seasons,
        "per_page": 100
    }

    all_games = []
    while True:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        logging.info(f"Request status: {response.status_code}")

        if response.status_code == 429:
            logging.warning("Rate limited. Waiting 15 seconds...")
            time.sleep(15)
            continue

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed with status {response.status_code}: {response.text[:300]}"
            )

        payload = response.json()
        all_games.extend(payload["data"])

        next_cursor = payload["meta"].get("next_cursor")

        if next_cursor is None:
            break

        params["cursor"] = next_cursor

        time.sleep(1)
    return pd.json_normalize(all_games)


def extract_player_game_stats_raw(seasons):
    url = f"{BASE_URL}/stats"

    headers = {
        "Authorization": API_KEY
    }
    params = {
        "seasons[]": seasons,
        "per_page": 100
    }
    all_stats = []
    while True:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        logging.info(f"Request status: {response.status_code}")

        if response.status_code == 429:
            logging.warning("Rate limited. Waiting 15 seconds...")
            time.sleep(15)
            continue
        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed with status {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        all_stats.extend(payload['data'])

        next_cursor = payload['meta'].get('next_cursor')

        if next_cursor is None:
            break

        params['cursor'] = next_cursor

        time.sleep(1)

    return pd.json_normalize(all_stats)


# creating a Games fact table
def build_games_fact(df):
    df = df.copy()
    required_cols = ["home_team_score", "visitor_team_score", "id",
                     "date",
                     "season",
                     "postseason",
                     "home_team.id",
                     "home_team.full_name",
                     "visitor_team.id",
                     "visitor_team.full_name",
                     ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available Columns: {list(df.columns)}")

    df["home_point_diff"] = df["home_team_score"] - df["visitor_team_score"]
    df["total_points"] = df["home_team_score"] + df["visitor_team_score"]
    df["home_win"] = df["home_team_score"] > df["visitor_team_score"]

    selected_cols = [
        "id",
        "date",
        "season",
        "postseason",
        "home_team.id",
        "home_team.full_name",
        "visitor_team.id",
        "visitor_team.full_name",
        "home_team_score",
        "visitor_team_score",
        "home_point_diff",
        "total_points",
        "home_win"
    ]
    df["date"] = pd.to_datetime(df["date"])
    df = df[selected_cols]
    return df


# Creating a Teams dimension table
def build_teams_dim(df):
    home_teams = df[
        ['home_team.full_name',
         'home_team.id',
         'home_team.name',
         'home_team.abbreviation',
         'home_team.city',
         'home_team.conference',
         'home_team.division'
         ]
    ].rename(
        columns={
            'home_team.full_name': 'team_full_name',
            'home_team.id': 'team_id',
            'home_team.name': 'team_name',
            'home_team.abbreviation': 'team_abbreviation',
            'home_team.city': 'team_city',
            'home_team.conference': 'team_conference',
            'home_team.division': 'team_division'

        }
    )
    visitor_teams = df[
        ['visitor_team.full_name',
         'visitor_team.id',
         'visitor_team.name',
         'visitor_team.abbreviation',
         'visitor_team.city',
         'visitor_team.conference',
         'visitor_team.division'
         ]
    ].rename(
        columns={
            'visitor_team.full_name': 'team_full_name',
            'visitor_team.id': 'team_id',
            'visitor_team.name': 'team_name',
            'visitor_team.abbreviation': 'team_abbreviation',
            'visitor_team.city': 'team_city',
            'visitor_team.conference': 'team_conference',
            'visitor_team.division': 'team_division'
        }
    )
    teams = pd.concat([home_teams, visitor_teams])
    teams = teams.drop_duplicates()

    return teams


# Creating a player stats fact table
def build_player_game_stats_fact(df):
    df = df.copy()
    required_cols = ['id', 'min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
                     'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'turnover', 'pf', 'pts',
                     'plus_minus', 'player.id', 'game.id', 'team.id'
                     ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns:{missing_cols}. Available Columns: {list(df.columns)}")

    df = df[
        [
            'id', 'min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'turnover', 'pf', 'pts',
            'plus_minus', 'player.id', 'game.id', 'team.id'
        ]
    ].rename(
        columns={
            'id': 'player_game_stat_id',
            'game.id': 'game_id',
            'team.id': 'team_id',
            'player.id': 'player_id'
        }
    )

    return df


# creating a Players dimension table
def build_players_dim(df):
    player_data = df[
        ['player.id',
         'player.first_name',
         'player.last_name',
         'player.position',
         'player.height',
         'player.weight',
         'player.college',
         'player.country',
         'player.draft_year',
         'player.draft_round',
         'player.draft_number'
         ]
    ].rename(
        columns={
            'player.id': 'player_id',
            'player.first_name': 'first_name',
            'player.last_name': 'last_name',
            'player.position': 'position',
            'player.height': 'height',
            'player.weight': 'weight',
            'player.college': 'college',
            'player.country': 'country',
            'player.draft_year': 'draft_year',
            'player.draft_round': 'draft_round',
            'player.draft_number': 'draft_num'
        }
    )
    players = player_data.drop_duplicates()
    players = players.copy()
    players['player_name'] = players['first_name'] + ' ' + players['last_name']
    players = players[
        [
            "player_id",
            "player_name",
            "first_name",
            "last_name",
            "position",
            "height",
            "weight",
            "college",
            "country",
            "draft_year",
            "draft_round",
            "draft_num"
        ]
    ]

    return players


# function for saving the transformed games
def save_table(df, path):
    df.to_parquet(path, index=False)
    logging.info(f"Saved table to {path}")
    logging.info(f"Rows: {len(df)}")


# Run the code that gets the data

def main():
    # Get the raw data
    raw_games_df = extract_games_raw(SEASONS)
    raw_player_game_stats_df = extract_player_game_stats_raw(SEASONS)

    # Build the tables
    games_fact_df = build_games_fact(raw_games_df)
    teams_dim_df = build_teams_dim(raw_games_df)

    player_game_stats_fact_df = build_player_game_stats_fact(
        raw_player_game_stats_df
    )
    players_dim_df = build_players_dim(raw_player_game_stats_df)

    # Save the tables
    save_table(
        games_fact_df,
        PROCESSED_DATA_DIR/"games_fact.parquet"
    )
    save_table(
        teams_dim_df,
        PROCESSED_DATA_DIR/"teams_dim.parquet"
    )
    save_table(
        players_dim_df,
        PROCESSED_DATA_DIR/"players_dim.parquet"
    )
    save_table(
        player_game_stats_fact_df,
        PROCESSED_DATA_DIR/"player_game_stats_fact.parquet"
    )


if __name__ == "__main__":
    main()
