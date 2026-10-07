import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

games_fact_df = pd.read_parquet(PROCESSED_DATA_DIR / "games_fact.parquet")
player_games_fact_df = pd.read_parquet(PROCESSED_DATA_DIR / "player_game_stats_fact.parquet")
team_dim_df = pd.read_parquet(PROCESSED_DATA_DIR / "teams_dim.parquet")
player_dim_df = pd.read_parquet(PROCESSED_DATA_DIR / "players_dim.parquet")

# Data Quality Checks

# dupe check on player -> game records
duplicate_player_game_stats = (
    player_games_fact_df['player_game_stat_id']
    .duplicated()
    .sum()
)
if duplicate_player_game_stats > 0:
    raise ValueError(
        f"Duplicate player_game_stat_id records found: {duplicate_player_game_stats}"
    )
print("PASS: player_game_stat_id is unique")

# null check on keys
null_checker = (
    player_games_fact_df[['player_id', 'game_id', 'team_id']]
    .isnull()
    .sum()
)
if (null_checker > 0).any():
    raise ValueError(
        f"Null records found:\n{null_checker}"
    )
print("PASS: null_checker contains no null values")

# checking dim tables
duplicate_players = (
    player_dim_df['player_id']
    .duplicated()
    .sum()
)
if duplicate_players > 0:
    raise ValueError(
        f"Duplicate player records found: {duplicate_players}"
    )
print("PASS: player_id is unique in players_dim")

duplicate_teams = (
    team_dim_df['team_id']
    .duplicated()
    .sum()
)
if duplicate_teams > 0:
    raise ValueError(
        f"Duplicate team records found: {duplicate_teams}"
    )
print("PASS: team_id is unique in teams_dim")

# checking join integrity

merged = player_games_fact_df.merge(player_dim_df[['player_id']], on='player_id', how='left', indicator=True)

unmatched_players = (
    (merged['_merge'] == 'left_only')
    .sum()
)
if unmatched_players > 0:
    raise ValueError(
        f"Player IDs without a matching players_dim record: {unmatched_players}"
    )
print("PASS: all player IDs match players_dim")

# checking team join integrity

team_merge = player_games_fact_df.merge(
    team_dim_df[['team_id']],
    on='team_id',
    how='left',
    indicator=True
)

unmatched_teams = (
    (team_merge['_merge'] == 'left_only')
    .sum()
)

if unmatched_teams > 0:
    raise ValueError(
        f"Team IDs without a matching teams_dim record: {unmatched_teams}"
    )

print("PASS: all team IDs match teams_dim")

# checking game join integrity

game_merge = player_games_fact_df.merge(
    games_fact_df[['id']],
    left_on='game_id',
    right_on='id',
    how='left',
    indicator=True
)

unmatched_games = (
    (game_merge['_merge'] == 'left_only')
    .sum()
)

if unmatched_games > 0:
    raise ValueError(
        f"Game IDs without a matching games_fact record: {unmatched_games}"
    )

print("PASS: all game IDs match games_fact")