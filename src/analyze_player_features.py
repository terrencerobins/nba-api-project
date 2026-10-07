import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_min_minutes(player_stats, season, minutes):
    season_stats = player_stats[player_stats['season'] == season]

    return (
        season_stats
        .groupby(['player_id', 'player_name'])
        .filter(lambda df: df['minutes_played'].sum() >= minutes)
    )


def get_volatile_players_for_season(filtered_stats):
    return (
        filtered_stats
        .groupby(["player_id", "player_name"])["pts"]
        .std()
        .reset_index(name="std_dev")
        .sort_values("std_dev", ascending=False)
    )


def main():
    player_features = pd.read_parquet(PROCESSED_DATA_DIR /"player_features.parquet")

    filtered_stats = get_min_minutes(player_features, 2025, 1300)
    volatile_players = get_volatile_players_for_season(filtered_stats)
    print(volatile_players.head())


if __name__ == "__main__":
    main()
