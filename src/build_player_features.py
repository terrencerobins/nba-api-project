import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_base_player_games(
        player_games_fact_df,
        games_fact_df,
        player_dim_df
):
    base = player_games_fact_df.merge(
        games_fact_df[['id', 'date', 'season', 'postseason']],
        how='left',
        left_on='game_id',
        right_on='id'
    )
    base = base.drop(columns=['id'])

    base = base.merge(
        player_dim_df[['player_id', 'player_name']],
        on='player_id',
        how='left'
    )

    base = base[
        (base['min'] != '00') &
        (~base['postseason'])
        ].copy()

    base['minutes_played'] = pd.to_numeric(
        base['min'],
        errors='coerce'
    )

    return base


# sorting the dataframe to then be able to , by player, look back at the prev games stat
def add_previous_game_features(base):
    base = base.sort_values(['player_id', 'date']).copy()
    base['pts_prev_game'] = base.groupby(['player_id', 'season'])['pts'].shift(1)
    base['ast_prev_game'] = base.groupby(['player_id', 'season'])['ast'].shift(1)
    base['reb_prev_game'] = base.groupby(['player_id', 'season'])['reb'].shift(1)
    base['dreb_prev_game'] = base.groupby(['player_id', 'season'])['dreb'].shift(1)
    base['oreb_prev_game'] = base.groupby(['player_id', 'season'])['oreb'].shift(1)
    base['blk_prev_game'] = base.groupby(['player_id', 'season'])['blk'].shift(1)
    base['stl_prev_game'] = base.groupby(['player_id', 'season'])['stl'].shift(1)
    base['fga_prev_game'] = base.groupby(['player_id', 'season'])['fga'].shift(1)
    base['fgm_prev_game'] = base.groupby(['player_id', 'season'])['fgm'].shift(1)
    base['fg3a_prev_game'] = base.groupby(['player_id', 'season'])['fg3a'].shift(1)
    base['fg3m_prev_game'] = base.groupby(['player_id', 'season'])['fg3m'].shift(1)
    base['fta_prev_game'] = base.groupby(['player_id', 'season'])['fta'].shift(1)
    base['ftm_prev_game'] = base.groupby(['player_id', 'season'])['ftm'].shift(1)
    return base


# adding games played so far this season
def add_rolling_features(base):
    base = base.copy()

    # Number of prior games played within each player-season
    base['games_so_far'] = (
        base.groupby(['player_id', 'season'])
        .cumcount()
    )

    # Five-game baseline BEFORE the previous game.
    # shift(2) excludes both:
    #   1. the current game
    #   2. the previous game being compared against the baseline
    base['pts_avg_L5'] = (
        base.groupby(['player_id', 'season'])['pts']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['ast_avg_L5'] = (
        base.groupby(['player_id', 'season'])['ast']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['reb_avg_L5'] = (
        base.groupby(['player_id', 'season'])['reb']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['dreb_avg_L5'] = (
        base.groupby(['player_id', 'season'])['dreb']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['oreb_avg_L5'] = (
        base.groupby(['player_id', 'season'])['oreb']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['fga_avg_L5'] = (
        base.groupby(['player_id', 'season'])['fga']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['fgm_avg_L5'] = (
        base.groupby(['player_id', 'season'])['fgm']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['fg3a_avg_L5'] = (
        base.groupby(['player_id', 'season'])['fg3a']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['fg3m_avg_L5'] = (
        base.groupby(['player_id', 'season'])['fg3m']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['fta_avg_L5'] = (
        base.groupby(['player_id', 'season'])['fta']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['ftm_avg_L5'] = (
        base.groupby(['player_id', 'season'])['ftm']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['blk_avg_L5'] = (
        base.groupby(['player_id', 'season'])['blk']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    base['stl_avg_L5'] = (
        base.groupby(['player_id', 'season'])['stl']
        .transform(lambda s: s.shift(2).rolling(5).mean())
    )

    return base


def add_vs_l5_features(base):
    base = base.copy()
    # prev game to prev 5 game avg comparisons
    base['pts_vs_L5'] = base['pts_prev_game'] - base['pts_avg_L5']
    base['ast_vs_L5'] = base['ast_prev_game'] - base['ast_avg_L5']
    base['reb_vs_L5'] = base['reb_prev_game'] - base['reb_avg_L5']
    base['dreb_vs_L5'] = base['dreb_prev_game'] - base['dreb_avg_L5']
    base['oreb_vs_L5'] = base['oreb_prev_game'] - base['oreb_avg_L5']
    base['fga_vs_L5'] = base['fga_prev_game'] - base['fga_avg_L5']
    base['fgm_vs_L5'] = base['fgm_prev_game'] - base['fgm_avg_L5']
    base['fta_vs_L5'] = base['fta_prev_game'] - base['fta_avg_L5']
    base['ftm_vs_L5'] = base['ftm_prev_game'] - base['ftm_avg_L5']
    base['fg3a_vs_L5'] = base['fg3a_prev_game'] - base['fg3a_avg_L5']
    base['fg3m_vs_L5'] = base['fg3m_prev_game'] - base['fg3m_avg_L5']
    base['blk_vs_L5'] = base['blk_prev_game'] - base['blk_avg_L5']
    base['stl_vs_L5'] = base['stl_prev_game'] - base['stl_avg_L5']
    return base


def validate_player_features(base):
    min_games_so_far = (
        base.groupby(['player_id', 'season'])['games_so_far']
        .min()
    )
    if (min_games_so_far != 0).any():
        raise ValueError(
            "games_so_far does not start at 0 for every player-season"
        )
    print("PASS: games_so_far starts at 0 for every player-season")
    early_games = base[base['games_so_far'] < 6]

    if early_games['pts_avg_L5'].notna().any():
        raise ValueError(
            "pts_avg_L5 should be null when fewer than 6 prior games exist"
        )

    print("PASS: pts_avg_L5 is null before 6 prior games")
    eligible_games = base[base['games_so_far'] >= 6]

    if eligible_games['pts_avg_L5'].isna().any():
        raise ValueError(
            "pts_avg_L5 contains nulls when 6 or more prior games exist"
        )

    print("PASS: pts_avg_L5 is populated after 6 prior games")


def main():
    player_games_fact_df = pd.read_parquet(
        PROCESSED_DATA_DIR / "player_game_stats_fact.parquet"
    )

    games_fact_df = pd.read_parquet(
        PROCESSED_DATA_DIR / "games_fact.parquet"
    )

    player_dim_df = pd.read_parquet(
        PROCESSED_DATA_DIR / "players_dim.parquet"
    )

    base = build_base_player_games(
        player_games_fact_df,
        games_fact_df,
        player_dim_df
    )

    base = add_previous_game_features(base)
    base = add_rolling_features(base)
    base = add_vs_l5_features(base)

    validate_player_features(base)


if __name__ == "__main__":
    main()




