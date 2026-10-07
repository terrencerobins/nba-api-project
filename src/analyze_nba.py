import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# answering questions
def get_player_scoring_consistency(player_features, season, min_minutes):
    games = player_features.copy()
    return (
        games[games['season'] == season]
        .groupby(['player_id', 'player_name'])
        .filter(lambda df: df['minutes_played'].sum() >= min_minutes)
        .groupby(['player_id', 'player_name'])['pts_vs_L5']
        .apply(lambda x: x.abs().mean())
        .reset_index(name='avg_abs_pts_vs_L5')
        .sort_values('avg_abs_pts_vs_L5')
    )


# top 3pm per game
def get_top_3pm_per_game(player_features, season):
    games = player_features.copy()
    return (
        games[games['season'] == season]
        .groupby(['player_id', 'player_name'])['fg3m']
        .mean()
        .reset_index(name='fg3m')
        .sort_values('fg3m', ascending=False)
    )


# normalize home and away games into one row per team-game
def build_team_game_results(games_fact, teams_dim):
    games = games_fact.copy()

    games['visitor_point_diff'] = -games['home_point_diff']
    games['visitor_win'] = ~games['home_win']

    regular_season_games = games[~games['postseason']].copy()

    home_games = pd.merge(
        teams_dim[['team_id', 'team_name']],
        regular_season_games[['home_team.id', 'date', 'home_point_diff', 'home_win', 'season']],
        how='left',
        left_on='team_id',
        right_on='home_team.id'
    ).dropna()

    away_games = pd.merge(
        teams_dim[['team_id', 'team_name']],
        regular_season_games[['visitor_team.id', 'date', 'visitor_point_diff', 'visitor_win', 'season']],
        how='left',
        left_on='team_id',
        right_on='visitor_team.id'
    ).dropna()

    home_games = home_games.rename(columns={
        'home_point_diff': 'point_diff',
        'home_win': 'win_flag'
    })

    away_games = away_games.rename(columns={
        'visitor_point_diff': 'point_diff',
        'visitor_win': 'win_flag'
    })

    home_games = home_games[['team_id', 'team_name', 'date', 'season', 'point_diff', 'win_flag']]
    away_games = away_games[['team_id', 'team_name', 'date', 'season', 'point_diff', 'win_flag']]

    return pd.concat([home_games, away_games], ignore_index=True)


def get_highest_avg_win_margin(game_results, season):
    return (
        game_results[(game_results['win_flag'] == True) & (game_results['season'] == season)]
        .groupby('team_name')['point_diff']
        .mean()
        .reset_index(name='avg_win_margin')
        .sort_values('avg_win_margin', ascending=False)
    )


def get_b2b_results(game_results, season):
    back_to_back = game_results[game_results['season'] == season].copy()

    back_to_back = back_to_back.sort_values(by=['team_name', 'date'])

    back_to_back = back_to_back[['team_name', 'date', 'season', 'win_flag']]

    back_to_back['prev_game_date'] = back_to_back.groupby('team_name')['date'].shift(1)

    back_to_back['date_diff'] = back_to_back['date'] - back_to_back['prev_game_date']

    back_to_back['is_back_to_back'] = (back_to_back['date_diff'] == pd.Timedelta(days=1))

    b2b_results = (
        back_to_back[back_to_back['is_back_to_back']]
        .groupby('team_name')['win_flag']
        .agg(
            b2b_games='count',
            b2b_wins='sum'
        )
        .reset_index()
    )

    b2b_results['b2b_win_pct'] = (
            b2b_results['b2b_wins'] / b2b_results['b2b_games']
    )

    return b2b_results.sort_values('b2b_win_pct', ascending=False)

def main():
    player_features = pd.read_parquet(
        PROCESSED_DATA_DIR / "player_features.parquet"
    )

    teams_dim = pd.read_parquet(
        PROCESSED_DATA_DIR / "teams_dim.parquet"
    )

    games_fact = pd.read_parquet(
        PROCESSED_DATA_DIR / "games_fact.parquet"
    )

    game_results = build_team_game_results(
        games_fact,
        teams_dim
    )

    most_consistent = get_player_scoring_consistency(player_features, 2025, 1110)
    best_shooters = get_top_3pm_per_game(player_features, 2025)
    b2b = get_b2b_results(game_results, 2025)
    high_margin = get_highest_avg_win_margin(game_results, 2025)


if __name__ == "__main__":
    main()