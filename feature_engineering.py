import pandas as pd


def get_game_char_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["home"] = (df["recent_team"] == df["home_team"]).astype(int)

    # Create one-hot encoding for the opponent and team
    df = pd.concat([df, pd.get_dummies(df["opponent_team"], prefix="Oppt")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["recent_team"], prefix="Team")], axis=1)

    game_features = (
        ["home"]
        + list(df.filter(regex="^Oppt_").columns)
        + list(df.filter(regex="^Team_").columns)
    )

    return df, game_features


def rolling_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(min_periods=1, window=window).mean().shift(1)


def get_player_averages(
    df: pd.DataFrame, stats: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Estimate player averages for all stats and fantasy point histories,
    for season-to-date, last 4 weeks, and previous week
    """
    feature_names = []
    for stat in stats:
        season_stat = (
            df.groupby("player_id")[stat]
            .apply(lambda x: rolling_average(x, 16))
            .reset_index(level=0, drop=True)
        )
        recent_stat = (
            df.groupby("player_id")[stat]
            .apply(lambda x: rolling_average(x, 4))
            .reset_index(level=0, drop=True)
        )
        prev_stat = (
            df.groupby("player_id")[stat]
            .apply(lambda x: rolling_average(x, 1))
            .reset_index(level=0, drop=True)
        )

        df[f"season_{stat}"] = season_stat
        df[f"recent_{stat}"] = recent_stat
        df[f"prev_{stat}"] = prev_stat

        feature_names.extend([f"season_{stat}", f"recent_{stat}", f"prev_{stat}"])

    return df, feature_names


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    stats = [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "special_teams_tds",
        "rushing_fumbles",
        "receiving_fumbles",
        "sack_fumbles",
    ]
    
    df, game_features = get_game_char_indicators(df)
    df, player_features = get_player_averages(df, stats)
    
    df.to_csv("data/processed_data.csv", index=False)
    print(game_features + player_features)
    return df, game_features + player_features
