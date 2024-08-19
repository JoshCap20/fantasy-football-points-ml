import pandas as pd
from utils import get_logger

logger = get_logger(__name__)


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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features here as needed, big model perfomance boost from these so far.
    """
    # Interaction Features
    df["passing_yards_per_attempt"] = df["passing_yards"] / df["attempts"]
    df["rushing_yards_per_carry"] = df["rushing_yards"] / df["carries"]
    df["receiving_yards_per_target"] = df["receiving_yards"] / df["targets"]
    df["receiving_yards_per_reception"] = df["receiving_yards"] / df["receptions"]

    # Polynomial Features
    df["passing_yards_squared"] = df["passing_yards"] ** 2
    df["rushing_yards_squared"] = df["rushing_yards"] ** 2

    # Lag Features
    df["fantasy_points_lag1"] = df.groupby("player_id")["fantasy_points"].shift(1)
    df["fantasy_points_lag2"] = df.groupby("player_id")["fantasy_points"].shift(2)

    # Since these are kept in the model, only add non-target features to feature_names
    feature_names = [
        "fantasy_points_lag1",
        "fantasy_points_lag2",
    ]

    return df, feature_names


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df, feature_names = add_features(df)

    # These stats are averaged over the season, last 4 weeks, and previous week
    stats_to_average = [
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
        "passing_yards_per_attempt",
        "rushing_yards_per_carry",
        "passing_yards_squared",
        "rushing_yards_squared",
        "receiving_yards_per_target",
        "receiving_yards_per_reception",
    ]

    df, game_features = get_game_char_indicators(df)
    df, player_features = get_player_averages(df, stats_to_average)

    # df.to_csv("data/processed_data.csv", index=False)  # for debugging
    logger.debug(game_features + player_features + feature_names)

    return df, game_features + player_features + feature_names
