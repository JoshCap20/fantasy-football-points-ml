import pandas as pd


def get_game_char_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["home"] = (df["h/a"] == "h").astype(int)
    df = pd.concat([df, pd.get_dummies(df["Oppt"], prefix="Oppt")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["Team"], prefix="Team")], axis=1)
    game_features = (
        ["home"] + list(df.filter(regex="^Oppt_")) + list(df.filter(regex="^Team_"))
    )
    return df, game_features


def rolling_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(min_periods=1, window=window).mean().shift(1)


def get_player_averages(
    df: pd.DataFrame, stats: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Estimate player averages for all stats and FanDuel point histories,
    for season-to-date, last 4 weeks, and previous week
    """
    feature_names = []
    for stat in stats:
        season_stat = (
            df.groupby("playerID")[stat]
            .apply(lambda x: rolling_average(x, 16))
            .reset_index(level=0, drop=True)
        )
        recent_stat = (
            df.groupby("playerID")[stat]
            .apply(lambda x: rolling_average(x, 4))
            .reset_index(level=0, drop=True)
        )
        prev_stat = (
            df.groupby("playerID")[stat]
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
        "pass.att",
        "pass.comp",
        "passyds",
        "pass.tds",
        "pass.ints",
        "rush.att",
        "rushyds",
        "rushtds",
        "recept",
        "recyds",
        "rec.tds",
        "kick.rets",
        "punt.rets",
        "fgm",
        "xpmade",
        "totalfumbs",
    ]
    df, game_features = get_game_char_indicators(df)
    df, player_features = get_player_averages(df, stats)
    return df, game_features + player_features
