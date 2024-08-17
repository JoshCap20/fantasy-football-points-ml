import pandas as pd
from utils import get_logger
from sklearn.impute import SimpleImputer
from data_scraper import get_weekly_data, get_season_schedule
from config import POSITIONS
logging = get_logger(__name__)


def load_and_process_data(file_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error(f"File {file_name} not found.")
        raise FileNotFoundError

    df.sort_values(by=["playerID", "weeks"], inplace=True)
    df.fillna(0, inplace=True)
    return df


def load_data(years: list[int]) -> pd.DataFrame:
    weekly_df = get_weekly_data(years)
    schedule_df = get_season_schedule(years)
    
    weekly_df = weekly_df[weekly_df["position"].isin(POSITIONS)]
    
    return pd.merge(
        weekly_df,
        schedule_df,
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "home_team"],
    )


def drop_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=features)
    return df


def impute_missing_values_with_mean(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    df[features] = imputer.fit_transform(df[features])
    return df


def impute_missing_values_with_median(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    df[features] = imputer.fit_transform(df[features])
    return df


def impute_missing_values_with_zero(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    df[features] = imputer.fit_transform(df[features])
    return df
