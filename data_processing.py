"""
Data is scraped and saved to a file which acts as a sort of cache for perfomance speedup.
"""

import pandas as pd
from utils import get_logger
from sklearn.impute import SimpleImputer
from data_scraper import get_weekly_data, get_season_schedule
from config import POSITIONS
import os

logger = get_logger(__name__)


def load_and_process_data(file_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        logger.error(f"File {file_name} not found.")
        raise FileNotFoundError

    df.sort_values(by=["playerID", "weeks"], inplace=True)
    df.fillna(0, inplace=True)
    return df


def load_weekly_data(year: int, filepath: str) -> pd.DataFrame:
    file_path = f"{filepath}/weekly_{year}.csv"
    if os.path.exists(file_path):
        logger.debug(f"Loading weekly data for {year} from local cache.")
        return pd.read_csv(file_path).copy()
    else:
        logger.debug(f"Downloading weekly data for {year}.")
        weekly_df = get_weekly_data([year])
        weekly_df.to_csv(file_path, index=False)
        return weekly_df.copy()


def load_schedule_data(year: int, filepath: str) -> pd.DataFrame:
    file_path = f"{filepath}/schedule_{year}.csv"
    if os.path.exists(file_path):
        logger.debug(f"Loading schedule data for {year} from local cache.")
        return pd.read_csv(file_path).copy()
    else:
        logger.debug(f"Downloading schedule data for {year}.")
        schedule_df = get_season_schedule([year])
        schedule_df.to_csv(file_path, index=False)
        return schedule_df.copy()


def load_data(years: list[int], filepath: str = "data") -> pd.DataFrame:
    all_data = []

    for year in years:
        weekly_df = load_weekly_data(year, filepath)
        schedule_df = load_schedule_data(year, filepath)

        weekly_df = weekly_df[weekly_df["position"].isin(POSITIONS)]

        merged_df = pd.merge(
            weekly_df,
            schedule_df,
            left_on=["season", "week", "recent_team"],
            right_on=["season", "week", "home_team"],
        )
        all_data.append(merged_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df.copy()


def drop_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    logger.debug(f"Dropping missing values from features: {features}")
    df = df.dropna(subset=features)
    return df


def impute_missing_values_with_mean(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    logger.debug(f"Imputing missing values with mean for features: {features}")
    imputer = SimpleImputer(strategy="mean")
    df[features] = imputer.fit_transform(df[features])
    return df


def impute_missing_values_with_median(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    logger.debug(f"Imputing missing values with median for features: {features}")
    imputer = SimpleImputer(strategy="median")
    df[features] = imputer.fit_transform(df[features])
    return df


def impute_missing_values_with_zero(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    logger.debug(f"Imputing missing values with zero for features: {features}")
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    df[features] = imputer.fit_transform(df[features])
    return df
