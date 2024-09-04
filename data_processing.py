"""
Data Processing Module

Contains data loading and null value handling methods as described separately below.
"""

import os
import pandas as pd
from sklearn.impute import SimpleImputer

from utils import get_logger
from data_scraper import get_weekly_data, get_season_schedule

logger = get_logger(__name__)

"""
Methods for loading and processing data.
"""


def load_data(years: list[int], filepath: str = "data") -> pd.DataFrame:
    """
    Loads and merges weekly and schedule data for the given years.
    Data is scraped and saved to a file which acts as a sort of cache for perfomance speedup.

    Parameters:
    - years: List of years to load data for
    - filepath: Path to the data directory

    Returns:
    - DataFrame with merged data
    """
    all_data = []

    for year in years:
        weekly_df = load_weekly_data(year, filepath)
        schedule_df = load_schedule_data(year, filepath)

        merged_df = pd.merge(
            weekly_df,
            schedule_df,
            left_on=["season", "week", "recent_team"],
            right_on=["season", "week", "home_team"],
        )
        all_data.append(merged_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df.copy()


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


"""
Methods for dealing with null values in the data.

All are pure methods and return a new DataFrame. The original df is not modified.
"""


def drop_missing_values(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Drops rows with missing values in the specified columns.
    """
    logger.debug(f"Dropping missing values")
    df = df.copy()
    df = df.dropna(subset=feature_columns)
    return df


def impute_missing_values_with_mean(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Imputes missing values with the mean of the column.
    """
    logger.debug(f"Imputing missing values with mean")
    df = df.copy()
    imputer = SimpleImputer(strategy="mean")
    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    return df.copy()


def impute_missing_values_with_median(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Imputes missing values with the median of the column.
    """
    logger.debug(f"Imputing missing values with median")
    df = df.copy()
    imputer = SimpleImputer(strategy="median")
    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    return df.copy()


def impute_missing_values_with_zero(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Imputes missing values with zero.
    """
    logger.debug(f"Imputing missing values with zero")
    df = df.copy()
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    return df


def impute_missing_values_with_mode(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Imputes missing values with the mode of the column.
    """
    logger.debug(f"Imputing missing values with mode")
    df = df.copy()
    imputer = SimpleImputer(strategy="most_frequent")
    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    return df
