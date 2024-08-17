import pandas as pd
from utils import get_logger
from sklearn.impute import SimpleImputer

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


def drop_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=features)
    return df


def impute_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    df[features] = imputer.fit_transform(df[features])
    return df
