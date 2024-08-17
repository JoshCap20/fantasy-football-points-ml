import pandas as pd
from logger import get_logger

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
