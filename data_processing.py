import pandas as pd


def load_and_process_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df.sort_values(by=["playerID", "weeks"], inplace=True)
    df.fillna(0, inplace=True)
    return df