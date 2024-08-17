import pandas as pd


def save_results(results: dict[str, dict[str, float]], filename: str) -> None:
    df_rmse = pd.DataFrame(results).T
    df_rmse.to_csv(filename, header=True, index=True)


def calculate_fantasy_data_rmse(df: pd.DataFrame) -> pd.Series:
    df["diff"] = (df["proj"] - df["FD points"]) ** 2.0
    FantasyData_rmse = df.groupby(["Pos"])["diff"].mean() ** 0.5
    FantasyData_rmse.to_csv("FantasyData_rmse.csv", header=True, index=True)
    return FantasyData_rmse
