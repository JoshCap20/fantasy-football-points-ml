import pandas as pd


def save_results(results: dict[str, dict[str, float]], filename: str) -> None:
    df_rmse = pd.DataFrame(results).T
    df_rmse.to_csv(filename, header=True, index=True)


def calculate_fantasy_data_rmse(test_df: pd.DataFrame) -> pd.Series:
    test_df["diff"] = (test_df["proj"] - test_df["FD points"]) ** 2.0
    FantasyData_rmse = test_df.groupby(["Pos"])["diff"].mean() ** 0.5
    FantasyData_rmse.to_csv("FantasyData_rmse.csv", header=True, index=True)
    return FantasyData_rmse
