import pandas as pd


def calculate_fantasy_data_rmse(df: pd.DataFrame) -> pd.Series:
    df["diff"] = (df["proj"] - df["FD points"]) ** 2.0
    FantasyData_rmse = df.groupby(["Pos"])["diff"].mean() ** 0.5
    return FantasyData_rmse


class OutputManager:
    OUTPUT_DIRECTORY = "results/"

    @classmethod
    def save_results_from_dictionary(
        cls, results: dict[str, dict[str, float]], filename: str
    ) -> None:
        df_rmse = pd.DataFrame(results).T
        cls._save_dataframe(df_rmse, filename)

    @classmethod
    def save_results_from_dataframe(
        cls, df: pd.DataFrame, filename: str, header: bool = True, index: bool = True
    ) -> None:
        df.to_csv(f"{cls.OUTPUT_DIRECTORY}/{filename}", header=header, index=index)
