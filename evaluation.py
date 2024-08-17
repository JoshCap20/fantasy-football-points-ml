import pandas as pd


def calculate_fantasy_rmse(df: pd.DataFrame) -> pd.Series:
    df["diff"] = (df["proj"] - df["FD points"]) ** 2.0
    FantasyData_rmse = df.groupby(["Pos"])["diff"].mean() ** 0.5
    return FantasyData_rmse


def calculate_position_rmse(results: dict) -> pd.DataFrame:
    new_data = {"Model": [], "PK": [], "QB": [], "RB": [], "TE": [], "WR": []}

    for position in results.keys():
        for model in results[position].keys():
            for metric, value in results[position][model].items():
                row_name = f"{model}_{metric}"
                if row_name not in new_data["Model"]:
                    new_data["Model"].append(row_name)
                    for pos in ["PK", "QB", "RB", "TE", "WR"]:
                        new_data[pos].append(float("nan"))
                idx = new_data["Model"].index(row_name)
                new_data[position][idx] = value

    new_df = pd.DataFrame(new_data)
    new_df.set_index("Model", inplace=True)

    return new_df.T
