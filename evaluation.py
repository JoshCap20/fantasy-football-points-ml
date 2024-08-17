import pandas as pd


def calculate_fantasy_rmse(df: pd.DataFrame) -> pd.Series:
    df["diff"] = (df["proj"] - df["FD points"]) ** 2.0
    FantasyData_rmse = df.groupby(["Pos"])["diff"].mean() ** 0.5
    return FantasyData_rmse


def calculate_position_rmse(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results).T

    new_data = {"Model": [], "PK": [], "QB": [], "WR": [], "TE": [], "RB": []}
    position_data = {
        position: [] for position in new_data.keys() if position != "Model"
    }

    for position in df.index:
        for model in df.columns:
            rmse_dict = df.at[position, model]

            for key, rmse_value in rmse_dict.items():
                row_name = f"{model}_{key}"
                if row_name not in new_data["Model"]:
                    new_data["Model"].append(row_name)

                position_data[position].append(rmse_value)

    for position, values in position_data.items():
        new_data[position] = values

    new_df = pd.DataFrame(new_data)
    new_df.set_index("Model", inplace=True)

    return new_df
