import pandas as pd
import ast

def calculate_fantasy_rmse(df: pd.DataFrame) -> pd.Series:
    df["diff"] = (df["proj"] - df["FD points"]) ** 2.0
    FantasyData_rmse = df.groupby(["Pos"])["diff"].mean() ** 0.5
    return FantasyData_rmse

def calculate_position_rmse(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    
    new_data = {
        "Model": [],
        "PK": [],
        "QB": [],
        "WR": [],
        "TE": [],
        "RB": []
    }
    
    for position in df.index:
        for model in df.columns:
            rmse_dict = ast.literal_eval(df.at[position, model])

            for key, rmse_type in rmse_dict.items():
                row_name = f"{model}_{key}"
                if row_name not in new_data["Model"]:
                    new_data["Model"].append(row_name)

                new_data[position].append(rmse_type)
                
    new_df = pd.DataFrame(new_data)
    new_df.set_index("Model", inplace=True)

    return new_df

