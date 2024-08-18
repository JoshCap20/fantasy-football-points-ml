import pandas as pd
from config import POSITIONS


def calculate_position_rmse(results: dict) -> pd.DataFrame:
    new_data = {key: [] for key in ["Model"] + POSITIONS}

    for position in results.keys():
        for model in results[position].keys():
            for metric, value in results[position][model].items():
                row_name = f"{model}_{metric}"
                if row_name not in new_data["Model"]:
                    new_data["Model"].append(row_name)
                    for pos in POSITIONS:
                        new_data[pos].append(float("nan"))
                idx = new_data["Model"].index(row_name)
                new_data[position][idx] = value

    new_df = pd.DataFrame(new_data)
    new_df.set_index("Model", inplace=True)

    return new_df.T
