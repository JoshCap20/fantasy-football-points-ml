from data_processing import load_and_process_data
from feature_engineering import create_features
from model_training import train_model
from evaluation import save_results, calculate_fantasy_data_rmse


def main():
    path = "data/"
    train_df = load_and_process_data(path + "aggregated_2015.csv")
    test_df = load_and_process_data(path + "aggregated_2016.csv")

    train_df, features = create_features(train_df)
    test_df, _ = create_features(test_df)

    positions = train_df["Pos"].unique()
    results = {}

    for position in positions:
        print(f"Learning for Position {position} ...")
        results[position] = train_model(train_df, test_df, features, position)

    save_results(results, "rmse.csv")
    calculate_fantasy_data_rmse(test_df)
    print("Program finished normally")


if __name__ == "__main__":
    main()
