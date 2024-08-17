from data_processing import load_and_process_data
from feature_engineering import create_features
from model_training import train_model
from evaluation import save_results, calculate_fantasy_data_rmse
import logging

logger = logging.getLogger(__name__)

def main():
    path = "data/"
    train_df = load_and_process_data(path + "aggregated_2015.csv")
    test_df = load_and_process_data(path + "aggregated_2016.csv")
    logger.info("Data loaded and processed successfully")
    
    train_df, features = create_features(train_df)
    test_df, _ = create_features(test_df)
    logger.info("Features created successfully")

    positions = train_df["Pos"].unique()
    results = {}

    for position in positions:
        logger.info(f"Learning for Position {position} ...")
        results[position] = train_model(train_df, test_df, features, position)

    save_results(results, "rmse.csv")
    calculate_fantasy_data_rmse(test_df)
    logger.info("Results saved to rmse.csv and FantasyData_rmse.csv")
    logger.info("Program finished normally")


if __name__ == "__main__":
    main()
