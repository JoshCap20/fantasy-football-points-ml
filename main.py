"""
Entry point.

Assumes data aggregation already and files data/aggregated_2015.csv and data/aggregated_2016.csv exist.
"""

from data_processing import load_and_process_data, load_data
from feature_engineering import create_features
from model_training import train_model
from evaluation import calculate_position_rmse
from utils import get_logger, OutputManager
from config import POSITIONS, TRAIN_YEARS, TEST_YEARS

logger = get_logger(__name__)

def main():
    train_df = load_data(TRAIN_YEARS)
    test_df = load_data(TEST_YEARS)
    logger.info("Data loaded and processed successfully")

    train_df, features = create_features(train_df)
    test_df, _ = create_features(test_df)
    logger.info("Features created successfully")

    all_results = {}

    for position in POSITIONS:
        logger.info(f"Learning for Position {position} ...")
        all_results[position] = train_model(train_df, test_df, features, position)

    OutputManager.save_results_from_dictionary(
        calculate_position_rmse(all_results), "position_rmse.csv"
    )
    logger.info("Results saved successfully")

    logger.info("Program finished normally")


if __name__ == "__main__":
    main()
