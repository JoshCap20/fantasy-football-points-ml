"""
Entry point.

Abstracts data cleaning, data processing, feature engineering, model training, and model evaluation.
"""

from data_processing import load_data
from feature_engineering import create_features
from model_training import train_models
from utils import get_logger
from config import TRAIN_YEARS, TEST_YEARS
from analyze import run_analysis

logger = get_logger(__name__)


def main():
    train_df = load_data(TRAIN_YEARS)
    test_df = load_data(TEST_YEARS)
    logger.info("Data loaded and processed successfully")

    train_df, features = create_features(train_df)
    test_df, _ = create_features(test_df)
    logger.info("Features created successfully")

    train_models(train_df, test_df, features)
    logger.info("Program finished normally")
    
    run_analysis()
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
