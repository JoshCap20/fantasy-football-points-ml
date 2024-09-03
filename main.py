"""
Entry point.

Abstracts data cleaning, data processing, feature engineering, model training, and model evaluation.
"""

from data_processing import load_data
from feature_engineering import create_feature_columns
from model_training import train_models
from utils import get_logger
from config import TRAIN_YEARS, TEST_YEARS
from analyze import run_analysis

logger = get_logger(__name__)


def main():
    train_df = load_data(TRAIN_YEARS)
    test_df = load_data(TEST_YEARS)
    logger.info("Data loaded and processed successfully")

    train_df, feature_columns = create_feature_columns(train_df)
    test_df, _ = create_feature_columns(test_df)
    logger.info("Feature engineering finished normally")

    results_df, output_filepath = train_models(train_df, test_df, feature_columns)
    logger.info("Model training finished normally")

    run_analysis(df=results_df, path=output_filepath)
    logger.info("Model perfomance analysis finished normally")


if __name__ == "__main__":
    main()
