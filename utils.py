"""
Utils
"""

from config import DEBUG
import pandas as pd
import logging


def get_logger(name: str) -> logging.Logger:
    level = logging.DEBUG if DEBUG else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


class OutputManager:
    OUTPUT_DIRECTORY = "output/"

    @classmethod
    def save_results_from_dictionary(
        cls, results: dict[str, dict[str, float]], filename: str
    ) -> None:
        df_rmse = pd.DataFrame(results).T
        cls.save_results_from_dataframe(df_rmse, filename)

    @classmethod
    def save_results_from_dataframe(
        cls, df: pd.DataFrame, filename: str, header: bool = True, index: bool = True
    ) -> None:
        df.to_csv(f"{cls.OUTPUT_DIRECTORY}/{filename}", header=header, index=index)
