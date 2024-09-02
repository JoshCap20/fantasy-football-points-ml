"""
Utils
"""

from config import DEBUG
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
