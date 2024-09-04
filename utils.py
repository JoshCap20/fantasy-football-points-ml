"""
Utils

Provides utility functions for logging and signal handling.
"""

import sys
import signal
import logging

from config import DEBUG


def get_logger(name: str, log_file: str = "debug.log") -> logging.Logger:
    level = logging.DEBUG if DEBUG else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    if DEBUG:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger


# Signal handlers
def handle_sigint(signal, frame):
    logger = get_logger(__name__)
    logger.info("\nSIGINT received. Terminating model training.")
    sys.exit(0)


def handle_sigtstp(signal, frame):
    logger = get_logger(__name__)
    logger.info("\nSIGTSTP received. Terminating model training.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTSTP, handle_sigtstp)
