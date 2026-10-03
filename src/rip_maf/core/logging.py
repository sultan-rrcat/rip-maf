import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # up from core/
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging():
    logger = logging.getLogger()

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    log_format = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "app.log"), encoding="utf-8"
    )
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
