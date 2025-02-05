import logging
from core.config import LOG_FILE, LOG_LEVEL


def setup_logger():
    logger = logging.getLogger("LLMFramework")
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
