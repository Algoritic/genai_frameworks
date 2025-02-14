import logging
from core.settings import app_settings


def setup_logger():
    logger = logging.getLogger("LLMFramework")
    logger.setLevel(getattr(logging, app_settings.logger.log_level))

    # File Handler
    file_handler = logging.FileHandler(app_settings.logger.log_file)
    file_handler.setLevel(getattr(logging, app_settings.logger.log_level))

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
