import logging
from logging import handlers
import os


def setup_logger():
    logger = logging.getLogger('LLMFrameworkX')
    logger.setLevel(
        logging.DEBUG
    )  # Set the root logger level to DEBUG (so all levels are captured)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a directory for logs if it doesn't exist
    log_path = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # FileHandler for debug.log (Debug messages)
    debug_log_file = os.path.join(log_path, "debug.log")
    debug_handler = handlers.TimedRotatingFileHandler(debug_log_file,
                                                      when='D',
                                                      interval=1,
                                                      backupCount=7)
    debug_handler.setLevel(logging.DEBUG)  # Capture debug-level messages
    debug_handler.setFormatter(formatter)

    # FileHandler for app.log (Info messages)
    app_log_file = os.path.join(log_path, "app.log")
    app_handler = logging.FileHandler(app_log_file)
    app_handler.setLevel(logging.INFO)  # Capture info-level messages and above
    app_handler.setFormatter(formatter)

    # FileHandler for error.log (Error messages)
    error_log_file = os.path.join(log_path, "error.log")
    error_handler = handlers.RotatingFileHandler(error_log_file,
                                                 maxBytes=5000,
                                                 backupCount=3)
    error_handler.setLevel(
        logging.ERROR)  # Capture error-level messages and above
    error_handler.setFormatter(formatter)

    # Console Handler (Info and higher messages)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.INFO)  # Capture info-level messages for the console
    console_handler.setFormatter(formatter)

    # Adding handlers to the logger
    if not logger.handlers:
        logger.addHandler(debug_handler)
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)

    return logger


# Initialize the logger
logger = setup_logger()
