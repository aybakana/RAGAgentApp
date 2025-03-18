"""
This module configures a logging system for the RAG App.
The logging configuration is read from config/settings.py.
It supports file logging, console logging, and error-level logging.
"""

import logging
import os
from config import settings

# Create a logger for the application
logger = logging.getLogger("RAGApp")
logger.setLevel(settings.LOG_LEVEL)

# Formatter including timestamp, source file, line number, and log level
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(settings.CONSOLE_LOG_LEVEL)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for persistent logging
if settings.LOG_TO_FILE:
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
    file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
    file_handler.setLevel(settings.FILE_LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Optional additional handler for critical errors
if settings.ENABLE_ERROR_LOGGING:
    error_handler = logging.FileHandler(settings.ERROR_LOG_FILE_PATH)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

def get_logger(name: str = "RAGApp"):
    """
    Returns a child logger with the specified name.
    This helps in contextual logging (e.g., from RAGAgent, GUI components, etc.)
    
    Parameters:
        name (str): The name of the requested logger.
    
    Returns:
        logging.Logger: A configured logger instance.
    """
    return logger.getChild(name)