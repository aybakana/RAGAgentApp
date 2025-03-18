"""
This module defines custom exception classes for the RAG App as well as
error handling utilities for formatting error messages for end users.
"""
import logging  # Import the logging module

class DocumentLoadingError(Exception):
    def __init__(self, message="An error occurred while loading the document.", details=None):
        self.details = details
        super().__init__(message)

class ModelInitializationError(Exception):
    def __init__(self, message="An error occurred during model initialization.", details=None):
        self.details = details
        super().__init__(message)

class QueryProcessingError(Exception):
    def __init__(self, message="An error occurred while processing the query.", details=None):
        self.details = details
        super().__init__(message)

class FileHandlingError(Exception):
    def __init__(self, message="An error occurred while handling the file.", details=None):
        self.details = details
        super().__init__(message)


class ErrorHandler:
    """
    A centralized error handler class to manage errors in your application.
    """

    def __init__(self, log_level=logging.WARNING):
        """
        Initializes the ErrorHandler with a specified log level.
        """
        self.logger = logging.getLogger(__name__)  # Get a logger for this module
        self.logger.setLevel(log_level) # Set the overall log level

    def handle_document_load_error(self, message, context=None):
        """Handles DocumentLoadingError."""
        self.logger.error(f"DocumentLoadError: {message}", extra={'context': context})
        return f"Document load failed: {message}"  # User-friendly message

    def handle_model_init_error(self, message, context=None):
        """Handles ModelInitializationError."""
        self.logger.error(f"ModelInitError: {message}", extra={'context': context})
        return f"Model initialization failed: {message}"

    def handle_query_processing_error(self, message, context=None):
        """Handles QueryProcessingError."""
        self.logger.error(f"QueryProcessError: {message}", extra={'context': context})
        return f"Query processing failed: {message}"

    def handle_file_handling_error(self, message, context=None):
        """Handles FileHandlingError."""
        self.logger.error(f"FileHandleError: {message}", extra={'context': context})
        return f"File operation failed: {message}"


    def log_error(self, category, message, context=None):
        """
        Logs an error with a specified category and context.
        """
        if category == "critical":
            self.logger.critical(message, extra={'context': context})
        elif category == "warning":
            self.logger.warning(message, extra={'context': context})
        else:  # Default to warning if category is not recognized
            self.logger.error(message, extra={'context': context})

    def format_error_message(self, category, message, context=None):
        """Formats error messages for different categories."""
        if category == "critical":
            return f"CRITICAL ERROR: {message}. Context: {context}"
        elif category == "warning":
            return f"WARNING: {message}. Context: {context}"
        else:
            return f"{category.capitalize()} ERROR: {message}. Context: {context}"        

def format_error_message(error, friendly_template="An unexpected error has occurred."):
    """
    Formats an exception message with additional context.
    
    Parameters:
        error (Exception): The caught exception.
        friendly_template (str): A user-friendly error message template.
    
    Returns:
        str: A formatted error message for end users.
    """
    error_message = f"{friendly_template}"
    # Include error type and additional details if available
    error_type = type(error).__name__
    detail_info = f" Details: {error.details}" if hasattr(error, "details") and error.details else ""
    formatted_message = f"[{error_type}] {error_message}{detail_info}"
    return formatted_message

def handle_exception(error, logger, friendly_template="An unexpected error has occurred."):
    """
    Utility function to handle exceptions gracefully by logging and returning
    a user-friendly message.
    
    Parameters:
        error (Exception): The caught exception.
        logger (logging.Logger): Logger instance for logging error details.
        friendly_template (str): Customizable error message to show to users.
    
    Returns:
        str: A user-friendly error message.
    """
    formatted_message = format_error_message(error, friendly_template)
    # Log the error with exception traceback
    logger.error(formatted_message, exc_info=True)
    return formatted_message