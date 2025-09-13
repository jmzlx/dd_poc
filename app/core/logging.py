#!/usr/bin/env python3
"""
Logging Configuration Module

Provides consistent logging setup for the application.
This replaces the old src-based logging with a cleaner, app-specific solution.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def configure_langchain_logging(log_level: str = "WARNING") -> None:
    """
    Configure LangChain library logging levels to reduce verbosity.

    Args:
        log_level: Logging level for LangChain modules (default: WARNING)
    """
    langchain_modules = [
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_huggingface"
    ]

    level = getattr(logging, log_level.upper())
    for module in langchain_modules:
        logging.getLogger(module).setLevel(level)


def setup_logging(
    name: str = "dd_poc",
    log_level: str = "INFO",
    log_file: str = None
) -> logging.Logger:
    """
    Set up standard Python logging with rotating file handler

    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate setup if logger already has handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (if possible)
    if log_file or True:  # Always try to set up file logging
        try:
            log_dir = Path(".logs")
            log_dir.mkdir(exist_ok=True)

            if not log_file:
                log_file = log_dir / f"dd_poc_{Path.cwd().name}.log"

            # Use RotatingFileHandler for better log management
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            # File logging not available (e.g., on Streamlit Cloud)
            pass

    return logger


# Global logger instance
logger = setup_logging()
