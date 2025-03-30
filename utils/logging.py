"""
Logging utilities for Bayesian Adaptive LLM.

This module provides logging configuration and helper functions.
"""

import logging
import os
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, logs are written to stdout.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format: Format string for log messages.
        
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger('bayesian_adaptive_llm')
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create handler
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler
        handler = logging.FileHandler(log_file)
    else:
        # Create console handler
        handler = logging.StreamHandler()
    
    # Add formatter to handler
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str = 'bayesian_adaptive_llm') -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger object
    """
    return logging.getLogger(name)
