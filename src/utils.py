#!/usr/bin/env python3
"""
Utilities Module

This module contains error handling, logging, and other utility functions
for the DD-Checklist application.
"""

import logging
import functools
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Union
import streamlit as st
from datetime import datetime
import sys
import os


class DDChecklistLogger:
    """
    Custom logger for DD-Checklist application
    Handles both file and console logging with Streamlit integration
    """
    
    def __init__(self, name: str = "dd_checklist", log_level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if possible)
        try:
            log_dir = Path(".logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"dd_checklist_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            # File logging not available (e.g., on Streamlit Cloud)
            pass
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
        # Also show in Streamlit if available
        if 'st' in globals() and st:
            st.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
        # Also show in Streamlit if available
        if 'st' in globals() and st:
            st.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)
        # Show error in Streamlit if available
        if 'st' in globals() and st:
            st.error(f"{message} - Check logs for details.")


# Global logger instance
logger = DDChecklistLogger()


def handle_exceptions(
    return_value: Any = None,
    show_error: bool = True,
    log_error: bool = True
) -> Callable:
    """
    Decorator for handling exceptions in functions
    
    Args:
        return_value: Value to return on exception
        show_error: Whether to show error in UI
        log_error: Whether to log the error
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                
                if log_error:
                    logger.exception(error_msg)
                
                if show_error and 'st' in globals() and st:
                    st.error(error_msg)
                
                return return_value
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_message: Optional[str] = None,
    show_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        error_message: Custom error message
        show_error: Whether to show error in UI
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error executing {func.__name__}: {str(e)}"
        logger.exception(msg)
        
        if show_error and 'st' in globals() and st:
            st.error(msg)
        
        return default_return


class ErrorHandler:
    """
    Context manager for error handling
    """
    
    def __init__(
        self,
        error_message: str = "An error occurred",
        show_error: bool = True,
        reraise: bool = False
    ):
        """
        Initialize error handler
        
        Args:
            error_message: Message to display on error
            show_error: Whether to show error in UI
            reraise: Whether to reraise the exception
        """
        self.error_message = error_message
        self.show_error = show_error
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"{self.error_message}: {str(exc_val)}"
            logger.exception(error_msg)
            
            if self.show_error and 'st' in globals() and st:
                st.error(error_msg)
            
            if self.reraise:
                return False  # Reraise the exception
            
            return True  # Suppress the exception


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception as e:
        logger.warning(f"Invalid file path {file_path}: {e}")
        return False


def validate_directory_path(dir_path: Union[str, Path]) -> bool:
    """
    Validate that a directory path exists
    
    Args:
        dir_path: Directory path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(dir_path)
        return path.exists() and path.is_dir()
    except Exception as e:
        logger.warning(f"Invalid directory path {dir_path}: {e}")
        return False


def ensure_directory(dir_path: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        dir_path: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Could not create directory {dir_path}: {e}")
        return False


def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if error
    """
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        logger.warning(f"Could not get size for {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    size = size_bytes
    
    for i, unit in enumerate(size_names):
        if size < 1024 or i == len(size_names) - 1:
            return f"{size:.1f} {unit}"
        size /= 1024
    
    return f"{size:.1f} GB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Trim and ensure not empty
    sanitized = sanitized.strip('_. ')
    
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information
    
    Returns:
        Dictionary with memory usage stats
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return {}
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return {}


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    return wrapper


class ProgressTracker:
    """
    Progress tracking utility for long-running operations with weighted ETA calculation
    """
    
    def __init__(self, total_steps: int, description: str = "Processing", step_weights: Optional[Dict[int, float]] = None):
        """
        Initialize progress tracker
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
            step_weights: Optional dict mapping step numbers to relative weights (default: all steps equal weight)
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        self.step_start_times = {}  # Track when each step started
        self.step_durations = {}    # Track actual duration of completed steps
        
        # Set up step weights (default: equal weight for all steps)
        if step_weights:
            self.step_weights = step_weights
        else:
            self.step_weights = {i: 1.0 for i in range(1, total_steps + 1)}
        
        # Calculate total weight for progress calculation
        self.total_weight = sum(self.step_weights.values())
        
        # Initialize Streamlit progress bar if available
        if 'st' in globals() and st:
            self.progress_bar = st.progress(0, text=f"{description}...")
            self.status_text = st.empty()
        else:
            self.progress_bar = None
            self.status_text = None
    
    def update(self, step: int, message: str = ""):
        """
        Update progress with weighted ETA calculation
        
        Args:
            step: Current step number
            message: Optional status message
        """
        now = datetime.now()
        
        # Record step timing
        if self.current_step != step:
            # Mark completion of previous step
            if self.current_step > 0 and self.current_step in self.step_start_times:
                self.step_durations[self.current_step] = (now - self.step_start_times[self.current_step]).total_seconds()
            
            # Mark start of new step
            self.step_start_times[step] = now
            self.current_step = step
        
        # Calculate weighted progress
        completed_weight = sum(self.step_weights.get(i, 1.0) for i in range(1, step))
        current_step_weight = self.step_weights.get(step, 1.0)
        
        # For current step, assume 50% completion unless we have sub-progress info
        current_progress_weight = completed_weight + (current_step_weight * 0.5)
        progress = current_progress_weight / self.total_weight if self.total_weight > 0 else 0
        progress = min(progress, 1.0)  # Cap at 100%
        
        # Calculate improved ETA using weighted approach
        elapsed = (now - self.start_time).total_seconds()
        eta_str = ""
        
        if step > 1 and completed_weight > 0:
            # Use actual timing data from completed steps
            avg_time_per_weight = elapsed / completed_weight
            remaining_weight = self.total_weight - current_progress_weight
            eta = avg_time_per_weight * remaining_weight
            
            if eta > 1:
                if eta < 60:
                    eta_str = f" (ETA: {eta:.0f}s)"
                elif eta < 3600:
                    eta_str = f" (ETA: {eta/60:.1f}m)"
                else:
                    eta_str = f" (ETA: {eta/3600:.1f}h)"
        elif step == 1 and elapsed > 5:  # Only show ETA after 5 seconds
            # For first step, make a rough estimate based on step weights
            estimated_time_per_weight = elapsed / self.step_weights.get(1, 1.0)
            remaining_weight = self.total_weight - current_progress_weight
            eta = estimated_time_per_weight * remaining_weight
            
            if eta > 10:  # Only show if meaningful
                if eta < 60:
                    eta_str = f" (ETA: ~{eta:.0f}s)"
                else:
                    eta_str = f" (ETA: ~{eta/60:.1f}m)"
        
        status_msg = f"{self.description}: {step}/{self.total_steps}{eta_str}"
        if message:
            status_msg += f" - {message}"
        
        # Update Streamlit components
        if self.progress_bar:
            self.progress_bar.progress(progress, text=status_msg)
        
        # Log progress at key milestones
        if step == 1 or step % max(1, self.total_steps // 5) == 0:  # Log every 20%
            logger.info(status_msg)
    
    def update_step_progress(self, step: int, sub_progress: float, message: str = ""):
        """
        Update progress within a specific step (for long-running operations)
        
        Args:
            step: Current step number
            sub_progress: Progress within the step (0.0 to 1.0)
            message: Optional status message
        """
        now = datetime.now()
        
        # Ensure we're tracking this step
        if step not in self.step_start_times:
            self.step_start_times[step] = now
            self.current_step = step
        
        # Calculate weighted progress with sub-progress
        completed_weight = sum(self.step_weights.get(i, 1.0) for i in range(1, step))
        current_step_weight = self.step_weights.get(step, 1.0)
        
        # Use actual sub-progress instead of assuming 50%
        current_progress_weight = completed_weight + (current_step_weight * sub_progress)
        progress = current_progress_weight / self.total_weight if self.total_weight > 0 else 0
        progress = min(progress, 1.0)  # Cap at 100%
        
        # Calculate improved ETA
        elapsed = (now - self.start_time).total_seconds()
        eta_str = ""
        
        if step > 1 and completed_weight > 0:
            # Use actual timing data from completed steps
            avg_time_per_weight = elapsed / completed_weight
            remaining_weight = self.total_weight - current_progress_weight
            eta = avg_time_per_weight * remaining_weight
            
            if eta > 1:
                if eta < 60:
                    eta_str = f" (ETA: {eta:.0f}s)"
                elif eta < 3600:
                    eta_str = f" (ETA: {eta/60:.1f}m)"
                else:
                    eta_str = f" (ETA: {eta/3600:.1f}h)"
        elif step == 1 and elapsed > 5:
            # For first step, estimate based on current progress
            if sub_progress > 0.1:  # Only estimate if we have meaningful progress
                step_elapsed = (now - self.step_start_times[step]).total_seconds()
                estimated_step_time = step_elapsed / sub_progress
                remaining_step_time = estimated_step_time * (1 - sub_progress)
                
                # Add estimated time for remaining steps
                remaining_weight = self.total_weight - self.step_weights.get(step, 1.0)
                estimated_time_per_weight = estimated_step_time / self.step_weights.get(step, 1.0)
                eta = remaining_step_time + (estimated_time_per_weight * remaining_weight)
                
                if eta > 10:
                    if eta < 60:
                        eta_str = f" (ETA: ~{eta:.0f}s)"
                    else:
                        eta_str = f" (ETA: ~{eta/60:.1f}m)"
        
        status_msg = f"{self.description}: {step}/{self.total_steps}{eta_str}"
        if message:
            status_msg += f" - {message}"
        
        # Update Streamlit components
        if self.progress_bar:
            self.progress_bar.progress(progress, text=status_msg)
    
    def complete(self, message: str = "Complete"):
        """
        Mark progress as complete
        
        Args:
            message: Completion message
        """
        if self.progress_bar:
            self.progress_bar.progress(1.0, text=f"{self.description}: {message}")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.description} completed in {elapsed:.1f} seconds")


def batch_process(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 10,
    description: str = "Processing"
) -> List[Any]:
    """
    Process items in batches with progress tracking
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        batch_size: Size of each batch
        description: Description for progress tracking
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    tracker = ProgressTracker(total_batches, description)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
            
            tracker.update(batch_num, f"Batch {batch_num}/{total_batches}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # Continue with remaining batches
            continue
    
    tracker.complete()
    return results


# Streamlit-specific utilities
def show_success(message: str):
    """Show success message in Streamlit"""
    if 'st' in globals() and st:
        st.success(message)
    logger.info(message)


def show_info(message: str):
    """Show info message in Streamlit"""
    if 'st' in globals() and st:
        st.info(message)
    logger.info(message)


def show_warning(message: str):
    """Show warning message in Streamlit"""
    if 'st' in globals() and st:
        st.warning(message)
    logger.warning(message)


def show_error(message: str):
    """Show error message in Streamlit"""
    if 'st' in globals() and st:
        st.error(message)
    logger.error(message)
