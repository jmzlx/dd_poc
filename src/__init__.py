#!/usr/bin/env python3
"""
DD-Checklist Source Package

This package contains the refactored components of the DD-Checklist application.
"""

from .config import (
    get_config, init_config, logger, show_success, show_error, show_info,
    get_mime_type, format_document_title, count_documents_in_directory
)
from .document_processing import DocumentProcessor, escape_markdown_math, safe_execute
from .ui_components import render_project_selector, render_ai_settings

__version__ = "0.2.0"
__author__ = "DD-Checklist Team"

__all__ = [
    # Configuration
    "get_config",
    "init_config",
    
    # Document Processing
    "DocumentProcessor",
    "escape_markdown_math",
    "safe_execute",
    
    # Utilities (merged from utils.py)
    "logger",
    "show_success",
    "show_error", 
    "show_info",
    "get_mime_type",
    "format_document_title",
    "count_documents_in_directory",
    
    # UI Components
    "render_project_selector",
    "render_ai_settings"
]
