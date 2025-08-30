#!/usr/bin/env python3
"""
DD-Checklist Source Package

This package contains the refactored components of the DD-Checklist application.
"""

from .config import get_config, init_config, get_model_config, get_processing_config
from .document_processing import DocumentProcessor, escape_markdown_math
from .services import DDChecklistService, ChecklistParser, QuestionParser
from .utils import logger, handle_exceptions, safe_execute, ErrorHandler
from .ui_components import render_project_selector, render_ai_settings

__version__ = "0.2.0"
__author__ = "DD-Checklist Team"

__all__ = [
    # Configuration
    "get_config",
    "init_config", 
    "get_model_config",
    "get_processing_config",
    
    # Document Processing
    "DocumentProcessor",
    "escape_markdown_math",
    
    # Services
    "DDChecklistService",
    "ChecklistParser",
    "QuestionParser",
    
    # Utilities
    "logger",
    "handle_exceptions",
    "safe_execute",
    "ErrorHandler",
    
    # UI Components
    "render_project_selector",
    "render_ai_settings"
]
