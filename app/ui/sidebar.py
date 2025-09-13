#!/usr/bin/env python3
"""
Sidebar Component

Handles project selection, file selectors, and AI settings.
"""

import streamlit as st
from pathlib import Path
from typing import Tuple, Optional

from app.ui.session_manager import SessionManager
# Use lazy imports to avoid circular import issues
# from app.handlers.document_handler import DocumentHandler
# from app.handlers.ai_handler import AIHandler
# Import components directly to avoid circular import issues
import importlib.util
import os

# Load the ui_components.py module directly
components_path = os.path.join(os.path.dirname(__file__), 'ui_components.py')
spec = importlib.util.spec_from_file_location("components_module", components_path)
components_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components_module)

# Import the specific functions we need
render_project_selector = components_module.render_project_selector
render_ai_settings = components_module.render_ai_settings
render_file_selector = components_module.render_file_selector
display_processing_error = components_module.display_processing_error
status_message = components_module.status_message
from app.core import logger


class Sidebar:
    """
    Simplified sidebar component that handles all sidebar functionality.
    """

    def __init__(self, session: SessionManager, config):
        """Initialize sidebar with session manager and config"""
        self.session = session
        self.config = config
        # Handlers will be imported lazily when needed
        self._document_handler = None
        self._ai_handler = None

    @property
    def document_handler(self):
        """Lazy import of DocumentHandler"""
        if self._document_handler is None:
            from app.handlers.document_handler import DocumentHandler
            self._document_handler = DocumentHandler(self.session)
        return self._document_handler

    @property 
    def ai_handler(self):
        """Lazy import of AIHandler"""
        if self._ai_handler is None:
            from app.handlers.ai_handler import AIHandler
            self._ai_handler = AIHandler(self.session)
        return self._ai_handler

    def render(self) -> Tuple[Optional[str], bool]:
        """
        Render sidebar with project selection, file selectors, and AI settings

        Returns:
            Tuple of (data_room_path, process_button_pressed)
        """
        with st.sidebar:
            # Project and data room selection
            selected_project_path, data_room_path = render_project_selector()

            # Process button
            process_button = st.button(
                "🚀 Process Data Room",
                type="primary",
                width='stretch'
            )

            if process_button:
                st.success("Processing... Check main area for progress")

            st.divider()

            # Analysis Configuration
            st.subheader("📋 Analysis Configuration")

            # Strategy selector
            strategy_path, strategy_text = self._render_file_selector(
                self.config.paths['strategy_dir'], "Strategy", "🎯"
            )
            self.session.strategy_path = strategy_path
            self.session.strategy_text = strategy_text

            # Checklist selector
            checklist_path, checklist_text = self._render_file_selector(
                self.config.paths['checklist_dir'], "Checklist", "📊"
            )
            self.session.checklist_path = checklist_path
            self.session.checklist_text = checklist_text

            # Questions selector
            questions_path, questions_text = self._render_file_selector(
                self.config.paths['questions_dir'], "Questions", "❓"
            )
            self.session.questions_path = questions_path
            self.session.questions_text = questions_text

            st.divider()

            # AI settings
            api_key, model_choice = render_ai_settings()

            # Initialize AI agent if API key is available
            if api_key:
                existing_agent = self.session.agent
                if existing_agent is None:
                    if self.ai_handler.setup_agent(api_key, model_choice):
                        st.success("✅ AI Agent ready")
            else:
                self.session.agent = None

        return data_room_path, process_button

    def _render_file_selector(self, directory: str, label: str, icon: str) -> Tuple[Optional[str], str]:
        """
        Render a file selector for a specific directory

        Args:
            directory: Path to the directory containing files
            label: Label for the selector
            icon: Icon for the selector

        Returns:
            Tuple of (selected_file_path, selected_file_content)
        """
        try:
            return render_file_selector(directory, label, "sidebar", icon)
        except Exception as e:
            logger.error(f"Failed to render {label.lower()} selector: {e}")
            return None, ""

    def process_data_room(self, data_room_path: str):
        """
        Process a data room using the fast FAISS loading approach

        Args:
            data_room_path: Path to the data room directory
        """
        try:
            result = self.document_handler.process_data_room_fast(data_room_path)

            if result:
                doc_count, chunk_count = result
                st.success(f"✅ Loaded {doc_count} documents and {chunk_count} chunks from pre-built index!")
                st.rerun()
            else:
                display_processing_error("data room")
        except Exception as e:
            logger.error(f"Failed to process data room {data_room_path}: {e}")
            display_processing_error("data room", e)

