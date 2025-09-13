#!/usr/bin/env python3
"""
Tab Base Component

Provides shared functionality for all tab components including common
initialization patterns, render methods, and export functionality.
"""

# Standard library imports
from pathlib import Path
from typing import Optional, Any, Dict

# Third-party imports
import streamlit as st

# Local imports
from app.ui.error_handler import handle_ui_errors
from app.handlers.ai_handler import AIHandler
from app.handlers.export_handler import ExportHandler
from app.ui.session_manager import SessionManager
from app.ui.ui_components import status_message, render_generate_buttons


class TabBase:
    """
    Base class for tab components with shared functionality.

    Provides common patterns for initialization, rendering, and export functionality.
    """

    def __init__(self, session: SessionManager, config, ai_handler: AIHandler, export_handler: ExportHandler):
        """Initialize tab with session manager, config, and handlers"""
        self.session = session
        self.config = config
        self.ai_handler = ai_handler
        self.export_handler = export_handler

    def render(self):
        """Render the tab - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement render()")

    def _check_documents_available(self) -> bool:
        """Check if documents are available and show message if not"""
        if not self.session.documents:
            status_message("👈 Configure and process data room first", "info")
            return False
        return True

    def _render_generate_buttons(self, generate_label: str, regenerate_key: str,
                               session_attr: str, help_text: str) -> tuple[bool, bool]:
        """Render common generate and regenerate buttons using reusable component"""
        return render_generate_buttons(
            generate_label,
            regenerate_key,
            session_attr,
            help_text,
            self.session
        )

    def _should_generate_content(self, generate_clicked: bool, session_attr: str) -> bool:
        """Determine if content should be generated"""
        return generate_clicked and not getattr(self.session, session_attr)

    def _should_display_content(self, session_attr: str) -> bool:
        """Determine if content should be displayed"""
        return bool(getattr(self.session, session_attr))

    def _get_data_room_name(self) -> str:
        """Get the data room name from documents"""
        if not self.session.documents:
            return "Unknown"
        return Path(list(self.session.documents.keys())[0]).parent.name

    def _check_ai_availability(self) -> bool:
        """Check if AI agent is available"""
        if not self.ai_handler.is_agent_available():
            status_message("AI Agent not available. Please configure your API key in the sidebar.", "error")
            return False
        return True

    def _check_processing_active(self) -> bool:
        """Check if processing is already active"""
        if self.session.processing_active:
            status_message("⚠️ Another operation is currently running. Please wait.", "warning")
            return False
        return True

    def _set_processing_active(self, active: bool):
        """Set processing active state"""
        self.session.processing_active = active

    @handle_ui_errors("Report generation", "Please check your documents and try again")
    def _generate_report(self, report_type: str, session_attr: str, success_message: str):
        """Generate report using AI - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _generate_report()")

    def _render_export_button(self, export_method_name: str, download_key: str):
        """Render export button for reports"""
        # Get the session attribute dynamically
        session_attr = export_method_name.replace("export_", "").replace("_report", "_summary")
        if not getattr(self.session, session_attr):
            return

        # Call the export method dynamically
        export_method = getattr(self.export_handler, export_method_name)
        file_name, export_data = export_method()

        if file_name and export_data:
            st.download_button(
                "📥 Export Report",
                data=export_data,
                file_name=file_name,
                mime="text/markdown",
                key=download_key,
                help="Download report as Markdown file"
            )

    def _render_content_or_placeholder(self, session_attr: str, placeholder_message: str):
        """Render content if available, otherwise show placeholder"""
        content = getattr(self.session, session_attr)
        if content:
            if isinstance(content, str):
                st.markdown(content)
            else:
                # Handle dict/other types as needed by subclasses
                self._render_custom_content(content)
            self._render_export_button(self._get_export_method_name(), self._get_download_key())
        else:
            status_message(placeholder_message, "info")

    def _render_custom_content(self, content: Any):
        """Render custom content types - can be overridden by subclasses"""
        pass

    def _get_export_method_name(self) -> str:
        """Get export method name - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_export_method_name()")

    def _get_download_key(self) -> str:
        """Get download button key - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_download_key()")
