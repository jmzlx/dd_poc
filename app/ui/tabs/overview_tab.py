#!/usr/bin/env python3
"""
Overview Tab Component

Handles company overview generation and display.
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import streamlit as st

# Local imports
from app.ui.tabs.tab_base import TabBase
from app.ui.ui_components import status_message


class OverviewTab(TabBase):
    """
    Company overview tab that handles overview generation and display.
    """

    def render(self):
        """Render the overview tab"""
        if not self._check_documents_available():
            return

        # Generate button row
        button_clicked = self._render_generate_buttons(
            "🤖 Generate Overview",
            "regenerate_overview_btn",
            "overview_summary",
            "Use AI to generate company overview analysis"
        )

        # Generate or display content
        if self._should_generate_content(button_clicked, "overview_summary"):
            self._generate_report("overview", "overview_summary", "✅ Company overview generated successfully!")
        else:
            self._render_content_or_placeholder(
                "overview_summary",
                "👆 Click 'Generate Overview' to create AI-powered company analysis"
            )

    def _generate_report(self, report_type: str, session_attr: str, success_message: str):
        """Generate company overview report using AI"""
        if not self._check_ai_availability():
            return

        with st.spinner("Agent running, please wait..."):
            data_room_name = self._get_data_room_name()

            overview_summary = self.ai_handler.generate_report(
                report_type,
                documents=self.session.documents,
                data_room_name=data_room_name,
                strategy_text=self.session.strategy_text,
                checklist_results=self.session.checklist_results
            )

            if overview_summary:
                setattr(self.session, session_attr, overview_summary)
                status_message(success_message, "success")
                st.rerun()
            else:
                status_message("Failed to generate overview. Please try again.", "error")

    def _get_export_method_name(self) -> str:
        """Get export method name for overview reports"""
        return "export_overview_report"

    def _get_download_key(self) -> str:
        """Get download button key for overview reports"""
        return "export_overview_btn"

