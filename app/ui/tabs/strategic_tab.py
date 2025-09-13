#!/usr/bin/env python3
"""
Strategic Analysis Tab Component

Handles strategic analysis generation and display.
"""

import streamlit as st

from app.ui.tabs.tab_base import TabBase
from app.ui.ui_components import status_message
from app.core import logger


class StrategicTab(TabBase):
    """
    Strategic analysis tab that handles strategic report generation and display.
    """

    def render(self):
        """Render the strategic analysis tab"""
        if not self._check_documents_available():
            return

        # Generate button row
        button_clicked = self._render_generate_buttons(
            "🎯 Generate Analysis",
            "regenerate_strategic_btn",
            "strategic_summary",
            "Use AI to generate strategic analysis"
        )

        # Generate or display content
        if self._should_generate_content(button_clicked, "strategic_summary"):
            self._generate_report("strategic", "strategic_summary", "✅ Strategic analysis generated successfully!")
        else:
            self._render_content_or_placeholder(
                "strategic_summary",
                "👆 Click 'Generate Analysis' to create AI-powered strategic assessment"
            )

    def _generate_report(self, report_type: str, session_attr: str, success_message: str):
        """Generate strategic analysis report using AI"""
        if not self._check_ai_availability():
            return

        if not self._check_processing_active():
            return

        # Set processing active
        self._set_processing_active(True)

        try:
            with st.spinner("Agent running, please wait..."):
                data_room_name = self._get_data_room_name()

                strategic_summary = self.ai_handler.generate_report(
                    report_type,
                    documents=self.session.documents,
                    data_room_name=data_room_name,
                    strategy_text=self.session.strategy_text,
                    checklist_results=self.session.checklist_results
                )

                if strategic_summary:
                    setattr(self.session, session_attr, strategic_summary)
                    status_message(success_message, "success")
                    st.rerun()
                else:
                    status_message("Failed to generate strategic analysis. Please try again.", "error")
        except Exception as e:
            logger.error(f"Failed to generate strategic analysis: {e}")
            status_message(f"Failed to generate strategic analysis: {str(e)}", "error")
        finally:
            # Always reset processing state
            self._set_processing_active(False)

    def _get_export_method_name(self) -> str:
        """Get export method name for strategic reports"""
        return "export_strategic_report"

    def _get_download_key(self) -> str:
        """Get download button key for strategic reports"""
        return "export_strategic_btn"

