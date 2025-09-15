#!/usr/bin/env python3
"""
Export Handler

Handles report export operations.
"""

from pathlib import Path

from app.ui.session_manager import SessionManager
from app.core.exceptions import ProcessingError
from app.ui.error_handler import handle_ui_errors
from app.core.exceptions import create_processing_error


class ExportHandler:
    """
    Export handler that manages report export operations.
    """

    def __init__(self, session: SessionManager):
        """Initialize handler with session manager"""
        self.session = session

    @handle_ui_errors("Export overview report", "Please ensure overview analysis is complete")
    def export_overview_report(self) -> tuple[str, str]:
        """
        Export company overview report.

        Returns:
            Tuple of (file_name, content)
        """
        if not self.session.overview_summary:
            raise create_processing_error(
                "No overview analysis available for export",
                recovery_hint="Please complete the overview analysis first"
            )

        company_name = self._get_company_name()
        file_name = f"company_overview_{company_name}.md"
        content = f"# Company Overview\n\n{self.session.overview_summary}"

        return file_name, content

    @handle_ui_errors("Export strategic report", "Please ensure strategic analysis is complete")
    def export_strategic_report(self) -> tuple[str, str]:
        """
        Export strategic analysis report.

        Returns:
            Tuple of (file_name, content)
        """
        if not self.session.strategic_summary:
            raise create_processing_error(
                "No strategic analysis available for export",
                recovery_hint="Please complete the strategic analysis first"
            )

        company_name = self._get_company_name()
        file_name = f"dd_report_{company_name}.md"

        content = "# Due Diligence Report\n\n"
        if self.session.overview_summary:
            content += f"## Company Overview\n\n{self.session.overview_summary}\n\n"
        content += f"## Strategic Analysis\n\n{self.session.strategic_summary}"

        return file_name, content

    @handle_ui_errors("Export strategic company report", "Please ensure strategic company analysis is complete")
    def export_strategic_company_report(self) -> tuple[str, str]:
        """
        Export strategic company analysis report.

        Returns:
            Tuple of (file_name, content)
        """
        if not self.session.strategic_company_summary:
            raise create_processing_error(
                "No company analysis available for export",
                recovery_hint="Please complete the company analysis first"
            )

        company_name = self._get_company_name()
        file_name = f"company_analysis_{company_name}.md"
        content = f"# Company Analysis - {company_name.title()}\n\n{self.session.strategic_company_summary}"

        return file_name, content

    @handle_ui_errors("Export combined report", "Please ensure analysis is complete")
    def export_combined_report(self) -> tuple[str, str]:
        """
        Export combined due diligence report.

        Returns:
            Tuple of (file_name, content)
        """
        # Check for new company analysis first, then fall back to old format
        if self.session.strategic_company_summary:
            # Use the comprehensive company analysis
            company_name = self._get_company_name()
            file_name = f"complete_dd_report_{company_name}.md"
            content = f"# Complete Due Diligence Report - {company_name.title()}\n\n{self.session.strategic_company_summary}\n\n"
            
            # Add additional analyses if available
            if self.session.checklist_results:
                content += "## Checklist Analysis\n\n"
                for category, items in self.session.checklist_results.items():
                    content += f"### {category}\n\n"
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                content += f"- {item.get('text', str(item))}\n"
                            else:
                                content += f"- {str(item)}\n"
                    content += "\n"

            if self.session.question_answers:
                content += "## Due Diligence Questions\n\n"
                for question, answer in self.session.question_answers.items():
                    if isinstance(answer, dict) and answer.get('has_answer'):
                        content += f"### {question}\n\n{answer.get('answer', '')}\n\n"

            return file_name, content
        
        # Fall back to old combined format if no company analysis
        if not (self.session.overview_summary or self.session.strategic_summary):
            raise create_processing_error(
                "No analysis data available for export",
                recovery_hint="Please complete company analysis first"
            )

        company_name = self._get_company_name()
        file_name = f"complete_dd_report_{company_name}.md"

        content = f"# Complete Due Diligence Report - {company_name.title()}\n\n"

        if self.session.overview_summary:
            content += f"## Company Overview\n\n{self.session.overview_summary}\n\n"

        if self.session.strategic_summary:
            content += f"## Strategic Analysis\n\n{self.session.strategic_summary}\n\n"

        # Add checklist results if available
        if self.session.checklist_results:
            content += "## Checklist Analysis\n\n"
            for category, items in self.session.checklist_results.items():
                content += f"### {category}\n\n"
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            content += f"- {item.get('text', str(item))}\n"
                        else:
                            content += f"- {str(item)}\n"
                content += "\n"

        # Add question answers if available
        if self.session.question_answers:
            content += "## Due Diligence Questions\n\n"
            for question, answer in self.session.question_answers.items():
                if isinstance(answer, dict) and answer.get('has_answer'):
                    content += f"### {question}\n\n{answer.get('answer', '')}\n\n"

        return file_name, content

    @handle_ui_errors("Export checklist report", "Please ensure checklist analysis is complete")
    def export_checklist_report(self) -> tuple[str, str]:
        """
        Export checklist analysis report.

        Returns:
            Tuple of (file_name, content)
        """
        if not self.session.checklist_results:
            raise create_processing_error(
                "No checklist results available for export",
                recovery_hint="Please complete the checklist analysis first"
            )

        company_name = self._get_company_name()
        file_name = f"checklist_analysis_{company_name}.md"

        content = f"# Checklist Analysis Report - {company_name.title()}\n\n"

        for category, items in self.session.checklist_results.items():
            content += f"## {category}\n\n"
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        content += f"- {item.get('text', str(item))}\n"
                    else:
                        content += f"- {str(item)}\n"
            content += "\n"

        return file_name, content

    def _get_company_name(self) -> str:
        """Get company name from current documents"""
        documents = self.session.documents
        if documents:
            company_name = Path(list(documents.keys())[0]).parent.name
            return company_name
        return 'export'
