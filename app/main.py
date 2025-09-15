#!/usr/bin/env python3
"""
Main Application Entry Point
"""

# Standard library imports
import os
import warnings

# Third-party imports
import streamlit as st

# Local imports
from app.core.config import init_app_config
from app.core.logging import configure_langchain_logging
from app.handlers.ai_handler import AIHandler
from app.handlers.document_handler import DocumentHandler
from app.handlers.export_handler import ExportHandler
from app.ui.session_manager import SessionManager
from app.ui.sidebar import Sidebar
from app.ui.tabs.checklist_tab import ChecklistTab
from app.ui.tabs.graph_tab import GraphTab
from app.ui.tabs.overview_tab import OverviewTab
from app.ui.tabs.qa_tab import QATab
from app.ui.tabs.questions_tab import QuestionsTab
from app.ui.tabs.strategic_tab import StrategicTab

# Enable tokenizers parallelism for better performance
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Initialize for Streamlit Cloud deployment (must be done before other imports)
try:
    from scripts.streamlit_cloud_config import initialize_for_streamlit_cloud
    initialize_for_streamlit_cloud()
except ImportError:
    # Local development - skip cloud initialization
    pass

# Only suppress specific known non-critical warnings
warnings.filterwarnings("ignore", message=".*Relevance scores must be between.*")
warnings.filterwarnings("ignore", message=".*No relevant docs were retrieved.*")


class App:
    """Main application class that orchestrates all components."""

    def __init__(self):
        """Initialize the application"""
        # Initialize configuration
        self.config = init_app_config()

        # Initialize session manager
        self.session = SessionManager()

        # Initialize handlers
        self.document_handler = DocumentHandler(self.session)
        self.ai_handler = AIHandler(self.session)
        self.export_handler = ExportHandler(self.session)

        # Initialize UI components
        self.sidebar = Sidebar(self.session, self.config)
        self.tabs = {
            'overview': OverviewTab(self.session, self.config, self.ai_handler, self.export_handler),
            'strategic': StrategicTab(self.session, self.config, self.ai_handler, self.export_handler),
            'checklist': ChecklistTab(self.session, self.config, self.ai_handler),
            'questions': QuestionsTab(self.session, self.config, self.ai_handler),
            'qa': QATab(self.session, self.config, self.ai_handler),
            'graph': GraphTab(self.session, self.config, self.ai_handler, self.export_handler)
        }

        # Configure Streamlit page
        st.set_page_config(
            page_title=self.config.ui['page_title'],
            page_icon=self.config.ui['page_icon'],
            layout=self.config.ui['layout']
        )

    def run(self):
        """Run the main application"""
        # Render header
        st.title("🤖 AI Due Diligence")
        st.markdown("**Intelligent M&A Analysis:** Strategic assessment, automated document review, and AI-powered insights")

        # Render sidebar and get selections
        data_room_path, process_button = self.sidebar.render()

        # Store the selected data room path
        if data_room_path:
            self.session.data_room_path = data_room_path

        # Main tabs
        tab_names = [
            "🏢 Target Company Analysis",
            "🎯 Strategic Assessment",
            "📊 Checklist Matching",
            "❓ Due Diligence Questions",
            "💬 Q&A with Citations",
            "🧠 Knowledge Graph"
        ]

        tabs = st.tabs(tab_names)

        with tabs[0]:
            self.tabs['overview'].render()

        with tabs[1]:
            self.tabs['strategic'].render()

        with tabs[2]:
            self.tabs['checklist'].render()

        with tabs[3]:
            self.tabs['questions'].render()

        with tabs[4]:
            self.tabs['qa'].render()

        with tabs[5]:
            self.tabs['graph'].render()

        # Processing trigger
        if process_button and data_room_path:
            with st.spinner("🚀 Processing data room..."):
                self.sidebar.process_data_room(data_room_path)


def main():
    """Main application entry point"""
    # Configure LangChain logging
    configure_langchain_logging(log_level="WARNING")

    try:
        app = App()
        app.run()
    except Exception as e:
        from app.ui.error_handler import ErrorHandler
        ErrorHandler.handle_error(
            e,
            "Application startup failed",
            recovery_hint="Please refresh the page and try again"
        )
        st.stop()


if __name__ == "__main__":
    main()
