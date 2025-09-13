#!/usr/bin/env python3
"""
Checklist Tab Component

Handles checklist matching and display.
"""

import streamlit as st

from app.ui.session_manager import SessionManager
from app.ui.ui_components import (
    status_message,
    render_generate_buttons,
    processing_guard,
    display_generation_error,
    display_initialization_error
)
from app.handlers.ai_handler import AIHandler
from app.core.logging import logger


class ChecklistTab:
    """
    Checklist matching tab that handles checklist analysis and display.
    """

    def __init__(self, session: SessionManager, config, ai_handler: AIHandler):
        """Initialize tab with session manager, config, and AI handler"""
        self.session = session
        self.config = config
        self.ai_handler = ai_handler

    def render(self):
        """Render the checklist tab"""
        documents = self.session.documents
        if not documents:
            status_message("👈 Configure and process data room first", "info")
            return

        # Use checklist from sidebar
        file_text = self.session.checklist_text

        if not file_text:
            status_message("👈 Select a checklist in the sidebar first", "info")
            return

        # Generate button row
        button_clicked = render_generate_buttons(
            "📊 Generate Matching",
            "regenerate_checklist_btn",
            "checklist_results",
            "Generate checklist matching analysis",
            self.session
        )

        # Generate or display content
        if button_clicked and not self.session.checklist_results:
            self._generate_checklist_matching()
        elif self.session.checklist_results:
            from app.ui.ui_components import render_checklist_results
            results = self.session.checklist_results
            render_checklist_results(results, relevancy_threshold=self.config.processing['similarity_threshold'])
        else:
            status_message("👆 Click 'Generate Matching' to analyze checklist items against documents", "info")

    @processing_guard()
    def _generate_checklist_matching(self):
        """Generate checklist matching analysis"""
        # Initialize document processor with loaded FAISS store
        from app.core import create_document_processor
        
        # Get the store name from session (set during data room processing)
        store_name = self.session.vdr_store
        if not store_name:
            st.error("❌ No data room processed. Please process a data room first.")
            return
            
        document_processor = create_document_processor(store_name=store_name)

        try:
            checklist_text = self.session.checklist_text
            if not checklist_text or not self.session.chunks:
                st.error("❌ No checklist or document chunks available")
                return

            # Check if data room has been processed
            if not hasattr(self.session, 'documents') or not self.session.documents:
                st.error("❌ No data room processed. Please process a data room first before running checklist analysis.")
                return

            # Note: Document type embeddings will be auto-loaded if missing during processing

            with st.spinner("Processing checklist, please wait..."):
                from app.core.parsers import parse_checklist
                from app.core import search_and_analyze

                try:
                    # Parse raw checklist
                    llm = self.ai_handler.llm
                    if not llm:
                        raise ValueError("AI service not configured. Please set up your API key first.")
                    checklist = parse_checklist(checklist_text, llm)
                    self.session.checklist = checklist

                    # Use pre-built FAISS index from document processor
                    if not document_processor.vector_store:
                        raise ValueError("No pre-built FAISS index loaded. Please ensure data room is processed first.")

                    vector_store = document_processor.vector_store

                    # Process checklist items
                    checklist_results = search_and_analyze(
                        checklist,
                        vector_store,
                        self.ai_handler.session.agent.llm if self.ai_handler.is_agent_available() else None,
                        self.config.processing['similarity_threshold'],
                        'items',
                        store_name=getattr(document_processor, 'store_name', None),
                        session=self.session
                    )
                    self.session.checklist_results = checklist_results

                    status_message("✅ Checklist matching analysis completed!", "success")
                    st.rerun()

                except Exception as e:
                    logger.error(f"Checklist processing failed: {e}")
                    display_generation_error("checklist analysis", e)

        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            display_initialization_error("document processor", e)
        finally:
            # Processing state is managed by processing_guard decorator
            pass

