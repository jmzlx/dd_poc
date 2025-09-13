#!/usr/bin/env python3
"""
Q&A Tab Component

Handles Q&A with citations functionality.
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import streamlit as st

# Local imports
from app.core import RELEVANCY_THRESHOLD, logger
from app.handlers.ai_handler import AIHandler
from app.ui.session_manager import SessionManager
from app.ui.ui_components import (
    display_processing_error,
    display_generation_error,
    display_download_error,
    status_message
)


class QATab:
    """
    Q&A with citations tab that handles question answering and citation display.
    """

    def __init__(self, session: SessionManager, config, ai_handler: AIHandler):
        """Initialize tab with session manager, config, and AI handler"""
        self.session = session
        self.config = config
        self.ai_handler = ai_handler

    def render(self):
        """Render the Q&A tab"""
        chunks = self.session.chunks
        if not chunks:
            status_message("👈 Process data room first to enable Q&A", "info")
            return

        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main risks? What is the revenue model? Who are the key customers?",
            key="qa_question_input"
        )

        # Handle Q&A query if there's a question
        if question:
            st.divider()
            self._handle_qa_query(question)

    def _handle_qa_query(self, question: str):
        """Handle Q&A query and display results"""
        # Create a unique key for this Q&A session to prevent resets
        qa_key = f"qa_results_{hash(question) % 100000}"
        
        # Check if we already have results for this question in session state
        if qa_key not in st.session_state:
            try:
                from app.core import search_documents

                # Initialize document processor with loaded FAISS store
                from app.core import create_document_processor
                
                # Get the store name from session (set during data room processing)
                store_name = self.session.vdr_store
                if not store_name:
                    st.error("❌ No data room processed. Please process a data room first.")
                    return
                    
                document_processor = create_document_processor(store_name=store_name)

                # Use lower threshold for Q&A to get more relevant results
                qa_threshold = 0.15  # Lower threshold for QA to find more results

                with st.spinner("🔍 Searching documents..."):
                    results = search_documents(
                        question,
                        document_processor,
                        top_k=self.config.ui['top_k_search_results'],
                        threshold=qa_threshold
                    )

                    # Fallback: try with lower threshold if no results found
                    if not results:
                        logger.info(f"No results found with threshold {qa_threshold}, trying lower threshold...")
                        fallback_threshold = 0.05  # Very low threshold as last resort
                        results = search_documents(
                            question,
                            document_processor,
                            top_k=self.config.ui['top_k_search_results'],
                            threshold=fallback_threshold
                        )
                        if results:
                            st.info(f"ℹ️ Found results with lower relevance threshold ({fallback_threshold})")

                # Store results in session state to prevent resets
                st.session_state[qa_key] = {
                    'question': question,
                    'results': results,
                    'has_ai': self.ai_handler.is_agent_available()
                }

            except Exception as e:
                logger.error(f"Failed to handle Q&A query: {e}")
                display_processing_error("question", e)
                return

        # Render results from session state
        qa_data = st.session_state[qa_key]
        results = qa_data['results']
        
        if results:
            # Use agent to synthesize answer if available
            if qa_data['has_ai']:
                self._render_ai_answer(question, results)
            else:
                self._render_direct_results(results)
        else:
            status_message("No relevant information found for your question.", "warning")

    def _render_ai_answer(self, question: str, results: list):
        """Render AI-generated answer with citations"""
        st.markdown("### 🤖 AI Service Answer")
        with st.spinner("AI processing, please wait..."):
            try:
                # Convert results to document format for context
                context_docs = [f"From {r.get('source', 'Unknown')}:\n{r.get('text', '')}" for r in results[:3]]

                # Use the AI handler
                answer_text = self.ai_handler.answer_question(question, context_docs)

                st.markdown(answer_text)

            except Exception as e:
                logger.error(f"Failed to generate AI answer: {e}")
                display_generation_error("AI answer")

        st.divider()
        self._render_source_documents(results, question)

    def _render_direct_results(self, results: list):
        """Render direct search results without AI synthesis"""
        st.markdown("### 📚 Relevant Documents")
        self._render_source_documents(results)

    def _render_source_documents(self, results: list, question: str = ""):
        """Render source documents with download buttons"""
        st.markdown("### 📚 Source Documents")

        # Display source documents with download buttons
        for i, result in enumerate(results[:3], 1):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    text_content = result.get('text', '')
                    excerpt = text_content[:200] + "..." if len(text_content) > 200 else text_content
                    st.markdown(f"{i}. \"{excerpt}\")")

                    # Create clickable link for the document
                    doc_path = result.get('path', result.get('full_path', ''))
                    doc_name = result.get('source', 'Unknown Document')
                    doc_title = self._format_document_title(doc_name)

                    # Show document info and citation
                    doc_source = result.get('source', 'Unknown')
                    citation = result.get('citation', '')
                    st.caption(f"   📄 {doc_source} ({citation})" if citation else f"   📄 {doc_source}")

                with col2:
                    # Only show one download button
                    self._render_qa_download_button(result, i, question)

    def _format_document_title(self, doc_name: str) -> str:
        """Format document title for display"""
        try:
            from app.core import format_document_title
            return format_document_title(doc_name)
        except Exception:
            return doc_name

    def _render_qa_download_button(self, result: dict, idx: int, question: str):
        """Render download button for Q&A results"""
        doc_path = result.get('path', '')
        if doc_path:
            # Create a more stable key that won't cause resets
            doc_source = result.get('source', 'document')
            button_key = f"qa_dl_{idx}_{hash(doc_path + question) % 100000}"

            # Use consistent path resolution logic
            try:
                from app.ui.ui_components import _resolve_document_path
                resolved_path = _resolve_document_path(doc_path)
                
                if resolved_path and resolved_path.exists():
                    with open(resolved_path, 'rb') as f:
                        file_bytes = f.read()

                    st.download_button(
                        label="📥 Download",
                        data=file_bytes,
                        file_name=resolved_path.name,  # Use actual filename
                        mime="application/pdf",
                        key=button_key,
                        help=f"Download {doc_source}",
                        width='stretch'
                    )
                else:
                    st.caption("(unavailable)")
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                st.caption("(error)")
