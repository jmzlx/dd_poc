#!/usr/bin/env python3
"""
Questions Tab Component

Handles due diligence questions analysis and display.
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


class QuestionsTab:
    """
    Questions tab that handles due diligence questions analysis and display.
    """

    def __init__(self, session: SessionManager, config, ai_handler: AIHandler):
        """Initialize tab with session manager, config, and AI handler"""
        self.session = session
        self.config = config
        self.ai_handler = ai_handler

    def render(self):
        """Render the questions tab"""
        documents = self.session.documents
        if not documents:
            status_message("👈 Configure and process data room first", "info")
            return

        # Use questions from sidebar
        file_text = self.session.questions_text

        if not file_text:
            status_message("👈 Select a questions list in the sidebar first", "info")
            return

        # Generate button row
        button_clicked = render_generate_buttons(
            "❓ Generate Answers",
            "regenerate_questions_btn",
            "question_answers",
            "Generate answers for due diligence questions",
            self.session
        )

        # Generate or display content
        if button_clicked and not self.session.question_answers:
            self._generate_question_answers()
        elif self.session.question_answers:
            from app.ui.ui_components import render_question_results
            answers = self.session.question_answers
            # Convert from {'questions': [...]} format to {question_id: answer_data} format
            if isinstance(answers, dict) and 'questions' in answers:
                questions_dict = {}
                for i, question_data in enumerate(answers['questions']):
                    questions_dict[f"question_{i}"] = question_data
                render_question_results(questions_dict)
            else:
                render_question_results(answers)
        else:
            status_message("👆 Click 'Generate Answers' to find relevant documents for due diligence questions", "info")

    @processing_guard()
    def _generate_question_answers(self):
        """Generate question answering analysis"""
        from app.core.document_processor import DocumentProcessor

        # Initialize document processor with loaded FAISS store
        from app.core.utils import create_document_processor
        
        # Get the store name from session (set during data room processing)
        store_name = self.session.vdr_store
        if not store_name:
            st.error("❌ No data room processed. Please process a data room first.")
            return
            
        document_processor = create_document_processor(store_name=store_name)

        try:
            questions_text = self.session.questions_text
            if not questions_text or not self.session.chunks:
                st.error("❌ No questions or document chunks available")
                return

            # Show progress indicator
            with st.spinner("🚀 Starting question analysis..."):
                try:
                    from app.core.search import search_and_analyze, load_prebuilt_questions
                    from pathlib import Path

                    # Step 1: Load pre-parsed questions (no LLM needed)
                    st.info("📋 Loading pre-parsed questions...")
                    
                    # Extract filename from questions path
                    if hasattr(self.session, 'questions_path') and self.session.questions_path:
                        questions_filename = Path(self.session.questions_path).name
                    else:
                        raise ValueError("No questions file selected. Please select a questions file in the sidebar first.")
                    
                    questions = load_prebuilt_questions(questions_filename)
                    self.session.questions = questions
                    st.info(f"Found {len(questions)} questions to process")

                    # Step 2: Use pre-built FAISS index
                    st.info("🔍 Setting up document search...")
                    if not document_processor.vector_store:
                        raise ValueError("No pre-built FAISS index loaded. Please ensure data room is processed first.")
                    vector_store = document_processor.vector_store

                    # Step 3: Process questions with batch processing
                    st.info("🤖 **AI Agent Processing:** Running batch analysis with ReAct reasoning...")
                    st.info("🧠 **Agent Status:** Using concurrent processing for faster results...")

                    question_answers = search_and_analyze(
                        questions,
                        vector_store,
                        self.ai_handler.session.agent.llm if self.ai_handler.is_agent_available() else None,
                        self.config.processing['relevancy_threshold'],
                        'questions',
                        store_name=getattr(document_processor, 'store_name', None)
                    )
                    self.session.question_answers = question_answers

                    # Complete
                    questions_list = question_answers.get('questions', [])
                    answered_count = sum(1 for a in questions_list if a.get('has_answer', False))
                    st.success(f"✅ Completed! {answered_count}/{len(questions)} questions answered")

                    status_message("✅ Question answering analysis completed!", "success")
                    st.rerun()

                except Exception as e:
                    logger.error(f"Questions processing failed: {e}")
                    display_generation_error("question analysis", e)
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            display_initialization_error("document processor", e)
