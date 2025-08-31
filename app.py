#!/usr/bin/env python3
"""
DD-Checklist Main Application - Refactored Version

This is the main Streamlit application that orchestrates all components
using the new modular architecture for better maintainability.
"""

import os
# Fix tokenizers parallelism warning early
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import our refactored modules
from src import (
    get_config, init_config,
    DocumentProcessor, DDChecklistService,
    logger, handle_exceptions, safe_execute, ErrorHandler,
    render_project_selector, render_ai_settings, escape_markdown_math
)
from src.ui_components import (
    render_file_selector, render_progress_section, render_metrics_row,
    render_checklist_results, render_question_results, render_quick_questions
)
from src.services import ReportGenerator
from src.utils import ProgressTracker, show_success, show_error, show_info

# Import LangGraph + Anthropic configuration
try:
    from src.ai_integration import (
        DDChecklistAgent,
        LANGGRAPH_AVAILABLE,
        batch_summarize_documents,
        create_document_embeddings_with_summaries,
        match_checklist_with_summaries,
        generate_checklist_descriptions
    )
    LLM_AVAILABLE = LANGGRAPH_AVAILABLE
except ImportError:
    LLM_AVAILABLE = False
    DDChecklistAgent = None


class DDChecklistApp:
    """
    Main application class that orchestrates all components
    """
    
    def __init__(self):
        """Initialize the application"""
        # Initialize configuration
        self.config = init_config().get_config()
        
        # Initialize session state
        self._init_session_state()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title=self.config.ui.page_title,
            page_icon=self.config.ui.page_icon,
            layout=self.config.ui.layout
        )
        
        # Initialize services (will be loaded when needed)
        self.model = None
        self.service = None
        self.agent = None
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'documents': {},
            'chunks': [],
            'embeddings': None,
            'checklist': {},
            'checklist_results': {},
            'questions': [],
            'question_answers': {},
            'strategy_text': "",
            'strategy_analysis': "",
            'company_summary': "",
            'agent': None,
            'doc_embeddings_data': None,
            'just_processed': False,
            'is_processing': False,
            'trigger_processing': False,
            'processing_path': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @st.cache_resource
    def load_model(_self) -> SentenceTransformer:
        """Load the sentence transformer model"""
        with ErrorHandler("Failed to load AI model"):
            return SentenceTransformer(_self.config.model.sentence_transformer_model)
    
    def initialize_services(self):
        """Initialize core services"""
        if self.model is None:
            self.model = self.load_model()
        
        if self.service is None:
            self.service = DDChecklistService(self.model, self.agent)
            
            # Restore document processor state from session state if available
            if (hasattr(st.session_state, 'chunks') and st.session_state.chunks and
                hasattr(st.session_state, 'embeddings') and st.session_state.embeddings is not None):
                
                self.service.document_processor.chunks = st.session_state.chunks
                self.service.document_processor.embeddings = st.session_state.embeddings
                self.service.document_processor.documents = st.session_state.get('documents', {})
                
                # Ensure the document processor has the model
                self.service.document_processor.model = self.model
    
    def setup_ai_agent(self, api_key: str, model_choice: str) -> bool:
        """
        Setup AI agent if enabled
        
        Args:
            api_key: Anthropic API key
            model_choice: Claude model to use
            
        Returns:
            True if agent was successfully initialized
        """
        if not LLM_AVAILABLE or not DDChecklistAgent:
            show_error("AI packages not installed")
            return False
        
        try:
            with st.spinner("Initializing AI agent..."):
                agent = DDChecklistAgent(api_key, model_choice)
                
                if agent.is_available():
                    st.session_state.agent = agent
                    self.agent = agent
                    show_success("✅ AI Agent ready")
                    
                    # Update service with agent
                    if self.service:
                        self.service.report_generator = ReportGenerator(agent)
                    
                    return True
                else:
                    show_error("❌ Failed to initialize agent")
                    return False
        except Exception as e:
            show_error(f"Agent initialization failed: {str(e)}")
            return False
    
    def render_sidebar(self) -> tuple:
        """
        Render sidebar with project selection and AI settings
        
        Returns:
            Tuple of (selected_data_room_path, use_ai_features, process_button)
        """
        with st.sidebar:
            # Project and data room selection
            selected_project_path, selected_data_room_path = render_project_selector()
            
            # Process button
            button_text = "⏳ Processing..." if st.session_state.is_processing else "🚀 Process Data Room"
            process_button = st.button(
                button_text, 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.is_processing
            )
            
            if process_button:
                show_success("Processing... Check main area for progress")
            
            st.divider()
            
            # AI settings
            use_ai_features, api_key, model_choice = render_ai_settings()
            
            # Initialize AI agent if enabled
            if use_ai_features and api_key:
                if not hasattr(st.session_state, 'agent') or st.session_state.agent is None:
                    self.setup_ai_agent(api_key, model_choice)
                elif hasattr(st.session_state, 'agent') and st.session_state.agent:
                    self.agent = st.session_state.agent
            else:
                st.session_state.agent = None
                self.agent = None
        
        return selected_data_room_path, use_ai_features, process_button
    
    def render_summary_tab(self):
        """Render the summary and analysis tab"""
        # Strategy selector
        strategy_path, strategy_text = render_file_selector(
            self.config.paths.strategy_dir, "Strategy", "tab"
        )
        st.session_state.strategy_text = strategy_text
        
        # Check if we have documents to display summaries
        if st.session_state.documents:
            self._render_company_overview()
            self._render_strategic_analysis()
        else:
            show_info("👈 Configure and process data room to see analysis")
    
    def _render_company_overview(self):
        """Render company overview section"""
        st.subheader("🏢 Company Overview")
        
        # Auto-generate summary if not already present and AI is available
        if (not st.session_state.company_summary and 
            hasattr(st.session_state, 'agent') and st.session_state.agent):
            
            with st.spinner("🤖 Generating company overview..."):
                report_gen = ReportGenerator(st.session_state.agent)
                data_room_name = Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else "Unknown"
                st.session_state.company_summary = report_gen.generate_company_summary(
                    st.session_state.documents, data_room_name
                )
        
        # Display the company summary if available
        if st.session_state.company_summary:
            st.info(st.session_state.company_summary)
            
            # Add export and regenerate buttons
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    "📥 Export Summary",
                    data=f"# Company Overview\n\n{st.session_state.company_summary}",
                    file_name=f"company_overview_{Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else 'export'}.md",
                    mime="text/markdown"
                )
            with col2:
                if st.button("🔄 Regenerate Overview"):
                    st.session_state.company_summary = ""
                    st.rerun()
    
    def _render_strategic_analysis(self):
        """Render strategic analysis section"""
        st.subheader("🎯 Strategic Analysis")
        
        if not st.session_state.checklist_results:
            st.warning("⚠️ Process data room with checklist first to enable strategic analysis")
            return
        
        # Auto-generate analysis if not already present and AI is available
        if (not st.session_state.strategy_analysis and 
            hasattr(st.session_state, 'agent') and st.session_state.agent):
            
            with st.spinner("🤖 Generating strategic analysis..."):
                report_gen = ReportGenerator(st.session_state.agent)
                st.session_state.strategy_analysis = report_gen.generate_strategic_analysis(
                    st.session_state.strategy_text,
                    st.session_state.checklist_results,
                    st.session_state.documents
                )
        
        if st.session_state.strategy_analysis:
            st.info(st.session_state.strategy_analysis)
            
            # Add export and regenerate buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                # Combined report export
                combined_report = f"# Due Diligence Report\n\n"
                combined_report += f"## Company Overview\n\n{st.session_state.company_summary}\n\n"
                combined_report += f"## Strategic Analysis\n\n{st.session_state.strategy_analysis}"
                
                st.download_button(
                    "📥 Export Report",
                    data=combined_report,
                    file_name=f"dd_report_{Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else 'export'}.md",
                    mime="text/markdown"
                )
            with col2:
                if st.button("🔄 Regenerate Analysis"):
                    st.session_state.strategy_analysis = ""
                    st.rerun()
    
    def render_checklist_tab(self):
        """Render the checklist matching tab"""
        # Checklist selector
        checklist_path, checklist_text = render_file_selector(
            self.config.paths.checklist_dir, "Checklist", "tab"
        )
        
        if not checklist_text:
            show_error("No checklists found in data/checklist directory")
            return
        
        # Render results if available
        render_checklist_results(st.session_state.checklist_results)
    
    def render_questions_tab(self):
        """Render the questions tab"""
        # Question list selector
        questions_path, questions_text = render_file_selector(
            self.config.paths.questions_dir, "Question List", "tab"
        )
        
        if not questions_text:
            show_info("No question lists found in data/questions/")
            return
        
        # Render results if available
        render_question_results(st.session_state.question_answers)
    
    def render_qa_tab(self):
        """Render the Q&A with citations tab"""
        if not st.session_state.chunks:
            show_info("👈 Process data room first to enable Q&A")
            return
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main risks? What is the revenue model? Who are the key customers?"
        )
        
        # Quick question buttons
        quick_question = render_quick_questions()
        if quick_question:
            question = quick_question
        
        st.divider()
        
        if question:
            self._handle_qa_query(question)
    
    def _handle_qa_query(self, question: str):
        """Handle Q&A query and display results"""
        if not self.service:
            self.initialize_services()
        
        # Use lower threshold for Q&A to get more relevant results
        qa_threshold = 0.25
        
        results = self.service.search_documents(
            question, 
            top_k=self.config.ui.top_k_search_results,
            threshold=qa_threshold
        )
        
        if results:
            # Use agent to synthesize answer if available
            if (hasattr(st.session_state, 'agent') and st.session_state.agent and 
                hasattr(st.session_state.agent, 'llm')):
                
                st.markdown("### 🤖 AI Agent's Answer")
                with st.spinner("Agent analyzing documents..."):
                    # Convert results to document format for context
                    context = "\n\n".join([f"From {r['source']}:\n{r['text']}" for r in results[:3]])
                    # Use LLM directly for more reliable answers
                    from langchain_core.messages import HumanMessage
                    prompt = f"Question: {question}\n\nRelevant document excerpts:\n{context}\n\nProvide a comprehensive answer with citations to the sources."
                    response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                    # Clean up any leading whitespace and escape math characters
                    answer_text = escape_markdown_math(response.content.strip())
                    st.markdown(answer_text)
                st.divider()
            
            st.markdown("### 📚 Source Documents")
            
            # Display source documents with download buttons
            for i, result in enumerate(results[:3], 1):
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        excerpt = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                        st.markdown(f"{i}. \"{excerpt}\"")
                        st.caption(f"   📄 {result['source']} ({result['citation']})")
                    
                    with col2:
                        self._render_qa_download_button(result, i, question)
        else:
            st.warning("No relevant information found for your question.")
    
    def _render_qa_download_button(self, result: Dict, idx: int, question: str):
        """Render download button for Q&A results"""
        doc_path = result.get('path', '')
        if doc_path:
            try:
                file_path = Path(doc_path)
                if not file_path.is_absolute():
                    file_path = Path("data") / file_path
                
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    button_key = f"qacit_dl_{idx}_{question[:20]}".replace(" ", "_").replace("?", "")
                    
                    st.download_button(
                        label="📥 Download",
                        data=file_bytes,
                        file_name=result['source'],
                        key=button_key,
                        help=f"Download {result['source']}"
                    )
            except Exception:
                pass
    
    @handle_exceptions(show_error=True)
    def process_data_room(self, data_room_path: str):
        """
        Process the selected data room
        
        Args:
            data_room_path: Path to the data room to process
        """
        if not Path(data_room_path).exists():
            show_error(f"Data room path not found: {data_room_path}")
            st.session_state.is_processing = False  # Reset flag on error
            return
        
        try:
            # Initialize services
            self.initialize_services()
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 🚀 Processing Data Room")
                
                # Define step weights based on expected complexity/duration
                step_weights = {
                    1: 1.0,    # Scanning data room (fast)
                    2: 0.5,    # Found documents (instant)
                    3: 8.0,    # Generate AI summaries (very slow - depends on doc count)
                    4: 0.5,    # AI summaries complete (instant)
                    5: 1.0,    # Loading checklist and questions (fast)
                    6: 0.5,    # Checklist and questions loaded (instant)
                    7: 3.0,    # Generate checklist descriptions (moderate)
                    8: 0.5,    # Descriptions generated (instant)
                    9: 2.0,    # Match checklist to documents (moderate)
                    10: 0.5,   # Checklist matching complete (instant)
                    11: 2.0,   # Answer questions (moderate)
                    12: 0.5    # Complete (instant)
                }
                
                tracker = ProgressTracker(12, "Processing", step_weights)
                
                # Step 1: Load documents
                tracker.update(1, f"Scanning data room: {Path(data_room_path).name}")
                load_results = self.service.document_processor.load_data_room(data_room_path)
                st.session_state.documents = self.service.document_processor.documents
                st.session_state.chunks = self.service.document_processor.chunks
                st.session_state.embeddings = self.service.document_processor.embeddings
                
                tracker.update(2, f"Found {load_results['documents_count']} documents")
                
                # Step 2: Generate AI summaries if agent available
                if hasattr(st.session_state, 'agent') and st.session_state.agent:
                    doc_count = len(st.session_state.documents)
                    tracker.update(3, f"Generating AI summaries for {doc_count} documents...")
                    
                    # Adjust weight for step 3 based on actual document count
                    # More documents = longer processing time
                    if doc_count > 50:
                        step_weights[3] = min(15.0, doc_count * 0.15)  # Scale with doc count, cap at 15
                    elif doc_count > 20:
                        step_weights[3] = doc_count * 0.2  # 4-10 weight for 20-50 docs
                    
                    # Recalculate total weight
                    tracker.total_weight = sum(step_weights.values())
                    
                    # Convert documents for summarization
                    docs_for_summary = []
                    for path, doc_info in st.session_state.documents.items():
                        docs_for_summary.append({
                            'name': doc_info['name'],
                            'path': doc_info['rel_path'],
                            'content': doc_info.get('content', '')[:1500],
                            'metadata': doc_info.get('metadata', {})
                        })
                    
                    # Create a separate progress tracker for batch summarization
                    st.session_state.summary_progress = st.progress(0, text="📝 Starting document summarization...")
                    
                    # Batch summarize
                    summarized_docs = batch_summarize_documents(
                        docs_for_summary, 
                        st.session_state.agent.llm,
                        batch_size=self.config.processing.batch_size
                    )
                    
                    # Clean up summary progress tracker
                    if 'summary_progress' in st.session_state:
                        st.session_state.summary_progress.progress(1.0, text="✅ Document summarization complete")
                        del st.session_state.summary_progress
                    
                    # Store summaries
                    for doc in summarized_docs:
                        for path, doc_info in st.session_state.documents.items():
                            if doc_info['rel_path'] == doc['path']:
                                doc_info['summary'] = doc.get('summary', '')
                    
                    # Create embeddings using summaries
                    st.session_state.doc_embeddings_data = create_document_embeddings_with_summaries(
                        summarized_docs, self.model
                    )
                    
                    tracker.update(4, f"AI summaries complete ({doc_count} documents processed)")
                else:
                    tracker.update(4, "Skipping AI summaries (not enabled)")
                
                # Step 3: Parse checklist and questions
                tracker.update(5, "Loading checklist and questions...")
                
                # Load default checklist
                checklist_text = self._load_default_file(self.config.paths.checklist_path, "*.md")
                if checklist_text:
                    st.session_state.checklist = self.service.checklist_parser.parse_checklist(checklist_text)
                
                # Load default questions
                questions_text = self._load_default_file(self.config.paths.questions_path, "*.md")
                if questions_text:
                    st.session_state.questions = self.service.question_parser.parse_questions(questions_text)
                
                tracker.update(6, "Checklist and questions loaded")
                
                # Step 7: Generate checklist descriptions if AI is available
                if (hasattr(st.session_state, 'agent') and st.session_state.agent and 
                    st.session_state.checklist):
                    
                    tracker.update(7, "Generating checklist item descriptions...")
                    
                    # Create progress tracker for descriptions
                    st.session_state.description_progress = st.progress(0, text="📝 Generating descriptions...")
                    
                    # Generate enhanced descriptions for better matching
                    st.session_state.checklist = generate_checklist_descriptions(
                        st.session_state.checklist,
                        st.session_state.agent.llm,
                        batch_size=self.config.processing.batch_size
                    )
                    
                    # Clean up progress tracker
                    if 'description_progress' in st.session_state:
                        st.session_state.description_progress.progress(1.0, text="✅ Descriptions generated")
                        del st.session_state.description_progress
                    
                    tracker.update(8, "Checklist descriptions generated")
                else:
                    tracker.update(8, "Skipping description generation (AI not enabled)")
                
                # Step 9: Match checklist to documents
                if st.session_state.checklist and st.session_state.chunks:
                    tracker.update(9, "Matching checklist to documents...")
                    
                    if hasattr(st.session_state, 'doc_embeddings_data') and st.session_state.doc_embeddings_data:
                        # Use AI-enhanced matching with generated descriptions
                        st.session_state.checklist_results = match_checklist_with_summaries(
                            st.session_state.checklist,
                            st.session_state.doc_embeddings_data,
                            self.model,
                            self.config.processing.similarity_threshold
                        )
                    else:
                        # Use traditional matching
                        st.session_state.checklist_results = self.service.checklist_matcher.match_checklist_to_documents(
                            st.session_state.checklist,
                            st.session_state.chunks,
                            st.session_state.embeddings,
                            self.config.processing.similarity_threshold
                        )
                    
                    tracker.update(10, "Checklist matching complete")
                
                # Step 11: Answer questions
                if (st.session_state.questions and st.session_state.chunks and 
                    st.session_state.embeddings is not None):
                    
                    tracker.update(11, "Answering due diligence questions...")
                    
                    st.session_state.question_answers = self.service.question_answerer.answer_questions_with_chunks(
                        st.session_state.questions,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        self.config.processing.similarity_threshold
                    )
                    
                    answered_count = sum(1 for a in st.session_state.question_answers.values() if a['has_answer'])
                    tracker.update(12, f"Answered {answered_count}/{len(st.session_state.questions)} questions")
                
                tracker.complete("Processing complete!")
                
                # Small delay before clearing
                import time
                time.sleep(1.5)
                progress_container.empty()
            
            # Reset processing flag and mark as just processed on success
            st.session_state.is_processing = False
            st.session_state.just_processed = True
            st.rerun()
            
        except Exception:
            # Reset processing flag on any error
            st.session_state.is_processing = False
            raise  # Let decorator handle error display
    
    def _load_default_file(self, directory: Path, pattern: str) -> str:
        """Load the first file matching pattern from directory"""
        try:
            files = list(directory.glob(pattern))
            if files:
                return files[0].read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not load default file from {directory}: {e}")
        return ""
    
    def run(self):
        """Run the main application"""
        # Render header
        st.title("🤖 AI Due Diligence")
        st.markdown("**Intelligent M&A Analysis:** Strategic assessment, automated document review, and AI-powered insights")
        
        # Render sidebar and get selections
        selected_data_room_path, use_ai_features, process_button = self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Summary & Analysis", 
            "📊 Checklist Matching", 
            "❓ Due Diligence Questions", 
            "💬 Q&A with Citations"
        ])
        
        with tab1:
            self.render_summary_tab()
        
        with tab2:
            self.render_checklist_tab()
        
        with tab3:
            self.render_questions_tab()
        
        with tab4:
            self.render_qa_tab()
        
        # Show success message if just processed
        if st.session_state.just_processed:
            show_success("✅ Data room processing complete! View results in the tabs above.")
            st.session_state.just_processed = False
        
        # Handle processing trigger
        if process_button and selected_data_room_path and not st.session_state.is_processing:
            # Set trigger and path for next render
            st.session_state.trigger_processing = True
            st.session_state.processing_path = selected_data_room_path
            st.session_state.is_processing = True
            st.rerun()
        
        # Execute processing if triggered
        if st.session_state.trigger_processing and st.session_state.processing_path:
            st.session_state.trigger_processing = False  # Reset trigger
            processing_path = st.session_state.processing_path
            st.session_state.processing_path = None
            self.process_data_room(processing_path)


def main():
    """Main application entry point"""
    app = DDChecklistApp()
    app.run()


if __name__ == "__main__":
    main()
