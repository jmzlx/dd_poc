#!/usr/bin/env python3
"""
DD-Checklist Main Application - Refactored Version

This is the main Streamlit application that orchestrates all components
using the new modular architecture for better maintainability.
"""

import os
import warnings
# Fix tokenizers parallelism warning early
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress all LangChain verbose warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_huggingface")
warnings.filterwarnings("ignore", message=".*Relevance scores must be between.*")
warnings.filterwarnings("ignore", message=".*No relevant docs were retrieved.*")

# Set up LangChain logging levels early
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langchain_huggingface").setLevel(logging.ERROR)

import streamlit as st

from pathlib import Path
from typing import Dict

# Import our refactored modules
from src import (
    init_config, DocumentProcessor,
    logger,
    render_project_selector,
    render_ai_settings, escape_markdown_math,
    get_mime_type, format_document_title
)
from src.document_processing import safe_execute
# Using Streamlit directly for simplicity
from src.ui_components import (
    render_file_selector, render_checklist_results, render_question_results,
    render_quick_questions, create_document_link
)
from src.services import (
    search_documents
)

from src.config import show_success, show_error, show_info

# Import LangGraph + Anthropic configuration
from src.ai import (
    DDChecklistAgent
)


class DDChecklistApp:
    """
    Main application class that orchestrates all components
    """

    def __init__(self):
        """Initialize the application"""
        # Initialize configuration
        self.config = init_config()
        
        # Initialize session state
        self._init_session_state()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title=self.config.ui.page_title,
            page_icon=self.config.ui.page_icon,
            layout=self.config.ui.layout
        )
        
        # Initialize services (will be loaded when needed)
        self.model_name = self.config.model.sentence_transformer_model
        self.document_processor = None
        self.agent = None
    
    def _init_session_state(self):
        """Initialize essential session state variables only"""
        essential_defaults = {
            'documents': {},
            'chunks': [],
            'embeddings': None,
            'checklist_results': {},
            'question_answers': {},
            'company_summary': "",
            'strategy_analysis': "",
            'agent': None,
            'is_processing': False
        }
        
        for key, default_value in essential_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def initialize_services(self):
        """Initialize core services"""
        if self.document_processor is None:
            self.document_processor = DocumentProcessor(self.model_name)
            
            # Restore document processor state from session state if available
            if (hasattr(st.session_state, 'chunks') and st.session_state.chunks and
                hasattr(st.session_state, 'embeddings') and st.session_state.embeddings is not None):
                
                self.document_processor.chunks = st.session_state.chunks
                self.document_processor.embeddings = st.session_state.embeddings
                # Note: Don't restore documents here - they'll be recreated from chunks if needed
    
    def setup_ai_agent(self, api_key: str, model_choice: str) -> bool:
        """
        Setup AI agent
        
        Args:
            api_key: Anthropic API key
            model_choice: Claude model to use
            
        Returns:
            True if agent was successfully initialized
        """        
        try:
            with st.spinner("Initializing AI agent..."):
                agent = DDChecklistAgent(api_key, model_choice)
                
                if agent.is_available():
                    st.session_state.agent = agent
                    self.agent = agent
                    show_success("✅ AI Agent ready")
                    
                    
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
        """Render consolidated summary and analysis tab"""
        # Strategy selector
        strategy_path, strategy_text = render_file_selector(
            self.config.paths.strategy_dir, "Strategy", "tab"
        )
        
        # Check if we have documents to display summaries
        if st.session_state.documents:
            # Create nested tabs for different analysis views
            overview_tab, analysis_tab = st.tabs(["🏢 Company Overview", "🎯 Strategic Analysis"])
            
            with overview_tab:
                self._render_report_section("overview", strategy_text=strategy_text)
            
            with analysis_tab:
                self._render_report_section("strategic", strategy_text=strategy_text)
        else:
            show_info("👈 Configure and process data room to see analysis")
    
    def _render_report_section(self, report_type: str, strategy_text: str = ""):
        """Unified report rendering for both overview and strategic analysis"""
        from src.services import generate_reports
        
        summary_key = f"{report_type}_summary"
        
        # Check prerequisites for strategic analysis
        if report_type == "strategic" and not st.session_state.checklist_results:
            st.warning("⚠️ Process data room with checklist first to enable strategic analysis")
            return
        
        # Auto-generate report if not already present and AI is available
        if (not st.session_state.get(summary_key, "") and st.session_state.agent):
            with st.spinner(f"🤖 Generating {report_type} analysis..."):
                data_room_name = (Path(list(st.session_state.documents.keys())[0]).parent.name 
                                if st.session_state.documents else "Unknown")
                
                st.session_state[summary_key] = generate_reports(
                    st.session_state.documents,
                    data_room_name,
                    strategy_text,
                    st.session_state.checklist_results,
                    report_type,
                    st.session_state.agent.llm if st.session_state.agent else None
                )
        
        # Display the report if available
        if st.session_state.get(summary_key, ""):
            st.markdown(st.session_state[summary_key])
            
            # Add export and regenerate buttons
            self._render_report_actions(report_type, summary_key)
    
    def _render_report_actions(self, report_type: str, summary_key: str):
        """Render export and regenerate actions for reports"""
        if report_type == "overview":
            col1, col2 = st.columns([1, 5])
            with col1:
                company_name = (Path(list(st.session_state.documents.keys())[0]).parent.name 
                               if st.session_state.documents else 'export')
                file_name = f"company_overview_{company_name}.md"
                st.download_button(
                    "📥 Export Summary",
                    data=f"# Company Overview\n\n{st.session_state[summary_key]}",
                    file_name=file_name,
                    mime="text/markdown",
                    key=f"export_{summary_key}"
                )
            with col2:
                if st.button(f"🔄 Regenerate {report_type.title()}"):
                    st.session_state[summary_key] = ""
                    st.rerun()
        else:
            col1, col2 = st.columns([1, 5])
            with col1:
                # Combined report export for strategic analysis
                combined_report = f"# Due Diligence Report\n\n"
                combined_report += f"## Company Overview\n\n{st.session_state.get('overview_summary', '')}\n\n"
                combined_report += f"## Strategic Analysis\n\n{st.session_state[summary_key]}"
                
                company_name = (Path(list(st.session_state.documents.keys())[0]).parent.name 
                               if st.session_state.documents else 'export')
                file_name = f"dd_report_{company_name}.md"
                st.download_button(
                    "📥 Export Report",
                    data=combined_report,
                    file_name=file_name,
                    mime="text/markdown",
                    key=f"export_combined_{summary_key}"
                )
            with col2:
                if st.button(f"🔄 Regenerate {report_type.title()}"):
                    st.session_state[summary_key] = ""
                    st.rerun()
    
    def render_analysis_tab(self, tab_type: str):
        """Unified rendering for checklist and questions tabs"""
        if tab_type == "checklist":
            # Checklist selector
            file_path, file_text = render_file_selector(
                self.config.paths.checklist_dir, "Checklist", "tab"
            )
            
            if not file_text:
                show_error("No checklists found in data/checklist directory")
                return
            
            # Render results if available
            render_checklist_results(st.session_state.checklist_results)
            
        elif tab_type == "questions":
            # Question list selector
            file_path, file_text = render_file_selector(
                self.config.paths.questions_dir, "Question List", "tab"
            )
            
            if not file_text:
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
        if not self.document_processor:
            self.initialize_services()
        
        # Use lower threshold for Q&A to get more relevant results
        qa_threshold = 0.25
        
        results = search_documents(
            self.document_processor,
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
                    prompt = (f"Question: {question}\n\n"
                             f"Relevant document excerpts:\n{context}\n\n"
                             f"Provide a comprehensive answer with citations to the sources.")
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
                        
                        # Create clickable link for the document
                        doc_path = result.get('path', result.get('full_path', ''))
                        doc_name = result['source']
                        doc_title = format_document_title(doc_name)
                        
                        if doc_path:
                            link_html = create_document_link(doc_path, doc_name, doc_title)
                            st.markdown(f"   {link_html} ({result['citation']})", unsafe_allow_html=True)
                        else:
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
                    
                    # Determine MIME type based on file extension
                    mime_type = get_mime_type(file_path)
                    
                    button_key = f"qacit_dl_{idx}_{question[:20]}".replace(" ", "_").replace("?", "")
                    
                    st.download_button(
                        label="📥 Download",
                        data=file_bytes,
                        file_name=result['source'],
                        mime=mime_type,
                        key=button_key,
                        help=f"Download {result['source']}"
                    )
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
    
    def process_data_room(self, data_room_path: str):
        """Simplified data room processing"""
        if not Path(data_room_path).exists():
            show_error(f"Data room path not found: {data_room_path}")
            st.session_state.is_processing = False
            return
        
        # Use safe_execute for the entire processing operation
        def process_operation():
            self.initialize_services()
            # Simple processing - load documents
            self.document_processor.load_data_room(data_room_path)
            
            # Store results in session state with simplified structure
            # Convert list of LangChain documents to dictionary format expected by UI
            documents_dict = {}
            for doc in self.document_processor.documents:
                file_path = doc.metadata.get('source', doc.metadata.get('path', 'unknown'))
                documents_dict[file_path] = {
                    'name': doc.metadata.get('name', Path(file_path).name if file_path != 'unknown' else 'unknown'),
                    'path': doc.metadata.get('path', ''),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
            
            st.session_state.documents = documents_dict
            st.session_state.chunks = self.document_processor.chunks
            st.session_state.embeddings = self.document_processor.embeddings
            
            # Process checklist and questions if available
            self._process_checklist_and_questions()
            
            # Clear any existing analysis to trigger regeneration
            st.session_state.company_summary = ""
            st.session_state.strategy_analysis = ""
            st.session_state.overview_summary = ""
            st.session_state.strategic_summary = ""
            
            show_success("✅ Data room processing complete! View results in the tabs above.")
            st.rerun()
            
        safe_execute(
            process_operation,
            None,
            "Data room processing"
        )
        
        st.session_state.is_processing = False
    
    def _process_checklist_and_questions(self):
        """Process checklist and questions after documents are loaded"""
        from src.services import parse_checklist, parse_questions, create_vector_store, search_and_analyze, load_default_file
        
        # Load default checklist if available
        checklist_text = load_default_file(Path(self.config.paths.checklist_dir), "*.md")
        if checklist_text and self.document_processor.chunks:
            try:
                # Parse checklist
                checklist = parse_checklist(checklist_text)
                st.session_state.checklist = checklist
                
                # Create vector store from chunks for processing
                vector_store = create_vector_store(self.document_processor.chunks, self.model_name)
                
                # Process checklist items
                checklist_results = search_and_analyze(
                    checklist,
                    vector_store,
                    self.agent.llm if self.agent else None,
                    self.config.processing.similarity_threshold,
                    'items'
                )
                st.session_state.checklist_results = checklist_results
                logger.info("✅ Checklist processing completed")
            except Exception as e:
                logger.error(f"Checklist processing failed: {e}")
        
        # Load default questions if available  
        questions_text = load_default_file(Path(self.config.paths.questions_dir), "*.md")
        if questions_text and self.document_processor.chunks:
            try:
                # Parse questions
                questions = parse_questions(questions_text)
                st.session_state.questions = questions
                
                # Create vector store from chunks for processing (reuse if already created)
                if 'vector_store' not in locals():
                    vector_store = create_vector_store(self.document_processor.chunks, self.model_name)
                
                # Process questions
                question_answers = search_and_analyze(
                    questions,
                    vector_store,
                    self.agent.llm if self.agent else None,
                    self.config.processing.relevancy_threshold,
                    'questions'
                )
                st.session_state.question_answers = question_answers
                logger.info("✅ Questions processing completed")
            except Exception as e:
                logger.error(f"Questions processing failed: {e}")
    
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
            self.render_analysis_tab("checklist")
        
        with tab3:
            self.render_analysis_tab("questions")
        
        with tab4:
            self.render_qa_tab()
        
        # Processing complete message is handled in process_data_room function
        
        # Simplified processing trigger
        if process_button and selected_data_room_path and not st.session_state.is_processing:
            st.session_state.is_processing = True
            self.process_data_room(selected_data_room_path)


def main():
    """Main application entry point"""
    app = DDChecklistApp()
    app.run()


if __name__ == "__main__":
    main()
