#!/usr/bin/env python3
"""
Company Analysis Tab Component

Handles comprehensive due diligence analysis combining company overview 
and strategic assessment using a unified ReAct agent.
"""

import streamlit as st
from typing import List, Dict, Any

from app.ui.tabs.tab_base import TabBase
from app.ui.ui_components import status_message, progress_status_tracker
from app.core.logging import logger


class CompanyAnalysisTab(TabBase):
    """
    Company analysis tab that provides comprehensive due diligence status reporting
    for Corporate Development teams, including data room analysis and process tracking.
    """

    def render(self):
        """Render the company analysis tab"""
        if not self._check_documents_available():
            return

        # Generate button for comprehensive analysis
        button_clicked = self._render_generate_buttons(
            "📊 Generate Due Diligence Status Report",
            "regenerate_company_analysis_btn",
            "strategic_company_summary",
            "Generate comprehensive due diligence status report for Corporate Development team with data room analysis and process updates"
        )

        # Generate or display content
        if self._should_generate_content(button_clicked, "strategic_company_summary"):
            self._generate_comprehensive_analysis()
        else:
            self._render_comprehensive_content_or_placeholder()

    def _generate_comprehensive_analysis(self):
        """Generate comprehensive company analysis using unified ReAct agent with full context preparation"""
        if not self._check_ai_availability():
            return

        if not self._check_processing_active():
            return

        # Set processing active
        self._set_processing_active(True)

        try:
            # STEP 1: Prepare comprehensive context by auto-running missing analyses
            with st.spinner("🔄 Preparing analysis context..."):
                self._prepare_comprehensive_context()

            # STEP 2: Generate comprehensive analysis with all available context
            analysis_progress = progress_status_tracker()
            analysis_steps = [
                "Initialize AI agent",
                "Analyze documents",
                "Generate report",
                "Validate citations"
            ]
            analysis_progress.initialize(analysis_steps, "🤖 AI Agent Analysis")
            
            # Use vdr_store for proper vector store access
            data_room_name = getattr(self.session, 'vdr_store', None) or self._get_data_room_name()
            
            analysis_progress.start_step(0, "🤖 Booting up AI ReAct Agent with advanced reasoning...")
            analysis_progress.complete_step(0, f"🎯 AI Agent ready - targeting {data_room_name}")
            
            analysis_progress.start_step(1, "🧠 AI Agent reading documents, extracting insights, reasoning about findings...")
            # Note: This step will run for the longest time, so we keep it in progress
            
            # Use comprehensive ReAct agent with full prepared context
            report_content, citation_info = self.ai_handler.generate_react_report(
                "comprehensive",
                data_room_name=data_room_name,
                strategy_text=self.session.strategy_text,
                checklist_results=self.session.checklist_results,
                question_answers=self.session.question_answers,
                project_info={'company_name': data_room_name, 'data_room_path': self.session.data_room_path}
            )
            
            analysis_progress.complete_step(1, "Document analysis completed")
            analysis_progress.start_step(2, "Generating report...")
            analysis_progress.complete_step(2, f"Report generated ({len(report_content) if report_content else 0} chars)")
            
            analysis_progress.start_step(3, "Validating citations...")

            # DEBUG: Log what was actually returned
            logger.info(f"RETURNED from generate_react_report: report_content={report_content is not None} ({len(report_content) if report_content else 0} chars), citation_info={citation_info}")

            # Validate that we have citations (they're now inline in the report)
            if not citation_info.get('has_citations', False):
                analysis_progress.error_step(3, "No citations found in analysis")
                logger.error("CRITICAL: No citations found in ReAct agent analysis")
                raise ValueError("Company analysis must include citations from source documents. No citations were found in the agent's analysis.")

            analysis_progress.complete_step(3, f"Citations validated: {len(citation_info.get('citations', []))} sources")

            # Store comprehensive analysis and citation info for rendering
            self.session.strategic_company_summary = report_content
            # Store citation info separately for download functionality
            setattr(self.session, 'strategic_company_citations', citation_info.get('citations', []))
            status_message("✅ Company analysis completed successfully!", "success")
            st.rerun()
                    
        except Exception as e:
            logger.error(f"Company analysis generation failed: {e}")
            status_message(f"Failed to generate company analysis: {str(e)}", "error")
        finally:
            # Always reset processing state
            self._set_processing_active(False)

    def _prepare_comprehensive_context(self):
        """Prepare comprehensive context by auto-running missing analyses and vectorizing results"""
        
        # Initialize progress tracker
        progress_tracker = progress_status_tracker()
        
        # Define all steps for better progress visualization
        steps = [
            "Verify data room processing",
            "Check vector store availability", 
            "Validate session data",
            "Check strategy context",
            "Run checklist analysis",
            "Run Q&A analysis",
            "Vectorize analysis results"
        ]
        
        progress_tracker.initialize(steps, "🔄 Preparing Analysis Context")
        
        try:
            # STEP 1: Verify data room is processed
            progress_tracker.start_step(0, "Checking data room...")
            data_room_name = getattr(self.session, 'vdr_store', None)
            
            if not data_room_name:
                progress_tracker.error_step(0, "No data room processed")
                st.error("❌ **No data room processed!** Please select and process a data room first.")
                logger.error("Cannot prepare context - no data room processed (session.vdr_store is None)")
                raise ValueError("No data room processed - cannot run Company Analysis without processed documents")
            
            progress_tracker.complete_step(0, f"Using data room: {data_room_name}")
            
            # STEP 2: Verify vector store access
            progress_tracker.start_step(1, "Loading vector store...")
            from app.core.utils import create_document_processor
            document_processor = create_document_processor(store_name=data_room_name)
            
            if not document_processor.vector_store:
                progress_tracker.error_step(1, f"No FAISS index found for '{data_room_name}'")
                st.error(f"❌ **No FAISS index found for '{data_room_name}'!** Please ensure data room processing completed successfully.")
                logger.error(f"Vector store not available for {data_room_name}")
                raise ValueError(f"No FAISS index found for '{data_room_name}' - cannot access documents")
            
            vector_count = document_processor.vector_store.index.ntotal
            progress_tracker.complete_step(1, f"Vector store ready: {vector_count} vectors")
            
            # STEP 3: Check session data availability
            progress_tracker.start_step(2, "Validating session data...")
            
            session_status = []
            if self.session.documents:
                session_status.append(f"{len(self.session.documents)} files")
            if self.session.chunks:
                session_status.append(f"{len(self.session.chunks)} chunks")
            
            if session_status:
                progress_tracker.complete_step(2, f"Session data: {' | '.join(session_status)}")
            else:
                progress_tracker.error_step(2, "Missing session data")
                raise ValueError("Missing documents or chunks in session data")
            
            # STEP 4: Check strategy context
            progress_tracker.start_step(3, "Checking strategy context...")
            if self.session.strategy_text:
                progress_tracker.complete_step(3, "Strategy context available")
            else:
                progress_tracker.complete_step(3, "No strategy - using document-only context")
            
            # STEP 5: Auto-run checklist matching if not done
            progress_tracker.start_step(4, "Processing checklist analysis...")
            if not self.session.checklist_results and self.session.checklist_text:
                # CRITICAL: Checklist analysis must succeed for comprehensive analysis
                self._run_checklist_analysis()
                progress_tracker.complete_step(4, "Checklist analysis completed")
            elif self.session.checklist_results:
                progress_tracker.complete_step(4, "Checklist results available")
            else:
                progress_tracker.complete_step(4, "No checklist - using document-only context")
            
            # STEP 6: Auto-run Q&A analysis if not done
            progress_tracker.start_step(5, "Processing Q&A analysis...")
            if not self.session.question_answers and self.session.questions_text:
                try:
                    self._run_qa_analysis()
                    progress_tracker.complete_step(5, "Q&A analysis completed")
                except Exception as e:
                    logger.warning(f"Auto-Q&A analysis failed: {e}")
                    progress_tracker.error_step(5, "Q&A analysis failed - continuing without")
            elif self.session.question_answers:
                progress_tracker.complete_step(5, "Q&A results available")
            else:
                progress_tracker.complete_step(5, "No questions - using document-only context")
            
            # STEP 7: Vectorize combined analysis results for agent access
            progress_tracker.start_step(6, "Vectorizing analysis results...")
            if self.session.checklist_results or self.session.question_answers:
                # CRITICAL: Vectorization must succeed for agent access to analysis results
                self._vectorize_analysis_results()
                progress_tracker.complete_step(6, "Analysis results vectorized and ready")
            else:
                progress_tracker.complete_step(6, "No additional analysis to vectorize")
        
        except Exception as e:
            logger.error(f"Context preparation failed: {e}")
            raise

    def _run_checklist_analysis(self):
        """Auto-run checklist matching analysis with proper embedding preloading"""
        from app.core.utils import create_document_processor
        from app.core.search import search_and_analyze, preload_checklist_embeddings
        
        # Get document processor
        store_name = self.session.vdr_store
        document_processor = create_document_processor(store_name=store_name)
        
        if not document_processor.vector_store:
            raise ValueError("No vector store available for checklist analysis")
        
        # CRITICAL: Ensure checklist embeddings are preloaded before any similarity calculations
        logger.info("Ensuring checklist embeddings are preloaded for auto-analysis...")
        try:
            preload_count = preload_checklist_embeddings()
            logger.info(f"✅ Preloaded {preload_count} checklist embeddings for auto-analysis")
        except Exception as e:
            logger.error(f"Failed to preload checklist embeddings for auto-analysis: {e}")
            raise ValueError(f"Cannot run checklist analysis without embeddings: {e}")
        
        # CRITICAL: Ensure document type embeddings are available in session
        logger.info("Ensuring document type embeddings are loaded for auto-analysis...")
        if not hasattr(self.session, 'document_type_embeddings') or not self.session.document_type_embeddings:
            try:
                from app.core.search import preload_document_type_embeddings
                type_embeddings = preload_document_type_embeddings(store_name)
                self.session.document_type_embeddings = type_embeddings
                logger.info(f"✅ Loaded {len(type_embeddings)} document type embeddings for auto-analysis")
            except Exception as e:
                logger.error(f"Failed to load document type embeddings for auto-analysis: {e}")
                raise ValueError(f"Cannot run checklist analysis without document type embeddings: {e}")
        else:
            logger.info(f"✅ Document type embeddings already available: {len(self.session.document_type_embeddings)} entries")
        
        # Load pre-parsed checklist structure (no LLM needed)
        from app.core.search import load_prebuilt_checklist
        from pathlib import Path
        
        # Extract filename from checklist path
        if hasattr(self.session, 'checklist_path') and self.session.checklist_path:
            checklist_filename = Path(self.session.checklist_path).name
        else:
            raise ValueError("No checklist file selected. Please select a checklist in the sidebar first.")
        
        checklist = load_prebuilt_checklist(checklist_filename)
        self.session.checklist = checklist
        
        # Process checklist matching with preloaded embeddings
        checklist_results = search_and_analyze(
            checklist,
            document_processor.vector_store,
            self.ai_handler.llm if self.ai_handler.is_agent_available() else None,
            self.config.processing['similarity_threshold'],
            'items',
            store_name=store_name,
            session=self.session
        )
        self.session.checklist_results = checklist_results

    def _run_qa_analysis(self):
        """Auto-run Q&A analysis"""
        from app.core.utils import create_document_processor
        from app.core.search import search_and_analyze
        
        # Get document processor
        store_name = self.session.vdr_store
        document_processor = create_document_processor(store_name=store_name)
        
        if not document_processor.vector_store:
            raise ValueError("No vector store available for Q&A analysis")
        
        # Load pre-parsed questions (no LLM needed for parsing)
        from app.core.search import load_prebuilt_questions
        from pathlib import Path
        
        # Extract filename from questions path
        if hasattr(self.session, 'questions_path') and self.session.questions_path:
            questions_filename = Path(self.session.questions_path).name
        else:
            raise ValueError("No questions file selected. Please select a questions file in the sidebar first.")
        
        questions = load_prebuilt_questions(questions_filename)
        self.session.questions = questions
        
        # Get LLM for question answering (not parsing)
        llm = self.ai_handler.llm if self.ai_handler.is_agent_available() else None
        
        # Process questions
        question_answers = search_and_analyze(
            questions,
            document_processor.vector_store,
            llm,
            self.config.processing['relevancy_threshold'],
            'questions',
            store_name=store_name
        )
        self.session.question_answers = question_answers

    def _vectorize_analysis_results(self):
        """Vectorize analysis results for agent search access"""
        from app.core.model_cache import get_cached_embeddings
        from app.core.config import get_app_config
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        
        config = get_app_config()
        embeddings = get_cached_embeddings(config.model['sentence_transformer_model'])
        
        # Create documents from analysis results
        analysis_docs = []
        
        # Add strategy content
        if self.session.strategy_text:
            strategy_doc = Document(
                page_content=self.session.strategy_text,
                metadata={'type': 'strategy', 'source': 'corporate_strategy', 'name': 'Strategic Context'}
            )
            analysis_docs.append(strategy_doc)
        
        # Add checklist insights
        if self.session.checklist_results:
            checklist_insights = self._extract_checklist_insights()
            for insight in checklist_insights:
                analysis_docs.append(Document(
                    page_content=insight['content'],
                    metadata={'type': 'checklist_insight', 'source': insight['source'], 'name': insight['name']}
                ))
        
        # Add Q&A insights
        if self.session.question_answers:
            qa_insights = self._extract_qa_insights()
            for insight in qa_insights:
                analysis_docs.append(Document(
                    page_content=insight['content'],
                    metadata={'type': 'qa_insight', 'source': insight['source'], 'name': insight['name']}
                ))
        
        # Create analysis vector store if we have content
        if analysis_docs:
            analysis_vector_store = FAISS.from_documents(analysis_docs, embeddings)
            # Store in session for agent access
            self.session.analysis_vector_store = analysis_vector_store
            logger.info(f"Created analysis vector store with {len(analysis_docs)} insights")
        else:
            logger.info("No analysis results to vectorize")

    def _extract_checklist_insights(self) -> List[Dict[str, str]]:
        """Extract insights from checklist results for vectorization"""
        insights = []
        
        if not self.session.checklist_results:
            return insights
        
        for category, items in self.session.checklist_results.items():
            if isinstance(items, dict):
                category_name = items.get('name', category)
                matched_items = items.get('matched_items', 0)
                total_items = items.get('total_items', 0)
                completion = (matched_items / total_items * 100) if total_items > 0 else 0
                
                # Create insight document
                insight_content = f"""Checklist Category: {category_name}
Completion Status: {completion:.1f}% complete ({matched_items}/{total_items} items matched)

{'Strong compliance' if completion > 80 else 'Moderate compliance' if completion > 50 else 'Compliance gaps identified'} in {category_name.lower()}.
{'No significant gaps in this area.' if completion > 80 else f'Missing {total_items - matched_items} required items.' if completion > 50 else 'Significant documentation gaps require attention.'}"""
                
                insights.append({
                    'content': insight_content,
                    'source': f'checklist_{category}',
                    'name': f'Checklist - {category_name}'
                })
        
        return insights

    def _extract_qa_insights(self) -> List[Dict[str, str]]:
        """Extract insights from Q&A results for vectorization"""
        insights = []
        
        if not self.session.question_answers:
            return insights
        
        questions_data = self.session.question_answers.get('questions', [])
        
        for q_data in questions_data:
            if isinstance(q_data, dict) and q_data.get('has_answer'):
                question = q_data.get('question', '')
                answer = q_data.get('answer', '')
                category = q_data.get('category', 'general')
                
                if question and answer:
                    # Create insight document
                    insight_content = f"""Due Diligence Question: {question}
Category: {category}

Analysis: {answer}

Key Finding: {answer[:200]}...
"""
                    
                    insights.append({
                        'content': insight_content,
                        'source': f'qa_{category}',
                        'name': f'Q&A - {question[:50]}...'
                    })
        
        return insights

    def _render_comprehensive_content_or_placeholder(self):
        """Render comprehensive analysis content with inline clickable citations and downloads"""
        content = self.session.strategic_company_summary
        citations = getattr(self.session, 'strategic_company_citations', [])
        
        if content:
            # Debug logging
            logger.info(f"Rendering company analysis content: {len(content)} characters")
            logger.info(f"Available citations for download: {len(citations)}")
            
            # Import the simple clickable file rendering function
            from app.ui.ui_components import render_content_with_clickable_citations
            
            # Render the analysis with simple clickable citations
            render_content_with_clickable_citations(content, citations)
            
            # Add compact referenced documents section with actual download buttons
            if citations:
                st.markdown("---")
                st.markdown("### 📚 Referenced Documents")
                st.markdown("*Additional downloads for all source documents mentioned in the analysis above:*")
                
                # Create download buttons in a clean grid layout
                cols = st.columns(min(3, len(citations)))  # Max 3 columns
                for i, citation in enumerate(citations):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        self._render_citation_download_button(citation, i+1)
            
            # Export button
            self._render_export_button(self._get_export_method_name(), self._get_download_key())
        else:
            status_message(
                "👆 Click 'Generate Due Diligence Status Report' to create comprehensive status update for Corporate Development team with data room analysis and process tracking", 
                "info"
            )
    
    def _render_citation_download_button(self, citation: Dict[str, Any], index: int):
        """Render a compact download button for a cited document"""
        doc_path = citation.get('path', '')
        doc_name = citation.get('name', 'Unknown Document')
        
        if not doc_path:
            st.caption(f"📄 {doc_name} (not available)")
            return
        
        try:
            # Use the same path resolution logic as the original ReportRenderer
            from app.ai.citation_manager import ReportRenderer
            resolved_path = ReportRenderer._resolve_document_path(doc_path)
            
            if resolved_path and resolved_path.exists():
                with open(resolved_path, 'rb') as f:
                    file_bytes = f.read()
                
                # Clean document name for display
                clean_name = doc_name.replace('.pdf', '').replace('.docx', '').replace('.doc', '')
                
                st.download_button(
                    label=f"📄 {clean_name}",
                    data=file_bytes,
                    file_name=resolved_path.name,
                    mime="application/pdf",
                    key=f"company_cite_download_{index}_{hash(str(resolved_path)) % 10000}",
                    help=f"Download: {doc_name}",
                    use_container_width=True
                )
            else:
                st.caption(f"📄 {doc_name} (file not found)")
                
        except Exception as e:
            logger.error(f"Failed to create download button for {doc_name}: {e}")
            st.caption(f"📄 {doc_name} (download error)")


    def _get_export_method_name(self) -> str:
        """Get export method name for strategic company reports"""
        return "export_strategic_company_report"

    def _get_download_key(self) -> str:
        """Get download button key for strategic company reports"""
        return "export_strategic_company_btn"
