#!/usr/bin/env python3
"""
AI Handler

Handles AI operations and coordinates between UI and AI service.
"""

from typing import Optional, List, Tuple, Dict, Any

import streamlit as st
from app.ui.session_manager import SessionManager
from app.services.ai_service import AIService, create_ai_service
from app.core.exceptions import AIError, ConfigError, create_ai_error
from app.ui.error_handler import handle_processing_errors
from app.core.logging import logger


class AIHandler:
    """
    AI handler that manages AI operations using the AI service.

    Provides a clean interface between UI and AI service.
    """

    def __init__(self, session: SessionManager):
        """Initialize handler with session manager"""
        self.session = session
        self._ai_service: Optional[AIService] = None

    @handle_processing_errors("AI service setup", "Please check your API key and try again")
    def setup_agent(self, api_key: str, model_choice: str) -> bool:
        """
        Setup AI service with given credentials.

        Args:
            api_key: Anthropic API key
            model_choice: Claude model to use

        Returns:
            True if AI service was successfully initialized

        Raises:
            AIError: If AI service setup fails
            ConfigError: If API key or model is invalid
        """
        # Get appropriate max_tokens for the model
        from app.core.config import get_app_config
        config = get_app_config()
        
        # Adjust max_tokens based on model limitations
        max_tokens = config.model['max_tokens']
        original_max_tokens = max_tokens
        
        if 'haiku' in model_choice.lower():
            # Claude Haiku has a maximum of 8192 output tokens
            max_tokens = min(max_tokens, 8192)
        elif 'sonnet' in model_choice.lower():
            # Claude Sonnet models can handle higher token counts
            max_tokens = min(max_tokens, 8192)  # Conservative limit for reliability
        
        if max_tokens != original_max_tokens:
            logger.info(f"Adjusted max_tokens for {model_choice}: {original_max_tokens} -> {max_tokens}")
        
        logger.info(f"Initializing AI service: model={model_choice}, max_tokens={max_tokens}, temperature={config.model['temperature']}")
        
        # Create AI service with proper token limits
        self._ai_service = create_ai_service(
            api_key=api_key, 
            model=model_choice,
            temperature=config.model['temperature'],
            max_tokens=max_tokens
        )

        # Check if service was created successfully
        if self._ai_service is None:
            raise create_ai_error(
                "AI service creation failed",
                recovery_hint="Please check your API key and try again"
            )

        # Test the service
        if self._ai_service.is_available:
            # Store the AI service in the session for other components to access
            self.session.agent = self._ai_service
            return True
        else:
            raise create_ai_error(
                "AI service initialization failed",
                recovery_hint="Please check your API key and network connection"
            )

    def is_agent_available(self) -> bool:
        """
        Check if AI service is available and ready.

        Returns:
            True if AI service is available
        """
        # Check local AI service first
        if self._ai_service is not None and self._ai_service.is_available:
            return True
        
        # Check session for existing agent
        if self.session.agent is not None:
            # Update local reference if session has an agent
            self._ai_service = self.session.agent
            return self._ai_service.is_available
        
        return False


    @handle_processing_errors("Report generation", "Please check your documents and try again")
    def generate_report(self, report_type: str, **kwargs) -> Optional[str]:
        """
        Generate a report using RAG-based analysis for better document coverage.

        Args:
            report_type: Type of report ('overview', 'strategic', 'checklist', 'questions')
            **kwargs: Additional arguments for report generation

        Returns:
            Generated report content or None if failed

        Raises:
            AIError: If report generation fails
        """
        if not self.is_agent_available():
            raise create_ai_error(
                "AI service not available",
                recovery_hint="Please configure your API key in the sidebar"
            )

        # Check if we should use RAG for this report type
        use_rag = report_type in ['overview', 'strategic']
        
        if use_rag:
            return self._generate_report_with_rag(report_type, **kwargs)
        else:
            # Fallback to original method for other types
            documents = kwargs.get('documents', {})
            strategy_text = kwargs.get('strategy_text')
            checklist_results = kwargs.get('checklist_results')

            return self._ai_service.analyze_documents(
                documents=documents,
                analysis_type=report_type,
                strategy_text=strategy_text,
                checklist_results=checklist_results
            )

    def _generate_report_with_rag(self, report_type: str, **kwargs) -> Optional[str]:
        """
        Generate report using RAG functionality for comprehensive document analysis.
        
        Args:
            report_type: Type of report ('overview', 'strategic')
            **kwargs: Additional arguments including data_room_name, strategy_text, checklist_results
            
        Returns:
            Generated report content using RAG
        """
        from app.core.utils import create_document_processor
        from app.core.search import search_and_analyze
        from langchain_core.messages import HumanMessage
        
        # Get necessary parameters with defaults to avoid undefined variable errors
        data_room_name = kwargs.get('data_room_name')
        strategy_text = kwargs.get('strategy_text', '')
        checklist_results = kwargs.get('checklist_results', {})
        question_answers = kwargs.get('question_answers', {})
        project_info = kwargs.get('project_info', {})
        
        if not data_room_name:
            raise create_ai_error(
                "Data room name required for RAG-based report generation",
                recovery_hint="Please ensure a data room is processed first"
            )
        
        try:
            # Initialize document processor with vector store
            document_processor = create_document_processor(store_name=data_room_name)
            
            if not document_processor.vector_store:
                raise create_ai_error(
                    f"No FAISS index found for '{data_room_name}'",
                    recovery_hint="Please ensure data room processing completed successfully"
                )
            
            vector_store = document_processor.vector_store
            
            # Create analysis queries based on report type
            if report_type == 'overview':
                analysis_queries = self._create_overview_queries()
            elif report_type == 'strategic':
                analysis_queries = self._create_strategic_queries()
            else:
                raise ValueError(f"Unsupported RAG report type: {report_type}")
            
            # Use RAG to get comprehensive document analysis
            logger.info(f"Running RAG analysis for {report_type} report with {len(analysis_queries)} queries")
            
            rag_results = search_and_analyze(
                analysis_queries,
                vector_store,
                llm=self.llm,
                threshold=0.2,  # Lower threshold for more comprehensive coverage
                search_type='questions',
                store_name=data_room_name
            )
            
            # Extract context from RAG results
            context_sections = []
            for question_result in rag_results.get('questions', []):
                if question_result.get('has_answer') and question_result.get('answer'):
                    context_sections.append(
                        f"**{question_result['question']}**\n{question_result['answer']}\n"
                    )
            
            if not context_sections:
                # Fail explicitly if RAG returns no results
                raise AIError("No relevant context found for analysis. Ensure documents are properly processed and contain relevant information.")
            
            # Create comprehensive prompt with RAG context
            context_text = "\n".join(context_sections)
            
            if report_type == 'overview':
                analysis_prompt = self._create_overview_synthesis_prompt(context_text, strategy_text, checklist_results)
            else:  # strategic
                analysis_prompt = self._create_strategic_synthesis_prompt(context_text, strategy_text, checklist_results)
            
            # Generate final report using the AI service
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"RAG-based report generation failed: {e}")
            raise create_ai_error(
                f"Failed to generate {report_type} report using RAG",
                recovery_hint="Please try again or ensure the data room was processed correctly"
            )

    def _create_overview_queries(self) -> List[dict]:
        """Create structured queries for comprehensive overview analysis"""
        return [
            {"question": "What is the company's business model and core operations?", "category": "business"},
            {"question": "What are the company's main products and services?", "category": "business"},
            {"question": "What is the company's market position and competitive landscape?", "category": "market"},
            {"question": "What are the company's key financial metrics and performance indicators?", "category": "financial"},
            {"question": "What are the company's main revenue streams and customer base?", "category": "financial"},
            {"question": "What are the company's key assets, capabilities, and competitive advantages?", "category": "strategic"},
            {"question": "What are the main operational, financial, and strategic risks?", "category": "risk"},
            {"question": "Who are the company's key management team and employees?", "category": "management"},
            {"question": "What is the company's organizational structure and governance?", "category": "governance"},
            {"question": "What regulatory or compliance issues does the company face?", "category": "compliance"}
        ]

    def _create_strategic_queries(self) -> List[dict]:
        """Create structured queries for comprehensive strategic analysis"""
        return [
            {"question": "What strategic value would this company bring to an acquirer?", "category": "strategic"},
            {"question": "What are the company's core competitive advantages and differentiators?", "category": "competitive"},
            {"question": "What market opportunities and growth potential does the company have?", "category": "growth"},
            {"question": "What are the key strategic risks and challenges facing the company?", "category": "risk"},
            {"question": "How does the company fit within industry consolidation trends?", "category": "industry"},
            {"question": "What synergies could be realized through acquisition?", "category": "synergies"},
            {"question": "What are the company's key strategic partnerships and relationships?", "category": "partnerships"},
            {"question": "What is the company's technology stack and innovation capabilities?", "category": "technology"},
            {"question": "What are the company's expansion plans and strategic initiatives?", "category": "expansion"},
            {"question": "What would be the key integration challenges and opportunities?", "category": "integration"}
        ]

    def _create_overview_synthesis_prompt(self, context_text: str, strategy_text: str = None, checklist_results: dict = None) -> str:
        """Create synthesis prompt for overview analysis using RAG context"""
        prompt = f"""Based on comprehensive document analysis, provide a detailed TARGET COMPANY OVERVIEW from an acquisition perspective.

**ANALYSIS CONTEXT:**
{context_text}

"""
        if strategy_text:
            prompt += f"**ACQUIRER'S STRATEGIC CONTEXT:**\n{strategy_text[:1000]}\n\n"
        
        if checklist_results:
            prompt += f"**DUE DILIGENCE FINDINGS:**\n{str(checklist_results)[:1000]}\n\n"
        
        prompt += """**PROVIDE A COMPREHENSIVE TARGET COMPANY ANALYSIS COVERING:**

1. **Company Overview**: Business model, market position, and core operations of the target
2. **Strategic Value**: Why this target company would be attractive for acquisition
3. **Competitive Strengths**: Key assets, capabilities, and competitive advantages the target brings
4. **Risk Assessment**: Main operational, financial, and strategic risks associated with the target
5. **Financial Health**: Target company's financial position and performance indicators
6. **Acquisition Rationale**: How the target fits acquisition criteria and strategic objectives

Focus on analyzing the target company as a potential acquisition candidate. Be specific, factual, and highlight both opportunities and concerns from an acquirer's due diligence perspective. Use the comprehensive document analysis above to provide detailed insights."""

        return prompt

    def _create_strategic_synthesis_prompt(self, context_text: str, strategy_text: str = None, checklist_results: dict = None) -> str:
        """Create synthesis prompt for strategic analysis using RAG context"""
        prompt = f"""Based on comprehensive document analysis, provide a detailed STRATEGIC ASSESSMENT of this target company from an acquisition perspective.

**STRATEGIC ANALYSIS CONTEXT:**
{context_text}

"""
        if strategy_text:
            prompt += f"**ACQUIRER'S STRATEGIC CONTEXT:**\n{strategy_text[:1000]}\n\n"
        
        if checklist_results:
            prompt += f"**DUE DILIGENCE FINDINGS:**\n{str(checklist_results)[:1000]}\n\n"
        
        prompt += """**PROVIDE A COMPREHENSIVE STRATEGIC ASSESSMENT COVERING:**

1. **Strategic Positioning**: How the target fits within the broader market and industry landscape
2. **Value Creation Opportunities**: Specific ways the acquisition could create value
3. **Competitive Analysis**: Target's competitive position and differentiation strategies
4. **Growth Strategy**: Target's growth opportunities and expansion potential
5. **Synergy Assessment**: Potential operational, financial, and strategic synergies
6. **Risk Analysis**: Key strategic risks and mitigation strategies
7. **Integration Considerations**: Critical factors for successful integration
8. **Recommendation**: Go/No-Go recommendation with clear rationale

Focus on strategic implications and provide actionable insights for acquisition decision-making. Be specific about opportunities and risks, and provide clear recommendations based on the comprehensive analysis."""

        return prompt


    def generate_react_report(self, report_type: str, **kwargs) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate comprehensive report using ReAct agents with inline citations.
        
        Args:
            report_type: Type of report ('overview', 'strategic', 'comprehensive')
            **kwargs: Additional arguments including data_room_name
            
        Returns:
            Tuple of (formatted_report, citation_info) where citation_info contains download data
        """
        if not self.is_agent_available():
            raise create_ai_error(
                "AI service not available",
                recovery_hint="Please configure your API key in the sidebar"
            )
        
        # Extract all parameters at the beginning
        data_room_name = kwargs.get('data_room_name')
        strategy_text = kwargs.get('strategy_text', '')
        checklist_results = kwargs.get('checklist_results', {})
        question_answers = kwargs.get('question_answers', {})
        project_info = kwargs.get('project_info', {})
        
        if not data_room_name:
            raise create_ai_error(
                "Data room name required for ReAct agent analysis",
                recovery_hint="Please ensure a data room is processed first"
            )
        
        try:
            from app.core.utils import create_document_processor
            from app.ai.react_agents import create_comprehensive_dd_agent
            from app.ai.citation_manager import create_comprehensive_report
            from langchain_core.messages import HumanMessage
            
            # Initialize document processor with vector store
            document_processor = create_document_processor(store_name=data_room_name)
            
            if not document_processor.vector_store:
                raise create_ai_error(
                    f"No FAISS index found for '{data_room_name}'",
                    recovery_hint="Please ensure data room processing completed successfully"
                )
            
            vector_store = document_processor.vector_store
            logger.info(f"Starting comprehensive due diligence analysis using unified ReAct agent")
            logger.info(f"Document processor vector store: {type(vector_store)} with {vector_store.index.ntotal} vectors")
            logger.info(f"Data room name: {data_room_name}")
            logger.info(f"Session vdr_store: {getattr(self.session, 'vdr_store', 'NOT SET')}")
            
            # Create unified comprehensive due diligence agent with full context access
            # CRITICAL: Pass the vector_store so tools can actually search documents
            # Also pass analysis vector store if available
            analysis_vector_store = getattr(self.session, 'analysis_vector_store', None)
            
            agent, tools = create_comprehensive_dd_agent(
                self.llm, 
                data_room_name, 
                vector_store,  # This was the issue - tools need the vector store!
                strategy_text=strategy_text,
                checklist_results=checklist_results,
                question_answers=question_answers,
                project_info=project_info,
                analysis_vector_store=analysis_vector_store
            )
            
            # Verify tools have vector store access with detailed debugging
            logger.info("DEBUGGING: Verifying tool vector store access...")
            for tool in tools:
                if hasattr(tool, 'vector_store'):
                    vs_status = tool.vector_store is not None
                    vs_type = type(tool.vector_store) if tool.vector_store else "None"
                    logger.info(f"Tool {tool.name}: vector_store={vs_status} (type: {vs_type})")
                    
                    if tool.vector_store is None:
                        logger.error(f"CRITICAL: Tool {tool.name} has no vector store access!")
                        logger.error(f"  - Passed vector_store parameter: {vector_store is not None}")
                        logger.error(f"  - Tool store_name: {getattr(tool, 'store_name', 'NOT SET')}")
                elif hasattr(tool, 'analysis_vector_store'):
                    avs_status = tool.analysis_vector_store is not None
                    logger.info(f"Tool {tool.name}: analysis_vector_store={avs_status}")
                else:
                    logger.info(f"Tool {tool.name}: context tool (no vector store needed)")
            
            # CRITICAL CHECK: If any document tools have None vector store, fail immediately
            document_tools = ['document_search', 'cross_reference', 'financial_analysis', 'competitive_analysis']
            failed_tools = []
            
            for tool in tools:
                if tool.name in document_tools and hasattr(tool, 'vector_store') and tool.vector_store is None:
                    failed_tools.append(tool.name)
            
            if failed_tools:
                error_msg = f"CRITICAL FAILURE: Document tools have no vector store access: {failed_tools}"
                logger.error(error_msg)
                logger.error(f"Vector store passed to agent: {vector_store is not None} ({type(vector_store)})")
                logger.error(f"Store name: {data_room_name}")
                raise create_ai_error(
                    f"Tools cannot access documents: {failed_tools} have no vector store",
                    recovery_hint="This indicates a bug in tool initialization. Check logs for vector store passing details."
                )
            
            # Build comprehensive context for the agent
            context_sections = []
            
            # Add project information
            if project_info:
                company_name = project_info.get('company_name', data_room_name)
                context_sections.append(f"**PROJECT CONTEXT:**\nTarget Company: {company_name}")
                if project_info.get('data_room_path'):
                    context_sections.append(f"Data Room: {project_info['data_room_path']}")
            
            # Add corporate strategy context
            if strategy_text and strategy_text.strip():
                context_sections.append(f"**ACQUIRER'S STRATEGIC CONTEXT:**\n{strategy_text[:1000]}...")
            
            # Add checklist results context
            if checklist_results:
                context_sections.append(f"**DUE DILIGENCE CHECKLIST STATUS:**")
                for category, items in checklist_results.items():
                    if isinstance(items, dict):
                        matched = items.get('matched_items', 0)
                        total = items.get('total_items', 0)
                        context_sections.append(f"- {items.get('name', category)}: {matched}/{total} items matched")
            
            # Add question answers context
            if question_answers and isinstance(question_answers, dict):
                answered_questions = []
                for q_data in question_answers.get('questions', []):
                    if isinstance(q_data, dict) and q_data.get('has_answer'):
                        question = q_data.get('question', '')
                        answer = q_data.get('answer', '')
                        if question and answer:
                            answered_questions.append(f"Q: {question[:100]}...\nA: {answer[:200]}...")
                
                if answered_questions:
                    context_sections.append(f"**PREVIOUS DUE DILIGENCE Q&A:**")
                    context_sections.extend(answered_questions[:3])  # Limit to 3 most relevant
            
            # Build the complete analysis request with context
            context_text = "\n\n".join(context_sections) if context_sections else ""
            
            # Explicit analysis request with clear synthesis instructions
            analysis_request = f"""Conduct M&A due diligence analysis and provide a comprehensive investment recommendation report.

{context_text if context_text else ""}

INSTRUCTIONS:
1. Use your tools to gather information (3-4 tool calls maximum)
2. After using tools, SYNTHESIZE your findings into a complete report
3. Do NOT just repeat tool outputs - analyze and synthesize them

IMPORTANT: You must provide a FINAL ANALYSIS REPORT in proper format, not just tool results.

Your final response should be a complete, well-structured report following the format specified in your instructions."""
            
            # Run the comprehensive ReAct agent with progress tracking
            logger.info(f"Starting ReAct AI Agent for comprehensive due diligence analysis...")
            
            # Add progress indicator for user
            progress_placeholder = st.empty()
            progress_placeholder.info("🧠 **AI Agent Starting:** Initializing comprehensive analysis tools...")
            
            # Configure recursion limit and other settings
            config = {
                "recursion_limit": 25,  # Allow enough steps for 8-10 tool calls + comprehensive synthesis
                "configurable": {
                    "thread_id": f"react_agent_comprehensive_{hash(data_room_name) % 10000}"
                }
            }
            
            # Update progress
            progress_placeholder.info("🔍 **AI Agent Working:** Analyzing documents and gathering intelligence...")
            
            result = agent.invoke({
                "messages": [HumanMessage(content=analysis_request)]
            }, config=config)
            
            # Final progress update
            progress_placeholder.info("📊 **AI Agent Finalizing:** Synthesizing findings and generating report...")
            
            # Clear progress indicator
            progress_placeholder.empty()
            
            # Debug: Log the complete result structure
            logger.info(f"ReAct agent result type: {type(result)}")
            logger.info(f"ReAct agent result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract the agent's final response with enhanced debugging
            agent_output = ""
            if result and "messages" in result:
                logger.info(f"Found {len(result['messages'])} messages in result")
                
                # Log all messages for debugging with more detail
                for i, message in enumerate(result["messages"]):
                    msg_type = type(message).__name__
                    has_content = hasattr(message, 'content')
                    
                    # Handle both string and list content types for debugging
                    content_text = ""
                    content_length = 0
                    
                    if has_content and message.content:
                        if isinstance(message.content, list):
                            # If content is a list, extract text parts for logging
                            text_parts = []
                            for item in message.content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            content_text = ' '.join(text_parts)
                        else:
                            content_text = str(message.content)
                        
                        content_length = len(content_text)
                    
                    logger.info(f"Message {i}: Type={msg_type}, Length={content_length}")
                    
                    if content_text:
                        content_preview = content_text[:150]
                        logger.info(f"Message {i} preview: {content_preview}...")
                        
                        # Check if this looks like a final report
                        if (content_length > 500 and 
                            ('# Company Analysis' in content_text or '## Executive Summary' in content_text)):
                            logger.info(f"Message {i} appears to be a FINAL REPORT")
                        elif 'Analysis - ' in content_text[:50]:
                            logger.info(f"Message {i} appears to be TOOL OUTPUT")
                        elif content_text.startswith('I '):
                            logger.info(f"Message {i} appears to be REASONING")
                
                # Get the final analysis report (not tool outputs)
                final_report = None
                
                # Look for messages containing a proper analysis report
                for message in reversed(result["messages"]):
                    if (hasattr(message, 'content') and message.content):
                        
                        # Handle both string and list content types
                        if isinstance(message.content, list):
                            # If content is a list, extract text parts
                            text_parts = []
                            for item in message.content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            content = ' '.join(text_parts).strip()
                        else:
                            # Content is already a string
                            content = str(message.content).strip()
                        
                        # Must be substantial content
                        if len(content) <= 200:
                            continue
                        
                        # Skip tool outputs and reasoning steps
                        if (content.startswith('I need to') or 
                            content.startswith('I should') or 
                            content.startswith('Thought:') or
                            content.startswith('Action:') or
                            content.startswith('Observation:') or
                            'Found ' in content[:50] and 'documents:' in content[:100] or  # Tool search results
                            'Analysis - ' in content[:50] or  # Raw tool analysis output
                            content.count('\n') < 5):  # Too simple/short to be final report
                            continue
                        
                        # Look for proper report structure
                        if ('# Company Analysis Report' in content or 
                            '## Executive Summary' in content or
                            '## Business' in content or
                            '## Investment Recommendation' in content or
                            ('GO' in content or 'NO-GO' in content) and len(content) > 500):
                            final_report = content
                            logger.info(f"Found structured analysis report: {len(final_report)} characters")
                            break
                        
                        # Fallback: substantial content that looks like analysis
                        elif (len(content) > 500 and 
                              ('company' in content.lower() or 'business' in content.lower()) and
                              ('recommendation' in content.lower() or 'analysis' in content.lower())):
                            final_report = content
                            logger.info(f"Found analysis content (fallback): {len(final_report)} characters")
                            break
                
                agent_output = final_report or ""
            else:
                logger.error("ReAct agent result missing 'messages' key or result is None")
            
            if not agent_output:
                logger.error("No structured analysis report found in agent messages")
                logger.error("Agent may have produced only tool outputs without final synthesis")
                
                # Log the last few messages to understand what happened
                if result and "messages" in result and len(result["messages"]) > 0:
                    logger.error("Last 3 messages from agent:")
                    for i, message in enumerate(result["messages"][-3:]):
                        if hasattr(message, 'content') and message.content:
                            logger.error(f"  Message {len(result['messages'])-3+i}: {message.content[:200]}...")
                
                raise create_ai_error(
                    "ReAct agent failed to provide final analysis report",
                    recovery_hint="The agent completed tool calls but did not synthesize a final report. This may indicate the agent is stopping after tool usage without providing analysis. Check logs for details."
                )
            
            # Create comprehensive report with inline citations
            formatted_report, citation_info = create_comprehensive_report(
                agent_output, tools, report_type
            )
            
            logger.info(f"ReAct agent analysis completed with agent output: {len(agent_output)} characters")
            logger.info(f"Formatted report length: {len(formatted_report)} characters")
            logger.info(f"Citation info: {citation_info}")
            
            # DEBUG: Log exactly what we're about to return
            logger.info(f"GENERATE_REACT_REPORT about to return: formatted_report={formatted_report is not None} ({len(formatted_report) if formatted_report else 0} chars), citation_info={citation_info}")
            
            return formatted_report, citation_info
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"ReAct agent analysis failed: {e}")
            
            # NO FALLBACKS - Fail fast to debug the actual issue
            logger.error(f"ReAct agent failed for {report_type} analysis")
            logger.error(f"Error details: {error_message}")
            
            # Re-raise the original exception to see exactly what went wrong
            raise

    def debug_react_agent(self, data_room_name: str) -> str:
        """Debug method to test ReAct agent with simple request"""
        try:
            from app.core.utils import create_document_processor
            from app.ai.react_agents import create_comprehensive_dd_agent
            from langchain_core.messages import HumanMessage
            
            logger.info("DEBUG: Testing ReAct agent with simple request")
            
            # Initialize document processor
            document_processor = create_document_processor(store_name=data_room_name)
            if not document_processor.vector_store:
                return "❌ No vector store available for debugging"
            
            # Create agent with minimal context
            agent, tools = create_comprehensive_dd_agent(
                self.llm, data_room_name, document_processor.vector_store
            )
            
            # Simple test request
            simple_request = "Please provide a brief analysis of this company using your document search tool. Keep it short and simple."
            
            config = {"recursion_limit": 15}  # Increased for more thorough testing
            
            result = agent.invoke({
                "messages": [HumanMessage(content=simple_request)]
            }, config=config)
            
            # Extract output
            if result and "messages" in result:
                for message in reversed(result["messages"]):
                    if hasattr(message, 'content') and message.content:
                        logger.info(f"DEBUG: Agent produced {len(message.content)} characters")
                        return f"✅ Agent working: {message.content[:200]}..."
            
            return "❌ Agent produced no output"
            
        except Exception as e:
            logger.error(f"DEBUG: ReAct agent test failed: {e}")
            return f"❌ Agent test failed: {str(e)}"


    @handle_processing_errors("Question answering", "Please try rephrasing your question")
    def answer_question(self, question: str, context_docs: List[str]) -> str:
        """
        Answer a specific question using AI.

        Args:
            question: The question to answer
            context_docs: List of relevant document excerpts

        Returns:
            AI-generated answer

        Raises:
            AIError: If question answering fails
        """
        if not self.is_agent_available():
            raise create_ai_error(
                "AI service not available",
                recovery_hint="Please configure your API key in the sidebar"
            )

        return self._ai_service.answer_question(question, context_docs)

    @property
    def llm(self):
        """Get the underlying LLM instance"""
        # Check local AI service first
        if self._ai_service is not None:
            return self._ai_service.llm
        
        # Check session for existing agent
        if self.session.agent is not None:
            # Update local reference if session has an agent
            self._ai_service = self.session.agent
            return self._ai_service.llm
        
        return None
