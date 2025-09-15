#!/usr/bin/env python3
"""
ReAct Agents for Due Diligence Analysis

This module implements ReAct (Reasoning and Acting) agents for comprehensive
due diligence analysis with iterative reasoning, document validation, and 
comprehensive citation tracking.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic

# Local imports
from app.core.search import search_and_analyze
from app.core.utils import create_document_processor
from app.core.logging import logger
from app.ai.agent_utils import AgentState

# Enhanced Agent State for ReAct behavior
class ReActAgentState(AgentState):
    """Enhanced state for ReAct agents with comprehensive tracking"""
    # Analysis tracking
    current_analysis: Dict[str, Any]
    reasoning_history: List[Dict[str, str]]
    iteration_count: int
    max_iterations: int
    
    # Citation and source tracking
    cited_documents: Dict[str, Dict[str, Any]]  # doc_id -> {name, path, excerpts}
    source_confidence: Dict[str, float]  # source -> confidence score
    validation_results: Dict[str, Dict[str, Any]]  # claim -> validation info
    
    # Analysis progress
    analyzed_topics: Set[str]
    follow_up_questions: List[str]
    identified_gaps: List[str]
    
    # Final output
    final_report_sections: Dict[str, str]
    citations_list: List[Dict[str, Any]]


class DocumentSearchTool(BaseTool):
    """Tool for semantic search across all documents with citation tracking"""
    
    name: str = "document_search"
    description: str = """Search across all documents for specific information. 
    Use this to find relevant documents and information about specific topics.
    Input should be a specific search query or topic."""
    
    # Pydantic fields for the tool instance
    vector_store: Any = None
    store_name: str = ""
    
    def __init__(self, vector_store, store_name: str):
        super().__init__(vector_store=vector_store, store_name=store_name)
    
    def _run(self, query: str) -> str:
        """Execute document search and return results with citations"""
        try:
            # FAIL FAST: Raise error if vector store is not available
            if self.vector_store is None:
                raise ValueError(f"DocumentSearchTool: vector_store is None. Cannot search documents for query: {query}")
            
            # Perform search using existing search infrastructure
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=8)
            
            results = []
            citations = []
            
            for i, (doc, score) in enumerate(docs_with_scores):
                if score <= 1.5:  # Good relevance threshold
                    similarity = 1.0 - (score / 2.0) if score <= 2.0 else 0.0
                    
                    # Extract key information with better excerpts
                    # Try to get a more meaningful excerpt around the query match
                    content = doc.page_content
                    query_lower = query.lower()
                    content_lower = content.lower()
                    
                    # Find the best excerpt around the query match
                    if query_lower in content_lower:
                        match_pos = content_lower.find(query_lower)
                        start_pos = max(0, match_pos - 100)
                        end_pos = min(len(content), match_pos + 300)
                        excerpt = content[start_pos:end_pos]
                        if start_pos > 0:
                            excerpt = "..." + excerpt
                        if end_pos < len(content):
                            excerpt = excerpt + "..."
                    else:
                        # Fallback to beginning excerpt
                        excerpt = content[:400] + "..." if len(content) > 400 else content
                    
                    doc_name = doc.metadata.get('name', f'Document_{i}')
                    doc_path = doc.metadata.get('path', '')
                    
                    result_text = f"[{doc_name}]: {excerpt}"
                    results.append(result_text)
                    
                    # Track citation info with relevance score
                    citation = {
                        'doc_id': f"doc_{hash(doc_path) % 10000}",
                        'name': doc_name,
                        'path': doc_path,
                        'relevance': round(similarity, 3),
                        'excerpt': excerpt
                    }
                    citations.append(citation)
            
            if not results:
                return "No relevant documents found for the query."
            
            # Return formatted results
            search_summary = f"Found {len(results)} relevant documents:\n\n" + "\n\n".join(results[:5])
            
            # Store citations in a way the agent can access
            self._last_citations = citations
            
            return search_summary
            
        except ValueError as e:
            # Re-raise ValueError for vector store issues to fail fast
            logger.error(f"Document search failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def get_last_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last search"""
        return getattr(self, '_last_citations', [])


class CrossReferenceTool(BaseTool):
    """Tool for cross-referencing claims across multiple documents"""
    
    name: str = "cross_reference"
    description: str = """Verify claims by finding supporting or contradicting evidence 
    across multiple documents. Input should be a specific claim or statement to verify."""
    
    # Pydantic fields for the tool instance
    vector_store: Any = None
    store_name: str = ""
    
    def __init__(self, vector_store, store_name: str):
        super().__init__(vector_store=vector_store, store_name=store_name)
    
    def _run(self, claim: str) -> str:
        """Cross-reference a claim across documents"""
        try:
            # FAIL FAST: Raise error if vector store is not available
            if self.vector_store is None:
                raise ValueError(f"CrossReferenceTool: vector_store is None. Cannot cross-reference claim: {claim}")
            
            # Search for evidence
            search_queries = [
                claim,
                f"evidence {claim}",
                f"data supporting {claim}",
                f"information about {claim}"
            ]
            
            all_evidence = []
            for query in search_queries:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
                for doc, score in docs_with_scores:
                    if score <= 1.2:  # High relevance only for validation
                        all_evidence.append((doc, score))
            
            # Remove duplicates and sort by relevance
            unique_evidence = list({doc.page_content: (doc, score) for doc, score in all_evidence}.values())
            unique_evidence.sort(key=lambda x: x[1])  # Sort by score (lower is better)
            
            if len(unique_evidence) >= 2:
                validation_status = "SUPPORTED - Found evidence in multiple documents"
            elif len(unique_evidence) == 1:
                validation_status = "PARTIALLY SUPPORTED - Found evidence in one document"
            else:
                validation_status = "UNSUPPORTED - No clear evidence found"
            
            evidence_summary = []
            citations = []
            
            for i, (doc, score) in enumerate(unique_evidence[:3]):
                doc_name = doc.metadata.get('name', f'Document_{i}')
                
                # Better excerpt extraction around claim match
                content = doc.page_content
                claim_lower = claim.lower()
                content_lower = content.lower()
                
                if claim_lower in content_lower:
                    match_pos = content_lower.find(claim_lower)
                    start_pos = max(0, match_pos - 80)
                    end_pos = min(len(content), match_pos + 250)
                    evidence_text = content[start_pos:end_pos]
                    if start_pos > 0:
                        evidence_text = "..." + evidence_text
                    if end_pos < len(content):
                        evidence_text = evidence_text + "..."
                else:
                    evidence_text = content[:330] + "..." if len(content) > 330 else content
                
                evidence_summary.append(f"[{doc_name}]: {evidence_text}")
                
                citations.append({
                    'doc_id': f"doc_{hash(doc.metadata.get('path', '')) % 10000}",
                    'name': doc_name,
                    'path': doc.metadata.get('path', ''),
                    'excerpt': evidence_text,
                    'relevance': round(1.0 - (score / 2.0), 3)
                })
            
            result = f"{validation_status}\n\nEvidence found:\n" + "\n\n".join(evidence_summary)
            self._last_validation_citations = citations
            
            return result
            
        except ValueError as e:
            # Re-raise ValueError for vector store issues to fail fast
            logger.error(f"Cross-reference failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Cross-reference failed: {e}")
            return f"Cross-reference failed: {str(e)}"
    
    def get_last_validation_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last validation"""
        return getattr(self, '_last_validation_citations', [])


class FinancialAnalysisTool(BaseTool):
    """Tool for analyzing financial information across documents"""
    
    name: str = "financial_analysis"
    description: str = """Analyze financial information, metrics, and trends from documents.
    Input should specify what financial aspect to analyze (e.g., 'revenue trends', 'profitability', 'debt levels')."""
    
    # Pydantic fields for the tool instance
    vector_store: Any = None
    store_name: str = ""
    
    def __init__(self, vector_store, store_name: str):
        super().__init__(vector_store=vector_store, store_name=store_name)
    
    def _run(self, analysis_focus: str) -> str:
        """Analyze financial information"""
        try:
            # FAIL FAST: Raise error if vector store is not available
            if self.vector_store is None:
                raise ValueError(f"FinancialAnalysisTool: vector_store is None. Cannot analyze: {analysis_focus}")
            
            # Financial search terms
            financial_queries = [
                f"{analysis_focus} financial",
                f"{analysis_focus} revenue profit",
                f"{analysis_focus} financial statements",
                f"financial {analysis_focus} performance"
            ]
            
            financial_docs = []
            for query in financial_queries:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)
                for doc, score in docs_with_scores:
                    if score <= 1.3:
                        financial_docs.append((doc, score))
            
            if not financial_docs:
                return f"No financial information found for: {analysis_focus}"
            
            # Sort by relevance and remove duplicates
            unique_docs = list({doc.page_content: (doc, score) for doc, score in financial_docs}.values())
            unique_docs.sort(key=lambda x: x[1])
            
            # Extract financial insights
            insights = []
            citations = []
            
            for i, (doc, score) in enumerate(unique_docs[:4]):
                doc_name = doc.metadata.get('name', f'Financial_Document_{i}')
                content = doc.page_content
                
                # Extract relevant financial excerpt around the analysis focus
                analysis_lower = analysis_focus.lower()
                content_lower = content.lower()
                
                # Look for financial keywords and analysis focus
                financial_keywords = ['revenue', 'profit', 'financial', 'income', 'expense', 'cash', 'balance']
                
                best_excerpt = content[:400] + "..." if len(content) > 400 else content
                
                # Try to find excerpt with both financial keywords and analysis focus
                for keyword in financial_keywords:
                    if keyword in content_lower and analysis_lower in content_lower:
                        keyword_pos = content_lower.find(keyword)
                        focus_pos = content_lower.find(analysis_lower)
                        center_pos = min(keyword_pos, focus_pos)
                        start_pos = max(0, center_pos - 100)
                        end_pos = min(len(content), center_pos + 350)
                        excerpt = content[start_pos:end_pos]
                        if start_pos > 0:
                            excerpt = "..." + excerpt
                        if end_pos < len(content):
                            excerpt = excerpt + "..."
                        best_excerpt = excerpt
                        break
                
                insights.append(f"[{doc_name}]: {best_excerpt}")
                
                citations.append({
                    'doc_id': f"doc_{hash(doc.metadata.get('path', '')) % 10000}",
                    'name': doc_name,
                    'path': doc.metadata.get('path', ''),
                    'excerpt': best_excerpt,
                    'relevance': round(1.0 - (score / 2.0), 3)
                })
            
            analysis = f"Financial Analysis - {analysis_focus}:\n\n" + "\n\n".join(insights)
            self._last_financial_citations = citations
            
            return analysis
            
        except ValueError as e:
            # Re-raise ValueError for vector store issues to fail fast
            logger.error(f"Financial analysis failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Financial analysis failed: {e}")
            return f"Financial analysis failed: {str(e)}"
    
    def get_last_financial_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last financial analysis"""
        return getattr(self, '_last_financial_citations', [])


class CompetitiveAnalysisTool(BaseTool):
    """Tool for analyzing competitive positioning and market dynamics"""
    
    name: str = "competitive_analysis"
    description: str = """Analyze competitive positioning, market dynamics, and competitive advantages.
    Input should specify what competitive aspect to analyze."""
    
    # Pydantic fields for the tool instance
    vector_store: Any = None
    store_name: str = ""
    
    def __init__(self, vector_store, store_name: str):
        super().__init__(vector_store=vector_store, store_name=store_name)
    
    def _run(self, focus_area: str) -> str:
        """Analyze competitive positioning"""
        try:
            # FAIL FAST: Raise error if vector store is not available
            if self.vector_store is None:
                raise ValueError(f"CompetitiveAnalysisTool: vector_store is None. Cannot analyze: {focus_area}")
            
            # Competitive search terms
            competitive_queries = [
                f"{focus_area} competitive advantage",
                f"{focus_area} market position competition",
                f"competitive {focus_area}",
                f"market share {focus_area}",
                f"competitors {focus_area}"
            ]
            
            competitive_docs = []
            for query in competitive_queries:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=4)
                for doc, score in docs_with_scores:
                    if score <= 1.4:
                        competitive_docs.append((doc, score))
            
            if not competitive_docs:
                return f"No competitive information found for: {focus_area}"
            
            # Process and deduplicate
            unique_docs = list({doc.page_content: (doc, score) for doc, score in competitive_docs}.values())
            unique_docs.sort(key=lambda x: x[1])
            
            analysis_points = []
            citations = []
            
            for i, (doc, score) in enumerate(unique_docs[:4]):
                doc_name = doc.metadata.get('name', f'Competitive_Doc_{i}')
                content = doc.page_content
                
                # Extract relevant competitive excerpt around the focus area
                focus_lower = focus_area.lower()
                content_lower = content.lower()
                
                # Look for competitive keywords and focus area
                competitive_keywords = ['competitive', 'market', 'competitor', 'advantage', 'position', 'share']
                
                best_excerpt = content[:400] + "..." if len(content) > 400 else content
                
                # Try to find excerpt with both competitive keywords and focus area
                for keyword in competitive_keywords:
                    if keyword in content_lower and focus_lower in content_lower:
                        keyword_pos = content_lower.find(keyword)
                        focus_pos = content_lower.find(focus_lower)
                        center_pos = min(keyword_pos, focus_pos)
                        start_pos = max(0, center_pos - 120)
                        end_pos = min(len(content), center_pos + 380)
                        excerpt = content[start_pos:end_pos]
                        if start_pos > 0:
                            excerpt = "..." + excerpt
                        if end_pos < len(content):
                            excerpt = excerpt + "..."
                        best_excerpt = excerpt
                        break
                
                analysis_points.append(f"[{doc_name}]: {best_excerpt}")
                
                citations.append({
                    'doc_id': f"doc_{hash(doc.metadata.get('path', '')) % 10000}",
                    'name': doc_name,
                    'path': doc.metadata.get('path', ''),
                    'excerpt': best_excerpt,
                    'relevance': round(1.0 - (score / 2.0), 3)
                })
            
            competitive_analysis = f"Competitive Analysis - {focus_area}:\n\n" + "\n\n".join(analysis_points)
            self._last_competitive_citations = citations
            
            return competitive_analysis
            
        except ValueError as e:
            # Re-raise ValueError for vector store issues to fail fast
            logger.error(f"Competitive analysis failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Competitive analysis failed: {e}")
            return f"Competitive analysis failed: {str(e)}"
    
    def get_last_competitive_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last competitive analysis"""
        return getattr(self, '_last_competitive_citations', [])


class ContextAnalysisTool(BaseTool):
    """Tool for analyzing existing context from previous analyses"""
    
    name: str = "context_analysis"
    description: str = """Analyze existing context information including strategy, checklist results, and Q&A insights.
    Input should specify what aspect of the existing context to analyze (e.g., 'strategic alignment', 'checklist gaps', 'previous findings')."""
    
    # Pydantic fields for the tool instance
    strategy_text: str = ""
    checklist_results: Dict[str, Any] = {}
    question_answers: Dict[str, Any] = {}
    project_info: Dict[str, Any] = {}
    
    def __init__(self, strategy_text: str = "", checklist_results: Dict[str, Any] = None, 
                 question_answers: Dict[str, Any] = None, project_info: Dict[str, Any] = None):
        super().__init__(
            strategy_text=strategy_text or "",
            checklist_results=checklist_results or {},
            question_answers=question_answers or {},
            project_info=project_info or {}
        )
    
    def _run(self, analysis_focus: str) -> str:
        """Analyze existing context information with optional vectorized search"""
        try:
            context_analysis = []
            
            # First try vectorized analysis search if available
            vectorized_results = self._search_vectorized_analysis(analysis_focus)
            if vectorized_results:
                context_analysis.append(f"**Vectorized Analysis Search Results:**\n{vectorized_results}")
            
            # Then add structured context analysis
            # Analyze strategic context
            if "strategic" in analysis_focus.lower() and self.strategy_text:
                context_analysis.append(f"**Strategic Context Analysis:**\n{self.strategy_text[:500]}...")
            
            # Analyze checklist gaps and matches
            if "checklist" in analysis_focus.lower() and self.checklist_results:
                checklist_summary = []
                for category, items in self.checklist_results.items():
                    if isinstance(items, dict):
                        matched = items.get('matched_items', 0)
                        total = items.get('total_items', 0)
                        completion = (matched / total * 100) if total > 0 else 0
                        checklist_summary.append(f"- {items.get('name', category)}: {completion:.1f}% complete ({matched}/{total})")
                
                if checklist_summary:
                    context_analysis.append(f"**Checklist Analysis:**\n" + "\n".join(checklist_summary))
            
            # Analyze previous Q&A insights
            if "previous" in analysis_focus.lower() or "qa" in analysis_focus.lower():
                if self.question_answers and isinstance(self.question_answers, dict):
                    qa_insights = []
                    for q_data in self.question_answers.get('questions', []):
                        if isinstance(q_data, dict) and q_data.get('has_answer'):
                            question = q_data.get('question', '')
                            answer = q_data.get('answer', '')
                            if question and answer:
                                qa_insights.append(f"Q: {question}\nA: {answer[:300]}...")
                    
                    if qa_insights:
                        context_analysis.append(f"**Previous Q&A Insights:**\n" + "\n\n".join(qa_insights[:3]))
            
            # Project information
            if "project" in analysis_focus.lower() and self.project_info:
                project_details = []
                for key, value in self.project_info.items():
                    if value:
                        project_details.append(f"- {key.replace('_', ' ').title()}: {value}")
                
                if project_details:
                    context_analysis.append(f"**Project Information:**\n" + "\n".join(project_details))
            
            if context_analysis:
                return "\n\n".join(context_analysis)
            else:
                return f"No relevant context found for: {analysis_focus}"
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return f"Context analysis failed: {str(e)}"

    def _search_vectorized_analysis(self, query: str) -> str:
        """Search through vectorized analysis results if available"""
        try:
            # This would need to be passed from session, but for now return structured info
            # TODO: Access session.analysis_vector_store when available
            return ""  # Placeholder for vectorized search
        except Exception as e:
            logger.error(f"Vectorized analysis search failed: {e}")
            return ""
    
    def get_available_context(self) -> str:
        """Get summary of all available context"""
        available = []
        if self.strategy_text:
            available.append("✅ Strategic context")
        if self.checklist_results:
            available.append("✅ Checklist results")
        if self.question_answers:
            available.append("✅ Q&A insights")
        if self.project_info:
            available.append("✅ Project information")
        
        return "Available context: " + ", ".join(available) if available else "No additional context available"


class AnalysisSearchTool(BaseTool):
    """Tool for searching through vectorized analysis results (strategy, checklist, Q&A)"""
    
    name: str = "analysis_search"
    description: str = """Search through vectorized analysis results including strategy context, checklist insights, and Q&A findings.
    Use this to find specific insights from previous analyses. Input should be a search query for analysis insights."""
    
    # Pydantic fields for the tool instance
    analysis_vector_store: Any = None
    
    def __init__(self, analysis_vector_store = None):
        super().__init__(analysis_vector_store=analysis_vector_store)
    
    def _run(self, query: str) -> str:
        """Search through vectorized analysis results"""
        try:
            # Check if analysis vector store is available
            if self.analysis_vector_store is None:
                return f"Analysis search unavailable: No vectorized analysis results available. Query: {query}"
            
            # Search through analysis results
            docs_with_scores = self.analysis_vector_store.similarity_search_with_score(query, k=5)
            
            if not docs_with_scores:
                return f"No relevant analysis insights found for: {query}"
            
            results = []
            for doc, score in docs_with_scores:
                if score <= 1.5:  # Good relevance threshold
                    doc_type = doc.metadata.get('type', 'unknown')
                    doc_name = doc.metadata.get('name', 'Unknown Analysis')
                    content_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    
                    result_text = f"[{doc_name}] ({doc_type}): {content_preview}"
                    results.append(result_text)
            
            if results:
                return f"Found {len(results)} relevant analysis insights:\n\n" + "\n\n".join(results)
            else:
                return f"No relevant analysis insights found for: {query}"
            
        except Exception as e:
            logger.error(f"Analysis search failed: {e}")
            return f"Analysis search failed: {str(e)}"


def create_comprehensive_dd_agent(llm: ChatAnthropic, store_name: str, vector_store, 
                                strategy_text: str = "", checklist_results: Dict[str, Any] = None,
                                question_answers: Dict[str, Any] = None, project_info: Dict[str, Any] = None,
                                analysis_vector_store = None) -> Any:
    """Create unified ReAct agent for comprehensive due diligence analysis with full context access"""
    
    # Initialize tools including context analysis
    # CRITICAL: Ensure all tools get proper vector_store access
    tools = [
        DocumentSearchTool(vector_store, store_name),
        CrossReferenceTool(vector_store, store_name),
        FinancialAnalysisTool(vector_store, store_name),
        CompetitiveAnalysisTool(vector_store, store_name),
        ContextAnalysisTool(strategy_text, checklist_results, question_answers, project_info),
        AnalysisSearchTool(analysis_vector_store)  # New tool for searching analysis insights
    ]
    
    # Verify all tools have vector store access (except ContextAnalysisTool)
    logger.info(f"Creating ReAct agent with {len(tools)} tools")
    for tool in tools:
        if hasattr(tool, 'vector_store'):
            logger.info(f"Tool {tool.name}: vector_store={'Available' if tool.vector_store is not None else 'None'}")
        else:
            logger.info(f"Tool {tool.name}: context tool (no vector store needed)")
    
    # Clear, explicit system prompt focused on due diligence status reporting
    system_prompt = """You are a senior due diligence analyst reporting to the Corporate Development team. Your goal is to provide a comprehensive status update on the due diligence process and analysis of the target company's data room.

WORKFLOW:
1. Conduct thorough analysis using your tools (8-10 tool calls for comprehensive coverage)
2. Focus on data room completeness, documentation gaps, and process status
3. STOP using tools and WRITE your final status report

TOOL USAGE (8-10 calls recommended for thorough analysis):
- Use document_search multiple times for different business areas (financials, legal, operations, etc.)
- Use competitive_analysis for market position and competitive threats
- Use financial_analysis for financial health and performance trends
- Use cross_reference to validate critical claims and data consistency
- Use context_analysis to review strategic alignment and previous findings
- Use analysis_search for insights from completed analyses

CRITICAL: After conducting thorough analysis with tools, you MUST provide your final status report. Aim for comprehensive coverage before concluding.

FINAL REPORT REQUIREMENTS:
When you have sufficient information, provide a complete status report with this EXACT format:

## Due Diligence Status Summary
[Current status of the DD process, completion percentage, and key milestones achieved]

## Data Room Analysis
[Assessment of data room completeness, document quality, and organization. Identify missing documents or categories]

## Business Model & Operations Assessment
[Analysis of the company's business model, operations, and market position based on available documents]

## Financial Analysis Status
[Review of financial documents available, key metrics identified, and areas requiring additional information]

## Legal & Compliance Review
[Status of legal document review, compliance issues identified, and regulatory considerations]

## Strategic Fit Assessment
[Evaluation of strategic alignment with acquisition criteria and corporate strategy]

## Critical Gaps & Red Flags
[Specific documentation gaps, concerns identified, and areas requiring immediate attention]

## Next Steps & Recommendations
[Recommended actions for the Corporate Development team, additional diligence required, and timeline considerations]

## Process Status
**DILIGENCE STATUS**: [ON TRACK | CONCERNS IDENTIFIED | ADDITIONAL REVIEW REQUIRED] with specific rationale

FORMATTING REQUIREMENTS:
- Use proper markdown headers (# and ##)
- CRITICAL: Cite documents using curly braces: {Document Name} - this enables automatic citation linking
- When referencing specific data, use: "According to {Financial Statement 2023}, the revenue was $5.2M"
- Examples: "{Annual Report 2023} shows strong growth" or "The {Balance Sheet Q3} indicates healthy cash flow"
- Keep sections focused and substantive with specific details
- End with clear DILIGENCE STATUS assessment (ON TRACK | CONCERNS IDENTIFIED | ADDITIONAL REVIEW REQUIRED)
- Use standard currency formatting: $87.5M, $6.5M (up from $4.5M)
- Avoid LaTeX expressions, mathematical notation, or complex formulas
- Use simple, clear text formatting for all financial figures
- Every major claim should reference a specific document by name using the {Document Name} format
- For data room analysis, be specific about document categories and completeness percentages
- Identify specific gaps with document names and types needed
- Provide actionable recommendations for the Corporate Development team

DILIGENCE PROCESS FOCUS:
- Report on what has been reviewed vs. what remains to be analyzed
- Assess data room organization and accessibility
- Flag any urgent issues requiring immediate Corporate Development attention
- Provide realistic timeline estimates for remaining work
- Consider impact on transaction timeline and decision-making

Remember: Your job is to provide a COMPREHENSIVE DUE DILIGENCE STATUS REPORT for the Corporate Development team, not just investment analysis. After using tools, synthesize your findings into a professional status update that helps the team understand process progress, data quality, and next steps."""

    # Create the unified ReAct agent
    react_agent = create_react_agent(llm, tools, prompt=system_prompt)
    
    return react_agent, tools
