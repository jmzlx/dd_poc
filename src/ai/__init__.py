#!/usr/bin/env python3
"""
AI Integration Module

This module provides AI-powered functionality for the DD-Checklist application,
including LangGraph agents, document processing, and checklist matching.
"""

# Try to import core components and set availability flag
try:
    from .agent_core import DDChecklistAgent, get_langgraph_agent, LANGGRAPH_AVAILABLE
    from .llm_utilities import (
        batch_summarize_documents,
        create_document_embeddings_with_summaries,
        match_checklist_with_summaries,
        generate_checklist_descriptions,
        exponential_backoff_retry
    )
    from .agent_nodes import AgentState, TaskType
    from .prompts import (
        get_checklist_parsing_prompt,
        get_document_relevance_prompt,
        get_question_answering_prompt,
        get_findings_summary_prompt,
        get_description_generation_prompt,
        get_document_summarization_prompt
    )
    
    # Set availability flag based on successful imports
    AI_MODULE_AVAILABLE = LANGGRAPH_AVAILABLE
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"AI module dependencies not available: {e}")
    
    # Create placeholder classes/functions for graceful degradation
    class DDChecklistAgent:
        def __init__(self, *args, **kwargs):
            self.app = None
            self.llm = None
        
        def is_available(self):
            return False
    
    def get_langgraph_agent(*args, **kwargs):
        return None
    
    def batch_summarize_documents(documents, *args, **kwargs):
        return documents
    
    def create_document_embeddings_with_summaries(documents, *args, **kwargs):
        return {'embeddings': [], 'documents': []}
    
    def match_checklist_with_summaries(*args, **kwargs):
        return {}
    
    def generate_checklist_descriptions(checklist, *args, **kwargs):
        return checklist
    
    def exponential_backoff_retry(func, *args, **kwargs):
        return func()
    
    # Set availability flags
    LANGGRAPH_AVAILABLE = False
    AI_MODULE_AVAILABLE = False
    
    # Placeholder classes for type hints
    class AgentState:
        pass
    
    class TaskType:
        pass

# Export main public API
__all__ = [
    # Core agent functionality
    'DDChecklistAgent',
    'get_langgraph_agent',
    
    # LLM utility functions
    'batch_summarize_documents',
    'create_document_embeddings_with_summaries',
    'match_checklist_with_summaries',
    'generate_checklist_descriptions',
    'exponential_backoff_retry',
    
    # Agent types and state
    'AgentState',
    'TaskType',
    
    # Prompt functions
    'get_checklist_parsing_prompt',
    'get_document_relevance_prompt', 
    'get_question_answering_prompt',
    'get_findings_summary_prompt',
    'get_description_generation_prompt',
    'get_document_summarization_prompt',
    
    # Availability flags
    'LANGGRAPH_AVAILABLE',
    'AI_MODULE_AVAILABLE',
]
