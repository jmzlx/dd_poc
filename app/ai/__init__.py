#!/usr/bin/env python3
"""
AI Integration Module

This module provides AI-powered functionality for the DD-Checklist application,
including LangGraph agents, document processing, and checklist matching.
"""

# Import core components
from .prompts import (
    get_checklist_parsing_prompt,
    get_document_relevance_prompt,
    get_question_answering_prompt,
    get_findings_summary_prompt,
    get_description_generation_prompt,
    get_document_summarization_prompt
)

# Direct imports for AI functionality - assuming dependencies are present
from .agent_core import (
    Agent,
    get_langgraph_agent
)
from .agent_utils import (
    AgentState,
    TaskType
)

# Export main public API
__all__ = [
    # Core agent functionality
    'Agent',
    'get_langgraph_agent',

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
]
