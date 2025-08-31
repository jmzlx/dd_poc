#!/usr/bin/env python3
"""
LangGraph Agent Core Module

This module contains the main LangGraph agent setup and the high-level
DDChecklistAgent class for interacting with the agent system.
"""

import os
from typing import Optional, Dict, List, Any, Tuple

try:
    import streamlit as st
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    st = None
    ChatAnthropic = object
    BaseMessage = object
    HumanMessage = object
    AIMessage = object
    SystemMessage = object

from ..config import get_config
from .agent_nodes import (
    AgentState, TaskType, 
    route_task, parse_checklist_node, match_checklist_node,
    answer_question_node, summarize_node, route_condition
)


def get_langgraph_agent(api_key: Optional[str] = None, model: Optional[str] = None) -> Optional[Tuple[Any, ChatAnthropic]]:
    """
    Create a LangGraph agent with Anthropic
    
    Args:
        api_key: Anthropic API key (optional, will be sourced from environment/config)
        model: Model name to use (optional, will use config default)
        
    Returns:
        Tuple of (compiled_app, llm) or None if not available
    """
    
    if not LANGGRAPH_AVAILABLE:
        return None
    
    # Get configuration
    config = get_config()
    
    # Get API key from various sources
    if not api_key:
        api_key = config.api.anthropic_api_key
        if not api_key and st and hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            api_key = st.secrets['ANTHROPIC_API_KEY']
    
    if not api_key:
        return None
    
    # Use model from config if not specified
    if not model:
        model = config.model.claude_model
    
    # Initialize Claude with config values
    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens
    )
    
    # Define tools for the agent
    @tool
    def parse_checklist_tool(checklist_text: str) -> Dict:
        """Parse a due diligence checklist into structured format"""
        return {"status": "parsing", "text": checklist_text[:100]}
    
    @tool
    def analyze_relevance_tool(doc_text: str, checklist_item: str) -> float:
        """Analyze how relevant a document is to a checklist item"""
        return 0.75  # Placeholder
    
    @tool
    def extract_information_tool(doc_text: str, query: str) -> str:
        """Extract specific information from a document"""
        return f"Extracted info about {query} from document"
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Create node functions that have access to the llm
    def _route_task(state: AgentState) -> AgentState:
        return route_task(state)
    
    def _parse_checklist_node(state: AgentState) -> AgentState:
        return parse_checklist_node(state, llm)
    
    def _match_checklist_node(state: AgentState) -> AgentState:
        return match_checklist_node(state, llm)
    
    def _answer_question_node(state: AgentState) -> AgentState:
        return answer_question_node(state, llm)
    
    def _summarize_node(state: AgentState) -> AgentState:
        return summarize_node(state, llm)
    
    # Add nodes to workflow
    workflow.add_node("route", _route_task)
    workflow.add_node("parse_checklist", _parse_checklist_node)
    workflow.add_node("match_checklist", _match_checklist_node)
    workflow.add_node("answer_question", _answer_question_node)
    workflow.add_node("summarize", _summarize_node)
    
    # Define edges
    workflow.set_entry_point("route")
    
    # Conditional routing based on next_action
    workflow.add_conditional_edges(
        "route",
        route_condition,
        {
            "parse_checklist": "parse_checklist",
            "match_checklist": "match_checklist",
            "answer_question": "answer_question",
            "summarize": "summarize"
        }
    )
    
    # All task nodes go to END
    workflow.add_edge("parse_checklist", END)
    workflow.add_edge("match_checklist", END)
    workflow.add_edge("answer_question", END)
    workflow.add_edge("summarize", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app, llm


class DDChecklistAgent:
    """High-level interface for the LangGraph agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the DD Checklist Agent
        
        Args:
            api_key: Anthropic API key (optional)
            model: Model name to use
        """
        result = get_langgraph_agent(api_key, model)
        if result:
            self.app, self.llm = result
            self.thread_id = "dd-checklist-session"
        else:
            self.app = None
            self.llm = None
    
    def is_available(self) -> bool:
        """Check if the agent is available for use"""
        return self.app is not None
    
    def parse_checklist(self, checklist_text: str) -> Optional[Dict]:
        """
        Parse checklist using the agent
        
        Args:
            checklist_text: Raw checklist text to parse
            
        Returns:
            Parsed checklist dictionary or None if failed
        """
        if not self.app:
            return None
        
        try:
            # Run the agent
            result = self.app.invoke(
                {"messages": [HumanMessage(content=f"Parse this checklist: {checklist_text}")]},
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            return result.get("checklist")
        except Exception as e:
            if st:
                st.error(f"Agent error: {str(e)}")
            return None
    
    def match_documents(self, checklist: Dict, documents: List[Dict]) -> Dict:
        """
        Match documents to checklist items
        
        Args:
            checklist: Parsed checklist dictionary
            documents: List of document dictionaries
            
        Returns:
            Dictionary of findings or empty dict if failed
        """
        if not self.app:
            return {}
        
        try:
            # Prepare state
            initial_state = {
                "messages": [HumanMessage(content="Match documents to checklist items")],
                "checklist": checklist,
                "documents": documents,
                "findings": {}
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            return result.get("findings", {})
        except Exception as e:
            if st:
                st.error(f"Agent error: {str(e)}")
            return {}
    
    def answer_question(self, question: str, documents: List[Dict]) -> str:
        """
        Answer a question using document context
        
        Args:
            question: User question
            documents: List of document dictionaries for context
            
        Returns:
            Answer string or error message
        """
        if not self.app:
            return "Agent not available"
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "documents": documents
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            # Get the last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "No answer generated"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def summarize_findings(self, findings: Dict) -> str:
        """
        Generate executive summary
        
        Args:
            findings: Dictionary of due diligence findings
            
        Returns:
            Summary string or error message
        """
        if not self.app:
            return "Agent not available"
        
        try:
            initial_state = {
                "messages": [HumanMessage(content="Summarize the due diligence findings")],
                "findings": findings
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            # Get the last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "No summary generated"
        except Exception as e:
            return f"Error: {str(e)}"
