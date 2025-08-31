#!/usr/bin/env python3
"""
LangGraph Agent Nodes Module

This module contains all the individual node functions used in the
LangGraph workflow for the DD-Checklist agent.
"""

import json
from typing import Dict, List, Optional, Sequence, Any
from typing_extensions import TypedDict
from enum import Enum

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_anthropic import ChatAnthropic
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseMessage = object
    HumanMessage = object  
    AIMessage = object
    ChatAnthropic = object

from .prompts import (
    get_checklist_parsing_prompt,
    get_document_relevance_prompt,
    get_question_answering_prompt,
    get_findings_summary_prompt
)


# Define the state for our agent
class AgentState(TypedDict):
    """State for the due diligence agent"""
    messages: Sequence[BaseMessage]
    checklist: Optional[Dict]
    documents: Optional[List[Dict]]
    current_task: Optional[str]
    findings: Dict[str, List[str]]
    next_action: Optional[str]


class TaskType(Enum):
    """Types of tasks the agent can perform"""
    PARSE_CHECKLIST = "parse_checklist"
    ANALYZE_DOCUMENT = "analyze_document"
    MATCH_CHECKLIST = "match_checklist"
    ANSWER_QUESTION = "answer_question"
    SUMMARIZE_FINDINGS = "summarize_findings"


def route_task(state: AgentState) -> AgentState:
    """Route to appropriate task based on current state"""
    messages = state["messages"]
    if not messages:
        return state
    
    last_message = messages[-1].content if messages else ""
    
    # Determine next action based on message content
    if "parse" in last_message.lower() and "checklist" in last_message.lower():
        state["next_action"] = TaskType.PARSE_CHECKLIST.value
    elif "analyze" in last_message.lower() or "match" in last_message.lower():
        state["next_action"] = TaskType.MATCH_CHECKLIST.value
    elif "?" in last_message:
        state["next_action"] = TaskType.ANSWER_QUESTION.value
    else:
        state["next_action"] = TaskType.SUMMARIZE_FINDINGS.value
    
    return state


def parse_checklist_node(state: AgentState, llm: ChatAnthropic) -> AgentState:
    """Parse checklist using Claude"""
    messages = state["messages"]
    checklist_text = messages[-1].content if messages else ""
    
    prompt = get_checklist_parsing_prompt(checklist_text)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Parse JSON from response
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        parsed = json.loads(json_str.strip())
        state["checklist"] = parsed
        state["messages"].append(AIMessage(content=f"Parsed {len(parsed)} categories"))
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Parsing failed: {str(e)}"))
    
    return state


def match_checklist_node(state: AgentState, llm: ChatAnthropic) -> AgentState:
    """Match documents to checklist items"""
    checklist = state.get("checklist", {})
    documents = state.get("documents", [])
    
    if not checklist or not documents:
        state["messages"].append(AIMessage(content="Need both checklist and documents to match"))
        return state
    
    # For each checklist item, find relevant documents
    findings = {}
    for cat_letter, category in checklist.items():
        cat_findings = []
        for item in category.get("items", []):
            # Use Claude to assess relevance
            document_names = [d.get('name', 'Unknown') for d in documents[:10]]
            prompt = get_document_relevance_prompt(item['text'], document_names)
            
            response = llm.invoke([HumanMessage(content=prompt)])
            cat_findings.append({
                "item": item['text'],
                "relevant_docs": response.content
            })
        
        findings[category['name']] = cat_findings
    
    state["findings"] = findings
    state["messages"].append(AIMessage(content=f"Matched checklist to {len(documents)} documents"))
    
    return state


def answer_question_node(state: AgentState, llm: ChatAnthropic) -> AgentState:
    """Answer questions using document context"""
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    documents = state.get("documents", [])
    
    # Create context from documents
    context = "\n".join([f"- {d.get('name', 'Unknown')}: {d.get('text', '')[:200]}" 
                        for d in documents[:5]])
    
    prompt = get_question_answering_prompt(question, context)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(AIMessage(content=response.content))
    
    return state


def summarize_node(state: AgentState, llm: ChatAnthropic) -> AgentState:
    """Summarize findings"""
    findings = state.get("findings", {})
    
    if not findings:
        state["messages"].append(AIMessage(content="No findings to summarize"))
        return state
    
    prompt = get_findings_summary_prompt(findings)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(AIMessage(content=response.content))
    
    return state


def route_condition(state: AgentState) -> str:
    """Conditional routing function based on next_action"""
    next_action = state.get("next_action")
    if next_action == TaskType.PARSE_CHECKLIST.value:
        return "parse_checklist"
    elif next_action == TaskType.MATCH_CHECKLIST.value:
        return "match_checklist"
    elif next_action == TaskType.ANSWER_QUESTION.value:
        return "answer_question"
    else:
        return "summarize"
