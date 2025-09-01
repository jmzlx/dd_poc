#!/usr/bin/env python3
"""
LangGraph Agent Core Module

This module contains the main LangGraph agent setup, AI utilities, and the high-level
DDChecklistAgent class for interacting with the agent system.

Merged from: agent_core.py, agent_nodes.py, llm_utilities.py
"""

import os
import json
import time
import random
import logging
from typing import Optional, Dict, List, Any, Tuple, Sequence
from typing_extensions import TypedDict
from enum import Enum
import streamlit as st
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.runnables.config import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from ..config import get_config
from ..document_processing import safe_execute
from .prompts import (
    get_checklist_parsing_prompt,
    get_document_relevance_prompt,
    get_question_answering_prompt,
    get_findings_summary_prompt,
    get_description_generation_prompt,
    get_document_summarization_prompt
)

logger = logging.getLogger(__name__)


def with_retry(func, max_attempts=3, base_delay=1.0):
    """
    Wrapper function to add exponential backoff retry logic to any function.
    
    Args:
        func: Function to wrap with retry logic
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:  # Last attempt
                    raise e
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        return func(*args, **kwargs)  # Should not reach here
    return wrapper


# =============================================================================
# TYPE DEFINITIONS - Merged from agent_nodes.py
# =============================================================================

# Simple Pydantic models for structured output parsing
class SimpleChecklist(BaseModel):
    """Simple model matching existing checklist structure"""
    categories: Dict = Field(description="Checklist categories as they currently exist")


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


# =============================================================================
# AGENT NODE FUNCTIONS - Merged from agent_nodes.py
# =============================================================================

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


def parse_checklist_node(state: AgentState, llm: "ChatAnthropic") -> AgentState:
    """Parse checklist using structured output - much simpler!"""
    messages = state["messages"]
    checklist_text = messages[-1].content if messages else ""
    
    # Set up simple parser
    parser = PydanticOutputParser(pydantic_object=SimpleChecklist)
    prompt = get_checklist_parsing_prompt(checklist_text)
    
    # Create chain and parse - that's it!
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "checklist_text": checklist_text[:3000],
            "format_instructions": parser.get_format_instructions()
        })
        
        state["checklist"] = result.categories  # Already in the right format!
        state["messages"].append(AIMessage(content=f"Parsed {len(result.categories)} categories"))
        
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Parsing failed: {str(e)}"))
    
    return state


def match_checklist_node(state: AgentState, llm: "ChatAnthropic") -> AgentState:
    """Match documents to checklist items - keep it simple"""
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
            
            response = llm.invoke([HumanMessage(content=str(prompt))])
            cat_findings.append({
                "item": item['text'],
                "relevant_docs": response.content
            })
        
        findings[category['name']] = cat_findings
    
    state["findings"] = findings
    state["messages"].append(AIMessage(content=f"Matched checklist to {len(documents)} documents"))
    
    return state


def answer_question_node(state: AgentState, llm: "ChatAnthropic") -> AgentState:
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


def summarize_node(state: AgentState, llm: "ChatAnthropic") -> AgentState:
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


# =============================================================================
# LLM UTILITIES - Merged from llm_utilities.py
# =============================================================================

def create_batch_processor(llm: "ChatAnthropic", max_concurrency: int = None) -> RunnableLambda:
    """
    Create a batch processor using LangChain's retry and fallback mechanisms.
    
    Args:
        llm: ChatAnthropic instance
        max_concurrency: Maximum concurrent requests (uses config default if None)
    
    Returns:
        RunnableLambda configured with retry and fallback mechanisms
    """
    config = get_config()
    if max_concurrency is None:
        max_concurrency = config.api.max_concurrent_requests
    
    def process_single_item(input_data):
        """Process a single item with error handling"""
        try:
            messages, item_info = input_data
            response = llm.invoke(messages)
            return {
                'success': True,
                'response': response,
                'item_info': item_info,
                'error': None
            }
        except Exception as e:
            # Return error info instead of raising to allow partial batch success
            return {
                'success': False, 
                'response': None,
                'item_info': item_info,
                'error': str(e)
            }
    
    def process_batch(batch_inputs):
        """Process a batch of inputs with individual item error handling"""
        try:
            # Use LLM's batch method for efficiency
            messages_batch = [input_data[0] for input_data in batch_inputs]
            item_infos = [input_data[1] for input_data in batch_inputs]
            
            responses = llm.batch(
                messages_batch,
                config={"max_concurrency": max_concurrency}
            )
            
            # Process results with individual error handling
            results = []
            for i, (response, item_info) in enumerate(zip(responses, item_infos)):
                if response:
                    results.append({
                        'success': True,
                        'response': response,
                        'item_info': item_info,
                        'error': None
                    })
                else:
                    results.append({
                        'success': False,
                        'response': None,
                        'item_info': item_info,
                        'error': f'No response for item {i}'
                    })
            
            return results
            
        except Exception as e:
            # If batch fails completely, return error results for all items
            logger.warning(f"Batch processing failed: {e}")
            return [{
                'success': False,
                'response': None,
                'item_info': item_info,
                'error': str(e)
            } for _, item_info in batch_inputs]
    
    # Create the main processor with retry logic
    retryable_process_batch = with_retry(process_batch, max_attempts=3, base_delay=1.0)
    processor = RunnableLambda(retryable_process_batch)
    
    # Add fallback for complete failures
    fallback_processor = RunnableLambda(lambda batch_inputs: [
        {
            'success': False,
            'response': None,
            'item_info': item_info,
            'error': 'All processing attempts failed'
        } for _, item_info in batch_inputs
    ])
    
    return RunnableWithFallbacks(
        runnable=processor,
        fallbacks=[fallback_processor]
    )


def generate_checklist_descriptions(checklist: Dict, llm: "ChatAnthropic", batch_size: Optional[int] = None) -> Dict:
    """
    Generate detailed descriptions for each checklist item explaining what documents should satisfy it.
    Uses LangChain's built-in retry mechanisms and proper error handling for individual items.
    Returns checklist with added 'description' field for each item.
    
    Args:
        checklist: Checklist dictionary to enhance
        llm: ChatAnthropic instance for generating descriptions
        batch_size: Number of items to process in each batch (uses config default if None)
        
    Returns:
        Enhanced checklist with descriptions
    """
    
    config = get_config()
    if batch_size is None:
        batch_size = config.processing.description_batch_size
    
    # Create batch processor with retry and fallback mechanisms
    batch_processor = create_batch_processor(llm, config.api.max_concurrent_requests)
    
    # Process all checklist items
    enhanced_checklist = {}
    all_items_to_process = []
    
    # Collect all items with their context
    for cat_letter, category in checklist.items():
        cat_name = category.get('name', '')
        enhanced_checklist[cat_letter] = {
            'name': cat_name,
            'letter': cat_letter,
            'items': []
        }
        
        for item in category.get('items', []):
            item_data = {
                'category_letter': cat_letter,
                'category_name': cat_name,
                'item_text': item.get('text', ''),
                'original_item': item,
                'prompt': get_description_generation_prompt(cat_name, item.get('text', '')).format()
            }
            all_items_to_process.append(item_data)
    
    # Process items in batches
    total_items = len(all_items_to_process)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, total_items, batch_size), 1):
        batch = all_items_to_process[i:i + batch_size]
        batch_end = min(i + batch_size, total_items)
        
        # Update progress if available
        if hasattr(st, 'progress') and 'description_progress' in st.session_state:
            progress = i / total_items
            st.session_state.description_progress.progress(
                progress, 
                text=f"📝 Generating descriptions batch {batch_num}/{total_batches} (items {i+1}-{batch_end} of {total_items})"
            )
        
        # Prepare batch inputs for the processor
        batch_inputs = []
        for item_data in batch:
            messages = [HumanMessage(content=item_data['prompt'])]
            batch_inputs.append((messages, item_data))
        
        # Process batch using LangChain's built-in mechanisms
        try:
            batch_results = batch_processor.invoke(batch_inputs)
            
            # Process results with individual item error handling
            for result in batch_results:
                item_data = result['item_info']
                enhanced_item = item_data['original_item'].copy()
                
                if result['success'] and result['response']:
                    # Successfully generated description
                    enhanced_item['description'] = result['response'].content.strip()
                else:
                    # Use fallback description on error
                    logger.warning(f"Failed to generate description for item '{item_data['item_text']}': {result.get('error', 'Unknown error')}")
                    enhanced_item['description'] = f"Documents related to {item_data['item_text']}"
                
                enhanced_checklist[item_data['category_letter']]['items'].append(enhanced_item)
                
        except Exception as e:
            logger.error(f"Batch {batch_num} processing completely failed: {e}. Using fallback descriptions.")
            # Fallback: add all items with basic descriptions
            for item_data in batch:
                enhanced_item = item_data['original_item'].copy()
                enhanced_item['description'] = f"Documents related to {item_data['item_text']}"
                enhanced_checklist[item_data['category_letter']]['items'].append(enhanced_item)
    
    return enhanced_checklist


def batch_summarize_documents(documents: List[Dict], llm: "ChatAnthropic", batch_size: Optional[int] = None) -> List[Dict]:
    """
    Summarize documents using LangChain's built-in retry mechanisms and proper error handling.
    Uses RunnableLambda for better batch processing control with individual item error handling.
    Returns documents with added 'summary' field.
    
    Args:
        documents: List of document dictionaries to summarize
        llm: ChatAnthropic instance for generating summaries
        batch_size: Number of documents to process in each batch (uses config default if None)
        
    Returns:
        List of documents with added summary field
    """
    
    config = get_config()
    if batch_size is None:
        batch_size = config.processing.batch_size
    
    # Create batch processor with retry and fallback mechanisms
    batch_processor = create_batch_processor(llm, config.api.max_concurrent_requests)
    
    # Process documents in batches
    summarized_docs = []
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, total_docs, batch_size), 1):
        batch = documents[i:i + batch_size]
        batch_end = min(i + batch_size, total_docs)
        
        # Update progress with batch info
        if hasattr(st, 'progress') and 'summary_progress' in st.session_state:
            progress = i / total_docs
            st.session_state.summary_progress.progress(
                progress, 
                text=f"📝 Processing batch {batch_num}/{total_batches} (docs {i+1}-{batch_end} of {total_docs})"
            )
        
        # Prepare batch inputs for the processor
        batch_inputs = []
        for doc in batch:
            template = get_document_summarization_prompt(doc)
            prompt = template.format()
            messages = [HumanMessage(content=prompt)]
            batch_inputs.append((messages, doc))
        
        # Process batch using LangChain's built-in mechanisms
        try:
            batch_results = batch_processor.invoke(batch_inputs)
            
            # Process results with individual document error handling
            for result in batch_results:
                doc = result['item_info'].copy()
                
                if result['success'] and result['response']:
                    # Successfully generated summary
                    doc['summary'] = result['response'].content.strip()
                else:
                    # Use fallback summary on error
                    logger.warning(f"Failed to generate summary for document '{doc.get('name', 'Unknown')}': {result.get('error', 'Unknown error')}")
                    doc['summary'] = f"Document: {doc.get('name', 'Unknown')}"
                
                summarized_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Batch {batch_num} processing completely failed: {e}. Using fallback summaries.")
            # Fallback: add all documents with basic summaries
            for doc in batch:
                doc_copy = doc.copy()
                doc_copy['summary'] = f"Document: {doc.get('name', 'Unknown')}"
                summarized_docs.append(doc_copy)
    
    return summarized_docs


def create_document_embeddings_with_summaries(documents: List[Dict], model_name: str = None) -> Dict[str, Any]:
    """
    Prepare document data for LangChain-based similarity matching.
    No longer creates embeddings directly - LangChain handles embedding generation.
    
    Args:
        documents: List of documents with summaries
        
    Returns:
        Dictionary with document info formatted for LangChain matching
    """
    doc_info = []
    
    for doc in documents:
        # Prepare document info for LangChain matching
        doc_name = doc.get('name', 'Unknown')
        doc_path = doc.get('path', '')
        summary = doc.get('summary', '')
        
        doc_info.append({
            'name': doc_name,
            'path': doc_path,
            'full_path': doc.get('full_path', doc_path),
            'summary': summary,
            'original_doc': doc
        })
    
    return {
        'documents': doc_info
    }


def match_checklist_with_summaries(
    checklist: Dict, 
    doc_embeddings_data: Dict,
    model_name: str,
    threshold: Optional[float] = None
) -> Dict:
    """
    Match checklist items against document summaries using LangChain FAISS.
    Enhanced to use LLM-generated descriptions for better semantic matching.
    
    Args:
        checklist: Checklist dictionary with items and descriptions
        doc_embeddings_data: Dictionary containing document info and embeddings
        model_name: Name of the HuggingFace model for embeddings
        threshold: Similarity threshold for matching (uses config default if None)
        
    Returns:
        Dictionary with matching results
    """
    config = get_config()
    if threshold is None:
        threshold = config.processing.similarity_threshold
    
    doc_info = doc_embeddings_data['documents']
    
    # Create LangChain embeddings instance
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Convert document summaries to LangChain Documents
    documents = [
        Document(
            page_content=f"{doc['name']}\n{doc['path']}\n{doc['summary']}",
            metadata={
                'name': doc['name'],
                'path': doc['path'],
                'full_path': doc.get('full_path', doc['path']),
                'summary': doc['summary'],
                **doc.get('original_doc', {}).get('metadata', {})
            }
        )
        for doc in doc_info
    ]
    
    # Create LangChain FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold, "k": 5}
    )
    
    results = {}
    
    for cat_letter, category in checklist.items():
        cat_name = category.get('name', '')
        cat_results = {
            'name': cat_name,
            'letter': cat_letter,
            'total_items': len(category.get('items', [])),
            'matched_items': 0,
            'items': []
        }
        
        for item in category.get('items', []):
            item_text = item.get('text', '')
            item_description = item.get('description', '')
            
            # Create enhanced query using both item text and generated description
            if item_description:
                # Use the LLM-generated description for richer semantic matching
                query = f"{cat_name}: {item_text}\n{item_description}"
            else:
                # Fall back to original method if no description available
                query = f"{cat_name}: {item_text}"
            
            # Use LangChain retriever for similarity search
            docs = safe_execute(
                lambda: retriever.invoke(query),
                default=[],
                context="Document matching with summaries"
            )
            
            # Convert LangChain documents to matches format
            matches = []
            for doc in docs[:5]:  # Keep top 5 matches
                match_data = {
                    'name': doc.metadata['name'],
                    'path': doc.metadata['path'], 
                    'full_path': doc.metadata.get('full_path', doc.metadata['path']),
                    'summary': doc.metadata['summary'],
                    'score': 0.8,  # LangChain retriever doesn't return raw scores
                    'metadata': {k: v for k, v in doc.metadata.items() 
                                if k not in ['name', 'path', 'full_path', 'summary']}
                }
                matches.append(match_data)
            
            item_result = {
                'text': item_text,
                'original': item.get('original', item_text),
                'description': item_description,  # Include the generated description
                'matches': matches
            }
            
            # Count items with matches toward category total
            if matches:
                cat_results['matched_items'] += 1
            
            cat_results['items'].append(item_result)
        
        results[cat_letter] = cat_results
    
    return results


# =============================================================================
# LANGGRAPH AGENT FUNCTIONS
# =============================================================================

def get_langgraph_agent(api_key: Optional[str] = None, model: Optional[str] = None) -> Optional[Tuple[Any, "ChatAnthropic"]]:
    """
    Create a LangGraph agent with Anthropic
    
    Args:
        api_key: Anthropic API key (optional, will be sourced from environment/config)
        model: Model name to use (optional, will use config default)
        
    Returns:
        Tuple of (compiled_app, llm) or None if not available
    """
    
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
        return self.app is not None and self.llm is not None
    
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
