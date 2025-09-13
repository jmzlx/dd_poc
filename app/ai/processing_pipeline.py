#!/usr/bin/env python3
"""
Processing Pipeline Module

This module contains content processing pipeline and workflow functions,
including agent node functions and batch processing utilities.
"""

# Standard library imports
import logging
from typing import List, Dict, Optional

# Third-party imports
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Local imports
from app.ai.agent_utils import AgentState, create_batch_processor
from app.ai.prompts import (
    get_checklist_parsing_prompt,
    get_document_relevance_prompt,
    get_question_answering_prompt,
    get_findings_summary_prompt,
    get_description_generation_prompt,
    get_document_summarization_prompt
)
from app.core.config import get_config
from app.core.constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


# Pydantic models for structured output parsing
class ChecklistItem(BaseModel):
    """Individual checklist item"""
    text: str = Field(description="The checklist item text")
    original: str = Field(description="The original text before any cleanup")

class ChecklistCategory(BaseModel):
    """Checklist category with items"""
    name: str = Field(description="Category name (e.g., 'Organizational and Corporate Documents')")
    items: List[ChecklistItem] = Field(description="List of checklist items in this category")

class StructuredChecklist(BaseModel):
    """Complete checklist with all categories"""
    categories: Dict[str, ChecklistCategory] = Field(
        description="Dictionary of categories keyed by letter (A, B, C, etc.)"
    )

class Question(BaseModel):
    """Individual question"""
    category: str = Field(description="Question category")
    question: str = Field(description="The question text")
    id: str = Field(description="Unique question ID")

class StructuredQuestions(BaseModel):
    """List of structured questions"""
    questions: List[Question] = Field(description="List of all questions")



def route_task(state: AgentState) -> AgentState:
    """Route to appropriate task based on current state"""
    messages = state["messages"]
    if not messages:
        return state

    last_message = messages[-1].content if messages else ""

    # Determine next action based on message content
    if "parse" in last_message.lower() and "checklist" in last_message.lower():
        state["next_action"] = "parse_checklist"
    elif "analyze" in last_message.lower() or "match" in last_message.lower():
        state["next_action"] = "match_checklist"
    elif "?" in last_message:
        state["next_action"] = "answer_question"
    else:
        state["next_action"] = "summarize"

    return state


def parse_checklist_node(state: AgentState, llm: "ChatAnthropic") -> AgentState:
    """Parse checklist using structured output - standardized with StructuredChecklist!"""
    messages = state["messages"]
    checklist_text = messages[-1].content if messages else ""

    # Set up structured parser - using the same as parse_checklist function
    parser = PydanticOutputParser(pydantic_object=StructuredChecklist)
    prompt = get_checklist_parsing_prompt()

    try:
        # Format the prompt with the checklist text and format instructions
        formatted_prompt = prompt.format_messages(
            checklist_text=checklist_text,  # Don't truncate - let LLM handle full checklist
            format_instructions=parser.get_format_instructions()
        )
        
        # Get LLM response
        llm_response = llm.invoke(formatted_prompt)
        
        # Parse the response using the Pydantic parser
        result = parser.parse(llm_response.content)

        # Convert Pydantic model to expected dictionary format (same as parse_checklist)
        categories_dict = {}
        for key, category in result.categories.items():
            categories_dict[key] = {
                'name': category.name,
                'items': [
                    {
                        'text': item.text,
                        'original': item.original
                    }
                    for item in category.items
                ]
            }

        state["checklist"] = categories_dict
        state["messages"].append(AIMessage(content=f"Parsed {len(categories_dict)} categories"))

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
    if next_action == "parse_checklist":
        return "parse_checklist"
    elif next_action == "match_checklist":
        return "match_checklist"
    elif next_action == "answer_question":
        return "answer_question"
    else:
        return "summarize"




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
        batch_size = DEFAULT_BATCH_SIZE

    # Create batch processor with retry and fallback mechanisms
    batch_processor = create_batch_processor(llm, max_concurrency=3)

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
                    summarized_docs.append(doc)
                else:
                    # Fail on summary generation error
                    error_msg = f"Failed to generate summary for document '{doc.get('name', 'Unknown')}': {result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

        except Exception as e:
            error_msg = f"Batch {batch_num} processing completely failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    return summarized_docs
