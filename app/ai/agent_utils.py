#!/usr/bin/env python3
"""
Agent Utilities Module

This module contains utility functions, helper methods, and type definitions
for the LangGraph agent system.
"""

# Standard library imports
import logging
import random
import time
from enum import Enum
from typing import Optional, Dict, List, Sequence

# Third-party imports
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

# Local imports
from app.core.config import get_config

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

    return wrapper


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
        max_concurrency = 3  # Default max concurrency

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
            # Fail immediately on any error
            error_msg = f"Single item processing failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

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

            # Process results with individual error handling - fail on any error
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
                    # Fail immediately on any missing response
                    error_msg = f'No response for item {i}'
                    logger.error(error_msg)
                    raise Exception(error_msg)

            return results

        except Exception as e:
            # If batch fails completely, fail immediately
            error_msg = f"Batch processing failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # Create the main processor with retry logic
    retryable_process_batch = with_retry(process_batch, max_attempts=3, base_delay=1.0)
    processor = RunnableLambda(retryable_process_batch)

    return processor


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Define the state for our agent
class AgentState(TypedDict):
    """State for the due diligence agent"""
    messages: Sequence["BaseMessage"]
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
