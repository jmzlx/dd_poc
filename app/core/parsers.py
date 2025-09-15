#!/usr/bin/env python3
"""
LLM-based parsing functions for due diligence documents.

This module provides modern structured output parsing using Pydantic models
to ensure reliable, type-safe parsing of LLM responses.
"""

from typing import Dict, List
from app.core.logging import logger


def parse_checklist(checklist_text: str, llm) -> Dict:
    """
    Parse markdown checklist using Pydantic structured output.
    
    This approach uses LangChain's PydanticOutputParser to ensure the LLM
    returns properly structured data that matches our expected format.
    
    Args:
        checklist_text: The raw checklist text to parse
        llm: LLM instance to use for parsing
        
    Returns:
        Dictionary with categories and their items
        
    Raises:
        RuntimeError: If LLM is not available or parsing fails
        ValueError: If llm parameter is not provided
    """
    if llm is None:
        raise ValueError("LLM parameter is required")
    
    try:
        from langchain_core.output_parsers import PydanticOutputParser
        from app.ai.processing_pipeline import StructuredChecklist
        from app.ai.prompts import get_checklist_parsing_prompt
        
        # Set up structured output parser
        parser = PydanticOutputParser(pydantic_object=StructuredChecklist)
        
        # Use centralized prompt from prompts.py (avoid duplication)
        prompt = get_checklist_parsing_prompt()
        
        # Format the prompt with the checklist text and format instructions
        formatted_prompt = prompt.format_messages(
            checklist_text=checklist_text,
            format_instructions=parser.get_format_instructions()
        )
        
        # Get LLM response
        logger.info(f"Sending checklist to LLM for parsing (length: {len(checklist_text)} chars)")
        llm_response = llm.invoke(formatted_prompt)
        logger.debug(f"LLM response length: {len(llm_response.content)} chars")
        
        # Parse the response using the Pydantic parser
        result = parser.parse(llm_response.content)
        
        # Convert Pydantic model to expected dictionary format, filtering out invalid items
        categories_dict = {}
        for key, category in result.categories.items():
            # Only include valid items with actual text content
            valid_items = category.get_valid_items()
            if valid_items:  # Only include categories that have valid items
                items_list = []
                for item in valid_items:
                    
                    items_list.append({
                        'text': item.text,
                        'original': item.original or item.text  # Use text as fallback if original is None
                    })
                
                categories_dict[key] = {
                    'name': category.name,
                    'items': items_list
                }
        
        logger.info(f"Successfully parsed {len(categories_dict)} categories: {list(categories_dict.keys())}")
        return categories_dict
        
    except Exception as e:
        raise RuntimeError(f"Structured parsing failed: {str(e)}")


def parse_questions(questions_text: str, llm) -> List[Dict]:
    """
    Parse markdown questions using Pydantic structured output.
    
    Args:
        questions_text: The raw questions text to parse
        llm: LLM instance to use for parsing
        
    Returns:
        List of dictionaries with question data
        
    Raises:
        RuntimeError: If LLM is not available or parsing fails
        ValueError: If llm parameter is not provided
    """
    if llm is None:
        raise ValueError("LLM parameter is required")
    
    try:
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.ai.processing_pipeline import StructuredQuestions
        
        # Set up structured output parser
        parser = PydanticOutputParser(pydantic_object=StructuredQuestions)
        
        # Create prompt with format instructions
        from langchain_core.prompts import HumanMessagePromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a document parser. Parse the due diligence questions document into the EXACT JSON format specified.

CRITICAL:
- Return ONLY valid JSON, no additional text or explanations
- Extract categories (like "### A. Category Name")
- Extract numbered questions within each category
- Clean up markdown formatting but preserve core text
- Follow the exact format specified in the format instructions

The output must be valid JSON that can be parsed directly.
"""),
            HumanMessagePromptTemplate.from_template("""Parse these questions into the exact JSON format:

{questions_text}

Required JSON schema:
{format_instructions}

Return only the JSON:""")
        ])
        
        # Format the prompt with the questions text and format instructions
        formatted_prompt = prompt.format_messages(
            questions_text=questions_text,
            format_instructions=parser.get_format_instructions()
        )
        
        # Get LLM response
        llm_response = llm.invoke(formatted_prompt)
        
        # Parse the response using the Pydantic parser
        result = parser.parse(llm_response.content)
        
        # Convert Pydantic model to expected list format
        return [
            {
                'category': question.category,
                'question': question.question,
                'id': question.id
            }
            for question in result.questions
        ]
        
    except Exception as e:
        raise RuntimeError(f"Structured parsing failed: {str(e)}")