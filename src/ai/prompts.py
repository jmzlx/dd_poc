#!/usr/bin/env python3
"""
AI Prompts Module

This module contains all prompt templates used for AI interactions
in the DD-Checklist application.
"""

from typing import Dict, List


def get_checklist_parsing_prompt(checklist_text: str) -> str:
    """
    Generate prompt for parsing due diligence checklists
    
    Args:
        checklist_text: Raw checklist text to parse
        
    Returns:
        Formatted prompt string
    """
    return f"""Parse this due diligence checklist into a structured JSON format.
        
Extract categories (A., B., C.) and numbered items.

Return ONLY valid JSON:
{{
    "A": {{
        "name": "Category Name",
        "items": [{{"text": "item", "number": 1}}]
    }}
}}

Checklist:
{checklist_text[:3000]}

JSON:"""


def get_document_relevance_prompt(item_text: str, documents: List[str]) -> str:
    """
    Generate prompt for assessing document relevance to checklist items
    
    Args:
        item_text: Checklist item text
        documents: List of document names
        
    Returns:
        Formatted prompt string
    """
    return f"""Which of these documents is relevant to: {item_text}
                
Documents: {documents}

List the relevant document names only."""


def get_question_answering_prompt(question: str, context: str) -> str:
    """
    Generate prompt for answering questions based on document context
    
    Args:
        question: User question
        context: Document context
        
    Returns:
        Formatted prompt string
    """
    return f"""Answer this question based on the documents:

Question: {question}

Document Context:
{context}

Provide a comprehensive answer with citations."""


def get_findings_summary_prompt(findings: Dict, max_chars: int = 2000) -> str:
    """
    Generate prompt for summarizing due diligence findings
    
    Args:
        findings: Dictionary of findings to summarize
        max_chars: Maximum characters to include from findings
        
    Returns:
        Formatted prompt string
    """
    import json
    findings_text = json.dumps(findings, indent=2)[:max_chars]
    
    return f"""Provide an executive summary of the due diligence findings:

{findings_text}

Focus on:
1. Completeness of documentation
2. Key gaps or concerns
3. Overall assessment"""


def get_description_generation_prompt(category_name: str, item_text: str) -> str:
    """
    Generate prompt for creating checklist item descriptions
    
    Args:
        category_name: Name of the checklist category
        item_text: Text of the checklist item
        
    Returns:
        Formatted prompt string
    """
    return f"""For this due diligence checklist item, provide a brief description (2-3 sentences) explaining what types of documents or information would satisfy this requirement. Focus on the specific document types, content characteristics, and key information that would be relevant.

Category: {category_name}
Checklist Item: {item_text}

Description (2-3 sentences explaining what documents/information satisfy this requirement):"""


def get_document_summarization_prompt(doc: Dict) -> str:
    """
    Generate prompt for document type identification and summarization
    
    Args:
        doc: Dictionary containing document information
        
    Returns:
        Formatted prompt string
    """
    # Extract text preview (first 1000 chars)
    text_preview = doc.get('content', '')[:1000] if doc.get('content') else ''
    doc_name = doc.get('name', 'Unknown')
    doc_path = doc.get('path', '')
    
    return f"""Identify and describe what type of document this is in 1-2 sentences.
Focus specifically on the document type, category, and what kind of information it contains.

Examples of document types: financial statement, contract agreement, corporate governance document, employee handbook, technical specification, compliance report, audit report, etc.

Document: {doc_name}
Path: {doc_path}
Content preview:
{text_preview}

Document type description (1-2 sentences only):"""


# Template constants for common patterns
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2000
