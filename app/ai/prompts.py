#!/usr/bin/env python3
"""
AI Prompts Module

This module contains all prompt templates used for AI interactions
in the DD-Checklist application.
"""

# Standard library imports
import json
from typing import Dict, List

# Third-party imports
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Local imports
from app.core.constants import QA_MAX_TOKENS


def get_checklist_parsing_prompt() -> ChatPromptTemplate:
    """Generate prompt template for parsing due diligence checklists with structured output"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a JSON parser. Your ONLY task is to convert the checklist into valid JSON format.

CRITICAL PARSING RULES:
- Return ONLY valid JSON - no explanations, no notes, no additional text
- Do NOT add any conversational text before or after the JSON
- Do NOT offer to continue or ask questions
- Do NOT provide partial results or examples
- Parse the COMPLETE document - every single category and item

JSON Structure Required:
- Top-level object with "categories" field
- Categories keyed by letter (A, B, C, D, E, etc.)
- Each category has "name" and "items" fields
- Each item has "text" and "original" fields

You must process the ENTIRE checklist. Do not stop after a few categories.

Output format:
{
  "categories": {
    "A": {
      "name": "Category Name",
      "items": [
        {"text": "Item text", "original": "1. Item text"}
      ]
    }
  }
}

Return ONLY the JSON. No other text.
"""),
        HumanMessagePromptTemplate.from_template("""Parse this complete checklist into the exact JSON format:

{checklist_text}

Required JSON schema:
{format_instructions}

Return the complete JSON with all categories found in the checklist:""")
    ])


def get_document_relevance_prompt(item_text: str, documents: List[str]) -> PromptTemplate:
    """Generate prompt for assessing document relevance to checklist items"""
    return PromptTemplate.from_template(
        """Analyze which documents are most relevant to the following checklist item.

Checklist Item: {item_text}

Available Documents:
{documents}

Provide a brief analysis identifying the most relevant documents and explain why they are relevant to this checklist item. Be concise and specific."""
    )


def get_question_answering_prompt(question: str, context: str) -> ChatPromptTemplate:
    """Generate prompt for answering questions based on document context"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="Answer questions based on document context. Provide comprehensive answers with citations."),
        HumanMessage(content=f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer:")
    ])


def get_findings_summary_prompt(findings: Dict, max_chars: int = QA_MAX_TOKENS) -> PromptTemplate:
    """Generate prompt for summarizing due diligence findings"""
    findings_text = json.dumps(findings, indent=2)[:max_chars]
    return PromptTemplate.from_template(
        "Provide an executive summary of these due diligence findings:\n\n"
        "{findings_text}\n\n"
        "Focus on:\n"
        "1. Completeness of documentation\n"
        "2. Key gaps or concerns\n"
        "3. Overall assessment"
    ).partial(findings_text=findings_text)


def get_description_generation_prompt(category_name: str, item_text: str) -> PromptTemplate:
    """Generate prompt for creating checklist item descriptions"""
    return PromptTemplate.from_template(
        "For this due diligence checklist item, provide a concise description (1-2 sentences) "
        "explaining what types of documents or information would satisfy this requirement.\n\n"
        "Category: {category_name}\n"
        "Checklist Item: {item_text}\n\n"
        "Description:"
    ).partial(category_name=category_name, item_text=item_text)


def get_document_type_classification_prompt() -> PromptTemplate:
    """Generate prompt for fast document type classification based on first chunk content"""
    return PromptTemplate.from_template(
        "Classify the document type using one short phrase. Use exact terminology.\n"
        "Respond with ONLY the document type, no prefix or explanation.\n\n"
        "Examples:\n"
        "certificate of incorporation\n"
        "corporate bylaws\n"
        "amended and restated bylaws\n"
        "board resolution\n"
        "financial statement\n"
        "employment agreement\n"
        "software license agreement\n\n"
        "Document: {doc_name}\n"
        "Content: {content_preview}\n\n"
        "Document type:"
    )

def get_document_summarization_prompt(doc: Dict) -> PromptTemplate:
    """Generate prompt for document type identification and summarization"""
    doc_name = doc.get('name', 'Unknown')
    doc_path = doc.get('path', '')
    text_preview = doc.get('content', '')[:1000] if doc.get('content') else ''
    
    return PromptTemplate.from_template(
        "Identify and describe what type of document this is in 1-2 sentences.\n\n"
        "Examples: financial statement, contract agreement, corporate governance document, etc.\n\n"
        "Document: {doc_name}\n"
        "Path: {doc_path}\n"
        "Content preview:\n{text_preview}\n\n"
        "Document type description:"
    ).partial(doc_name=doc_name, doc_path=doc_path, text_preview=text_preview)