#!/usr/bin/env python3
"""
AI Prompts Module

This module contains all prompt templates used for AI interactions
in the DD-Checklist application.
"""

import json
from typing import Dict, List
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


def get_checklist_parsing_prompt(checklist_text: str) -> ChatPromptTemplate:
    """Generate prompt for parsing due diligence checklists with structured output"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Parse this due diligence checklist into structured format. Extract:
- Categories (A., B., C., etc.) with their names
- Numbered items within each category (1., 2., 3., etc.)
- Total count of items

Follow the exact format specified in the format instructions.
"""),
        HumanMessage(content="""Parse this checklist:

{checklist_text}

{format_instructions}

Please provide the structured output:""")
    ])


def get_document_relevance_prompt(item_text: str, documents: List[str]) -> PromptTemplate:
    """Generate prompt for assessing document relevance to checklist items with structured output"""
    return PromptTemplate.from_template(
        """Analyze which documents are relevant to the following checklist item:

Checklist Item: {item_text}

Available Documents:
{documents}

{format_instructions}

Please provide your analysis in the specified format:"""
    )


def get_question_answering_prompt(question: str, context: str) -> ChatPromptTemplate:
    """Generate prompt for answering questions based on document context"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="Answer questions based on document context. Provide comprehensive answers with citations."),
        HumanMessage(content=f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer:")
    ])


def get_findings_summary_prompt(findings: Dict, max_chars: int = 2000) -> PromptTemplate:
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