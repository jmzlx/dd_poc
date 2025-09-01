#!/usr/bin/env python3
"""
Business Logic Services Module

Simplified service layer with focused functions instead of over-abstracted classes.
"""

import re
import logging
import warnings
from pathlib import Path

# Suppress verbose LangChain warnings in services
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")
warnings.filterwarnings("ignore", message=".*Relevance scores must be between.*")
warnings.filterwarnings("ignore", message=".*No relevant docs were retrieved.*")
from typing import Dict, List, Optional, Any
import markdown



from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from .config import get_config
from .document_processing import DocumentProcessor, escape_markdown_math


logger = logging.getLogger(__name__)


# =============================================================================
# PARSING FUNCTIONS - Simplified from ChecklistParser and QuestionParser classes
# =============================================================================

def parse_checklist(checklist_text: str) -> Dict:
    """Parse markdown checklist into categories and items using standard markdown parser"""
    categories = {}
    current_category = None

    # Parse line by line for reliable extraction
    lines = checklist_text.split('\n')
    for line_num, original_line in enumerate(lines):
        line = original_line.strip()

        # Skip empty lines and separator lines
        if not line or line.startswith('⸻') or line.startswith('---'):
            continue

        # Skip title lines
        if 'due diligence checklist' in line.lower() or line.startswith('#'):
            continue

        # Category headers - look for pattern "A. Category Name"
        category_match = re.match(r'^([A-Z])\.\s+(.+)', line)
        if category_match and not re.match(r'^\d+\.\s+', line):
            letter, name = category_match.groups()
            current_category = letter
            categories[letter] = {
                'name': name.strip(),
                'items': []
            }
            continue

        # Numbered items within categories - look for indented items
        if current_category and line:
            # Check if original line was indented (starts with tab or multiple spaces)
            is_indented = original_line.startswith(('\t', '  ', '    '))
            item_match = re.match(r'^\d+\.\s+(.+)', line)

            if item_match and (is_indented or current_category):
                item_text = item_match.group(1).strip()
                if item_text and not item_text.lower().startswith('[other requests'):
                    # Clean up markdown formatting but preserve content
                    clean_text = re.sub(r'\[.*?\]', '', item_text).strip()
                    if not clean_text:
                        clean_text = item_text
                    
                    categories[current_category]['items'].append({
                        'text': clean_text,
                        'original': item_text
                    })
    
    return categories


def parse_questions(questions_text: str) -> List[Dict]:
    """Parse markdown questions into a list using standard markdown parser"""
    # Convert markdown to understand structure
    md = markdown.Markdown(extensions=['toc'])
    html = md.convert(questions_text)
    
    questions = []
    current_category = None
    
    # Parse line by line for reliable extraction
    lines = questions_text.split('\n')
    for line in lines:
        line = line.strip()
        
        # Category headers (### format)
        if line.startswith('### '):
            match = re.match(r'###\s+([A-Z])\.\s+(.+)', line)
            if match:
                letter, name = match.groups()
                current_category = f"{letter}. {name.strip()}"
        
        # Question items (numbered lists)
        elif current_category and line:
            match = re.match(r'^\d+\.\s+(.+)', line)
            if match:
                question_text = match.group(1).strip()
                if question_text:
                    # Clean markdown formatting
                    clean_question = re.sub(r'\*\*(.*?)\*\*', r'\1', question_text)  # Remove bold
                    clean_question = re.sub(r'\*(.*?)\*', r'\1', clean_question)      # Remove italics
                    
                    questions.append({
                        'category': current_category,
                        'question': clean_question,
                        'id': f"q_{len(questions)}"
                    })
    
    return questions


# =============================================================================
# SEARCH FUNCTIONS - Consolidated from ChecklistMatcher and QuestionAnswerer
# =============================================================================

def create_vector_store(source_data, model_name: str) -> FAISS:
    """Unified vector store creation from various data sources"""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Handle different input types
    if isinstance(source_data, list):
        if all(isinstance(item, Document) for item in source_data):
            # Already LangChain documents
            return FAISS.from_documents(source_data, embeddings)
        elif all(isinstance(item, dict) for item in source_data):
            # Document chunks
            documents = [
                Document(
                    page_content=chunk['text'], 
                    metadata={
                        'source': chunk.get('source', ''),
                        'path': chunk.get('path', ''),
                        'full_path': chunk.get('full_path', ''),
                        **chunk.get('metadata', {})
                    }
                ) for chunk in source_data
            ]
            return FAISS.from_documents(documents, embeddings)
    elif isinstance(source_data, dict) and 'documents' in source_data:
        # Document embeddings data with summaries
        documents = [
            Document(
                page_content=f"{doc['name']}\n{doc['path']}\n{doc['summary']}",
                metadata={
                    'name': doc['name'],
                    'path': doc['path'],
                    'summary': doc['summary'],
                    **doc.get('original_doc', {}).get('metadata', {})
                }
            ) for doc in source_data['documents']
        ]
        return FAISS.from_documents(documents, embeddings)
    
    raise ValueError("Unsupported data type for vector store creation")


def search_and_analyze(queries: List[Dict], vector_store: FAISS, llm=None, threshold: float = 0.7, search_type: str = 'items') -> Dict:
    """Unified search function for both checklist items and questions using LangChain RAG"""
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold, "k": 5 if search_type == 'questions' else 10}
    )
    
    # Create RAG chain if LLM is provided
    qa_chain = None
    if llm:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the provided context to answer the question. Be concise and factual.
            
Context: {context}
            
Question: {question}
            
Answer:"""
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )
    
    if search_type == 'items':
        return _process_checklist_items(queries, retriever, qa_chain)
    else:
        return _process_questions(queries, retriever, qa_chain)


def _process_checklist_items(checklist: Dict, retriever, qa_chain=None) -> Dict:
    """Process checklist items with unified search logic"""
    results = {}
    for cat_letter, category in checklist.items():
        cat_results = {
            'name': category['name'],
            'items': [],
            'total_items': len(category['items']),
            'matched_items': 0
        }
        
        for item in category['items']:
            query = f"{category['name']}: {item['text']}"
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                logger.error(f"Error in document matching: {e}")
                docs = []
            
            matches = [{
                'name': doc.metadata.get('source', ''),
                'path': doc.metadata.get('path', ''),
                'full_path': doc.metadata.get('full_path', ''),
                'score': 0.8,  # LangChain similarity scores not directly accessible
                'metadata': {k: v for k, v in doc.metadata.items() 
                           if k not in ['source', 'path', 'full_path']}
            } for doc in docs[:5]]
            
            if matches:
                cat_results['matched_items'] += 1
            
            cat_results['items'].append({
                'text': item['text'],
                'original': item['original'],
                'matches': matches
            })
        
        results[cat_letter] = cat_results
    
    return results


def _process_questions(questions: List[Dict], retriever, qa_chain=None) -> Dict:
    """Process questions with unified search logic"""
    answers = {}
    for question in questions:
        try:
            docs = retriever.invoke(question['question'])
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            docs = []
        
        if docs:
            chunks_data = [{
                'text': doc.page_content[:500],
                'source': doc.metadata.get('source', ''),
                'path': doc.metadata.get('path', ''),
                'score': 0.8,
                'metadata': {k: v for k, v in doc.metadata.items() 
                           if k not in ['source', 'path']}
            } for doc in docs]
            
            # Generate answer using RAG chain if available
            answer_text = "Retrieved relevant document chunks."
            if qa_chain:
                try:
                    answer_text = qa_chain.run(question['question'])
                except Exception as e:
                    logger.error(f"RAG chain failed: {e}")
                    answer_text = "Retrieved relevant document chunks."
            
            answers[question['id']] = {
                'question': question['question'],
                'category': question['category'],
                'answer': answer_text,
                'chunks': chunks_data,
                'has_answer': True
            }
        else:
            answers[question['id']] = {
                'question': question['question'],
                'category': question['category'],
                'answer': "No relevant documents found",
                'chunks': [],
                'has_answer': False
            }
    
    return answers


# =============================================================================
# REPORT GENERATION FUNCTIONS - Simplified from ReportGenerator class
# =============================================================================

def generate_reports(documents: Dict[str, Dict], data_room_name: str = "Unknown", 
                    strategy_text: str = "", checklist_results: Dict = None, 
                    report_type: str = "overview", llm=None) -> str:
    """Unified report generation using LangChain prompt templates"""
    
    if not llm:
        return _generate_basic_report(documents, data_room_name, checklist_results, report_type)
    
    # Define prompt templates
    if report_type == "overview":
        template = PromptTemplate(
            input_variables=["company_name", "document_summaries"],
            template="""Based on the following document summaries from a due diligence data room, provide a comprehensive company overview.

Company: {company_name}

Document Summaries:
{document_summaries}

Please provide:
1. Company name and industry
2. Business model and key products/services  
3. Market position and competitive advantages
4. Key financials (if available)
5. Organizational structure
6. Notable risks or concerns
7. Overall assessment for M&A consideration

Format the response in clear sections with bullet points where appropriate."""
        )
        
        # Prepare document summaries
        doc_summaries = []
        for path, doc_info in list(documents.items())[:10]:
            if 'summary' in doc_info:
                doc_summaries.append(f"{doc_info['name']}: {doc_info['summary']}")
            else:
                content_preview = doc_info.get('content', '')[:500]
                if content_preview:
                    doc_summaries.append(f"{doc_info['name']}: {content_preview}")
        
        if not doc_summaries:
            return "No documents available for summary generation."
        
        inputs = {
            "company_name": data_room_name,
            "document_summaries": "\n".join(doc_summaries[:10])
        }
        
    elif report_type == "strategic":
        template = PromptTemplate(
            input_variables=["strategy_text", "checklist_context"],
            template="""Based on the due diligence checklist results and the selected strategy, provide a strategic analysis.

Strategy Document:
{strategy_text}

Checklist Coverage:
{checklist_context}

Please provide:
1. Strategic alignment assessment
2. Key risks and gaps identified
3. Opportunities and synergies
4. Recommended next steps
5. Overall recommendation

Format the response with clear sections and bullet points."""
        )
        
        # Build checklist context
        if not checklist_results:
            return "No checklist results available for strategic analysis."
            
        checklist_context = []
        for cat_id, cat_data in checklist_results.items():
            cat_name = cat_data['name']
            matched_items = cat_data['matched_items']
            total_items = cat_data['total_items']
            coverage = (matched_items / total_items * 100) if total_items > 0 else 0
            
            checklist_context.append(f"- {cat_name}: {coverage:.0f}% coverage ({matched_items}/{total_items} items)")
            
            # Add details about gaps
            missing_items = [item['text'] for item in cat_data['items'] if not item['matches']]
            if missing_items and len(missing_items) <= 3:
                checklist_context.append(f"  Missing: {', '.join(missing_items[:3])}")
        
        inputs = {
            "strategy_text": strategy_text,
            "checklist_context": "\n".join(checklist_context)
        }
    
    # Execute the chain
    try:
        chain = template | llm | StrOutputParser()
        response = chain.invoke(inputs)
        return escape_markdown_math(response.strip())
    except Exception as e:
        logger.error(f"LLM report generation failed: {e}")
        return f"Failed to generate {report_type} report: {str(e)}"


def _generate_basic_report(documents: Dict[str, Dict], data_room_name: str, 
                          checklist_results: Dict, report_type: str) -> str:
    """Generate basic reports without AI"""
    if report_type == "overview":
        doc_count = len(documents)
        file_types = {}
        
        for doc_info in documents.values():
            doc_type = doc_info.get('metadata', {}).get('type', 'unknown')
            file_types[doc_type] = file_types.get(doc_type, 0) + 1
        
        return f"""# Company Overview: {data_room_name}

## Document Analysis
- **Total Documents**: {doc_count}
- **File Types**: {', '.join([f"{count} {type_name}" for type_name, count in file_types.items()])}

## Key Areas Covered
Based on the document structure, this data room appears to cover standard due diligence areas including corporate documents, financial records, and operational information.

*Note: Enable AI features for detailed company analysis and insights.*
"""
    
    elif report_type == "strategic":
        if not checklist_results:
            return "No checklist results available for strategic analysis."
            
        total_items = sum(cat['total_items'] for cat in checklist_results.values())
        matched_items = sum(cat['matched_items'] for cat in checklist_results.values())
        coverage = (matched_items / total_items * 100) if total_items > 0 else 0
        
        analysis = f"""# Strategic Analysis

## Documentation Coverage
- **Overall Coverage**: {coverage:.1f}% ({matched_items}/{total_items} items)

## Category Breakdown
"""
        
        for cat_id, cat_data in checklist_results.items():
            cat_coverage = (cat_data['matched_items'] / cat_data['total_items'] * 100) if cat_data['total_items'] > 0 else 0
            analysis += f"- **{cat_data['name']}**: {cat_coverage:.1f}% coverage\n"
        
        analysis += """
## Key Observations
- Review areas with low coverage for potential risks
- High coverage areas indicate good documentation practices
- Consider requesting missing documents for incomplete categories

*Note: Enable AI features for detailed strategic analysis and recommendations.*
"""
        
        return analysis
    
    return "Invalid report type specified."


# =============================================================================
# MAIN SERVICE FUNCTIONS - Simplified orchestration
# =============================================================================




def search_documents(doc_processor: DocumentProcessor, query: str, top_k: int = 5, 
                    threshold: Optional[float] = None) -> List[Dict]:
    """Search documents using the document processor"""
    return doc_processor.search(query, top_k, threshold)


def load_default_file(directory: Path, pattern: str) -> str:
    """Load the first file matching pattern from directory"""
    try:
        files = list(directory.glob(pattern))
        return files[0].read_text(encoding='utf-8') if files else ""
    except Exception as e:
        logger.error(f"File loading failed: {e}")
        return ""


