#!/usr/bin/env python3
"""
Business Logic Services Module

This module contains the core business logic services for the DD-Checklist application.
Services handle specific domain operations and coordinate between different components.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_processing import DocumentProcessor, escape_markdown_math


class ChecklistParser:
    """Service for parsing due diligence checklists"""
    
    @staticmethod
    def parse_checklist(checklist_text: str) -> Dict:
        """
        Parse markdown checklist into categories and items
        
        Args:
            checklist_text: Raw checklist text in markdown format
            
        Returns:
            Dictionary with parsed categories and items
        """
        categories = {}
        current_category = None
        
        for line in checklist_text.split('\n'):
            # Category header (e.g., "A. Corporate Organization" or "## A. Corporate Organization")
            # Try both formats
            match = None
            if line.startswith('## '):
                match = re.match(r'## ([A-Z])\. (.+)', line)
            elif line.strip() and not line.startswith('\t') and not line.startswith(' '):
                # Try plain format (no ##)
                match = re.match(r'^([A-Z])\. (.+)', line.strip())
            
            if match:
                letter, name = match.groups()
                current_category = letter
                categories[letter] = {
                    'name': name.strip(),
                    'items': []
                }
            # Numbered items (may be indented with tabs or spaces)
            elif current_category:
                # Check for numbered items with various indentation
                line_stripped = line.strip()
                if re.match(r'^\d+\.', line_stripped):
                    item_text = re.sub(r'^\d+\.\s*', '', line_stripped)
                    if item_text:
                        # Clean up [bracketed] content but keep the text
                        clean_text = re.sub(r'\[.*?\]', '', item_text).strip()
                        if not clean_text:
                            clean_text = item_text
                        categories[current_category]['items'].append({
                            'text': clean_text,
                            'original': item_text
                        })
        
        return categories


class QuestionParser:
    """Service for parsing due diligence questions"""
    
    @staticmethod
    def parse_questions(questions_text: str) -> List[Dict]:
        """
        Parse markdown questions into a list of questions with categories
        
        Args:
            questions_text: Raw questions text in markdown format
            
        Returns:
            List of parsed questions with categories
        """
        questions = []
        current_category = None
        
        for line in questions_text.split('\n'):
            # Category header (e.g., "### A. Organizational and Corporate Documents")
            if line.startswith('### '):
                match = re.match(r'### ([A-Z])\. (.+)', line)
                if match:
                    letter, name = match.groups()
                    current_category = f"{letter}. {name.strip()}"
            # Question lines (numbered items)
            elif current_category and line.strip():
                match = re.match(r'^\d+\.\s+(.+)', line.strip())
                if match:
                    question_text = match.group(1).strip()
                    if question_text:
                        questions.append({
                            'category': current_category,
                            'question': question_text,
                            'id': f"q_{len(questions)}"
                        })
        
        return questions


class ChecklistMatcher:
    """Service for matching checklists to documents"""
    
    def __init__(self, model: SentenceTransformer):
        """
        Initialize the matcher
        
        Args:
            model: SentenceTransformer model for embeddings
        """
        self.model = model
    
    def match_checklist_to_documents(
        self, 
        checklist: Dict, 
        chunks: List[Dict], 
        embeddings: np.ndarray, 
        threshold: float = 0.35
    ) -> Dict:
        """
        Match each checklist item to relevant documents
        
        Args:
            checklist: Parsed checklist
            chunks: Document chunks
            embeddings: Precomputed embeddings
            threshold: Similarity threshold
            
        Returns:
            Matching results
        """
        results = {}
        
        for cat_letter, category in checklist.items():
            cat_results = {
                'name': category['name'],
                'items': [],
                'total_items': len(category['items']),
                'matched_items': 0
            }
            
            for item_idx, item in enumerate(category['items']):
                # Encode checklist item with category context
                item_text = f"{category['name']} {item['text']}"
                item_embedding = self.model.encode(item_text)
                
                # Find matching chunks
                similarities = np.dot(embeddings, item_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(item_embedding)
                )
                
                # Get unique documents that match
                doc_matches = {}
                for idx, sim in enumerate(similarities):
                    if sim > threshold:
                        doc_path = chunks[idx]['path']
                        if doc_path not in doc_matches or sim > doc_matches[doc_path]['score']:
                            doc_matches[doc_path] = {
                                'name': chunks[idx]['source'],
                                'path': doc_path,
                                'full_path': chunks[idx].get('full_path', doc_path),
                                'score': float(sim),
                                'metadata': chunks[idx]['metadata']
                            }
                
                # Sort by score
                sorted_matches = sorted(doc_matches.values(), key=lambda x: x['score'], reverse=True)
                
                item_result = {
                    'text': item['text'],
                    'original': item['original'],
                    'matches': sorted_matches
                }
                
                if sorted_matches:
                    cat_results['matched_items'] += 1
                
                cat_results['items'].append(item_result)
            
            results[cat_letter] = cat_results
        
        return results
    
    def match_checklist_with_summaries(
        self,
        checklist: Dict, 
        doc_embeddings_data: Dict,
        threshold: float = 0.35
    ) -> Dict:
        """
        Match checklist items against document summaries using embeddings
        
        Args:
            checklist: Parsed checklist
            doc_embeddings_data: Document embeddings with summaries
            threshold: Similarity threshold
            
        Returns:
            Matching results using AI summaries
        """
        doc_embeddings = np.array(doc_embeddings_data['embeddings'])
        doc_info = doc_embeddings_data['documents']
        
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
                
                # Create embedding for checklist item with category context
                checklist_embedding_text = f"{cat_name}: {item_text}"
                item_embedding = self.model.encode(checklist_embedding_text)
                
                # Calculate similarities with all documents
                similarities = np.dot(doc_embeddings, item_embedding) / (
                    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(item_embedding)
                )
                
                # Find matching documents above threshold
                matches = []
                for idx, sim in enumerate(similarities):
                    if sim > threshold:
                        matches.append({
                            'name': doc_info[idx]['name'],
                            'path': doc_info[idx]['path'],
                            'summary': doc_info[idx]['summary'],
                            'score': float(sim),
                            'metadata': doc_info[idx].get('original_doc', {}).get('metadata', {})
                        })
                
                # Sort by score and keep top matches
                matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
                
                item_result = {
                    'text': item_text,
                    'original': item.get('original', item_text),
                    'matches': matches
                }
                
                if matches:
                    cat_results['matched_items'] += 1
                
                cat_results['items'].append(item_result)
            
            results[cat_letter] = cat_results
        
        return results


class QuestionAnswerer:
    """Service for answering questions using document chunks"""
    
    def __init__(self, model: SentenceTransformer):
        """
        Initialize the question answerer
        
        Args:
            model: SentenceTransformer model for embeddings
        """
        self.model = model
    
    def answer_questions_with_chunks(
        self, 
        questions: List[Dict], 
        chunks: List[Dict], 
        embeddings: np.ndarray,
        threshold: float = 0.4
    ) -> Dict:
        """
        Answer questions using document chunks with citations
        
        Args:
            questions: List of parsed questions
            chunks: Document chunks
            embeddings: Precomputed embeddings
            threshold: Similarity threshold
            
        Returns:
            Dictionary of answers with citations
        """
        answers = {}
        
        for question in questions:
            # Encode question
            question_embedding = self.model.encode(question['question'])
            
            # Find matching chunks
            similarities = np.dot(embeddings, question_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
            )
            
            # Get top matching chunks
            top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 chunks
            relevant_chunks = []
            
            for idx in top_indices:
                if similarities[idx] > threshold:
                    chunk_info = chunks[idx]
                    relevant_chunks.append({
                        'text': chunk_info['text'][:500],  # Limit text length
                        'source': chunk_info['source'],
                        'path': chunk_info['path'],
                        'score': float(similarities[idx]),
                        'metadata': chunk_info.get('metadata', {})
                    })
            
            answers[question['id']] = {
                'question': question['question'],
                'category': question['category'],
                'chunks': relevant_chunks,
                'has_answer': len(relevant_chunks) > 0
            }
        
        return answers


class ReportGenerator:
    """Service for generating reports and summaries"""
    
    def __init__(self, agent=None):
        """
        Initialize the report generator
        
        Args:
            agent: Optional AI agent for enhanced reporting
        """
        self.agent = agent
    
    def generate_company_summary(self, documents: Dict[str, Dict], data_room_name: str = "Unknown") -> str:
        """
        Generate company overview summary
        
        Args:
            documents: Dictionary of processed documents
            data_room_name: Name of the data room/company
            
        Returns:
            Company summary text
        """
        if not self.agent or not hasattr(self.agent, 'llm'):
            return self._generate_basic_summary(documents, data_room_name)
        
        # Gather key information from documents
        doc_summaries = []
        for path, doc_info in list(documents.items())[:10]:  # Use top 10 docs
            if 'summary' in doc_info:
                doc_summaries.append(f"{doc_info['name']}: {doc_info['summary']}")
            else:
                # Use first 500 chars of content if no summary
                content_preview = doc_info.get('content', '')[:500]
                if content_preview:
                    doc_summaries.append(f"{doc_info['name']}: {content_preview}")
        
        if not doc_summaries:
            return "No documents available for summary generation."
        
        # Create prompt for company summary
        from langchain_core.messages import HumanMessage
        prompt = f"""Based on the following document summaries from a due diligence data room, provide a comprehensive company overview.
        
        Company: {data_room_name}
        
        Document Summaries:
        {chr(10).join(doc_summaries[:10])}
        
        Please provide:
        1. Company name and industry
        2. Business model and key products/services
        3. Market position and competitive advantages
        4. Key financials (if available)
        5. Organizational structure
        6. Notable risks or concerns
        7. Overall assessment for M&A consideration
        
        Format the response in clear sections with bullet points where appropriate."""
        
        try:
            response = self.agent.llm.invoke([HumanMessage(content=prompt)])
            return escape_markdown_math(response.content.strip())
        except Exception as e:
            return f"Failed to generate AI summary: {str(e)}"
    
    def _generate_basic_summary(self, documents: Dict[str, Dict], data_room_name: str) -> str:
        """Generate basic summary without AI"""
        doc_count = len(documents)
        file_types = {}
        
        for doc_info in documents.values():
            doc_type = doc_info.get('metadata', {}).get('type', 'unknown')
            file_types[doc_type] = file_types.get(doc_type, 0) + 1
        
        summary = f"""# Company Overview: {data_room_name}

## Document Analysis
- **Total Documents**: {doc_count}
- **File Types**: {', '.join([f"{count} {type_name}" for type_name, count in file_types.items()])}

## Key Areas Covered
Based on the document structure, this data room appears to cover standard due diligence areas including corporate documents, financial records, and operational information.

*Note: Enable AI features for detailed company analysis and insights.*
"""
        return summary
    
    def generate_strategic_analysis(
        self, 
        strategy_text: str, 
        checklist_results: Dict, 
        documents: Dict[str, Dict]
    ) -> str:
        """
        Generate strategic analysis based on strategy and checklist results
        
        Args:
            strategy_text: Strategic document content
            checklist_results: Results from checklist matching
            documents: Document dictionary
            
        Returns:
            Strategic analysis text
        """
        if not self.agent or not hasattr(self.agent, 'llm'):
            return self._generate_basic_strategic_analysis(checklist_results)
        
        # Build context from checklist results
        checklist_context = []
        for cat_id, cat_data in checklist_results.items():
            cat_name = cat_data['name']
            matched_items = sum(1 for item in cat_data['items'] if item['matches'])
            total_items = len(cat_data['items'])
            coverage = (matched_items / total_items * 100) if total_items > 0 else 0
            
            checklist_context.append(f"- {cat_name}: {coverage:.0f}% coverage ({matched_items}/{total_items} items)")
            
            # Add details about specific gaps
            missing_items = [item['text'] for item in cat_data['items'] if not item['matches']]
            if missing_items and len(missing_items) <= 3:
                checklist_context.append(f"  Missing: {', '.join(missing_items[:3])}")
        
        # Build prompt
        prompt = f"""Based on the due diligence checklist results and the selected strategy, provide a strategic analysis.
        
        Strategy Document:
        {strategy_text}
        
        Checklist Coverage:
        {chr(10).join(checklist_context)}
        
        Please provide:
        1. Strategic alignment assessment
        2. Key risks and gaps identified
        3. Opportunities and synergies
        4. Recommended next steps
        5. Overall recommendation
        
        Format the response with clear sections and bullet points."""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.agent.llm.invoke([HumanMessage(content=prompt)])
            return escape_markdown_math(response.content.strip())
        except Exception as e:
            return f"Failed to generate strategic analysis: {str(e)}"
    
    def _generate_basic_strategic_analysis(self, checklist_results: Dict) -> str:
        """Generate basic strategic analysis without AI"""
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


class DDChecklistService:
    """
    Main service orchestrator for DD-Checklist operations
    Coordinates between different services and manages the overall workflow
    """
    
    def __init__(self, model: SentenceTransformer, agent=None):
        """
        Initialize the service
        
        Args:
            model: SentenceTransformer model
            agent: Optional AI agent
        """
        self.model = model
        self.agent = agent
        self.document_processor = DocumentProcessor(model)
        self.checklist_parser = ChecklistParser()
        self.question_parser = QuestionParser()
        self.checklist_matcher = ChecklistMatcher(model)
        self.question_answerer = QuestionAnswerer(model)
        self.report_generator = ReportGenerator(agent)
    
    def process_data_room(
        self, 
        data_room_path: str, 
        checklist_text: str = "", 
        questions_text: str = ""
    ) -> Dict[str, Any]:
        """
        Process entire data room with checklist and questions
        
        Args:
            data_room_path: Path to data room
            checklist_text: Optional checklist text
            questions_text: Optional questions text
            
        Returns:
            Dictionary with all processing results
        """
        results = {}
        
        # Load data room
        load_results = self.document_processor.load_data_room(data_room_path)
        results['load_results'] = load_results
        
        # Parse checklist if provided
        checklist = {}
        if checklist_text:
            checklist = self.checklist_parser.parse_checklist(checklist_text)
            results['checklist'] = checklist
        
        # Parse questions if provided
        questions = []
        if questions_text:
            questions = self.question_parser.parse_questions(questions_text)
            results['questions'] = questions
        
        # Match checklist to documents
        checklist_results = {}
        if checklist and self.document_processor.chunks:
            checklist_results = self.checklist_matcher.match_checklist_to_documents(
                checklist,
                self.document_processor.chunks,
                self.document_processor.embeddings
            )
            results['checklist_results'] = checklist_results
        
        # Answer questions
        question_answers = {}
        if questions and self.document_processor.chunks and self.document_processor.embeddings is not None:
            question_answers = self.question_answerer.answer_questions_with_chunks(
                questions,
                self.document_processor.chunks,
                self.document_processor.embeddings
            )
            results['question_answers'] = question_answers
        
        return results
    
    def search_documents(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search documents using the document processor
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Similarity threshold
            
        Returns:
            Search results
        """
        return self.document_processor.search(query, top_k, threshold)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return self.document_processor.get_statistics()
