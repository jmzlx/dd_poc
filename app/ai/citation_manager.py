#!/usr/bin/env python3
"""
Citation Management and Report Formatting

Handles citation tracking, report formatting with citations, and 
document download link generation for ReAct agents.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import streamlit as st
from app.core.logging import logger
from app.core.document_processor import escape_markdown_math


class CitationManager:
    """Manages citations and formats reports with proper source tracking"""
    
    def __init__(self):
        self.citations = {}  # doc_id -> citation info
        self.citation_counter = 1
    
    def add_citation(self, doc_info: Dict[str, Any]) -> str:
        """Add a citation and return its reference ID"""
        doc_id = doc_info.get('doc_id', f"doc_{self.citation_counter}")
        
        if doc_id not in self.citations:
            self.citations[doc_id] = {
                'id': self.citation_counter,
                'name': doc_info.get('name', 'Unknown Document'),
                'path': doc_info.get('path', ''),
                'excerpts': [doc_info.get('excerpt', '')],
                'relevance_scores': [doc_info.get('relevance', 0.0)]
            }
            self.citation_counter += 1
        else:
            # Add additional excerpt if not already present
            excerpt = doc_info.get('excerpt', '')
            if excerpt and excerpt not in self.citations[doc_id]['excerpts']:
                self.citations[doc_id]['excerpts'].append(excerpt)
                self.citations[doc_id]['relevance_scores'].append(doc_info.get('relevance', 0.0))
        
        return f"[{self.citations[doc_id]['id']}]"
    
    def format_report_with_citations(self, report_text: str, tool_citations: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Format report text with inline download links instead of numbered citations"""
        
        # Process citations from tools
        for tool_name, citations_list in tool_citations.items():
            for citation in citations_list:
                self.add_citation(citation)
        
        # Apply inline citation replacements
        formatted_text = report_text
        
        # Build a mapping of document names to inline download links
        doc_replacements = {}
        for doc_id, citation_info in self.citations.items():
            doc_name = citation_info['name']
            doc_path = citation_info['path']
            
            # Clean document name for display (remove extensions)
            clean_doc_name = doc_name.replace('.pdf', '').replace('.docx', '').replace('.doc', '')
            
            # Create inline download link using Streamlit's download_button syntax
            # We'll use a simple format that can be processed by Streamlit
            inline_link = self._create_inline_download_link(clean_doc_name, doc_path, doc_id)
            
            # Map ALL possible variations to the same inline link for more flexible matching
            doc_replacements[doc_name] = inline_link
            doc_replacements[clean_doc_name] = inline_link
            
            # Also map common variations
            # Remove common prefixes/suffixes from file names
            base_name = doc_name
            if '.' in base_name:
                base_name = base_name.split('.')[0]  # Everything before first dot
            if base_name != doc_name and base_name != clean_doc_name:
                doc_replacements[base_name] = inline_link
            
            # Also handle path-based names (just the filename part)
            from pathlib import Path
            if doc_path:
                path_filename = Path(doc_path).name
                path_clean = path_filename.replace('.pdf', '').replace('.docx', '').replace('.doc', '')
                if path_clean not in doc_replacements:
                    doc_replacements[path_clean] = inline_link
        
        # Sort by longest document name first to avoid partial matches
        sorted_docs = sorted(doc_replacements.keys(), key=len, reverse=True)
        
        replacements_made = 0
        for doc_name in sorted_docs:
            inline_link = doc_replacements[doc_name]
            
            # Simple string replacement for {Document Name} format
            citation_marker = f"{{{doc_name}}}"
            
            # Count and replace
            before_count = formatted_text.count(citation_marker)
            formatted_text = formatted_text.replace(citation_marker, inline_link)
            actual_replacements = before_count - formatted_text.count(citation_marker)
            replacements_made += actual_replacements
        
        
        # For compatibility, still return citation list (but it won't be used for bottom section)
        citation_list = []
        for doc_id, citation_info in self.citations.items():
            citation_entry = {
                'id': citation_info['id'],
                'name': citation_info['name'],
                'path': citation_info['path'],
                'excerpts': citation_info['excerpts'][:2],
                'max_relevance': max(citation_info['relevance_scores']) if citation_info['relevance_scores'] else 0.0
            }
            citation_list.append(citation_entry)
        
        citation_list.sort(key=lambda x: x['id'])
        
        return formatted_text, citation_list
    
    def _create_inline_download_link(self, clean_name: str, doc_path: str, doc_id: str) -> str:
        """Create an inline citation that reads naturally"""
        # Create a natural reading citation that will be processed for download functionality
        # Format: "according to {Document Name}"
        return f"according to **{clean_name}**"
    
    def clear_citations(self):
        """Clear all citations for new report"""
        self.citations = {}
        self.citation_counter = 1


class ReportRenderer:
    """Utility class for document path resolution"""
    
    @staticmethod
    def _resolve_document_path(doc_path: str) -> Optional[Path]:
        """Resolve document path to actual file location"""
        try:
            # Handle various path formats
            path = Path(doc_path)
            
            # If it's already absolute and exists, use it
            if path.is_absolute() and path.exists():
                logger.info(f"Found absolute path: {path}")
                return path
            
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            
            # Try various combinations
            possible_paths = [
                project_root / doc_path,
                project_root / "data" / "vdrs" / doc_path,
                project_root / doc_path.lstrip('/'),
            ]
            
            # If path contains vdrs, try different structures
            if 'vdrs' in doc_path.lower():
                # Extract the part after vdrs
                parts = doc_path.split('/')
                if 'vdrs' in [p.lower() for p in parts]:
                    vdr_index = [p.lower() for p in parts].index('vdrs')
                    relative_path = Path(*parts[vdr_index:])
                    possible_paths.append(project_root / "data" / relative_path)
            
            # Find first existing path
            for candidate_path in possible_paths:
                logger.debug(f"Checking path: {candidate_path}")
                if candidate_path.exists():
                    logger.info(f"Found document at: {candidate_path}")
                    return candidate_path
            
            # If no exact match, try finding file by name in data/vdrs
            if doc_path:
                filename = Path(doc_path).name
                vdrs_path = project_root / "data" / "vdrs"
                logger.info(f"Searching for filename '{filename}' in {vdrs_path}")
                if vdrs_path.exists():
                    # Search recursively
                    for pdf_file in vdrs_path.rglob(filename):
                        if pdf_file.is_file():
                            logger.info(f"Found file by name search: {pdf_file}")
                            return pdf_file
            
            logger.warning(f"Could not resolve document path: {doc_path}")
            logger.warning(f"Tried paths: {[str(p) for p in possible_paths]}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving document path {doc_path}: {e}")
            return None


def extract_tool_citations(tools: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract citations from all tools after they've been used"""
    all_citations = {}
    
    for tool in tools:
        tool_name = tool.name
        citations = []
        
        if hasattr(tool, 'get_last_citations'):
            citations.extend(tool.get_last_citations())
        
        if hasattr(tool, 'get_last_validation_citations'):
            citations.extend(tool.get_last_validation_citations())
        
        if hasattr(tool, 'get_last_financial_citations'):
            citations.extend(tool.get_last_financial_citations())
        
        if hasattr(tool, 'get_last_competitive_citations'):
            citations.extend(tool.get_last_competitive_citations())
        
        if citations:
            all_citations[tool_name] = citations
    return all_citations


def create_comprehensive_report(agent_output: str, tools: List[Any], report_type: str) -> Tuple[str, Dict[str, Any]]:
    """Create a comprehensive report with inline citations and download info"""
    
    # Extract citations from tools
    tool_citations = extract_tool_citations(tools)
    
    # Calculate total citations
    total_citations = sum(len(citations) for citations in tool_citations.values())
    
    # Create citation manager
    citation_manager = CitationManager()
    
    # Format report with inline citations
    formatted_report, citation_list = citation_manager.format_report_with_citations(
        agent_output, tool_citations
    )
    
    # Add report header
    report_title = "Target Company Analysis" if report_type == "overview" else "Strategic Assessment"
    
    final_report = f"""# 🏢 {report_title}

{formatted_report}
"""
    
    # Return report and citation info for download functionality
    citation_info = {
        'has_citations': total_citations > 0,
        'citations': citation_list,
        'total_count': total_citations
    }
    
    return final_report, citation_info
