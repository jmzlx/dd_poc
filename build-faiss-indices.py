#!/usr/bin/env python3
"""
FAISS Index Build Script

This script generates FAISS indices for all data sources in the project:
- All VDR (Virtual Data Room) directories
- Checklist markdown files
- Questions markdown files
- Strategy documents

Run this script before pushing to repo to ensure all FAISS indices are up-to-date.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Progress indicators
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import get_config, setup_logging
from src.document_processing import DocumentProcessor
from src.services import parse_checklist, parse_questions, create_vector_store

# Set up logging
logger = setup_logging("build_faiss", log_level="INFO")

def clean_existing_faiss_files(faiss_dir: Path) -> None:
    """Remove existing FAISS files to ensure clean rebuild"""
    logger.info(f"Cleaning existing FAISS files in {faiss_dir}")
    
    if faiss_dir.exists():
        faiss_files = list(faiss_dir.glob("*.faiss")) + list(faiss_dir.glob("*.pkl"))
        for file_path in faiss_files:
            try:
                file_path.unlink()
                logger.info(f"Removed: {file_path.name}")
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
    else:
        faiss_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created FAISS directory: {faiss_dir}")

def get_vdr_directories(vdrs_path: Path) -> List[tuple]:
    """Get all VDR directories and their company names"""
    vdr_dirs = []
    
    if not vdrs_path.exists():
        logger.warning(f"VDRs directory not found: {vdrs_path}")
        return vdr_dirs
    
    for project_dir in vdrs_path.iterdir():
        if project_dir.is_dir():
            # Look for company directories within each project
            for item in project_dir.iterdir():
                if item.is_dir() and item.name != '__pycache__':
                    # Check if it's a company directory (has documents)
                    has_docs = any(
                        f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md']
                        for f in item.rglob('*') if f.is_file()
                    )
                    if has_docs:
                        # Use company directory name as store name
                        store_name = item.name.replace(' ', '-').lower()
                        vdr_dirs.append((item, store_name, project_dir.name))
                        logger.info(f"Found VDR: {item} -> {store_name}")
    
    return vdr_dirs

def process_vdr_data_room(vdr_path: Path, store_name: str, project_name: str, pbar: tqdm = None) -> Dict[str, Any]:
    """Process a single VDR data room and create FAISS index"""
    if pbar:
        pbar.set_description(f"Processing {store_name}...")
    logger.info(f"Processing VDR: {project_name}/{vdr_path.name} -> {store_name}")
    
    start_time = time.time()
    
    try:
        # Initialize document processor with specific store name
        processor = DocumentProcessor(store_name=store_name)
        
        # Count total files for progress tracking
        total_files = sum(1 for f in vdr_path.rglob('*') 
                         if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])
        
        # Process the data room with file-level progress
        with tqdm(total=total_files, desc=f"Files in {store_name}", 
                  unit="files", leave=False, disable=pbar is None) as file_pbar:
            
            # Custom progress callback
            def update_file_progress():
                if file_pbar:
                    file_pbar.update(1)
            
            # Process the data room
            result = processor.load_data_room(str(vdr_path))
            
            # Update progress bar based on actual files processed
            if file_pbar and result.get('documents_count', 0) > 0:
                file_pbar.update(result['documents_count'])
        
        processing_time = time.time() - start_time
        if pbar:
            pbar.set_description(f"✅ Completed {store_name}")
        logger.info(
            f"✅ Completed {store_name}: {result['documents_count']} docs, "
            f"{result['chunks_count']} chunks, {processing_time:.1f}s"
        )
        
        return {
            'success': True,
            'store_name': store_name,
            'project_name': project_name,
            'vdr_path': str(vdr_path),
            'processing_time': processing_time,
            **result
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to process {store_name}: {e}")
        return {
            'success': False,
            'store_name': store_name,
            'project_name': project_name,
            'vdr_path': str(vdr_path),
            'error': str(e)
        }

def process_markdown_content(content_dir: Path, content_type: str, pbar: tqdm = None) -> Dict[str, Any]:
    """Process markdown content (checklists or questions) into FAISS index"""
    if pbar:
        pbar.set_description(f"Processing {content_type}...")
    logger.info(f"Processing {content_type} from {content_dir}")
    
    if not content_dir.exists():
        logger.warning(f"{content_type} directory not found: {content_dir}")
        return {'success': False, 'error': f"Directory not found: {content_dir}"}
    
    start_time = time.time()
    config = get_config()
    
    try:
        # Find all markdown files
        md_files = list(content_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"No markdown files found in {content_dir}")
            return {'success': False, 'error': 'No markdown files found'}
        
        # Process all markdown files with progress
        all_documents = []
        
        with tqdm(md_files, desc=f"Processing {content_type} files", 
                  unit="file", leave=False, disable=pbar is None) as file_pbar:
        
            for md_file in file_pbar:
                file_pbar.set_description(f"Processing {md_file.name}")
                logger.info(f"Processing: {md_file.name}")
                content = md_file.read_text(encoding='utf-8')
                
                if content_type == "checklist":
                    parsed_data = parse_checklist(content)
                    # Convert checklist items to documents
                    for cat_key, category in parsed_data.items():
                        for item in category.get('items', []):
                            from langchain_core.documents import Document
                            doc = Document(
                                page_content=f"{category['name']}: {item['text']}",
                                metadata={
                                    'source': md_file.name,
                                    'category': category['name'],
                                    'type': 'checklist_item'
                                }
                            )
                            all_documents.append(doc)
                            
                elif content_type == "questions":
                    parsed_data = parse_questions(content)
                    # Convert questions to documents
                    for question in parsed_data:
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=f"{question['category']}: {question['question']}",
                            metadata={
                                'source': md_file.name,
                                'category': question['category'],
                                'question_id': question['id'],
                                'type': 'question'
                            }
                        )
                        all_documents.append(doc)
        
        if not all_documents:
            logger.warning(f"No content extracted from {content_type} files")
            return {'success': False, 'error': 'No content extracted'}
        
        # Create FAISS index
        vector_store = create_vector_store(all_documents, config.model.sentence_transformer_model)
        
        # Save the vector store
        faiss_dir = config.paths.faiss_path
        faiss_dir.mkdir(parents=True, exist_ok=True)
        
        store_name = f"{content_type}-data"
        vector_store.save_local(str(faiss_dir), index_name=store_name)
        
        processing_time = time.time() - start_time
        if pbar:
            pbar.set_description(f"✅ Completed {content_type}")
        logger.info(
            f"✅ Completed {content_type}: {len(all_documents)} items, "
            f"{processing_time:.1f}s"
        )
        
        return {
            'success': True,
            'content_type': content_type,
            'store_name': store_name,
            'documents_count': len(all_documents),
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to process {content_type}: {e}")
        return {
            'success': False,
            'content_type': content_type,
            'error': str(e)
        }

def generate_build_report(results: List[Dict[str, Any]], total_time: float) -> None:
    """Generate and display build summary report"""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    logger.info("\n" + "="*60)
    logger.info("FAISS BUILD SUMMARY REPORT")
    logger.info("="*60)
    logger.info(f"Total processing time: {total_time:.1f}s")
    logger.info(f"Successful builds: {len(successful)}")
    logger.info(f"Failed builds: {len(failed)}")
    
    if successful:
        logger.info("\n✅ SUCCESSFUL BUILDS:")
        total_docs = 0
        total_chunks = 0
        
        for result in successful:
            if 'documents_count' in result:
                total_docs += result['documents_count']
            if 'chunks_count' in result:
                total_chunks += result['chunks_count']
                
            if result.get('content_type'):  # Markdown content
                logger.info(f"  - {result['content_type']}: {result['documents_count']} items")
            else:  # VDR
                logger.info(
                    f"  - {result['store_name']}: {result.get('documents_count', 0)} docs, "
                    f"{result.get('chunks_count', 0)} chunks"
                )
        
        logger.info(f"\nTotal documents processed: {total_docs}")
        if total_chunks > 0:
            logger.info(f"Total chunks created: {total_chunks}")
    
    if failed:
        logger.info("\n❌ FAILED BUILDS:")
        for result in failed:
            name = result.get('store_name') or result.get('content_type', 'unknown')
            error = result.get('error', 'Unknown error')
            logger.info(f"  - {name}: {error}")
    
    logger.info("\n" + "="*60)

def main():
    """Main build script execution"""
    logger.info("🚀 Starting FAISS indices build process...")
    
    start_time = time.time()
    config = get_config()
    results = []
    
    try:
        # Step 1: Clean existing FAISS files
        print("🧹 Cleaning existing FAISS files...")
        clean_existing_faiss_files(config.paths.faiss_path)
        
        # Calculate total tasks for overall progress
        vdr_dirs = get_vdr_directories(config.paths.vdrs_path)
        total_tasks = len(vdr_dirs) + 2  # VDRs + checklist + questions
        
        print(f"\n📊 Processing {total_tasks} data sources...")
        
        # Overall progress bar
        with tqdm(total=total_tasks, desc="Overall Progress", unit="source", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:
            
            # Step 2: Process all VDR data rooms
            logger.info("\n📁 Processing VDR Data Rooms...")
            if not vdr_dirs:
                logger.warning("No VDR directories found to process")
            else:
                for vdr_path, store_name, project_name in vdr_dirs:
                    result = process_vdr_data_room(vdr_path, store_name, project_name, main_pbar)
                    results.append(result)
                    main_pbar.update(1)
            
            # Step 3: Process checklist data
            logger.info("\n📋 Processing Checklist Data...")
            checklist_result = process_markdown_content(
                config.paths.checklist_path, 
                "checklist",
                main_pbar
            )
            results.append(checklist_result)
            main_pbar.update(1)
            
            # Step 4: Process questions data
            logger.info("\n❓ Processing Questions Data...")
            questions_result = process_markdown_content(
                config.paths.questions_path, 
                "questions",
                main_pbar
            )
            results.append(questions_result)
            main_pbar.update(1)
            
            main_pbar.set_description("✅ Build Complete")
        
        # Step 5: Generate report
        total_time = time.time() - start_time
        generate_build_report(results, total_time)
        
        # Return appropriate exit code
        failed_count = len([r for r in results if not r.get('success', False)])
        if failed_count > 0:
            logger.error(f"Build completed with {failed_count} failures")
            sys.exit(1)
        else:
            logger.info("🎉 Build completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Build failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
