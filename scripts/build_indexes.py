#!/usr/bin/env python3
"""
FAISS Index Build Script - Enhanced Version

This script generates FAISS indices for all data sources in the project:
- All VDR (Virtual Data Room) directories
- Checklist markdown files
- Questions markdown files
- Strategy documents

Run this script before pushing to repo to ensure all FAISS indices are up-to-date.
"""

import sys
import time
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Progress indicators
from tqdm import tqdm

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.core.model_cache import get_cached_embeddings

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.core.config import get_config
from app.core.constants import TEMPERATURE
from app.core.logging import setup_logging
from app.core.content_ingestion import UnifiedContentProcessor
from app.core.stage_manager import StageManager

# Set up logging
logger = setup_logging("build_faiss", log_level="INFO")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Build FAISS indices for document search with stage-based incremental builds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_indexes.py                    # Incremental build
  python scripts/build_indexes.py --clean           # Force clean rebuild
  python scripts/build_indexes.py --stages embed    # Only run embedding stage
  python scripts/build_indexes.py --status          # Show build status
  python scripts/build_indexes.py --verbose         # Verbose output
        """
    )

    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Force clean rebuild of all stages (ignore cache)'
    )

    parser.add_argument(
        '--stages', '-s',
        nargs='+',
        choices=['scan', 'extract', 'classify', 'chunk', 'embed', 'index', 'sparse'],
        help='Run only specific stages (default: all stages)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current build status and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()

def print_header():
    """Print the script header with colors"""
    print(f"{BLUE}🚀 Fast Document Type Classification FAISS Build Script{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    print("")

def check_uv_installation():
    """Check if uv is available"""
    import subprocess
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{RED}❌ uv is not available or not working{NC}")
            print("Please install uv: https://github.com/astral-sh/uv")
            sys.exit(1)
    except FileNotFoundError:
        print(f"{RED}❌ uv is not installed{NC}")
        print("Please install uv: https://github.com/astral-sh/uv")
        sys.exit(1)

def show_data_structure():
    """Show current data structure"""
    print(f"{YELLOW}📁 Data Structure:{NC}")
    config = get_config()

    if config.paths['vdrs_dir'].exists():
        vdr_count = len(list(config.paths['vdrs_dir'].rglob("*.pdf"))) + \
                   len(list(config.paths['vdrs_dir'].rglob("*.docx"))) + \
                   len(list(config.paths['vdrs_dir'].rglob("*.doc"))) + \
                   len(list(config.paths['vdrs_dir'].rglob("*.txt"))) + \
                   len(list(config.paths['vdrs_dir'].rglob("*.md")))
        project_count = len([d for d in config.paths['vdrs_dir'].iterdir() if d.is_dir()])
        print(f"  - VDRs: {project_count} projects, ~{vdr_count} documents")

    if config.paths['checklist_dir'].exists():
        checklist_count = len(list(config.paths['checklist_dir'].glob("*.md")))
        print(f"  - Checklists: {checklist_count} files")

    if config.paths['questions_dir'].exists():
        questions_count = len(list(config.paths['questions_dir'].glob("*.md")))
        print(f"  - Questions: {questions_count} files")

    print("")


def initialize_ai_agent():
    """Initialize AI agent using API key from .env file"""
    config = get_config()
    api_key = config.anthropic['api_key']

    if not api_key:
        print(f"{RED}❌ ANTHROPIC_API_KEY not found in .env file!{NC}")
        print("Please add ANTHROPIC_API_KEY=your_api_key_here to your .env file")
        sys.exit(1)

    try:
        from app.services.ai_service import create_ai_service
        ai_service = create_ai_service(api_key, config.model['claude_model'])
        if not ai_service.is_available:
            print(f"{RED}❌ Failed to initialize AI service{NC}")
            sys.exit(1)

        logger.info(f"✅ AI Service initialized with model: {config.model['claude_model']}")
        return ai_service
    except Exception as e:
        print(f"{RED}❌ AI service initialization failed: {e}{NC}")
        sys.exit(1)

def initialize_haiku_classifier():
    """Initialize fast Haiku model for document type classification"""
    config = get_config()
    api_key = config.anthropic['api_key']

    if not api_key:
        print(f"{RED}❌ ANTHROPIC_API_KEY not found in .env file!{NC}")
        sys.exit(1)

    try:
        from langchain_anthropic import ChatAnthropic
        haiku_llm = ChatAnthropic(
            api_key=api_key,
            model=config.model['claude_haiku_model'],
            temperature=TEMPERATURE,  # Deterministic for consistent classification
            max_tokens=config.model['classification_max_tokens']
        )

        logger.info(f"✅ Haiku classifier initialized: {config.model['claude_haiku_model']}")
        return haiku_llm
    except Exception as e:
        print(f"{RED}❌ Haiku classifier initialization failed: {e}{NC}")
        sys.exit(1)

def clean_existing_faiss_files(faiss_dir: Path) -> None:
    """Remove existing FAISS files to ensure clean rebuild"""
    logger.info(f"Cleaning existing FAISS files in {faiss_dir}")

    if faiss_dir.exists():
        # Remove FAISS index files for clean rebuild (with document type classification)
        file_patterns = [
            "*.faiss",      # FAISS index files
            "*.pkl",        # FAISS metadata files
            "*_summaries.json",  # Legacy AI summary files (cleanup)
            "*_document_types.json",  # Document type classification files
            "checklists.json",  # Direct checklist files
            "company_summaries.json",    # Legacy metadata
            "default_tracker.json",      # Processing tracker
            "*.backup"      # Backup files
        ]

        removed_count = 0
        for pattern in file_patterns:
            for file_path in faiss_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path.name}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

        logger.info(f"🧹 Clean rebuild: removed {removed_count} existing files")
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

def process_vdr_data_room(vdr_path: Path, store_name: str, project_name: str, processor: UnifiedContentProcessor, haiku_classifier=None, pbar: tqdm = None) -> Dict[str, Any]:
    """Process a single VDR data room using unified content processing pipeline"""
    logger.info(f"Processing VDR: {project_name}/{vdr_path.name} -> {store_name}")

    try:
        # Use unified processing pipeline with new simplified API
        result = processor.process_content_source(
            content_source=vdr_path,
            content_type='vdr',
            store_name=store_name,
            classifier=haiku_classifier,
            progress_bar=pbar
        )

        # Add VDR-specific metadata
        result.update({
            'project_name': project_name,
            'vdr_path': str(vdr_path)
        })

        return result

    except Exception as e:
        logger.error(f"❌ Failed to process {store_name}: {e}")
        return {
            'success': False,
            'store_name': store_name,
            'project_name': project_name,
            'vdr_path': str(vdr_path),
            'error': str(e)
        }

# Direct markdown parsing functions (no LLM required)
def parse_markdown_items(content: str) -> list:
    """Parse numbered items from markdown content."""
    import re
    items = []
    for line in content.split('\n'):
        line = line.strip()
        if re.match(r'^\d+\.\s+', line):
            item_text = re.sub(r'^\d+\.\s+', '', line).strip()
            if len(item_text) > 10 and not item_text.isupper():
                items.append(item_text)
    return items

class BuildStageManager(StageManager):
    """Stage manager specifically for FAISS index building"""

    def __init__(self, faiss_dir: Path, config, args):
        super().__init__(faiss_dir)
        self.config = config
        self.args = args
        self.processor = None
        self.ai_service = None
        self.haiku_classifier = None

    def initialize_components(self):
        """Initialize shared components"""
        if not self.processor:
            self.processor = UnifiedContentProcessor()

        if not self.ai_service:
            self.ai_service = initialize_ai_agent()

        if not self.haiku_classifier:
            self.haiku_classifier = initialize_haiku_classifier()

    def execute_stage(self, stage_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific build stage"""
        force_clean = kwargs.get('force_clean', False)

        if stage_name == 'scan':
            return self.execute_scan_stage()
        elif stage_name == 'extract':
            return self.execute_extract_stage(force_clean=force_clean)
        elif stage_name == 'classify':
            return self.execute_classify_stage()
        elif stage_name == 'chunk':
            return self.execute_chunk_stage()
        elif stage_name == 'embed':
            return self.execute_embed_stage()
        elif stage_name == 'index':
            return self.execute_index_stage()
        elif stage_name == 'sparse':
            return self.execute_sparse_stage(force_clean=force_clean)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def execute_scan_stage(self) -> Dict[str, Any]:
        """Scan and catalog all documents"""
        logger.info("📊 Scanning document structure...")

        # Count documents by type
        vdr_count = len(list(self.config.paths['vdrs_dir'].rglob("*.pdf"))) + \
                   len(list(self.config.paths['vdrs_dir'].rglob("*.docx"))) + \
                   len(list(self.config.paths['vdrs_dir'].rglob("*.doc"))) + \
                   len(list(self.config.paths['vdrs_dir'].rglob("*.txt"))) + \
                   len(list(self.config.paths['vdrs_dir'].rglob("*.md")))

        checklist_count = len(list(self.config.paths['checklist_dir'].glob("*.md")))
        questions_count = len(list(self.config.paths['questions_dir'].glob("*.md")))

        # Get VDR structure
        vdr_dirs = get_vdr_directories(self.config.paths['vdrs_dir'])
        project_count = len([d for d in self.config.paths['vdrs_dir'].iterdir() if d.is_dir()])

        result = {
            'total_documents': vdr_count + checklist_count + questions_count,
            'vdr_documents': vdr_count,
            'checklist_files': checklist_count,
            'questions_files': questions_count,
            'vdr_projects': project_count,
            'vdr_company_dirs': len(vdr_dirs)
        }

        # Save scan cache
        scan_cache = self.faiss_dir / '.scan_cache.json'
        scan_cache.write_text(json.dumps(result, indent=2))

        logger.info(f"📊 Found {result['total_documents']} total documents")
        return result

    def execute_extract_stage(self, force_clean: bool = False) -> Dict[str, Any]:
        """Extract text from all documents"""
        logger.info("📄 Extracting text from documents...")

        self.initialize_components()
        vdr_dirs = get_vdr_directories(self.config.paths['vdrs_dir'])

        results = []
        total_docs = 0
        total_chunks = 0

        # Process VDR documents
        with tqdm(total=len(vdr_dirs), desc="Processing VDRs", unit="vdr") as pbar:
            for vdr_path, store_name, project_name in vdr_dirs:
                result = process_vdr_data_room(vdr_path, store_name, project_name,
                                             self.processor, self.haiku_classifier, pbar)
                results.append(result)
                total_docs += result.get('documents_count', 0)
                total_chunks += result.get('chunks_count', 0)
                pbar.update(1)

        # Process checklists and questions directly (no LLM)
        logger.info("📄 Processing checklists and questions with direct parsing...")

        # Simple direct processing
        checklist_docs = []
        questions_docs = []

        # Process checklist files
        for md_file in self.config.paths['checklist_dir'].glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            items = parse_markdown_items(content)
            for item in items:
                checklist_docs.append(Document(
                    page_content=item,
                    metadata={'source': md_file.name, 'type': 'checklist'}
                ))

        # Process questions files
        for md_file in self.config.paths['questions_dir'].glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            items = parse_markdown_items(content)
            for item in items:
                questions_docs.append(Document(
                    page_content=item,
                    metadata={'source': md_file.name, 'type': 'questions'}
                ))

        # Create FAISS indices
        if checklist_docs:
            embeddings_model = get_cached_embeddings(self.config.model['sentence_transformer_model'])
            vector_store = FAISS.from_documents(checklist_docs, embeddings_model)
            vector_store.save_local(str(self.config.paths['faiss_dir']), index_name="checklist-simple")
            logger.info(f"✅ Created checklist index with {len(checklist_docs)} items")

        if questions_docs:
            embeddings_model = get_cached_embeddings(self.config.model['sentence_transformer_model'])
            vector_store = FAISS.from_documents(questions_docs, embeddings_model)
            vector_store.save_local(str(self.config.paths['faiss_dir']), index_name="questions-simple")
            logger.info(f"✅ Created questions index with {len(questions_docs)} items")

        checklist_result = {
            'success': True,
            'store_name': 'checklist-simple',
            'documents_count': len(checklist_docs),
            'content_type': 'checklist'
        }
        results.append(checklist_result)

        questions_result = {
            'success': True,
            'store_name': 'questions-simple',
            'documents_count': len(questions_docs),
            'content_type': 'questions'
        }
        results.append(questions_result)

        result = {
            'documents_processed': total_docs,
            'chunks_created': total_chunks,
            'vdr_results': len([r for r in results if r.get('store_name')]),
            'markdown_results': len([r for r in results if r.get('content_type')])
        }

        # Save extraction cache
        extraction_cache = self.faiss_dir / '.extraction_cache.json'
        extraction_cache.write_text(json.dumps(result, indent=2))

        return result

    def execute_classify_stage(self) -> Dict[str, Any]:
        """Classify document types using AI"""
        logger.info("🏷️ Document classification already handled in extract stage")
        # Classification happens during extraction for efficiency
        return {'status': 'classification_integrated'}

    def execute_chunk_stage(self) -> Dict[str, Any]:
        """Handle text chunking (integrated with extraction)"""
        logger.info("✂️ Text chunking already handled in extract stage")
        # Chunking happens during extraction for efficiency
        return {'status': 'chunking_integrated'}

    def execute_embed_stage(self) -> Dict[str, Any]:
        """Generate vector embeddings"""
        logger.info("🧮 Generating vector embeddings...")
        # This will be implemented as part of the extraction process
        return {'status': 'embeddings_generated'}

    def execute_index_stage(self) -> Dict[str, Any]:
        """Build FAISS indices"""
        logger.info("🔍 Building FAISS indices...")
        # This will be implemented as part of the extraction process
        return {'status': 'indices_built'}

    def execute_sparse_stage(self, force_clean: bool = False) -> Dict[str, Any]:
        """Build BM25 sparse indices"""
        logger.info("🔍 Building BM25 sparse indices...")

        from app.core.sparse_index import BM25Index
        from app.core.document_processor import DocumentProcessor

        vdr_dirs = get_vdr_directories(self.config.paths['vdrs_dir'])
        results = []
        total_docs = 0

        for vdr_path, store_name, project_name in vdr_dirs:
            logger.info(f"Building sparse index for {store_name}")

            # Check if index already exists
            index_path = self.config.paths['faiss_dir'] / f"{store_name}_bm25.pkl"
            if not force_clean and index_path.exists():
                logger.info(f"✅ Sparse index already exists for {store_name}")
                results.append({
                    'success': True,
                    'store_name': store_name,
                    'skipped': True
                })
                continue

            try:
                # Load documents using existing processor
                processor = DocumentProcessor(store_name=store_name)
                result = processor.load_data_room(str(vdr_path))

                if result['chunks_count'] == 0:
                    logger.warning(f"No chunks found for {store_name}")
                    results.append({
                        'success': False,
                        'store_name': store_name,
                        'error': 'No chunks found'
                    })
                    continue

                # Prepare documents for BM25
                documents = []
                for i, chunk in enumerate(processor.chunks):
                    documents.append({
                        'id': f"{store_name}_chunk_{i}",
                        'content': chunk['text']
                    })

                # Build sparse index
                bm25_index = BM25Index(str(index_path))
                bm25_index.build_index(documents)

                total_docs += len(documents)
                results.append({
                    'success': True,
                    'store_name': store_name,
                    'documents_count': len(documents),
                    'chunks_count': result['chunks_count']
                })

                logger.info(f"✅ Built BM25 index for {store_name}: {len(documents)} chunks")

            except Exception as e:
                logger.error(f"❌ Failed to build sparse index for {store_name}: {e}")
                results.append({
                    'success': False,
                    'store_name': store_name,
                    'error': str(e)
                })

        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        skipped = [r for r in results if r.get('skipped', False)]

        return {
            'success': len(failed) == 0,
            'total_stores': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'skipped': len(skipped),
            'total_documents': total_docs
        }

def generate_build_report(results: List[Dict[str, Any]], total_time: float) -> None:
    """Generate and display build summary report"""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}BUILD SUMMARY REPORT{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Successful builds: {len(successful)}")
    print(f"Failed builds: {len(failed)}")

    if successful:
        print(f"\n{GREEN}✅ SUCCESSFUL BUILDS:{NC}")
        total_docs = 0
        total_chunks = 0
        total_summaries = 0
        total_enhanced = 0

        total_classifications = 0

        for result in successful:
            if 'documents_count' in result:
                total_docs += result['documents_count']
            if 'chunks_count' in result:
                total_chunks += result['chunks_count']
            if 'summaries_count' in result:
                total_summaries += result['summaries_count']
            if 'classifications_count' in result:
                total_classifications += result['classifications_count']
            if 'enhanced_files' in result:
                total_enhanced += result['enhanced_files']

            if result.get('content_type'):  # Markdown content
                print(f"  - {result['content_type']}: {result['documents_count']} items, direct similarity")
            else:  # VDR
                classifications_info = f", {result.get('classifications_count', 0)} document types" if result.get('classifications_count', 0) > 0 else ""
                print(
                    f"  - {result['store_name']}: {result.get('documents_count', 0)} docs, "
                    f"{result.get('chunks_count', 0)} chunks{classifications_info}"
                )

        print(f"\nTotal documents processed: {total_docs}")
        if total_chunks > 0:
            print(f"Total chunks created: {total_chunks}")
        print(f"🔍 Direct similarity search enabled for all document chunks")
        if total_classifications > 0:
            print(f"🏷️ Total document types classified: {total_classifications}")
            print(f"⚡ Fast Haiku classification bridges semantic gap for legal documents!")

    if failed:
        print(f"\n{RED}❌ FAILED BUILDS:{NC}")
        for result in failed:
            name = result.get('store_name') or result.get('content_type', 'unknown')
            error = result.get('error', 'Unknown error')
            print(f"  - {name}: {error}")

    print(f"\n{BLUE}{'='*60}{NC}")

def show_build_status(stage_manager: BuildStageManager):
    """Show current build status"""
    print(f"\n{BLUE}📊 Build Status Report{NC}")
    print(f"{BLUE}{'='*50}{NC}")

    summary = stage_manager.tracker.get_build_summary()

    print(f"Last build: {summary['last_build'] or 'Never'}")
    print(f"Total builds: {summary['total_builds']}")

    print(f"\n{GREEN}✅ Completed Stages:{NC}")
    for stage in summary['completed_stages']:
        status = stage_manager.tracker.get_stage_status(stage)
        completed_at = status.get('completed_at', 'Unknown')
        print(f"  • {stage}: {completed_at}")

    if summary['incomplete_stages']:
        print(f"\n{YELLOW}⏳ Incomplete Stages:{NC}")
        for stage in summary['incomplete_stages']:
            print(f"  • {stage}")

    if summary['failed_stages']:
        print(f"\n{RED}❌ Failed Stages:{NC}")
        for stage in summary['failed_stages']:
            status = stage_manager.tracker.get_stage_status(stage)
            print(f"  • {stage}: {status.get('error', 'Unknown error')}")

    print(f"\n{BLUE}{'='*50}{NC}")

def main():
    """Main build script execution using stage-based system"""
    args = parse_arguments()

    print_header()
    check_uv_installation()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = get_config()
    stage_manager = BuildStageManager(config.paths['faiss_dir'], config, args)

    # Handle status request
    if args.status:
        show_build_status(stage_manager)
        return

    show_data_structure()

    print(f"{GREEN}🚀 Starting stage-based FAISS build process...{NC}")
    if args.clean:
        print(f"{YELLOW}🧹 Clean rebuild requested - all stages will be executed{NC}")
    else:
        print(f"{BLUE}⚡ Incremental build - completed stages will be skipped{NC}")

    print(f"{BLUE}🏷️ Using Haiku AI for document type classification{NC}")
    print("")

    start_time = time.time()

    try:
        # Handle clean rebuild - delete all build files upfront
        if args.clean:
            logger.info("🧹 Clean rebuild requested - removing all existing build files")
            faiss_dir = config.paths['faiss_dir']
            if faiss_dir.exists():
                # Remove all FAISS-related files
                patterns = [
                    "*.faiss",      # FAISS index files
                    "*.pkl",        # FAISS metadata files
                    "*_document_types.json",  # Document type classifications
                    ".extraction_cache.json",
                    ".scan_cache.json",
                    ".build_state.json"
                ]

                removed_count = 0
                for pattern in patterns:
                    for file_path in faiss_dir.glob(pattern):
                        try:
                            file_path.unlink()
                            logger.info(f"Removed: {file_path.name}")
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {e}")

                logger.info(f"🧹 Clean rebuild: removed {removed_count} existing files")

        # Run the build pipeline
        build_result = stage_manager.run_build_pipeline(
            target_stages=args.stages,
            force_clean=args.clean
        )

        # Generate report
        total_time = time.time() - start_time

        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}STAGE-BASED BUILD SUMMARY REPORT{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        print(f"Total execution time: {total_time:.1f}s")
        print(f"Stages executed: {build_result['stages_executed']}")
        print(f"Stages skipped: {build_result['stages_skipped']}")
        print(f"Stages failed: {build_result['stages_failed']}")

        if build_result['success']:
            print(f"\n{GREEN}✅ All stages completed successfully!{NC}")

            # Show generated files
            if config.paths['faiss_dir'].exists():
                print(f"\n{BLUE}📊 Generated FAISS Files:{NC}")
                import os
                faiss_files = list(config.paths['faiss_dir'].glob("*.faiss"))
                pkl_files = list(config.paths['faiss_dir'].glob("*.pkl"))

                for file_path in faiss_files:
                    print(f"  {file_path.name}")
                for file_path in pkl_files:
                    print(f"  {file_path.name}")

                # Calculate total size
                total_size = sum(f.stat().st_size for f in config.paths['faiss_dir'].rglob("*") if f.is_file())
                print(f"\n{GREEN}📈 Total FAISS data size: {total_size / (1024*1024):.1f} MB{NC}")

            print(f"\n{GREEN}🎉 Ready to commit and push FAISS indices!{NC}")
            print("")
            print("Next steps:")
            print("  git add data/search_indexes/")
            print("  git commit -m 'Update FAISS indices with incremental build'")
            print("  git push")
            sys.exit(0)
        else:
            print(f"\n{RED}❌ Build completed with failures{NC}")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️ Build interrupted by user{NC}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}💥 Build failed with unexpected error: {e}{NC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
