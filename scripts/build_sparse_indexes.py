#!/usr/bin/env python3
"""
BM25 Sparse Index Build Script

This script generates BM25 sparse indices for all VDR data sources.
These indexes complement the existing FAISS dense indices for hybrid retrieval.

Run this script locally before pushing to repo to ensure sparse indices are up-to-date.
The generated indexes will be committed to the repo and loaded on Streamlit Cloud.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Progress indicators
from tqdm import tqdm

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.core.config import get_config
from app.core.logging import setup_logging
from app.core.document_processor import DocumentProcessor
from app.core.sparse_index import BM25Index, build_sparse_index_for_store

# Set up logging
logger = setup_logging("build_sparse", log_level="INFO")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Build BM25 sparse indices for document search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_sparse_indexes.py                    # Build all sparse indexes
  python scripts/build_sparse_indexes.py --store summit    # Build specific store
  python scripts/build_sparse_indexes.py --clean           # Force rebuild
  python scripts/build_sparse_indexes.py --status          # Show build status
        """
    )

    parser.add_argument(
        '--store', '-s',
        help='Build index for specific store name only'
    )

    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Force clean rebuild of all indexes'
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
    print(f"{BLUE}🔍 BM25 Sparse Index Build Script{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    print("")


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


def build_sparse_index_for_vdr(vdr_path: Path, store_name: str, project_name: str,
                              clean: bool = False) -> Dict[str, Any]:
    """Build BM25 sparse index for a single VDR"""

    start_time = time.time()
    logger.info(f"Building sparse index for {store_name}")

    try:
        # Check if index already exists and we're not forcing clean rebuild
        index_path = f"data/search_indexes/{store_name}_bm25.pkl"
        if not clean and Path(index_path).exists():
            logger.info(f"✅ Sparse index already exists for {store_name} (use --clean to rebuild)")
            return {
                'success': True,
                'store_name': store_name,
                'skipped': True,
                'processing_time': 0
            }

        # Load documents using existing processor
        processor = DocumentProcessor(store_name=store_name)
        result = processor.load_data_room(str(vdr_path))

        if result['chunks_count'] == 0:
            logger.warning(f"No chunks found for {store_name}")
            return {
                'success': False,
                'store_name': store_name,
                'error': 'No chunks found'
            }

        # Prepare documents for BM25
        documents = []
        for i, chunk in enumerate(processor.chunks):
            documents.append({
                'id': f"{store_name}_chunk_{i}",
                'content': chunk['text']
            })

        # Build sparse index
        bm25_index = build_sparse_index_for_store(store_name, documents)

        processing_time = time.time() - start_time

        logger.info(f"✅ Built BM25 index for {store_name}: {len(documents)} chunks in {processing_time:.2f}s")

        return {
            'success': True,
            'store_name': store_name,
            'documents_count': len(documents),
            'chunks_count': result['chunks_count'],
            'processing_time': processing_time
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Failed to build sparse index for {store_name}: {e}")
        return {
            'success': False,
            'store_name': store_name,
            'error': str(e),
            'processing_time': processing_time
        }


def show_build_status():
    """Show current sparse index build status"""
    print(f"\n{BLUE}📊 Sparse Index Build Status{NC}")
    print(f"{BLUE}{'='*50}{NC}")

    config = get_config()
    faiss_dir = config.paths['faiss_dir']

    if not faiss_dir.exists():
        print(f"{YELLOW}No index directory found: {faiss_dir}{NC}")
        return

    # Count existing sparse indexes
    sparse_indexes = list(faiss_dir.glob("*_bm25.pkl"))
    faiss_indexes = list(faiss_dir.glob("*.faiss"))

    print(f"Sparse indexes: {len(sparse_indexes)}")
    print(f"FAISS indexes: {len(faiss_indexes)}")

    if sparse_indexes:
        print(f"\n{GREEN}Existing Sparse Indexes:{NC}")
        total_size = 0
        for index_file in sparse_indexes:
            size_mb = index_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            store_name = index_file.stem.replace('_bm25', '')
            print(f"  • {store_name}: {size_mb:.2f} MB")

        print(f"\n{GREEN}Total sparse index size: {total_size:.2f} MB{NC}")

    # Check for missing sparse indexes
    vdr_dirs = get_vdr_directories(config.paths['vdrs_dir'])
    vdr_store_names = {store_name for _, store_name, _ in vdr_dirs}

    existing_sparse = {idx.stem.replace('_bm25', '') for idx in sparse_indexes}
    missing = vdr_store_names - existing_sparse

    if missing:
        print(f"\n{YELLOW}Missing sparse indexes for:{NC}")
        for store_name in missing:
            print(f"  • {store_name}")

    print(f"\n{BLUE}{'='*50}{NC}")


def main():
    """Main build script execution"""
    args = parse_arguments()

    print_header()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        show_build_status()
        return

    config = get_config()

    # Get VDR directories
    vdr_dirs = get_vdr_directories(config.paths['vdrs_dir'])

    if not vdr_dirs:
        print(f"{YELLOW}No VDR directories found{NC}")
        return

    print(f"{BLUE}Found {len(vdr_dirs)} VDR data rooms{NC}")

    # Filter for specific store if requested
    if args.store:
        vdr_dirs = [(path, name, proj) for path, name, proj in vdr_dirs
                   if args.store.lower() in name.lower()]
        if not vdr_dirs:
            print(f"{RED}No VDR found matching '{args.store}'{NC}")
            return

    print(f"{GREEN}🚀 Building sparse indexes for {len(vdr_dirs)} data rooms...{NC}")
    print("")

    start_time = time.time()
    results = []

    with tqdm(total=len(vdr_dirs), desc="Building sparse indexes",
              unit="index", leave=False) as pbar:

        for vdr_path, store_name, project_name in vdr_dirs:
            pbar.set_description(f"Building {store_name}")

            result = build_sparse_index_for_vdr(vdr_path, store_name, project_name, args.clean)
            results.append(result)

            pbar.update(1)

    # Generate report
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    skipped = [r for r in results if r.get('skipped', False)]

    total_time = time.time() - start_time

    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}SPARSE INDEX BUILD SUMMARY{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")

    if successful:
        print(f"\n{GREEN}✅ SUCCESSFUL BUILDS:{NC}")
        total_docs = sum(r.get('documents_count', 0) for r in successful)
        print(f"Total documents indexed: {total_docs}")

        # Calculate total size
        faiss_dir = config.paths['faiss_dir']
        if faiss_dir.exists():
            sparse_files = list(faiss_dir.glob("*_bm25.pkl"))
            total_size = sum(f.stat().st_size for f in sparse_files)
            print(f"Total sparse index size: {total_size / (1024*1024):.2f} MB")

    if failed:
        print(f"\n{RED}❌ FAILED BUILDS:{NC}")
        for result in failed:
            print(f"  • {result['store_name']}: {result.get('error', 'Unknown error')}")

    if successful:
        print(f"\n{GREEN}🎉 Sparse indexes ready for commit!{NC}")
        print("")
        print("Next steps:")
        print("  git add data/search_indexes/*_bm25.pkl")
        print("  git commit -m 'Add BM25 sparse indexes for hybrid retrieval'")
        print("  git push")
        print("")
        print("The indexes will be automatically loaded on Streamlit Cloud.")

    print(f"\n{BLUE}{'='*60}{NC}")


if __name__ == "__main__":
    main()
