#!/usr/bin/env python3
"""
Build Script

This script runs the complete build process for the due diligence application:
1. Download required models (if needed)
2. Build FAISS indices for document search
3. Build knowledge graphs for relationship analysis
4. Verify all components are working

This is the single command to prepare your application for deployment.
Usage: uv run build
"""

import sys
import subprocess
import time
from pathlib import Path

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{'='*80}{NC}")
    print(f"{GREEN}{BOLD}🚀 {title}{NC}")
    print(f"{BLUE}{'='*80}{NC}")

def print_step(step_num: int, total_steps: int, description: str):
    """Print a step header"""
    print(f"\n{YELLOW}{BOLD}Step {step_num}/{total_steps}: {description}{NC}")
    print(f"{BLUE}{'-'*60}{NC}")

def run_command(command: list, description: str, required: bool = True) -> bool:
    """Run a command and return success status"""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"{GREEN}✅ {description} completed successfully{NC}")
        return True
        
    except subprocess.CalledProcessError as e:
        if required:
            print(f"{RED}❌ {description} failed with exit code {e.returncode}{NC}")
            print(f"{RED}This is a required step - build process cannot continue{NC}")
        else:
            print(f"{YELLOW}⚠️ {description} failed with exit code {e.returncode}{NC}")
            print(f"{YELLOW}This is optional - continuing with build process{NC}")
        return False
    except Exception as e:
        if required:
            print(f"{RED}❌ {description} failed: {str(e)}{NC}")
        else:
            print(f"{YELLOW}⚠️ {description} failed: {str(e)}{NC}")
        return False

def check_prerequisites() -> bool:
    """Check if prerequisites are met"""
    print_step(0, 4, "Checking Prerequisites")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(f"{RED}❌ Error: Please run this script from the project root directory{NC}")
        return False
    
    # Check if data directory exists
    if not Path("data/vdrs").exists():
        print(f"{RED}❌ Error: No data/vdrs directory found{NC}")
        print(f"{RED}Please ensure you have document data in data/vdrs/ before proceeding{NC}")
        return False
    
    # Check if there are any VDR directories
    vdr_dirs = list(Path("data/vdrs").glob("*/"))
    if not vdr_dirs:
        print(f"{RED}❌ Error: No VDR directories found in data/vdrs/{NC}")
        print(f"{RED}Please add your document data before proceeding{NC}")
        return False
    
    print(f"{GREEN}✅ Found {len(vdr_dirs)} VDR directories to process{NC}")
    for vdr_dir in vdr_dirs:
        print(f"   📁 {vdr_dir.name}")
    
    return True

def download_models() -> bool:
    """Download required models"""
    print_step(1, 4, "Downloading Required Models")
    
    # Check if models already exist
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.iterdir()):
        print(f"{GREEN}✅ Models directory already exists - skipping download{NC}")
        return True
    
    return run_command(
        ["uv", "run", "download-models"],
        "Model download",
        required=False  # Models might already be cached
    )

def build_faiss_indices() -> bool:
    """Build FAISS indices for document search"""
    print_step(2, 4, "Building FAISS Indices")
    
    return run_command(
        ["uv", "run", "build-indexes"],
        "FAISS index building",
        required=True
    )

def build_knowledge_graphs() -> bool:
    """Build knowledge graphs for relationship analysis"""
    print_step(3, 4, "Building Knowledge Graphs")
    
    return run_command(
        ["uv", "run", "build-graphs"],
        "Knowledge graph building",
        required=False  # Knowledge graphs are optional enhancement
    )

def verify_build() -> bool:
    """Verify that the build was successful"""
    print_step(4, 4, "Verifying Build Results")
    
    success = True
    
    # Check FAISS indices
    faiss_dir = Path("data/search_indexes")
    if faiss_dir.exists():
        faiss_files = list(faiss_dir.glob("*.faiss"))
        if faiss_files:
            print(f"{GREEN}✅ Found {len(faiss_files)} FAISS indices{NC}")
            for faiss_file in faiss_files:
                print(f"   📊 {faiss_file.stem}")
        else:
            print(f"{RED}❌ No FAISS indices found{NC}")
            success = False
    else:
        print(f"{RED}❌ FAISS directory not found{NC}")
        success = False
    
    # Check knowledge graphs
    graphs_dir = faiss_dir / "knowledge_graphs" if faiss_dir.exists() else None
    if graphs_dir and graphs_dir.exists():
        graph_files = list(graphs_dir.glob("*_knowledge_graph.pkl"))
        if graph_files:
            print(f"{GREEN}✅ Found {len(graph_files)} knowledge graphs{NC}")
            for graph_file in graph_files:
                company_name = graph_file.stem.replace('_knowledge_graph', '')
                print(f"   🧠 {company_name}")
        else:
            print(f"{YELLOW}⚠️ No knowledge graphs found (optional feature){NC}")
    else:
        print(f"{YELLOW}⚠️ Knowledge graphs directory not found (optional feature){NC}")
    
    # Check models
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.iterdir()):
        print(f"{GREEN}✅ Models directory exists{NC}")
    else:
        print(f"{YELLOW}⚠️ Models directory not found (may use cached models){NC}")
    
    return success

def print_usage_instructions():
    """Print instructions for using the built application"""
    print_header("🎉 Build Complete - Usage Instructions")
    
    print(f"""
{GREEN}{BOLD}Your Due Diligence Application is Ready!{NC}

{YELLOW}{BOLD}🚀 To start the application:{NC}
   {BLUE}uv run start{NC}
   or
   {BLUE}uv run streamlit run app/main.py{NC}

{YELLOW}{BOLD}📁 What was built:{NC}
   • FAISS indices for fast document search
   • Knowledge graphs for relationship analysis  
   • Pre-computed embeddings for semantic search
   • All assets stored in your repository

{YELLOW}{BOLD}🌐 For Streamlit Cloud deployment:{NC}
   • All build artifacts are now in your repo
   • Commit and push to deploy
   • No external databases required
   • Everything loads from local files

{YELLOW}{BOLD}🔄 To rebuild (after adding new documents):{NC}
   {BLUE}uv run build{NC}

{YELLOW}{BOLD}🧠 New Features Available:{NC}
   • Knowledge Graph tab with entity exploration
   • Semantic search using AI embeddings
   • Relationship discovery and path finding
   • Graph analysis and visualization

{GREEN}Happy analyzing! 🎯{NC}
    """)

def main():
    """Main function to run the complete build process"""
    start_time = time.time()
    
    print_header("Build Process for Due Diligence Application")
    print(f"Building FAISS indices, knowledge graphs, and all required assets...")
    
    # Step 0: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Download models (optional)
    models_success = download_models()
    
    # Step 2: Build FAISS indices (required)
    faiss_success = build_faiss_indices()
    if not faiss_success:
        print(f"\n{RED}❌ Build process failed at FAISS index building{NC}")
        print(f"{RED}Cannot proceed without FAISS indices{NC}")
        sys.exit(1)
    
    # Step 3: Build knowledge graphs (optional)
    graphs_success = build_knowledge_graphs()
    
    # Step 4: Verify build
    verify_success = verify_build()
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    # Summary
    print_header("Build Process Summary")
    
    print(f"⏱️  Total build time: {minutes}m {seconds}s")
    print(f"📊 FAISS indices: {'✅ Success' if faiss_success else '❌ Failed'}")
    print(f"🧠 Knowledge graphs: {'✅ Success' if graphs_success else '⚠️ Failed (optional)'}")
    print(f"📥 Models: {'✅ Success' if models_success else '⚠️ Failed (optional)'}")
    print(f"🔍 Verification: {'✅ Success' if verify_success else '❌ Failed'}")
    
    if faiss_success and verify_success:
        print(f"\n{GREEN}{BOLD}🎉 Build process completed successfully!{NC}")
        print_usage_instructions()
        sys.exit(0)
    else:
        print(f"\n{YELLOW}{BOLD}⚠️ Build completed with some issues{NC}")
        if faiss_success:
            print(f"The core application will work, but some features may be limited.")
            print_usage_instructions()
        sys.exit(1)

if __name__ == "__main__":
    main()
