#!/bin/bash

# FAISS Index Build Script
# 
# This script builds FAISS indices for all data sources in the project.
# Run this before pushing to repo to ensure all indices are up-to-date.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🚀 FAISS Index Build Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ Error: uv is not installed or not in PATH${NC}"
    echo "Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if Python build script exists
BUILD_SCRIPT="build-faiss-indices.py"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo -e "${RED}❌ Error: $BUILD_SCRIPT not found${NC}"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo -e "${RED}❌ Error: data directory not found${NC}"
    exit 1
fi

# Show current data structure
echo -e "${YELLOW}📁 Data Structure:${NC}"
if [ -d "data/vdrs" ]; then
    VDR_COUNT=$(find data/vdrs -name "*.pdf" -o -name "*.docx" -o -name "*.doc" -o -name "*.txt" -o -name "*.md" | wc -l)
    PROJECT_COUNT=$(find data/vdrs -maxdepth 1 -type d | grep -v "^data/vdrs$" | wc -l)
    echo "  - VDRs: $PROJECT_COUNT projects, ~$VDR_COUNT documents"
fi

if [ -d "data/checklist" ]; then
    CHECKLIST_COUNT=$(find data/checklist -name "*.md" | wc -l)
    echo "  - Checklists: $CHECKLIST_COUNT files"
fi

if [ -d "data/questions" ]; then
    QUESTIONS_COUNT=$(find data/questions -name "*.md" | wc -l)
    echo "  - Questions: $QUESTIONS_COUNT files"
fi

echo ""

# Ask for confirmation
read -p "Continue with FAISS index build? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️ Build cancelled by user${NC}"
    exit 0
fi

echo -e "${GREEN}✅ Starting build process...${NC}"
echo ""

# Set Python path to ensure imports work
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Run the Python build script using uv
echo ""
echo -e "${BLUE}🔧 Running FAISS index generation with progress indicators...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if uv run python "$BUILD_SCRIPT"; then
    echo ""
    echo -e "${GREEN}🎉 FAISS index build completed successfully!${NC}"
    echo ""
    
    # Show generated files
    if [ -d "data/enhanced_faiss" ]; then
        echo -e "${BLUE}📊 Generated FAISS Files:${NC}"
        ls -la data/enhanced_faiss/ | grep -E '\.(faiss|pkl)$' | while read -r line; do
            echo "  $line"
        done
        echo ""
        
        # Calculate total size
        TOTAL_SIZE=$(du -sh data/enhanced_faiss/ 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "${GREEN}📈 Total FAISS data size: $TOTAL_SIZE${NC}"
        echo ""
    fi
    
    echo -e "${GREEN}✅ Ready to commit and push FAISS indices to repository!${NC}"
    echo ""
    echo "Next steps:"
    echo "  git add data/enhanced_faiss/"
    echo "  git commit -m 'Update FAISS indices'"
    echo "  git push"
    
    exit 0
else
    echo ""
    echo -e "${RED}❌ FAISS index build failed!${NC}"
    echo "Check the output above for error details."
    echo "Log files are available in .logs/ directory."
    exit 1
fi
