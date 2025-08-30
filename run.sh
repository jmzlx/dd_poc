#!/bin/bash
# Run the DD-Checklist Streamlit app with uv

echo "🚀 Starting DD-Checklist App..."
echo "📋 Checklists: $(ls data/checklist/*.md 2>/dev/null | wc -l) found"
echo "📂 Data rooms: $(find data/vdrs -maxdepth 1 -type d ! -name vdrs ! -name ".*" 2>/dev/null | wc -l) found"
echo ""

# Ensure we have the virtual environment and dependencies
echo "📦 Setting up environment with uv..."
uv sync

# Run with uv
echo "🎯 Launching Streamlit app..."
uv run streamlit run app.py --server.port 8501 --server.headless true

echo "✅ App stopped"
