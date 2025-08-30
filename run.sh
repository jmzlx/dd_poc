#!/bin/bash

# DD-Checklist Launch Script
# Simple script to start the Streamlit application with uv

echo "🚀 Starting DD-Checklist Application..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies if needed
echo "📦 Ensuring dependencies are installed..."
uv sync

echo ""
echo "🌐 Starting Streamlit server..."
echo "   App will open in your browser automatically"
echo "   Press Ctrl+C to stop the server"
echo ""

# Set environment variables to suppress warnings
export TOKENIZERS_PARALLELISM=false

# Start the application
uv run streamlit run app.py
