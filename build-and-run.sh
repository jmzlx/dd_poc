#!/bin/bash

# DD-Checklist Docker Build and Run Script
# This script builds the Docker image and runs it locally for testing

set -e  # Exit on any error

echo "🐳 DD-Checklist Docker Build & Run"
echo "=================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker daemon is not running"
    echo "   Please start Docker Desktop or Docker daemon"
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t dd-checklist:latest .

# Stop any existing container
echo "🛑 Stopping any existing containers..."
docker stop dd-checklist-app 2>/dev/null || true
docker rm dd-checklist-app 2>/dev/null || true

# Run the container
echo "🚀 Starting DD-Checklist container..."
docker run -d \
    --name dd-checklist-app \
    -p 8501:8501 \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/logs:/app/logs" \
    -e TOKENIZERS_PARALLELISM=false \
    dd-checklist:latest

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Test health endpoint
if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 Open http://localhost:8501 in your browser"
    echo ""
    echo "Commands:"
    echo "  View logs: docker logs dd-checklist-app"
    echo "  Stop app:  docker stop dd-checklist-app"
    echo "  Remove:    docker rm dd-checklist-app"
else
    echo "❌ Health check failed. Checking logs..."
    docker logs dd-checklist-app --tail 20
    exit 1
fi
