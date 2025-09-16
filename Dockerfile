# HuggingFace Spaces Optimized Dockerfile
FROM python:3.13-slim

# Copy UV from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy project configuration
COPY pyproject.toml ./

# Generate requirements.txt and install (avoid package building)
RUN uv export --no-dev --no-hashes > requirements.txt && \
    uv pip install -r requirements.txt --system

# Copy the entire application (Git LFS files will be pulled automatically)
COPY . .

# Ensure LFS files are pulled
RUN git lfs pull || echo "LFS pull failed, continuing..."

# Files are now in the correct structure

# HuggingFace Spaces environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV TOKENIZERS_PARALLELISM=true
ENV UV_NO_CACHE=1

# Disable Streamlit metrics to avoid permission issues
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configure HuggingFace cache to writable location (50GB ephemeral storage)
ENV HF_HOME=/tmp/huggingface
ENV HF_HUB_CACHE=/tmp/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence_transformers

# Spaces-specific optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 8501

# Run the main application entry point (in app subdirectory)
CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
