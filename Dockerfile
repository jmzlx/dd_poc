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

# HuggingFace Spaces environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV TOKENIZERS_PARALLELISM=true
ENV UV_NO_CACHE=1

# Spaces-specific optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Run the main application entry point
CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
