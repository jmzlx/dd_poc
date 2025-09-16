# Use Python 3.13 base image
FROM python:3.13-slim

# Copy uv from the official image (much more efficient than installing)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy pyproject.toml and uv.lock for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the entire application
COPY . .

# Download and cache models on build (optional - can be done at runtime too)
# RUN uv run python -c "from app.core.model_cache import get_cached_embeddings, get_cached_cross_encoder; get_cached_embeddings(); get_cached_cross_encoder()"

# Expose the port Streamlit runs on (HuggingFace Spaces standard)
EXPOSE 8501

# Set environment variables for better performance
ENV TOKENIZERS_PARALLELISM=true
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Disable uv cache for runtime to avoid permission issues
ENV UV_NO_CACHE=1

# Run the Streamlit app using uv (HuggingFace Spaces format)
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
