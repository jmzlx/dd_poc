#!/usr/bin/env python3
"""
Streamlit Cloud Configuration Module

Configures HuggingFace model caching for Streamlit Cloud deployment.
Ensures models are cached persistently and preloaded to avoid downloads on each session.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _is_streamlit_cloud():
    """
    Detect if we're running on Streamlit Cloud.

    Returns True if running on Streamlit Cloud, False for local development.
    """
    # Check for Streamlit Cloud environment indicators
    indicators = [
        # Check if /app directory exists and is writable
        (Path("/app").exists() and os.access("/app", os.W_OK)),
        # Check for Streamlit Cloud specific environment variables
        os.environ.get("STREAMLIT_SERVER_HEADLESS", "").lower() == "true",
        # Check for typical Streamlit Cloud paths
        os.environ.get("HOME", "").startswith("/app"),
    ]

    # Return True if any indicator suggests Streamlit Cloud
    is_cloud = any(indicators)

    if is_cloud:
        logger.info("🌐 Detected Streamlit Cloud environment")
    else:
        logger.info("🏠 Detected local development environment")

    return is_cloud

def configure_streamlit_cloud_cache():
    """
    Configure HuggingFace caching for Streamlit Cloud deployment.

    This ensures models are cached in a persistent location and preloaded
    to avoid downloading on every session restart.
    """
    # Detect if we're running on Streamlit Cloud
    is_streamlit_cloud = _is_streamlit_cloud()

    if is_streamlit_cloud:
        # Use persistent cache directory for Streamlit Cloud
        cache_base = Path("/app/.cache/huggingface")

        # Ensure parent directories exist and are writable
        try:
            cache_base.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Streamlit Cloud cache directory created: {cache_base}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create Streamlit Cloud cache directory: {e}")
            logger.warning("Falling back to default HuggingFace cache")
            return

        # Configure HuggingFace environment variables for Streamlit Cloud
        os.environ.setdefault("HF_HOME", str(cache_base))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_base / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_base / "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_base / "transformers"))

        logger.info(f"✅ Configured Streamlit Cloud HuggingFace cache: {cache_base}")
    else:
        # Local development - use default HuggingFace cache, don't override
        logger.info("🏠 Local development detected - using default HuggingFace cache")

    # Enable tokenizers parallelism for better performance
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def preload_models():
    """
    Preload embedding and reranking models to ensure they're cached before app starts.

    This prevents download delays during user interactions.
    """
    import sys
    from pathlib import Path


    try:
        # Import model cache functions directly
        from app.core.model_cache import get_cached_embeddings, get_cached_cross_encoder

        logger.info("Preloading models for Streamlit Cloud...")

        # Preload main embedding model using cache
        embeddings = get_cached_embeddings("sentence-transformers/all-mpnet-base-v2")
        logger.info("✅ Main embedding model preloaded")

        # Preload cross-encoder for reranking using cache
        cross_encoder = get_cached_cross_encoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Cross-encoder model preloaded")

        # Test models with dummy data to ensure they're ready
        test_text = "This is a test document for model validation."
        embeddings.embed_query(test_text)

        test_pairs = [[test_text, "This is a relevant query."]]
        cross_encoder.predict(test_pairs)

        logger.info("✅ All models validated and cached")

    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
        raise

def initialize_for_streamlit_cloud():
    """
    Initialize the application for Streamlit Cloud deployment.

    This should be called at the very beginning of the main application file to ensure
    models are cached before any user interactions.
    """
    logger.info("Initializing application...")

    # Configure caching
    configure_streamlit_cloud_cache()

    # Skip model preloading to avoid PyTorch device placement issues in containers
    # Models will be loaded on-demand via model_cache.py
    logger.info("⏭️ Skipping model preloading - models loaded on-demand for better compatibility")

    logger.info("Application initialization complete")
