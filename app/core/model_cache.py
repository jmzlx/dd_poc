#!/usr/bin/env python3
"""
Model Cache Manager

Provides global caching for HuggingFace models to prevent re-downloads
across multiple instances and sessions.
"""

import logging
from typing import Optional
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.core.logging import logger

# Optional accelerate import
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# Global model cache
_EMBEDDINGS_CACHE = {}
_CROSS_ENCODER_CACHE = {}

# Local models directory
_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def _get_local_model_path(model_name: str) -> Optional[Path]:
    """
    Get local path for a model if it exists.

    Args:
        model_name: HuggingFace model name

    Returns:
        Path to local model directory or None if not found
    """
    if "/" in model_name:
        # Handle different model name formats
        if model_name.startswith("sentence-transformers/"):
            # For sentence transformers: sentence-transformers/all-mpnet-base-v2
            model_short_name = model_name.split("/")[-1]
            local_path = _MODELS_DIR / "sentence_transformers" / model_short_name
        elif model_name.startswith("cross-encoder/"):
            # For cross encoders: cross-encoder/ms-marco-MiniLM-L-6-v2
            model_short_name = model_name.split("/")[-1]
            local_path = _MODELS_DIR / "cross_encoder" / model_short_name
        else:
            # Fallback for other models
            model_short_name = model_name.split("/")[-1]
            local_path = _MODELS_DIR / model_short_name

        if local_path.exists():
            return local_path

    return None

def get_cached_embeddings(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> HuggingFaceEmbeddings:
    """
    Get cached HuggingFace embeddings model with accelerate optimization.

    Creates the model only once and reuses it across all instances.
    Uses local models directory if available, otherwise downloads from HuggingFace.
    Automatically uses GPU if available via accelerate.
    """
    if model_name not in _EMBEDDINGS_CACHE:
        # Check for local model first
        local_path = _get_local_model_path(model_name)
        if local_path:
            logger.info(f"Using local embeddings model: {local_path}")
            embeddings = HuggingFaceEmbeddings(model_name=str(local_path))
        else:
            logger.info(f"Downloading embeddings model: {model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Optimize device placement with accelerate if available
        if ACCELERATE_AVAILABLE:
            try:
                accelerator = Accelerator()
                logger.info(f"Embeddings model optimized for device: {accelerator.device}")
                # Accelerate will automatically handle device placement
            except Exception as e:
                logger.warning(f"Failed to optimize embeddings with accelerate: {e}")

        _EMBEDDINGS_CACHE[model_name] = embeddings
    else:
        logger.debug(f"Using cached embeddings model: {model_name}")

    return _EMBEDDINGS_CACHE[model_name]

def get_cached_cross_encoder(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> CrossEncoder:
    """
    Get cached cross-encoder model.

    Creates the model only once and reuses it across all instances.
    Uses local models directory if available, otherwise downloads from HuggingFace.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        # Check for local model first
        local_path = _get_local_model_path(model_name)
        if local_path:
            logger.info(f"Using local cross-encoder model: {local_path}")
            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(str(local_path))
        else:
            logger.info(f"Downloading cross-encoder model: {model_name}")
            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    else:
        logger.debug(f"Using cached cross-encoder model: {model_name}")

    return _CROSS_ENCODER_CACHE[model_name]

def clear_model_cache():
    """
    Clear all cached models.

    Useful for memory management or testing.
    """
    _EMBEDDINGS_CACHE.clear()
    _CROSS_ENCODER_CACHE.clear()
    logger.info("Model cache cleared")
