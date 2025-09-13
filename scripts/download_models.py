#!/usr/bin/env python3
"""
Model Download Script for Streamlit Cloud

Downloads and caches HuggingFace models locally for Streamlit Cloud deployment.
This ensures models are available without download delays during runtime.
"""

import os
import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_sentence_transformer_model(model_name: str, local_dir: Path):
    """Download sentence transformer model to local directory."""
    logger.info(f"Downloading sentence transformer model: {model_name}")

    try:
        # Download only essential files for faster download
        essential_files = [
            "config.json",
            "config_sentence_transformers.json",
            "data_config.json",
            "modules.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "README.md",
            "1_Pooling/config.json"
        ]

        from huggingface_hub import hf_hub_download

        # Create directory
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download essential files
        for filename in essential_files:
            try:
                hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
                continue

        logger.info(f"✅ Successfully downloaded {model_name} to {local_dir}")

        # Test the model
        model = SentenceTransformer(str(local_dir))
        test_embedding = model.encode("This is a test sentence.")
        logger.info(f"✅ Model test successful - embedding shape: {test_embedding.shape}")

    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        raise

def download_cross_encoder_model(model_name: str, local_dir: Path):
    """Download cross-encoder model to local directory."""
    logger.info(f"Downloading cross-encoder model: {model_name}")

    try:
        # Download only essential files for faster download
        essential_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "README.md"
        ]

        from huggingface_hub import hf_hub_download

        # Create directory
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download essential files
        for filename in essential_files:
            try:
                hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
                continue

        logger.info(f"✅ Successfully downloaded {model_name} to {local_dir}")

        # Test the model
        model = CrossEncoder(str(local_dir))
        test_scores = model.predict([["This is a test query", "This is a test document"]])
        logger.info(f"✅ Model test successful - scores shape: {test_scores.shape}")

    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        raise

def main():
    """Main function to download all required models."""
    print(f"{BLUE}🚀 Starting Model Download Process for Streamlit Cloud{NC}")
    print("")

    # Define models and their local paths
    models_dir = Path("models")
    models_config = [
        {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "local_dir": models_dir / "sentence_transformers" / "all-mpnet-base-v2",
            "download_func": download_sentence_transformer_model
        },
        {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "local_dir": models_dir / "cross_encoder" / "ms-marco-MiniLM-L-6-v2",
            "download_func": download_cross_encoder_model
        }
    ]

    # Create models directory
    models_dir.mkdir(exist_ok=True)

    # Download each model
    for model_config in models_config:
        try:
            print(f"{YELLOW}📥 Downloading {model_config['name']}...{NC}")
            model_config["local_dir"].parent.mkdir(parents=True, exist_ok=True)
            model_config["download_func"](model_config["name"], model_config["local_dir"])
            print(f"{GREEN}✅ Successfully downloaded {model_config['name']}{NC}")
        except Exception as e:
            print(f"{RED}❌ Failed to process {model_config['name']}: {e}{NC}")
            sys.exit(1)

    print(f"\n{GREEN}🎉 All models downloaded and cached successfully!{NC}")
    print(f"📁 Models cached in: {models_dir.absolute()}")

    # Print total size
    total_size = sum(
        sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        for model_dir in [m["local_dir"] for m in models_config]
        if model_dir.exists()
    )
    print(f"📊 Total cached size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
