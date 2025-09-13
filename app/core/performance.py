#!/usr/bin/env python3
"""
Performance Optimization Module

This module provides performance optimizations using prebuilt libraries:
- diskcache: Smart caching system
- joblib: Function result caching
- httpx: Async HTTP client
- backoff: Retry logic with exponential backoff
- psutil: System resource monitoring
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from functools import wraps

import diskcache
import joblib
import httpx
import backoff
import psutil
from tqdm import tqdm

# Optional imports for GPU/CPU optimization
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from app.core.config import get_config

logger = logging.getLogger(__name__)

# Type hints
T = TypeVar('T')

class PerformanceManager:
    """Central manager for performance optimizations"""

    def __init__(self):
        self.config = get_config()
        self._setup_caches()
        self._setup_clients()

    def _setup_caches(self):
        """Initialize caching systems"""
        faiss_dir = self.config.paths['faiss_dir']
        faiss_dir.mkdir(parents=True, exist_ok=True)

        # Document content cache
        self.doc_cache = diskcache.Cache(
            str(faiss_dir / '.doc_cache'),
            size_limit=500 * 1024 * 1024,  # 500MB
            eviction_policy='least-recently-used'
        )

        # Embedding cache
        self.embedding_cache = diskcache.Cache(
            str(faiss_dir / '.embedding_cache'),
            size_limit=2 * 1024 * 1024 * 1024,  # 2GB
            eviction_policy='least-recently-used'
        )

        # Joblib memory cache for expensive computations
        self.memory = joblib.Memory(
            location=str(faiss_dir / '.joblib_cache'),
            verbose=0,
            compress=True
        )

    def _setup_clients(self):
        """Initialize HTTP clients"""
        # Async HTTP client for AI API calls
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def cache_document_content(self, file_path: Path, content: str) -> None:
        """Cache document content with hash-based key"""
        file_hash = self.get_file_hash(file_path)
        cache_key = f"doc_content:{file_hash}"
        self.doc_cache.set(cache_key, content, expire=86400 * 30)  # 30 days

    def get_cached_document_content(self, file_path: Path) -> Optional[str]:
        """Get cached document content"""
        file_hash = self.get_file_hash(file_path)
        cache_key = f"doc_content:{file_hash}"
        return self.doc_cache.get(cache_key)

    def cache_embeddings(self, text_hash: str, embeddings: List[List[float]]) -> None:
        """Cache embeddings with content hash"""
        cache_key = f"embeddings:{text_hash}"
        self.embedding_cache.set(cache_key, embeddings, expire=86400 * 30)

    def get_cached_embeddings(self, text_hash: str) -> Optional[List[List[float]]]:
        """Get cached embeddings"""
        cache_key = f"embeddings:{text_hash}"
        return self.embedding_cache.get(cache_key)

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.TimeoutException),
        max_tries=3,
        jitter=backoff.random_jitter
    )
    async def make_api_request(self, url: str, **kwargs) -> httpx.Response:
        """Make API request with automatic retry logic"""
        return await self.http_client.request(url=url, **kwargs)

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()

        result = {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

        # Add GPU memory info if available
        if ACCELERATE_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0)
                    result.update({
                        'gpu_total': gpu_memory.total_memory / 1024 / 1024 / 1024,  # GB
                        'gpu_allocated': torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024,  # GB
                        'gpu_reserved': torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024,  # GB
                    })
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")

        return result

    def should_gc_collect(self, memory_usage: Dict[str, float]) -> bool:
        """Determine if garbage collection should be triggered"""
        return memory_usage['percent'] > 80.0 or memory_usage['rss'] > 2000  # 80% or 2GB

    def cleanup_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries"""
        doc_cleaned = self.doc_cache.expire()
        embedding_cleaned = self.embedding_cache.expire()

        return {
            'doc_cache_cleaned': doc_cleaned,
            'embedding_cache_cleaned': embedding_cleaned
        }

    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
        self.doc_cache.close()
        self.embedding_cache.close()

    def optimize_batch_size(self, available_memory: float, item_size_estimate: float = 0.1) -> int:
        """Dynamically optimize batch size based on available memory"""
        # Reserve 20% of memory for overhead
        usable_memory = available_memory * 0.8

        # Estimate optimal batch size
        optimal_batch = int(usable_memory / item_size_estimate)

        # Clamp to reasonable bounds
        return max(1, min(optimal_batch, 1000))

    def get_optimal_device(self) -> str:
        """Get the optimal device for computations"""
        if ACCELERATE_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
            except:
                pass
        return 'cpu'

    def setup_accelerate(self):
        """Setup accelerate for optimal performance"""
        if ACCELERATE_AVAILABLE:
            try:
                from accelerate import Accelerator
                self.accelerator = Accelerator()
                logger.info(f"Accelerate initialized with device: {self.accelerator.device}")
                return self.accelerator
            except Exception as e:
                logger.warning(f"Failed to initialize accelerate: {e}")
        return None


# Global performance manager instance
_perf_manager = None

def get_performance_manager() -> PerformanceManager:
    """Get global performance manager instance"""
    global _perf_manager
    if _perf_manager is None:
        _perf_manager = PerformanceManager()
    return _perf_manager


# Decorators for easy optimization
def cached_by_content(func: Callable[..., T]) -> Callable[..., T]:
    """Cache function results based on content hash"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate content hash from relevant arguments
        content_parts = []
        for arg in args[1:]:  # Skip self
            if isinstance(arg, (str, Path)):
                content_parts.append(str(arg))

        content_hash = hashlib.sha256(
            '|'.join(content_parts).encode()
        ).hexdigest()[:16]

        perf_manager = get_performance_manager()
        cache_key = f"{func.__name__}:{content_hash}"

        # Try cache first
        result = perf_manager.doc_cache.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit for {func.__name__}")
            return result

        # Compute and cache
        result = func(*args, **kwargs)
        perf_manager.doc_cache.set(cache_key, result, expire=86400 * 7)  # 7 days
        return result

    return wrapper


def memory_cached(func: Callable[..., T]) -> Callable[..., T]:
    """Cache function results using joblib memory cache"""
    perf_manager = get_performance_manager()
    cached_func = perf_manager.memory.cache(func)
    return cached_func


def monitor_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Monitor function performance and memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        perf_manager = get_performance_manager()

        # Memory before
        mem_before = perf_manager.monitor_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Memory after
            mem_after = perf_manager.monitor_memory_usage()
            duration = time.time() - start_time

            logger.debug(
                f"{func.__name__}: {duration:.2f}s, "
                f"Memory: {mem_before['rss']:.1f}MB -> {mem_after['rss']:.1f}MB"
            )

            # Trigger GC if needed
            if perf_manager.should_gc_collect(mem_after):
                import gc
                gc.collect()
                logger.debug("Garbage collection triggered")

    return wrapper


# Utility functions
def get_text_hash(text: str) -> str:
    """Generate hash for text content"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def parallel_process(items: List[T], func: Callable[[T], Any],
                    max_workers: int = 4, desc: str = "Processing") -> List[Any]:
    """Process items in parallel using ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}

        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    return results


def optimize_embedding_batch(texts: List[str], embeddings_model,
                           batch_size: int = 32) -> List[List[float]]:
    """Optimize embedding generation with dynamic batching"""
    perf_manager = get_performance_manager()

    # Get available memory for batch optimization
    mem_info = perf_manager.monitor_memory_usage()
    available_memory = mem_info['rss']

    # Dynamically adjust batch size based on memory
    optimal_batch = perf_manager.optimize_batch_size(available_memory, item_size_estimate=0.001)
    batch_size = min(batch_size, optimal_batch)

    logger.info(f"Using optimized batch size: {batch_size} (memory: {available_memory:.1f}MB)")

    all_embeddings = []

    # Process in optimized batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Monitor memory before processing
        mem_before = perf_manager.monitor_memory_usage()

        try:
            # Generate embeddings for this batch
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

            # Monitor memory after processing
            mem_after = perf_manager.monitor_memory_usage()

            # Trigger GC if memory usage is high
            if perf_manager.should_gc_collect(mem_after):
                import gc
                gc.collect()
                logger.debug("GC triggered during embedding generation")

        except Exception as e:
            logger.error(f"Failed to process embedding batch {i//batch_size}: {e}")
            # Continue with empty embeddings for this batch
            all_embeddings.extend([[] for _ in batch])

    return all_embeddings


async def gather_with_concurrency(n: int, *tasks):
    """Run async tasks with controlled concurrency"""
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


# Cleanup function for graceful shutdown
async def cleanup_performance_resources():
    """Clean up performance resources"""
    global _perf_manager
    if _perf_manager:
        await _perf_manager.close()
        _perf_manager = None
