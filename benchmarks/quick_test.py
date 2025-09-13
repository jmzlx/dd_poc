#!/usr/bin/env python3
"""
Quick Benchmark Test Script

This script provides a fast way to test the benchmarking infrastructure
without requiring full ground truth datasets.

Usage:
    python benchmarks/quick_test.py
"""

import sys
import time
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from app.core.config import get_config
from app.core.model_cache import get_cached_embeddings
from langchain_community.vectorstores import FAISS


def test_basic_setup():
    """Test basic setup and dependencies"""
    print("🧪 Testing basic setup...")

    try:
        # Test configuration loading
        config = get_config()
        print("✅ Configuration loaded successfully")

        # Test embeddings loading
        embeddings = get_cached_embeddings()
        print("✅ Embeddings model loaded successfully")

        # Test FAISS index loading (if available)
        faiss_dir = Path("data/search_indexes")
        if faiss_dir.exists():
            store_files = list(faiss_dir.glob("*_summit*"))
            if store_files:
                try:
                    vector_store = FAISS.load_local(
                        str(faiss_dir),
                        embeddings,
                        index_name="summit-digital-solutions-inc",
                        allow_dangerous_deserialization=True
                    )
                    print("✅ FAISS vector store loaded successfully")
                    print(f"   📊 Index contains {vector_store.index.ntotal} documents")
                except Exception as e:
                    print(f"⚠️ FAISS loading failed: {e}")
            else:
                print("⚠️ No FAISS index found - run document indexing first")
        else:
            print("⚠️ FAISS directory not found")

        return True

    except Exception as e:
        print(f"❌ Basic setup test failed: {e}")
        return False


def test_search_performance():
    """Test basic search performance"""
    print("\n🔍 Testing search performance...")

    try:
        from app.core.model_cache import get_cached_embeddings
        from langchain_community.vectorstores import FAISS

        embeddings = get_cached_embeddings()
        faiss_dir = Path("data/search_indexes")

        if not faiss_dir.exists():
            print("⚠️ Skipping search test - no FAISS index available")
            return True

        vector_store = FAISS.load_local(
            str(faiss_dir),
            embeddings,
            index_name="summit-digital-solutions-inc",
            allow_dangerous_deserialization=True
        )

        # Test queries
        test_queries = [
            "financial statements",
            "board meeting",
            "company overview",
            "legal agreements"
        ]

        print(f"Running {len(test_queries)} test queries...")

        total_time = 0
        total_results = 0

        for query in test_queries:
            start_time = time.time()
            results = vector_store.similarity_search_with_score(query, k=5)
            query_time = time.time() - start_time

            total_time += query_time
            total_results += len(results)

            print(f"   Query: '{query}' -> {len(results)} results in {query_time:.3f}s")
        avg_query_time = total_time / len(test_queries)
        queries_per_sec = len(test_queries) / total_time

        print(f"   Average query time: {avg_query_time:.3f}s")
        print(f"   Queries per second: {queries_per_sec:.3f}")
        print("✅ Search performance test completed")

        return True

    except Exception as e:
        print(f"❌ Search performance test failed: {e}")
        return False


def test_benchmark_imports():
    """Test that benchmark modules can be imported"""
    print("\n📦 Testing benchmark module imports...")

    try:
        from benchmarks.benchmark_runner import BenchmarkRunner
        print("✅ BenchmarkRunner imported successfully")

        from benchmarks.create_ground_truth import GroundTruthCreator
        print("✅ GroundTruthCreator imported successfully")

        from benchmarks.regression_detector import RegressionDetector
        print("✅ RegressionDetector imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Benchmark import failed: {e}")
        return False


def run_quick_benchmark():
    """Run a quick benchmark test"""
    print("🚀 Running Quick Benchmark Test")
    print("=" * 50)

    tests = [
        ("Basic Setup", test_basic_setup),
        ("Benchmark Imports", test_benchmark_imports),
        ("Search Performance", test_search_performance)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Benchmarking infrastructure is ready.")
        print("\nNext steps:")
        print("1. Create ground truth datasets:")
        print("   python benchmarks/create_ground_truth.py --type classification --dataset summit")
        print("2. Run full benchmarks:")
        print("   python benchmarks/benchmark_runner.py --task all --dataset summit")
        print("3. Generate reports:")
        print("   python benchmarks/benchmark_runner.py --report <run_id>")
    else:
        print("⚠️ Some tests failed. Check the errors above and ensure all dependencies are installed.")

    return passed == total


if __name__ == "__main__":
    success = run_quick_benchmark()
    sys.exit(0 if success else 1)
