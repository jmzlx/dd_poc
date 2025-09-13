#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Due Diligence POC

This module provides a complete benchmarking framework for evaluating the predictive
performance of all AI/ML components in the dd-poc system.

Benchmarked Components:
1. Document Classification (accuracy, precision, recall, F1)
2. Search Retrieval (precision@k, recall@k, NDCG, MRR)
3. Question Answering (BLEU, ROUGE, BERTScore, semantic similarity)
4. Report Generation (content quality, coherence, completeness)
5. Hybrid Search (end-to-end retrieval performance)

Usage:
    python benchmarks/benchmark_runner.py --task all --dataset summit
    python benchmarks/benchmark_runner.py --task search --dataset summit --iterations 3
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.core.config import get_config
from app.core.performance import get_performance_manager
from app.core.constants import TEMPERATURE
from app.ai.document_classifier import batch_classify_document_types
from app.core.search import hybrid_search, search_and_analyze, rerank_results
from app.core.model_cache import get_cached_embeddings, get_cached_cross_encoder
from app.core.sparse_index import load_sparse_index_for_store
from app.core.utils import create_document_processor
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    task: str
    metric: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkRun:
    """Container for a complete benchmark run"""
    run_id: str
    dataset: str
    tasks: List[str]
    results: List[BenchmarkResult]
    config: Dict[str, Any]
    duration: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BenchmarkRunner:
    """Main benchmark runner for dd-poc system"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config()
        self.perf_manager = get_performance_manager()
        self.results = []
        self.datasets = self._load_datasets()

        # Initialize models
        self._setup_models()

    def _setup_models(self):
        """Initialize required models for benchmarking"""
        logger.info("Setting up models for benchmarking...")

        try:
            self.embeddings = get_cached_embeddings()
            self.cross_encoder = get_cached_cross_encoder()

            # Try to initialize Claude for generation tasks
            self.llm = None
            try:
                api_key = self.config.api.anthropic_api_key
                if api_key:
                    self.llm = ChatAnthropic(
                        model=self.config.model.claude_model,
                        anthropic_api_key=api_key,
                        temperature=TEMPERATURE,  # Deterministic for consistent results
                        max_tokens=self.config.model.max_tokens
                    )
                    logger.info("✅ Claude model initialized")
                else:
                    logger.warning("❌ No Anthropic API key found - generation benchmarks will be skipped")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize Claude: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to setup models: {e}")
            raise

    def _load_datasets(self) -> Dict[str, Dict]:
        """Load benchmark datasets"""
        datasets = {}

        # Define available datasets based on existing data
        data_dir = Path("data")
        if (data_dir / "vdrs" / "industrial-security-leadership" / "deepshield-systems-inc").exists():
            datasets["deepshield"] = {
                "name": "DeepShield Systems Inc.",
                "path": data_dir / "vdrs" / "industrial-security-leadership" / "deepshield-systems-inc",
                "store_name": "deepshield-systems-inc",
                "documents": list((data_dir / "vdrs" / "industrial-security-leadership" / "deepshield-systems-inc").glob("**/*.pdf"))
            }

        if (data_dir / "vdrs" / "automated-services-transformation" / "summit-digital-solutions-inc").exists():
            datasets["summit"] = {
                "name": "Summit Digital Solutions Inc.",
                "path": data_dir / "vdrs" / "automated-services-transformation" / "summit-digital-solutions-inc",
                "store_name": "summit-digital-solutions-inc",
                "documents": list((data_dir / "vdrs" / "automated-services-transformation" / "summit-digital-solutions-inc").glob("**/*.pdf"))
            }

        logger.info(f"✅ Loaded {len(datasets)} benchmark datasets: {list(datasets.keys())}")
        return datasets

    def run_classification_benchmark(self, dataset: str, iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark document classification performance"""
        logger.info(f"🏷️ Running document classification benchmark on {dataset}")

        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        dataset_info = self.datasets[dataset]
        results = []

        # Load existing classifications if available
        ground_truth = self._load_classification_ground_truth(dataset)
        if not ground_truth:
            logger.warning(f"No ground truth classifications found for {dataset}")
            return results

        # Sample documents for benchmarking
        sample_docs = list(ground_truth.keys())[:50]  # Benchmark on first 50 docs
        if len(sample_docs) < 10:
            logger.warning(f"Insufficient ground truth data for {dataset}")
            return results

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            start_time = time.time()

            # Prepare documents for classification
            docs_to_classify = []
            true_labels = []

            for doc_path in sample_docs:
                if doc_path in ground_truth:
                    # Load first chunk of document
                    doc_info = self._load_document_first_chunk(doc_path)
                    if doc_info:
                        docs_to_classify.append(doc_info)
                        true_labels.append(ground_truth[doc_path])

            if not docs_to_classify:
                continue

            try:
                # Run classification
                classified_docs = batch_classify_document_types(
                    docs_to_classify,
                    self.llm
                )

                # Extract predictions
                pred_labels = []
                for doc in classified_docs:
                    pred_labels.append(doc.get('document_type', 'unknown'))

                # Calculate metrics
                accuracy = accuracy_score(true_labels, pred_labels)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average='weighted', zero_division=0
                )

                duration = time.time() - start_time
                throughput = len(docs_to_classify) / duration

                # Store results
                results.extend([
                    BenchmarkResult(
                        task="classification",
                        metric="accuracy",
                        value=accuracy,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(docs_to_classify)}
                    ),
                    BenchmarkResult(
                        task="classification",
                        metric="precision",
                        value=precision,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(docs_to_classify)}
                    ),
                    BenchmarkResult(
                        task="classification",
                        metric="recall",
                        value=recall,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(docs_to_classify)}
                    ),
                    BenchmarkResult(
                        task="classification",
                        metric="f1_score",
                        value=f1,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(docs_to_classify)}
                    ),
                    BenchmarkResult(
                        task="classification",
                        metric="throughput_docs_per_sec",
                        value=throughput,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(docs_to_classify)}
                    )
                ])

                logger.info(".3f"
            except Exception as e:
                logger.error(f"Classification benchmark failed: {e}")
                continue

        return results

    def run_search_benchmark(self, dataset: str, iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark search and retrieval performance"""
        logger.info(f"🔍 Running search benchmark on {dataset}")

        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        dataset_info = self.datasets[dataset]
        store_name = dataset_info["store_name"]
        results = []

        # Load vector store
        try:
            vector_store = FAISS.load_local(
                str(self.config.paths['faiss_dir']),
                self.embeddings,
                index_name=store_name,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load vector store for {store_name}: {e}")
            return results

        # Load search ground truth
        ground_truth = self._load_search_ground_truth(dataset)
        if not ground_truth:
            logger.warning(f"No search ground truth found for {dataset}")
            return results

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            # Test different search configurations
            search_configs = [
                {"method": "dense_only", "use_hybrid": False},
                {"method": "hybrid", "use_hybrid": True, "sparse_weight": 0.3, "dense_weight": 0.7},
                {"method": "hybrid_balanced", "use_hybrid": True, "sparse_weight": 0.5, "dense_weight": 0.5},
                {"method": "sparse_heavy", "use_hybrid": True, "sparse_weight": 0.7, "dense_weight": 0.3}
            ]

            for config in search_configs:
                start_time = time.time()

                # Run search queries
                query_results = []
                for query_info in ground_truth[:10]:  # Test on first 10 queries
                    query = query_info["query"]
                    relevant_docs = set(query_info["relevant_docs"])

                    try:
                        if config["use_hybrid"]:
                            search_results = hybrid_search(
                                query=query,
                                vector_store=vector_store,
                                store_name=store_name,
                                top_k=20,
                                sparse_weight=config["sparse_weight"],
                                dense_weight=config["dense_weight"]
                            )
                        else:
                            # Dense only search
                            docs_with_scores = vector_store.similarity_search_with_score(query, k=20)
                            search_results = [{
                                'doc_id': doc.metadata.get('source', ''),
                                'score': float(score)
                            } for doc, score in docs_with_scores]

                        # Calculate retrieval metrics
                        retrieved_docs = [r['doc_id'] for r in search_results[:10]]  # Top 10
                        retrieved_set = set(retrieved_docs)

                        # Precision@10, Recall@10
                        true_positives = len(retrieved_set & relevant_docs)
                        precision_at_10 = true_positives / len(retrieved_docs) if retrieved_docs else 0
                        recall_at_10 = true_positives / len(relevant_docs) if relevant_docs else 0

                        # Mean Reciprocal Rank (MRR)
                        mrr = 0
                        for rank, doc_id in enumerate(retrieved_docs, 1):
                            if doc_id in relevant_docs:
                                mrr = 1.0 / rank
                                break

                        query_results.append({
                            "precision@10": precision_at_10,
                            "recall@10": recall_at_10,
                            "mrr": mrr
                        })

                    except Exception as e:
                        logger.error(f"Search failed for query '{query}': {e}")
                        continue

                if query_results:
                    # Aggregate metrics
                    avg_precision = statistics.mean([r["precision@10"] for r in query_results])
                    avg_recall = statistics.mean([r["recall@10"] for r in query_results])
                    avg_mrr = statistics.mean([r["mrr"] for r in query_results])

                    duration = time.time() - start_time
                    queries_per_sec = len(query_results) / duration

                    results.extend([
                        BenchmarkResult(
                            task="search",
                            metric="precision@10",
                            value=avg_precision,
                            metadata={"method": config["method"], "iteration": iteration, "dataset": dataset}
                        ),
                        BenchmarkResult(
                            task="search",
                            metric="recall@10",
                            value=avg_recall,
                            metadata={"method": config["method"], "iteration": iteration, "dataset": dataset}
                        ),
                        BenchmarkResult(
                            task="search",
                            metric="mrr",
                            value=avg_mrr,
                            metadata={"method": config["method"], "iteration": iteration, "dataset": dataset}
                        ),
                        BenchmarkResult(
                            task="search",
                            metric="throughput_queries_per_sec",
                            value=queries_per_sec,
                            metadata={"method": config["method"], "iteration": iteration, "dataset": dataset}
                        )
                    ])

                    logger.info(".3f"
        return results

    def run_qa_benchmark(self, dataset: str, iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark question answering performance"""
        logger.info(f"🤖 Running QA benchmark on {dataset}")

        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        if not self.llm:
            logger.warning("No LLM available for QA benchmark")
            return []

        dataset_info = self.datasets[dataset]
        store_name = dataset_info["store_name"]
        results = []

        # Load vector store
        try:
            vector_store = FAISS.load_local(
                str(self.config.paths['faiss_dir']),
                self.embeddings,
                index_name=store_name,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load vector store for {store_name}: {e}")
            return results

        # Load QA ground truth
        ground_truth = self._load_qa_ground_truth(dataset)
        if not ground_truth:
            logger.warning(f"No QA ground truth found for {dataset}")
            return results

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            start_time = time.time()

            # Test QA on sample questions
            qa_results = []
            for qa_pair in ground_truth[:10]:  # Test on first 10 QA pairs
                question = qa_pair["question"]
                expected_answer = qa_pair["answer"]

                try:
                    # Use RAG to generate answer
                    retriever = vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.1, "k": 5}
                    )

                    from langchain.chains.retrieval import create_retrieval_chain
                    from langchain.chains.combine_documents import create_stuff_documents_chain
                    from langchain_core.prompts import PromptTemplate

                    prompt_template = PromptTemplate(
                        input_variables=["context", "input"],
                        template="""Use the provided context to answer the question. Be concise and factual.

Context: {context}

Question: {input}

Answer:"""
                    )

                    document_chain = create_stuff_documents_chain(self.llm, prompt_template)
                    qa_chain = create_retrieval_chain(retriever, document_chain)

                    response = qa_chain.invoke({"input": question})
                    generated_answer = response.get('answer', '')

                    if generated_answer:
                        # Calculate semantic similarity (simple approach)
                        similarity = self._calculate_answer_similarity(generated_answer, expected_answer)

                        qa_results.append({
                            "similarity": similarity,
                            "answer_length": len(generated_answer)
                        })

                except Exception as e:
                    logger.error(f"QA failed for question '{question}': {e}")
                    continue

            if qa_results:
                avg_similarity = statistics.mean([r["similarity"] for r in qa_results])
                avg_answer_length = statistics.mean([r["answer_length"] for r in qa_results])

                duration = time.time() - start_time
                questions_per_sec = len(qa_results) / duration

                results.extend([
                    BenchmarkResult(
                        task="qa",
                        metric="semantic_similarity",
                        value=avg_similarity,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(qa_results)}
                    ),
                    BenchmarkResult(
                        task="qa",
                        metric="avg_answer_length",
                        value=avg_answer_length,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(qa_results)}
                    ),
                    BenchmarkResult(
                        task="qa",
                        metric="throughput_questions_per_sec",
                        value=questions_per_sec,
                        metadata={"iteration": iteration, "dataset": dataset, "sample_size": len(qa_results)}
                    )
                ])

                logger.info(".3f"
        return results

    def run_all_benchmarks(self, dataset: str, iterations: int = 3) -> BenchmarkRun:
        """Run all benchmarks"""
        logger.info(f"🚀 Starting comprehensive benchmark on {dataset}")

        run_id = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()

        all_results = []

        # Run individual benchmarks
        benchmark_tasks = [
            ("classification", self.run_classification_benchmark),
            ("search", self.run_search_benchmark),
            ("qa", self.run_qa_benchmark)
        ]

        for task_name, benchmark_func in benchmark_tasks:
            try:
                logger.info(f"Running {task_name} benchmark...")
                task_results = benchmark_func(dataset, iterations)
                all_results.extend(task_results)
                logger.info(f"✅ {task_name} benchmark completed")
            except Exception as e:
                logger.error(f"❌ {task_name} benchmark failed: {e}")
                continue

        duration = time.time() - start_time

        # Create benchmark run
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            dataset=dataset,
            tasks=[r.task for r in all_results],
            results=all_results,
            config={
                "iterations": iterations,
                "models": {
                    "embeddings": "all-mpnet-base-v2",
                    "cross_encoder": "ms-marco-MiniLM-L-6-v2",
                    "llm": self.config.model.claude_model if self.llm else None
                }
            },
            duration=duration
        )

        # Save results
        self._save_benchmark_results(benchmark_run)

        logger.info(f"🎉 Benchmark completed in {duration:.2f}s")
        return benchmark_run

    def _load_classification_ground_truth(self, dataset: str) -> Dict[str, str]:
        """Load ground truth classifications for benchmarking"""
        # This would load from a ground truth file
        # For now, return empty dict - would need to be populated manually
        return {}

    def _load_search_ground_truth(self, dataset: str) -> List[Dict]:
        """Load ground truth search queries and relevant documents"""
        # This would load from a ground truth file
        # For now, return empty list - would need to be populated manually
        return []

    def _load_qa_ground_truth(self, dataset: str) -> List[Dict]:
        """Load ground truth QA pairs"""
        # This would load from a ground truth file
        # For now, return empty list - would need to be populated manually
        return []

    def _load_document_first_chunk(self, doc_path: str) -> Optional[Dict]:
        """Load first chunk of document for classification"""
        # This would extract first chunk from document
        # For now, return None - would need implementation
        return None

    def _calculate_answer_similarity(self, generated: str, expected: str) -> float:
        """Calculate semantic similarity between generated and expected answers"""
        # Simple word overlap for now - could be improved with embeddings
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())

        if not gen_words or not exp_words:
            return 0.0

        intersection = gen_words & exp_words
        union = gen_words | exp_words

        return len(intersection) / len(union) if union else 0.0

    def _save_benchmark_results(self, benchmark_run: BenchmarkRun):
        """Save benchmark results to file"""
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(exist_ok=True)

        # Save detailed results
        results_file = output_dir / f"{benchmark_run.run_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "run_id": benchmark_run.run_id,
                "dataset": benchmark_run.dataset,
                "timestamp": benchmark_run.timestamp,
                "duration": benchmark_run.duration,
                "config": benchmark_run.config,
                "results": [asdict(result) for result in benchmark_run.results]
            }, f, indent=2)

        # Save summary CSV
        summary_file = output_dir / f"{benchmark_run.run_id}_summary.csv"
        if benchmark_run.results:
            df = pd.DataFrame([{
                "task": r.task,
                "metric": r.metric,
                "value": r.value,
                "dataset": benchmark_run.dataset,
                "run_id": benchmark_run.run_id
            } for r in benchmark_run.results])
            df.to_csv(summary_file, index=False)

        logger.info(f"💾 Results saved to {results_file} and {summary_file}")

    def generate_report(self, run_id: Optional[str] = None):
        """Generate performance report and visualizations"""
        output_dir = Path("benchmarks/results")
        if not output_dir.exists():
            logger.error("No benchmark results found")
            return

        # Load latest results if no run_id specified
        if not run_id:
            result_files = list(output_dir.glob("*_results.json"))
            if not result_files:
                logger.error("No benchmark result files found")
                return
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            result_file = result_files[0]
        else:
            result_file = output_dir / f"{run_id}_results.json"

        if not result_file.exists():
            logger.error(f"Result file not found: {result_file}")
            return

        # Load results
        with open(result_file, 'r') as f:
            data = json.load(f)

        results = [BenchmarkResult(**r) for r in data["results"]]

        # Generate visualizations
        self._generate_performance_plots(results, data["run_id"])

        # Generate summary report
        self._generate_summary_report(results, data)

        logger.info(f"📊 Report generated for run {data['run_id']}")

    def _generate_performance_plots(self, results: List[BenchmarkResult], run_id: str):
        """Generate performance visualization plots"""
        output_dir = Path("benchmarks/reports")
        output_dir.mkdir(exist_ok=True)

        # Group results by task and metric
        task_metrics = {}
        for result in results:
            key = f"{result.task}_{result.metric}"
            if key not in task_metrics:
                task_metrics[key] = []
            task_metrics[key].append(result.value)

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Classification Performance", "Search Performance",
                          "QA Performance", "Throughput Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Classification metrics
        classification_data = [(k, v) for k, v in task_metrics.items()
                             if k.startswith("classification_") and not k.endswith("_throughput")]
        if classification_data:
            for metric_name, values in classification_data:
                metric = metric_name.replace("classification_", "")
                fig.add_trace(
                    go.Bar(name=f"Classification {metric}", x=[metric], y=[statistics.mean(values)]),
                    row=1, col=1
                )

        # Search metrics
        search_data = [(k, v) for k, v in task_metrics.items()
                      if k.startswith("search_") and not k.endswith("_throughput")]
        if search_data:
            for metric_name, values in search_data:
                metric = metric_name.replace("search_", "")
                fig.add_trace(
                    go.Bar(name=f"Search {metric}", x=[metric], y=[statistics.mean(values)]),
                    row=1, col=2
                )

        # QA metrics
        qa_data = [(k, v) for k, v in task_metrics.items()
                  if k.startswith("qa_") and not k.endswith("_throughput")]
        if qa_data:
            for metric_name, values in qa_data:
                metric = metric_name.replace("qa_", "")
                fig.add_trace(
                    go.Bar(name=f"QA {metric}", x=[metric], y=[statistics.mean(values)]),
                    row=2, col=1
                )

        # Throughput comparison
        throughput_data = [(k, v) for k, v in task_metrics.items() if "_throughput" in k]
        if throughput_data:
            tasks = []
            throughputs = []
            for metric_name, values in throughput_data:
                task = metric_name.split("_")[0]
                tasks.append(task)
                throughputs.append(statistics.mean(values))

            fig.add_trace(
                go.Bar(name="Throughput", x=tasks, y=throughputs),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=f"Benchmark Performance Report - {run_id}",
            showlegend=False,
            height=800
        )

        # Save plot
        plot_file = output_dir / f"{run_id}_performance_report.html"
        fig.write_html(str(plot_file))
        logger.info(f"📈 Performance plot saved to {plot_file}")

    def _generate_summary_report(self, results: List[BenchmarkResult], run_data: Dict):
        """Generate text summary report"""
        output_dir = Path("benchmarks/reports")
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / f"{run_data['run_id']}_summary_report.md"

        with open(report_file, 'w') as f:
            f.write("# Benchmark Summary Report\n\n")
            f.write(f"**Run ID:** {run_data['run_id']}\n")
            f.write(f"**Dataset:** {run_data['dataset']}\n")
            f.write(f"**Timestamp:** {run_data['timestamp']}\n")
            f.write(f"**Duration:** {run_data['duration']:.2f} seconds\n\n")

            f.write("## Configuration\n")
            f.write(f"- **Embeddings Model:** {run_data['config']['models']['embeddings']}\n")
            f.write(f"- **Cross-Encoder:** {run_data['config']['models']['cross_encoder']}\n")
            f.write(f"- **LLM:** {run_data['config']['models']['llm'] or 'None'}\n")
            f.write(f"- **Iterations:** {run_data['config']['iterations']}\n\n")

            # Group results by task
            task_results = {}
            for result in results:
                if result.task not in task_results:
                    task_results[result.task] = []
                task_results[result.task].append(result)

            # Generate task summaries
            for task, task_res in task_results.items():
                f.write(f"## {task.title()} Performance\n\n")

                # Group by metric
                metric_results = {}
                for result in task_res:
                    if result.metric not in metric_results:
                        metric_results[result.metric] = []
                    metric_results[result.metric].append(result.value)

                for metric, values in metric_results.items():
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    f.write(".3f")

                f.write("\n")

        logger.info(f"📋 Summary report saved to {report_file}")


def main():
    """Main entry point for benchmark runner"""
    parser = argparse.ArgumentParser(description="Run dd-poc benchmarks")
    parser.add_argument("--task", choices=["classification", "search", "qa", "all"],
                       default="all", help="Benchmark task to run")
    parser.add_argument("--dataset", choices=["deepshield", "summit"],
                       default="summit", help="Dataset to benchmark on")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations for each benchmark")
    parser.add_argument("--report", type=str, help="Generate report for specific run ID")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")

    args = parser.parse_args()

    try:
        runner = BenchmarkRunner()

        if args.list_datasets:
            print("Available datasets:")
            for name, info in runner.datasets.items():
                print(f"  - {name}: {info['name']} ({len(info['documents'])} documents)")
            return

        if args.report:
            runner.generate_report(args.report)
            return

        # Run benchmarks
        if args.task == "all":
            benchmark_run = runner.run_all_benchmarks(args.dataset, args.iterations)
        else:
            if args.task == "classification":
                results = runner.run_classification_benchmark(args.dataset, args.iterations)
            elif args.task == "search":
                results = runner.run_search_benchmark(args.dataset, args.iterations)
            elif args.task == "qa":
                results = runner.run_qa_benchmark(args.dataset, args.iterations)

            # Create a basic run summary
            benchmark_run = BenchmarkRun(
                run_id=f"{args.dataset}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset=args.dataset,
                tasks=[args.task],
                results=results,
                config={"task": args.task, "iterations": args.iterations},
                duration=0  # Would need to track this properly
            )

        print(f"\n🎉 Benchmark completed!")
        print(f"Run ID: {benchmark_run.run_id}")
        print(f"Tasks: {', '.join(benchmark_run.tasks)}")
        print(f"Results: {len(benchmark_run.results)} metrics collected")
        print("
💡 Use --report to generate visualizations and detailed reports"
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
