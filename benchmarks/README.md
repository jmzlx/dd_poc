# dd-poc Predictive Performance Benchmarking Guide

This guide provides comprehensive instructions for benchmarking the predictive performance of the dd-poc (Due Diligence Proof of Concept) system.

## Overview

The dd-poc system performs several predictive tasks that can be benchmarked:

1. **Document Classification** - Classifies documents into categories (corporate, financial, legal, etc.)
2. **Search & Retrieval** - Finds relevant documents using dense/sparse retrieval with reranking
3. **Question Answering** - Generates answers to questions using retrieved documents
4. **Report Generation** - Creates structured reports from document analysis

## Quick Start

### 1. Create Ground Truth Datasets

First, create ground truth datasets for benchmarking:

```bash
# Create classification ground truth (100 samples)
python benchmarks/create_ground_truth.py --type classification --dataset summit --sample-size 100

# Create search ground truth (50 queries)
python benchmarks/create_ground_truth.py --type search --dataset summit --num-queries 50

# Create QA ground truth (30 pairs)
python benchmarks/create_ground_truth.py --type qa --dataset summit --num-pairs 30
```

### 2. Complete Manual Annotations

Review and complete the generated ground truth files:

```bash
# Edit the generated JSON files to add manual annotations
# Files are saved in benchmarks/ground_truth/
```

### 3. Run Benchmarks

Execute comprehensive benchmarks:

```bash
# Run all benchmarks on summit dataset
python benchmarks/benchmark_runner.py --task all --dataset summit --iterations 3

# Run specific benchmark task
python benchmarks/benchmark_runner.py --task search --dataset summit --iterations 3

# Generate performance reports
python benchmarks/benchmark_runner.py --report <run_id>
```

### 4. Monitor Performance Trends

Set up performance regression detection:

```bash
# Compare two benchmark runs
python benchmarks/regression_detector.py --baseline-run baseline_run --compare-run new_run

# Analyze performance trends over time
python benchmarks/regression_detector.py --trend-analysis --days 30

# Send email alerts for regressions
python benchmarks/regression_detector.py --baseline-run old_run --compare-run new_run --alerts --email-to user@example.com
```

## Detailed Benchmarking Guide

### Document Classification Benchmark

**Purpose**: Evaluate how accurately the system classifies documents into categories.

**Metrics**:
- Accuracy: Overall classification accuracy
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall
- Throughput: Documents classified per second

**Ground Truth Creation**:
```bash
python benchmarks/create_ground_truth.py --type classification --dataset summit --sample-size 100
```

**Manual Annotation Required**:
1. Review each document's filename and preview text
2. Assign appropriate document type from the provided categories
3. Use "unknown" for documents that don't fit standard categories

**Running the Benchmark**:
```bash
python benchmarks/benchmark_runner.py --task classification --dataset summit --iterations 3
```

### Search & Retrieval Benchmark

**Purpose**: Evaluate document retrieval quality and speed.

**Metrics**:
- Precision@10: Fraction of top 10 results that are relevant
- Recall@10: Fraction of relevant documents found in top 10
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- Throughput: Queries processed per second

**Ground Truth Creation**:
```bash
python benchmarks/create_ground_truth.py --type search --dataset summit --num-queries 50
```

**Manual Annotation Required**:
1. Review candidate documents returned for each query
2. Identify which documents are truly relevant to the query
3. Optionally assign relevance scores (0-3 scale)

**Running the Benchmark**:
```bash
python benchmarks/benchmark_runner.py --task search --dataset summit --iterations 3
```

### Question Answering Benchmark

**Purpose**: Evaluate the quality of AI-generated answers.

**Metrics**:
- Semantic Similarity: Cosine similarity between generated and expected answers
- Answer Length: Average length of generated answers
- Throughput: Questions answered per second

**Ground Truth Creation**:
```bash
python benchmarks/create_ground_truth.py --type qa --dataset summit --num-pairs 30
```

**Manual Annotation Required**:
1. Review automatically generated question-answer pairs
2. Verify answers are accurate and complete
3. Adjust difficulty ratings if needed
4. Remove incorrect or inappropriate pairs

**Running the Benchmark**:
```bash
python benchmarks/benchmark_runner.py --task qa --dataset summit --iterations 3
```

## Performance Metrics Explained

### Classification Metrics

- **Accuracy**: `(Correct Classifications) / (Total Classifications)`
- **Precision**: `(True Positives) / (True Positives + False Positives)`
- **Recall**: `(True Positives) / (True Positives + False Negatives)`
- **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`

### Search Metrics

- **Precision@K**: Fraction of top K results that are relevant
- **Recall@K**: Fraction of all relevant documents found in top K
- **MRR**: `Average(1/rank_first_relevant)` across all queries

### QA Metrics

- **Semantic Similarity**: Measures how close generated answers are to expected answers
- **BLEU/ROUGE**: Traditional NLP metrics for text generation quality

## A/B Testing Different Configurations

### Comparing Embedding Models

```python
# In benchmark_runner.py, modify the embeddings initialization
from sentence_transformers import SentenceTransformer

# Test different models
models_to_test = [
    'all-mpnet-base-v2',      # Current model
    'all-MiniLM-L6-v2',       # Smaller, faster
    'paraphrase-multilingual-mpnet-base-v2'  # Multilingual
]

for model_name in models_to_test:
    embeddings = SentenceTransformer(model_name)
    # Run benchmarks with this model
```

### Comparing Search Strategies

```python
# Test different search configurations
search_configs = [
    {"method": "dense_only", "use_hybrid": False},
    {"method": "hybrid_balanced", "use_hybrid": True, "sparse_weight": 0.5, "dense_weight": 0.5},
    {"method": "sparse_heavy", "use_hybrid": True, "sparse_weight": 0.7, "dense_weight": 0.3}
]

for config in search_configs:
    # Run search benchmarks with different configurations
    results = run_search_benchmark(dataset, config)
```

### Comparing LLM Models

```python
# Test different Claude models
models_to_test = [
    'claude-3-haiku-20240307',    # Fast, cost-effective
    'claude-3-sonnet-20240229',   # Balanced performance
    'claude-3-opus-20240229'      # Highest quality
]

for model_name in models_to_test:
    llm = ChatAnthropic(model=model_name, ...)
    # Run QA and classification benchmarks
```

## Regression Detection and Monitoring

### Setting Up Automated Monitoring

1. **Create Baseline Benchmarks**:
```bash
# Run initial benchmark as baseline
python benchmarks/benchmark_runner.py --task all --dataset summit --iterations 5
# Note the run ID for future comparisons
```

2. **Set Up Regular Benchmarking**:
```bash
# Add to CI/CD pipeline or cron job
#!/bin/bash
RUN_ID="automated_$(date +%Y%m%d_%H%M%S)"
python benchmarks/benchmark_runner.py --task all --dataset summit --iterations 3

# Compare with baseline
python benchmarks/regression_detector.py --baseline-run baseline_run_id --compare-run $RUN_ID --alerts --email-to team@example.com
```

3. **Configure Alert Thresholds**:
```python
# In regression_detector.py, customize thresholds
alert_thresholds = {
    "accuracy": 0.03,  # 3% drop triggers alert
    "precision@10": 0.08,  # 8% drop for search
    "throughput": 0.10   # 10% drop in throughput
}
```

## Performance Optimization Strategies

### Identified from Benchmarks

1. **Batch Processing**: Use optimal batch sizes based on memory availability
2. **Caching Strategy**: Implement multi-level caching for embeddings and documents
3. **Model Selection**: Balance accuracy vs. speed based on use case
4. **Hybrid Search**: Combine sparse and dense retrieval for better results

### Memory Optimization

```python
# Monitor memory usage during benchmarks
from app.core.performance import get_performance_manager

perf_manager = get_performance_manager()
memory_usage = perf_manager.monitor_memory_usage()

if memory_usage['percent'] > 80:
    # Trigger garbage collection
    import gc
    gc.collect()
```

### GPU Acceleration

```python
# Enable GPU acceleration when available
if torch.cuda.is_available():
    device = 'cuda'
    # Move models to GPU
    embeddings = embeddings.to(device)
    cross_encoder = cross_encoder.to(device)
```

## Interpreting Results

### Good Performance Indicators

- **Classification**: Accuracy > 0.85, F1 > 0.80
- **Search**: Precision@10 > 0.70, MRR > 0.60
- **QA**: Semantic similarity > 0.75
- **Throughput**: > 10 queries/second for search, > 5 docs/second for classification

### Common Issues and Solutions

1. **Low Classification Accuracy**:
   - Check ground truth quality
   - Increase training data or fine-tune model
   - Review document preprocessing

2. **Poor Search Recall**:
   - Adjust similarity thresholds
   - Improve embedding quality
   - Add more comprehensive indexing

3. **Slow Performance**:
   - Implement caching
   - Use smaller models
   - Optimize batch sizes
   - Enable GPU acceleration

## Advanced Benchmarking Techniques

### Statistical Significance Testing

```python
from scipy import stats

# Test if performance difference is statistically significant
baseline_scores = [0.85, 0.87, 0.83, 0.86, 0.84]
new_scores = [0.82, 0.79, 0.81, 0.80, 0.83]

t_stat, p_value = stats.ttest_ind(baseline_scores, new_scores)

if p_value < 0.05:
    print("Performance difference is statistically significant")
```

### Confidence Intervals

```python
import numpy as np

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    h = std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
    return mean - h, mean + h

lower, upper = confidence_interval(scores)
print(".3f"```

### Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    # Train on fold training data
    # Test on fold test data
    # Record performance metrics
    fold_scores.append(score)
```

## Integration with CI/CD

### Automated Benchmarking Pipeline

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .

    - name: Run benchmarks
      run: |
        python benchmarks/benchmark_runner.py --task all --dataset summit --iterations 3

    - name: Detect regressions
      run: |
        python benchmarks/regression_detector.py --baseline-run baseline --compare-run ${{ github.run_id }}

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmarks/results/
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
```bash
pip install scipy plotly pandas scikit-learn torch sentence-transformers
```

2. **No GPU Available**:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
```

3. **Out of Memory Errors**:
```python
# Reduce batch sizes
batch_size = min(batch_size, 16)  # Limit to 16

# Enable gradient checkpointing for large models
# model.gradient_checkpointing_enable()
```

4. **Slow Embedding Generation**:
```python
# Use approximate nearest neighbors
# from annoy import AnnoyIndex

# Or reduce embedding dimensions
# embeddings = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
```

## Contributing

When adding new benchmark tasks:

1. Define clear evaluation metrics
2. Create appropriate ground truth datasets
3. Implement automated evaluation functions
4. Add results to the reporting system
5. Update this documentation

## Support

For questions about benchmarking:

1. Check this documentation first
2. Review the code comments in benchmark files
3. Create an issue with benchmark results and error messages
4. Include system information and configuration details
