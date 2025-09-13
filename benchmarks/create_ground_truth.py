#!/usr/bin/env python3
"""
Ground Truth Creation Tools for dd-poc Benchmarks

This module provides tools to create ground truth datasets for benchmarking
the predictive performance of the dd-poc system.

Ground Truth Types:
1. Document Classification - manually labeled document types
2. Search Relevance - queries with relevant document lists
3. QA Pairs - questions with expected answers

Usage:
    python benchmarks/create_ground_truth.py --type classification --dataset summit --sample-size 100
    python benchmarks/create_ground_truth.py --type search --dataset summit --num-queries 50
    python benchmarks/create_ground_truth.py --type qa --dataset summit --num-pairs 30
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from app.core.config import get_config
from app.core.content_ingestion import ContentIngestion
from app.core.document_processor import DocumentProcessor
from app.core.utils import create_document_processor


class GroundTruthCreator:
    """Creates ground truth datasets for benchmarking"""

    def __init__(self):
        self.config = get_config()
        self.content_ingestion = ContentIngestion()

        # Define document type categories
        self.document_types = [
            "corporate governance",
            "financial statements",
            "legal agreements",
            "intellectual property",
            "human resources",
            "operations",
            "tax documents",
            "insurance",
            "technology",
            "marketing",
            "unknown"
        ]

    def create_classification_ground_truth(self, dataset: str, sample_size: int = 100,
                                         output_file: Optional[str] = None) -> str:
        """Create ground truth for document classification"""
        print(f"🏷️ Creating classification ground truth for {dataset}")

        # Load dataset documents
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Get all PDF files
        pdf_files = list(dataset_path.glob("**/*.pdf"))
        if len(pdf_files) < sample_size:
            sample_size = len(pdf_files)
            print(f"⚠️ Reduced sample size to {sample_size} (available documents)")

        # Sample documents
        sampled_files = random.sample(pdf_files, sample_size)

        ground_truth = {}

        print(f"Processing {sample_size} documents for manual classification...")

        for i, pdf_file in enumerate(sampled_files, 1):
            print(f"📄 [{i}/{sample_size}] {pdf_file.name}")

            try:
                # Extract first page text for classification context
                first_page_text = self._extract_first_page_text(pdf_file)

                doc_info = {
                    "filename": pdf_file.name,
                    "path": str(pdf_file.relative_to(dataset_path.parent.parent)),
                    "full_path": str(pdf_file),
                    "first_page_preview": first_page_text[:500],  # First 500 chars
                    "suggested_type": self._suggest_document_type(pdf_file.name, first_page_text),
                    "document_type": ""  # To be filled manually
                }

                ground_truth[str(pdf_file)] = doc_info

            except Exception as e:
                print(f"❌ Failed to process {pdf_file.name}: {e}")
                continue

        # Save ground truth
        if not output_file:
            output_file = f"benchmarks/ground_truth/{dataset}_classification_gt.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "dataset": dataset,
                "created_at": datetime.now().isoformat(),
                "sample_size": sample_size,
                "document_types": self.document_types,
                "ground_truth": ground_truth,
                "instructions": """
To complete this ground truth dataset:

1. Review each document's filename and first_page_preview
2. Assign the most appropriate document_type from the document_types list
3. Use 'unknown' if the document type cannot be determined
4. Save the file after completing all classifications

Example classifications:
- "Board Meeting Minutes.pdf" -> "corporate governance"
- "Financial Statements Q3.pdf" -> "financial statements"
- "Employment Agreement.pdf" -> "human resources"
- "Patent Application.pdf" -> "intellectual property"
                """
            }, f, indent=2)

        print(f"✅ Classification ground truth saved to {output_path}")
        print(f"📝 Manual classification needed for {len(ground_truth)} documents")

        return str(output_path)

    def create_search_ground_truth(self, dataset: str, num_queries: int = 50,
                                 output_file: Optional[str] = None) -> str:
        """Create ground truth for search relevance"""
        print(f"🔍 Creating search ground truth for {dataset}")

        # Load dataset and processor
        dataset_path = self._get_dataset_path(dataset)
        store_name = f"{dataset.replace('-', '-')}-inc"  # Convert to store name format

        try:
            processor = create_document_processor(store_name=store_name)
        except Exception as e:
            print(f"❌ Failed to create document processor: {e}")
            return ""

        if not processor or not processor.vector_store:
            print("❌ No vector store available for search ground truth creation")
            return ""

        # Generate diverse search queries
        queries = self._generate_search_queries(dataset, num_queries)

        ground_truth = []

        print(f"Processing {num_queries} search queries...")

        for i, query_info in enumerate(queries, 1):
            query = query_info["query"]
            category = query_info["category"]

            print(f"🔍 [{i}/{num_queries}] Query: '{query[:50]}...'")

            try:
                # Search for relevant documents
                search_results = processor.search(query, top_k=20)

                # Get document names for manual relevance judgment
                candidate_docs = []
                for result in search_results:
                    doc_name = result.get('source', result.get('name', 'Unknown'))
                    doc_path = result.get('path', '')
                    preview = result.get('text', '')[:200]

                    candidate_docs.append({
                        "name": doc_name,
                        "path": doc_path,
                        "preview": preview,
                        "search_score": result.get('score', 0)
                    })

                query_gt = {
                    "query": query,
                    "category": category,
                    "candidate_documents": candidate_docs,
                    "relevant_docs": [],  # To be filled manually
                    "relevance_scores": {}  # To be filled manually
                }

                ground_truth.append(query_gt)

            except Exception as e:
                print(f"❌ Failed to process query '{query}': {e}")
                continue

        # Save ground truth
        if not output_file:
            output_file = f"benchmarks/ground_truth/{dataset}_search_gt.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "dataset": dataset,
                "created_at": datetime.now().isoformat(),
                "num_queries": num_queries,
                "ground_truth": ground_truth,
                "instructions": """
To complete this search ground truth dataset:

1. For each query, review the candidate_documents list
2. Identify documents that are truly relevant to the query
3. Add relevant document paths to the relevant_docs list
4. Optionally assign relevance scores (0-3) in relevance_scores dict:
   - 0: Not relevant
   - 1: Somewhat relevant
   - 2: Relevant
   - 3: Highly relevant

Example:
"query": "board meeting minutes",
"relevant_docs": ["/path/to/board_minutes.pdf", "/path/to/corporate_governance.pdf"],
"relevance_scores": {
    "/path/to/board_minutes.pdf": 3,
    "/path/to/corporate_governance.pdf": 2
}
                """
            }, f, indent=2)

        print(f"✅ Search ground truth saved to {output_path}")
        print(f"📝 Manual relevance judgment needed for {len(ground_truth)} queries")

        return str(output_path)

    def create_qa_ground_truth(self, dataset: str, num_pairs: int = 30,
                             output_file: Optional[str] = None) -> str:
        """Create ground truth for question answering"""
        print(f"🤖 Creating QA ground truth for {dataset}")

        # Load dataset documents
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Get some sample documents to generate QA pairs from
        pdf_files = list(dataset_path.glob("**/*.pdf"))[:10]  # Use first 10 docs

        qa_pairs = []

        print(f"Processing {len(pdf_files)} documents for QA pair generation...")

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"📄 [{i}/{len(pdf_files)}] {pdf_file.name}")

            try:
                # Extract text for QA generation
                full_text = self._extract_document_text(pdf_file)
                if not full_text or len(full_text) < 1000:
                    continue

                # Generate QA pairs for this document
                doc_qa_pairs = self._generate_qa_pairs_for_document(pdf_file.name, full_text, num_pairs // len(pdf_files) + 1)

                for qa_pair in doc_qa_pairs:
                    qa_pairs.append({
                        "document": pdf_file.name,
                        "document_path": str(pdf_file),
                        "question": qa_pair["question"],
                        "expected_answer": qa_pair["answer"],
                        "question_type": qa_pair["type"],
                        "difficulty": qa_pair["difficulty"]
                    })

                if len(qa_pairs) >= num_pairs:
                    break

            except Exception as e:
                print(f"❌ Failed to process {pdf_file.name}: {e}")
                continue

        # Trim to requested size
        qa_pairs = qa_pairs[:num_pairs]

        # Save ground truth
        if not output_file:
            output_file = f"benchmarks/ground_truth/{dataset}_qa_gt.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "dataset": dataset,
                "created_at": datetime.now().isoformat(),
                "num_pairs": len(qa_pairs),
                "ground_truth": qa_pairs,
                "instructions": """
This QA ground truth dataset has been automatically generated.
You may need to review and refine the generated questions and answers:

1. Check that questions are clear and answerable from the document
2. Verify that expected answers are accurate and complete
3. Adjust question difficulty ratings if needed
4. Remove any inappropriate or incorrect QA pairs

Question types:
- factual: Questions about specific facts, dates, names
- analytical: Questions requiring analysis or interpretation
- comparative: Questions comparing different aspects
- definitional: Questions about definitions or explanations
                """
            }, f, indent=2)

        print(f"✅ QA ground truth saved to {output_path}")
        print(f"📝 Review and validation needed for {len(qa_pairs)} QA pairs")

        return str(output_path)

    def _get_dataset_path(self, dataset: str) -> Path:
        """Get the path to a dataset"""
        base_path = Path("data/vdrs")

        if dataset == "deepshield":
            return base_path / "industrial-security-leadership" / "deepshield-systems-inc"
        elif dataset == "summit":
            return base_path / "automated-services-transformation" / "summit-digital-solutions-inc"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _extract_first_page_text(self, pdf_path: Path) -> str:
        """Extract text from first page of PDF"""
        try:
            # Use the content ingestion module
            content = self.content_ingestion.extract_text_from_pdf(str(pdf_path))

            # Get first page (assuming content is split by pages)
            if isinstance(content, list) and content:
                return content[0][:1000]  # First 1000 chars of first page
            elif isinstance(content, str):
                return content[:1000]  # First 1000 chars
            else:
                return "No content extracted"

        except Exception as e:
            return f"Error extracting text: {e}"

    def _extract_document_text(self, pdf_path: Path) -> str:
        """Extract full text from PDF"""
        try:
            content = self.content_ingestion.extract_text_from_pdf(str(pdf_path))

            if isinstance(content, list):
                return "\n".join(content)
            elif isinstance(content, str):
                return content
            else:
                return ""

        except Exception as e:
            return f"Error extracting text: {e}"

    def _suggest_document_type(self, filename: str, text: str) -> str:
        """Suggest document type based on filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()

        # Keyword-based suggestions
        type_keywords = {
            "corporate governance": ["board", "meeting", "minutes", "governance", "shareholder", "director"],
            "financial statements": ["financial", "statement", "income", "balance", "cash flow", "audit"],
            "legal agreements": ["agreement", "contract", "legal", "nda", "license", "terms"],
            "intellectual property": ["patent", "trademark", "copyright", "ip", "intellectual property"],
            "human resources": ["employment", "hr", "employee", "salary", "benefits", "handbook"],
            "operations": ["operations", "process", "procedure", "manual", "sop"],
            "tax documents": ["tax", "irs", "taxation", "withholding", "1099"],
            "insurance": ["insurance", "policy", "coverage", "liability"],
            "technology": ["technology", "software", "system", "architecture", "api"],
            "marketing": ["marketing", "brand", "advertising", "campaign"]
        }

        for doc_type, keywords in type_keywords.items():
            if any(keyword in filename_lower or keyword in text_lower for keyword in keywords):
                return doc_type

        return "unknown"

    def _generate_search_queries(self, dataset: str, num_queries: int) -> List[Dict]:
        """Generate diverse search queries for the dataset"""
        # Domain-specific queries based on dataset
        if dataset == "deepshield":
            base_queries = [
                "board meeting minutes",
                "financial statements",
                "intellectual property agreements",
                "employee handbook",
                "corporate governance",
                "technology architecture",
                "security policies",
                "insurance coverage",
                "tax documents",
                "marketing materials",
                "operational procedures",
                "legal agreements",
                "shareholder information",
                "audit reports",
                "patent applications"
            ]
        else:  # summit
            base_queries = [
                "company overview",
                "financial performance",
                "strategic plan",
                "board composition",
                "intellectual property",
                "employee benefits",
                "technology stack",
                "market analysis",
                "legal compliance",
                "operational metrics",
                "corporate structure",
                "risk assessment",
                "competitive analysis",
                "regulatory filings",
                "partnership agreements"
            ]

        # Generate variations and expand to requested size
        queries = []
        categories = ["corporate", "financial", "legal", "technical", "operational", "strategic"]

        for i in range(num_queries):
            base_query = random.choice(base_queries)
            category = random.choice(categories)

            # Add some variation
            variations = [
                base_query,
                f"latest {base_query}",
                f"{base_query} information",
                f"details about {base_query}",
                f"{base_query} documents",
                f"find {base_query}"
            ]

            query = random.choice(variations)

            queries.append({
                "query": query,
                "category": category
            })

        return queries

    def _generate_qa_pairs_for_document(self, doc_name: str, text: str, num_pairs: int) -> List[Dict]:
        """Generate QA pairs for a document"""
        # This is a simplified QA pair generation
        # In practice, you might want to use a more sophisticated NLP model

        qa_pairs = []

        # Extract some basic information for QA generation
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20][:10]

        for sentence in sentences:
            if len(qa_pairs) >= num_pairs:
                break

            # Generate simple factual questions
            if "company" in sentence.lower() or "organization" in sentence.lower():
                qa_pairs.append({
                    "question": "What is the main focus of the company mentioned in this document?",
                    "answer": sentence[:200] + "...",
                    "type": "factual",
                    "difficulty": "easy"
                })

            elif "financial" in sentence.lower() or "revenue" in sentence.lower():
                qa_pairs.append({
                    "question": "What financial information is discussed in this document?",
                    "answer": sentence[:200] + "...",
                    "type": "factual",
                    "difficulty": "medium"
                })

            elif any(word in sentence.lower() for word in ["agreement", "contract", "legal"]):
                qa_pairs.append({
                    "question": "What legal or contractual information is covered in this document?",
                    "answer": sentence[:200] + "...",
                    "type": "factual",
                    "difficulty": "medium"
                })

        # Fill remaining slots with generic questions
        while len(qa_pairs) < num_pairs:
            qa_pairs.append({
                "question": f"What information does this document '{doc_name}' contain?",
                "answer": text[:300] + "...",
                "type": "general",
                "difficulty": "easy"
            })

        return qa_pairs


def main():
    """Main entry point for ground truth creation"""
    parser = argparse.ArgumentParser(description="Create ground truth datasets for dd-poc benchmarks")
    parser.add_argument("--type", choices=["classification", "search", "qa"],
                       required=True, help="Type of ground truth to create")
    parser.add_argument("--dataset", choices=["deepshield", "summit"],
                       required=True, help="Dataset to create ground truth for")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Sample size for classification (default: 100)")
    parser.add_argument("--num-queries", type=int, default=50,
                       help="Number of queries for search ground truth (default: 50)")
    parser.add_argument("--num-pairs", type=int, default=30,
                       help="Number of QA pairs to create (default: 30)")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    try:
        creator = GroundTruthCreator()

        if args.type == "classification":
            output_file = creator.create_classification_ground_truth(
                args.dataset, args.sample_size, args.output
            )
        elif args.type == "search":
            output_file = creator.create_search_ground_truth(
                args.dataset, args.num_queries, args.output
            )
        elif args.type == "qa":
            output_file = creator.create_qa_ground_truth(
                args.dataset, args.num_pairs, args.output
            )

        print("
🎉 Ground truth creation completed!"        print(f"📁 Output file: {output_file}")
        print("\n📝 Next steps:"
        print("1. Review the generated file")
        print("2. Complete manual annotations as needed")
        print("3. Run benchmarks using the completed ground truth")

    except Exception as e:
        print(f"❌ Ground truth creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
