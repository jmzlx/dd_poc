#!/usr/bin/env python3
"""
Stage-based Build System for FAISS Index Generation

This module provides a stage-based build system that allows for incremental
builds, dependency management, and smart skipping of completed stages.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import glob

logger = logging.getLogger(__name__)

# Stage definitions with dependencies and outputs
STAGES = {
    'scan': {
        'name': 'Document Scanning',
        'description': 'Scan and catalog all documents',
        'dependencies': [],
        'outputs': ['.scan_cache.json'],
        'estimated_duration': '30s'
    },
    'extract': {
        'name': 'Text Extraction',
        'description': 'Extract text from PDFs and documents',
        'dependencies': ['scan'],
        'outputs': ['.extraction_cache.json'],
        'estimated_duration': '5-10m'
    },
    'classify': {
        'name': 'Document Classification',
        'description': 'Classify document types using AI and generate embeddings',
        'dependencies': ['extract'],
        'outputs': ['*_document_types.json', '*_document_type_embeddings.pkl'],
        'estimated_duration': '3-5m'
    },
    'chunk': {
        'name': 'Text Chunking',
        'description': 'Split documents into semantic chunks',
        'dependencies': ['extract'],
        'outputs': ['.chunking_cache.json'],
        'estimated_duration': '2-3m'
    },
    'embed': {
        'name': 'Vector Embeddings',
        'description': 'Generate embeddings for all chunks',
        'dependencies': ['chunk'],
        'outputs': ['*.pkl'],
        'estimated_duration': '5-8m'
    },
    'index': {
        'name': 'FAISS Indexing',
        'description': 'Build and save FAISS vector indices',
        'dependencies': ['embed'],
        'outputs': ['*.faiss'],
        'estimated_duration': '1-2m'
    },
    'sparse': {
        'name': 'BM25 Sparse Indexing',
        'description': 'Build BM25 sparse indices for hybrid search',
        'dependencies': ['extract'],
        'outputs': ['*_bm25.pkl'],
        'estimated_duration': '2-3m'
    }
}


class StageTracker:
    """Tracks the state and completion status of build stages"""

    def __init__(self, faiss_dir: Path):
        self.faiss_dir = faiss_dir
        self.state_file = faiss_dir / '.build_state.json'
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load current build state from disk"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted state file, starting fresh: {e}")
                return self._create_initial_state()
        else:
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial state structure"""
        return {
            'stages': {},
            'last_build': None,
            'version': '1.0',
            'total_builds': 0
        }

    def _save_state(self):
        """Save current state to disk"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if stage is complete and all outputs exist"""
        if stage_name not in self.state['stages']:
            return False

        stage_info = self.state['stages'][stage_name]
        stage_config = STAGES[stage_name]

        # Check if all output files exist
        for output_pattern in stage_config['outputs']:
            pattern_path = self.faiss_dir / output_pattern
            if not glob.glob(str(pattern_path)):
                logger.debug(f"Missing output: {pattern_path}")
                return False

        return True

    def mark_stage_complete(self, stage_name: str, metadata: dict = None):
        """Mark stage as completed with metadata"""
        self.state['stages'][stage_name] = {
            'completed_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_state()

    def mark_stage_failed(self, stage_name: str, error: str):
        """Mark stage as failed"""
        self.state['stages'][stage_name] = {
            'failed_at': datetime.now().isoformat(),
            'error': error,
            'status': 'failed'
        }
        self._save_state()

    def should_skip_stage(self, stage_name: str, force_clean: bool) -> bool:
        """Determine if stage should be skipped"""
        if force_clean:
            return False
        return self.is_stage_complete(stage_name)

    def get_stage_status(self, stage_name: str) -> Dict[str, Any]:
        """Get detailed status of a stage"""
        if stage_name not in self.state['stages']:
            return {'status': 'not_started'}

        stage_info = self.state['stages'][stage_name]
        is_complete = self.is_stage_complete(stage_name)

        return {
            'status': 'completed' if is_complete else 'incomplete',
            'completed_at': stage_info.get('completed_at'),
            'metadata': stage_info.get('metadata', {}),
            'error': stage_info.get('error'),
            'is_complete': is_complete
        }

    def get_build_summary(self) -> Dict[str, Any]:
        """Get summary of current build state"""
        completed_stages = []
        incomplete_stages = []
        failed_stages = []

        for stage_name in STAGES.keys():
            status = self.get_stage_status(stage_name)
            if status['status'] == 'completed':
                completed_stages.append(stage_name)
            elif status.get('error'):
                failed_stages.append(stage_name)
            else:
                incomplete_stages.append(stage_name)

        return {
            'completed_stages': completed_stages,
            'incomplete_stages': incomplete_stages,
            'failed_stages': failed_stages,
            'last_build': self.state.get('last_build'),
            'total_builds': self.state.get('total_builds', 0)
        }

    def reset_stage(self, stage_name: str):
        """Reset a specific stage to not started"""
        if stage_name in self.state['stages']:
            del self.state['stages'][stage_name]
            self._save_state()

    def reset_all_stages(self):
        """Reset all stages to not started"""
        self.state['stages'] = {}
        self._save_state()


class StageManager:
    """Manages execution of build stages with dependency resolution"""

    def __init__(self, faiss_dir: Path):
        self.faiss_dir = faiss_dir
        self.tracker = StageTracker(faiss_dir)

    def resolve_dependencies(self, target_stages: List[str], completed_stages: Set[str]) -> List[str]:
        """Resolve which stages need to run based on dependencies"""
        to_run = []

        for stage_name in target_stages:
            if stage_name not in STAGES:
                raise ValueError(f"Unknown stage: {stage_name}")

            # Check dependencies recursively
            for dep in STAGES[stage_name]['dependencies']:
                if dep not in completed_stages:
                    dep_chain = self.resolve_dependencies([dep], completed_stages)
                    to_run.extend(dep_chain)

            if stage_name not in completed_stages:
                to_run.append(stage_name)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for stage in to_run:
            if stage not in seen:
                seen.add(stage)
                result.append(stage)

        return result

    def get_completed_stages(self, force_clean: bool = False) -> Set[str]:
        """Get set of completed stages"""
        if force_clean:
            return set()

        completed = set()
        for stage_name in STAGES.keys():
            if self.tracker.is_stage_complete(stage_name):
                completed.add(stage_name)
        return completed

    def execute_stage(self, stage_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific stage - to be implemented by subclasses"""
        raise NotImplementedError(f"Stage execution not implemented for: {stage_name}")

    def run_build_pipeline(self, target_stages: Optional[List[str]] = None,
                          force_clean: bool = False) -> Dict[str, Any]:
        """Run the build pipeline with dependency resolution"""

        # Default to all stages if none specified
        if target_stages is None:
            target_stages = list(STAGES.keys())

        # Get completed stages
        completed_stages = self.get_completed_stages(force_clean)

        # Resolve which stages need to run
        stages_to_run = self.resolve_dependencies(target_stages, completed_stages)

        logger.info(f"Build pipeline: {len(stages_to_run)} stages to execute")

        results = []
        for stage_name in stages_to_run:
            stage_config = STAGES[stage_name]

            if self.tracker.should_skip_stage(stage_name, force_clean):
                logger.info(f"⏭️  Skipping stage '{stage_name}' (already complete)")
                results.append({
                    'stage': stage_name,
                    'status': 'skipped',
                    'reason': 'already_complete'
                })
                continue

            logger.info(f"🚀 Executing stage '{stage_name}': {stage_config['description']}")
            start_time = time.time()

            try:
                # Execute the stage
                result = self.execute_stage(stage_name, force_clean=force_clean)

                # Mark as complete
                execution_time = time.time() - start_time
                self.tracker.mark_stage_complete(stage_name, {
                    'execution_time': execution_time,
                    'result': result
                })

                logger.info(f"✅ Stage '{stage_name}' completed in {execution_time:.1f}s")
                results.append({
                    'stage': stage_name,
                    'status': 'completed',
                    'execution_time': execution_time,
                    'result': result
                })

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Stage '{stage_name}' failed after {execution_time:.1f}s: {e}"
                logger.error(f"❌ {error_msg}")

                self.tracker.mark_stage_failed(stage_name, str(e))

                results.append({
                    'stage': stage_name,
                    'status': 'failed',
                    'execution_time': execution_time,
                    'error': str(e)
                })

                # Don't continue with dependent stages on failure
                break

        # Update build metadata
        self.tracker.state['last_build'] = datetime.now().isoformat()
        self.tracker.state['total_builds'] = self.tracker.state.get('total_builds', 0) + 1
        self.tracker._save_state()

        return {
            'success': all(r['status'] in ['completed', 'skipped'] for r in results),
            'stages_executed': len([r for r in results if r['status'] == 'completed']),
            'stages_skipped': len([r for r in results if r['status'] == 'skipped']),
            'stages_failed': len([r for r in results if r['status'] == 'failed']),
            'results': results,
            'total_time': sum(r.get('execution_time', 0) for r in results)
        }
