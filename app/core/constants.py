# Constants for the application

# Chunk sizes
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Thresholds (Legacy - kept for backward compatibility)
SIMILARITY_THRESHOLD = 0.2
RELEVANCY_THRESHOLD = 0.25

# Statistical Filtering Configuration
STATISTICAL_CANDIDATE_POOL_SIZE = 30    # How many candidates to analyze for statistics
STATISTICAL_STD_MULTIPLIER = 1.5        # Standard deviations above mean (1.0=loose, 2.0=strict)
STATISTICAL_MIN_CANDIDATES = 5          # Minimum candidates needed for statistical analysis
STATISTICAL_MIN_STD_DEV = 0.05          # Minimum standard deviation to prevent tiny variance issues

# Token limits
CLASSIFICATION_MAX_TOKENS = 1000
QA_MAX_TOKENS = 8000
CHECKLIST_PARSING_MAX_TOKENS = 16000  # Large enough for full checklist parsing

# AI Model Configuration
TEMPERATURE = 0.0  # Deterministic responses for due diligence consistency

# Batch sizes
DEFAULT_BATCH_SIZE = 10
CLASSIFICATION_BATCH_SIZE = 20

# AI Analysis types
SUPPORTED_ANALYSIS_TYPES = ["overview", "strategic", "checklist", "questions"]
