# Configuration file for LIVE benchmark evaluation
# Copy this file to config.py and fill in your actual values

# VLM API Configuration
API_KEY = "your-api-key-here"
BASE_URL = "https://your-vlm-api.com/v1"
MODEL_NAME = "your-vision-language-model-name"

# Optional: Default timeout for API calls (seconds)
TIMEOUT = 60.0
CONNECT_TIMEOUT = 10.0

# Evaluation settings
MAX_RETRIES = 3  # Maximum retries for failed API calls
INITIAL_DELAY = 2  # Initial delay between retries (seconds)
TEMPERATURE = 0  # Temperature for model responses (0 for deterministic)
MAX_TOKENS = 4096  # Maximum tokens in model response

# Batch processing settings
DEFAULT_NUM_WORKERS = 20  # Number of concurrent workers
DEFAULT_BATCH_SIZE = 5000  # Process entries in batches
DEFAULT_CHECKPOINT_INTERVAL = 50  # Save checkpoint every N entries
DEFAULT_BATCH_DELAY = 1  # Seconds between batches

# Image settings
MAX_IMAGE_SIZE = 2048  # Maximum image size in pixels (for resizing if needed)
DEFAULT_IMAGE_DIR = "/ossfs/workspace/LIVE_benchmark/data/COCO/val2014"

# Model instructions
DEFAULT_INSTRUCTION = "Answer with 'Yes' or 'No' first, and then provide your reasoning."

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_FULL_RESPONSES = True  # Whether to save full model responses