"""
Contains paths and configurations for the project.
"""

inf = float("inf")

# Directory for the project containing models and datasets
BASE_DIR = "/Users/naman/Workspace/Data/Long-Document-Summarizer"
# BASE_DIR = "/home/nchibbar/Data"

# Directory containing all model configuration directories (in lower case names)
MODELS_DIR = f"{BASE_DIR}/models"

# Data loading configurations for training and evaluation
MIN_WORDS = 20_000
MAX_WORDS = inf
MAX_TEXTS = inf

# Text segmenter configuration
SEGMENT_MIN_WORDS = 20

# Encoder configurations
MIN_TOKEN_FRAC = .5
HEAD_SIZE = .5
THRESHOLD = .8
PROB_BOOST = .03
NUM_KEYWORDS = 20
SYSTEM_PROMPT = "You will be given some segments of a very long document. \
	Your task is to summarize the entire document as a whole by extracting key \
	information and ideas from the segments. Generate a detailed, concise, and \
	coherent summary in 300 words. Do not refer to the document in the summary in any way."

# Extra stop words for keywords extraction
EXTRA_STOP_WORDS = [
	"also", "however", "therefore", "thus", "hence", "moreover",
	"must", "may", "might", "could", "would", "shall", "need",
	"needs", "given", "since", "though",
]

# Model generation configurations
MIN_SUMMARY_TOKENS = 300
TEMPERATURE = 2.
REPETITION_PENALTY = 3.
TOP_P = .95

# Model training configurations
LEARNING_RATE = 1e-3
SCHEDULER_FACTOR = .1
SCHEDULER_PATIENCE = 5

# GPU usage tolerance (in MiB)
GPU_USAGE_TOLERANCE = 1000

# Seed for reproducibility
SEED = 69

# Float precision
FLT_PREC = 4

# Numeber of spaces to clear stdout
SPACES = 100
