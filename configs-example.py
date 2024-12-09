'''
Contains paths and configurations for the project.

This is a template for configs.py.
Rename this file to configs.py after filling in the paths and configurations.
'''

inf = float('inf')

# Directory for the project containing models and datasets
BASE_DIR = './'

# Directory containing all model configuration directories (in lower case names)
MODELS_DIR = f'{BASE_DIR}/models'

# OpenAI API key for OpenAI models
OPENAI_API_KEY = ''

# Data loading configurations for training and evaluation
MIN_WORDS = 0
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
SYSTEM_PROMPT = 'Your task is to summarize a very long document, given some of its segments.'

# Extra stop words for keywords extraction
EXTRA_STOP_WORDS = []

# Model generation configurations
MIN_SUMMARY_TOKENS = 0
TEMPERATURE = 1.
REPETITION_PENALTY = 1.
TOP_P = .95

# Model training configurations
LEARNING_RATE = 1e-3
SCHEDULER_FACTOR = .1
SCHEDULER_PATIENCE = 5

# GPU usage tolerance (in MiB)
GPU_USAGE_TOLERANCE = 1000

# Seed for reproducibility, set to None for random seed
SEED = None

# Float precision
FLT_PREC = 4

# Numeber of spaces to clear stdout
SPACES = 100
