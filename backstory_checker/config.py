"""Paths and training choices for the backstory checker."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NOVEL_DIR = DATA_DIR / "novels"
CACHE_DIR = DATA_DIR / "cache"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"

USER_AGENT = "backstory-checker/1.0 (educational; public-domain novels)"

# Plain-text Project Gutenberg editions. Both books are public domain in the US.
BOOKS = {
    "In Search of the Castaways": {
        "filename": "in_search_of_the_castaways.txt",
        "urls": [
            "https://www.gutenberg.org/cache/epub/2083/pg2083.txt",
            "https://www.gutenberg.org/ebooks/2083.txt.utf-8",
        ],
        "snippets": ["Paganel", "Thalcave", "Ayrton"],
        "source": "Project Gutenberg ebook 2083",
    },
    "The Count of Monte Cristo": {
        "filename": "the_count_of_monte_cristo.txt",
        "urls": [
            "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
            "https://www.gutenberg.org/ebooks/1184.txt.utf-8",
        ],
        "snippets": ["Faria", "Noirtier"],
        "source": "Project Gutenberg ebook 1184",
    },
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_MODEL = "microsoft/deberta-v3-small"

# Retrieval reads the whole novel and keeps the five closest passages.
TOP_K = 5
CHUNK_WORDS = 60
OVERLAP_WORDS = 15
# The classifier reads a short head of each of those five passages.
# This CPU has no GPU: a 512-token update took about 3.5 minutes.
MODEL_PASSAGE_WORDS = 8
MAX_LENGTH = 96

# Six epochs is still a long fine-tune for 64 rows. The learning rate stays
# below the usual 2e-5 default so the extra passes do not jump into overfit.
EPOCHS = 6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
BATCH_SIZE = 4
SEED = 42
TEST_SIZE = 0.2

LABEL2ID = {"contradict": 0, "consistent": 1}
ID2LABEL = {0: "contradict", 1: "consistent"}
