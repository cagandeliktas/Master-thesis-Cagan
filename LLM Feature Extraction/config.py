from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LLAMA_SERVER_URL = "http://127.0.0.1:8080"


LACTATE_KEYWORDS = ["lactate", "lactic"]
CHUNK_SENTENCE_SIZE = 5
CHUNK_SENTENCE_OVERLAP = 1
MAX_TOKENS = 200
TEMPERATURE = 0.0