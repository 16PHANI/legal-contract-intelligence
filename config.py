import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL        = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EXTRACT_TEMPERATURE = float(os.getenv("EXTRACT_TEMP", "0.1"))
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMP", "0.3"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
RATE_LIMIT_DELAY    = float(os.getenv("GEMINI_RATE_DELAY", "4"))

DEFAULT_N_CONTRACTS = int(os.getenv("N_CONTRACTS", "50"))

MAX_CONTRACT_CHARS  = int(os.getenv("MAX_CONTRACT_CHARS", "30000"))
SUMMARY_MIN_WORDS   = 100
SUMMARY_MAX_WORDS   = 150

DEFAULT_OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "output")

SENTINEL_NOT_FOUND  = "NOT_FOUND"
SENTINEL_FAILED     = "EXTRACTION_FAILED"
SENTINEL_SUM_FAILED = "SUMMARIZATION_FAILED"

OUTPUT_FIELDS = [
    "contract_id",
    "summary",
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
]