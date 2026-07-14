import json
import logging
import time
from typing import Any, Dict

from config import EXTRACT_TEMPERATURE, GEMINI_MODEL, MAX_RETRIES, SENTINEL_FAILED
from prompts import CLAUSE_EXTRACTION_PROMPT, fill

logger = logging.getLogger(__name__)

EXPECTED_KEYS = {"termination_clause", "confidentiality_clause", "liability_clause"}

EMPTY_RESULT = {
    "termination_clause":     SENTINEL_FAILED,
    "confidentiality_clause": SENTINEL_FAILED,
    "liability_clause":       SENTINEL_FAILED,
}


def _parse_response(raw):
    """Parse and validate JSON from LLM response."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    missing = EXPECTED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    return {k: str(data[k]).strip() for k in EXPECTED_KEYS}


def extract_clauses(client, contract_text, max_retries=MAX_RETRIES):
    """Extract termination, confidentiality, and liability clauses using Groq."""
    prompt = fill(CLAUSE_EXTRACTION_PROMPT, contract_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=EXTRACT_TEMPERATURE,
            )
            result = _parse_response(response.choices[0].message.content)
            logger.debug(f"Extraction succeeded on attempt {attempt}")
            return result

        except json.JSONDecodeError as exc:
            logger.warning(f"[{attempt}/{max_retries}] JSON parse error: {exc}")
        except ValueError as exc:
            logger.warning(f"[{attempt}/{max_retries}] Validation error: {exc}")
        except Exception as exc:
            logger.warning(f"[{attempt}/{max_retries}] API error: {exc}")

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    logger.error("Clause extraction failed after all retries")
    return EMPTY_RESULT