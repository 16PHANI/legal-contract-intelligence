import logging
import time
from typing import Any

from config import GEMINI_MODEL, MAX_RETRIES, SENTINEL_SUM_FAILED, SUMMARY_MAX_WORDS, SUMMARY_MIN_WORDS, SUMMARY_TEMPERATURE
from prompts import SUMMARY_PROMPT, fill

logger = logging.getLogger(__name__)


def _word_count(text):
    return len(text.split())


def summarize_contract(client, contract_text, max_retries=MAX_RETRIES):
    """Generate a 100-150 word executive summary of the contract."""
    prompt = fill(SUMMARY_PROMPT, contract_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=SUMMARY_TEMPERATURE,
            )
            summary = response.choices[0].message.content.strip()
            wc = _word_count(summary)
            if not (SUMMARY_MIN_WORDS <= wc <= SUMMARY_MAX_WORDS):
                logger.warning(f"Summary word count {wc} outside target [{SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS}]")
            logger.debug(f"Summary: {wc} words (attempt {attempt})")
            return summary

        except Exception as exc:
            logger.warning(f"[{attempt}/{max_retries}] Summarization error: {exc}")
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

    logger.error("Summarization failed after all retries")
    return SENTINEL_SUM_FAILED