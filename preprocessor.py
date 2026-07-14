import logging
import re
import unicodedata

from config import MAX_CONTRACT_CHARS

logger = logging.getLogger(__name__)

_SOFT_HYPHEN   = re.compile(r"-\s*\n\s*([a-z])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE   = re.compile(r"[ \t]+")
_TRAILING_SP   = re.compile(r" +\n")
_EXCESS_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text):
    """Normalize raw contract text for LLM processing."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _SOFT_HYPHEN.sub(r"\1", text)      # repair: "indemni-\nfication" -> "indemnification"
    text = _MULTI_SPACE.sub(" ", text)
    text = _TRAILING_SP.sub("\n", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)
    return text.strip()


def truncate_for_context(text, max_chars=MAX_CONTRACT_CHARS):
    """Truncate to max_chars to stay within the Gemini context window.

    Gemini 1.5 Flash: 1M token context. 800k chars ~ 200k tokens, which
    comfortably covers the longest CUAD contracts (typically < 150k chars).
    """
    if len(text) <= max_chars:
        return text
    logger.warning(f"Contract truncated: {len(text):,} -> {max_chars:,} chars")
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.9:
        truncated = truncated[:last_period + 1]
    return truncated


def preprocess(text):
    """Full pipeline: normalize then truncate."""
    return truncate_for_context(normalize_text(text))
