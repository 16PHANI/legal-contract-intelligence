import csv
import json
import logging
import os
import time
from typing import Dict, List, Optional

from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    DEFAULT_N_CONTRACTS,
    DEFAULT_OUTPUT_DIR,
    GEMINI_MODEL,
    OUTPUT_FIELDS,
    RATE_LIMIT_DELAY,
    SENTINEL_FAILED,
    SENTINEL_SUM_FAILED,
)
from extractor import extract_clauses
from loader import load_contracts
from preprocessor import normalize_text, truncate_for_context
from summarizer import summarize_contract

load_dotenv()
logger = logging.getLogger(__name__)


def init_gemini():
    """Initialize Groq client from GROQ_API_KEY in .env."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "1. Copy .env.example to .env\n"
            "2. Set GROQ_API_KEY=<your_key>\n"
            "   Free key: https://console.groq.com"
        )
    client = Groq(api_key=api_key)
    logger.info(f"Model: {GEMINI_MODEL}")
    return client

def save_results(results, output_dir):
    """Save results to CSV and JSON in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved: {csv_path}")

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON saved: {json_path}")


def run_pipeline(n_contracts=DEFAULT_N_CONTRACTS, pdf_dir=None,
                 output_dir=DEFAULT_OUTPUT_DIR, enable_search=True):
    """Run the full contract analysis pipeline.

    For each contract:
      1. Normalize and truncate text
      2. Extract clauses via Gemini (JSON mode, few-shot)
      3. Generate 100-150 word summary via Gemini
    Then save CSV + JSON and optionally build a semantic search index.
    """
    client = init_gemini()

    logger.info(f"Loading {n_contracts} contracts...")
    contracts = load_contracts(n_contracts=n_contracts, pdf_dir=pdf_dir)
    total = len(contracts)
    logger.info(f"Loaded {total} contracts")

    results = []
    fail_count = 0

    for i, contract in enumerate(tqdm(contracts, desc="Processing", unit="contract")):
        cid = contract["id"]
        raw_text = contract["text"]

        clean = normalize_text(raw_text)
        clean = truncate_for_context(clean)
        logger.info(f"[{i+1}/{total}] {cid[:70]} ({len(clean):,} chars)")

        clauses = extract_clauses(client, clean)
        time.sleep(RATE_LIMIT_DELAY)

        summary = summarize_contract(client, clean)
        time.sleep(RATE_LIMIT_DELAY)

        if any(v == SENTINEL_FAILED for v in clauses.values()) or summary == SENTINEL_SUM_FAILED:
            fail_count += 1
            logger.warning(f"Partial failure: {cid[:70]}")

        results.append({
            "contract_id": cid,
            "summary": summary,
            **clauses,
        })

    save_results(results, output_dir)
    logger.info(f"Done: {total} contracts, {fail_count} failures. Output: {output_dir}/")

    if enable_search:
        try:
            from search import ClauseSearchEngine
            engine = ClauseSearchEngine()
            n_indexed = engine.index(results)
            logger.info(f"Search index built: {n_indexed} clauses")

            demo = "termination without cause or notice"
            demo_hits = engine.search(demo, top_k=3)
            logger.info(f'Demo search: "{demo}"')
            for h in demo_hits:
                score = h["similarity_score"]
                cid_s = h["contract_id"][:55]
                ctype = h["clause_type"]
                logger.info(f"  [{score:.3f}]  {cid_s}  ({ctype})")
        except ImportError as exc:
            logger.warning(f"Search skipped (missing package): {exc}")
        except Exception as exc:
            logger.warning(f"Search skipped: {exc}")

    return results
