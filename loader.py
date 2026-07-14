import json
import logging
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from config import DEFAULT_N_CONTRACTS

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install PyMuPDF")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    pages = []
    for page_num, page in enumerate(doc, start=1):
        try:
            pages.append(page.get_text("text"))
        except Exception as exc:
            logger.warning(f"Page {page_num} error in {pdf_path}: {exc}")
    doc.close()
    return "\n".join(pages)


def load_from_pdf_directory(pdf_dir, n_contracts=DEFAULT_N_CONTRACTS):
    """Load contracts from a local directory of PDF files."""
    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))[:n_contracts]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")

    contracts = []
    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(str(pdf_file))
            contracts.append({
                "id": pdf_file.stem,
                "text": text,
                "source": "local_pdf",
                "ground_truth_answers": {},
            })
            logger.info(f"Loaded: {pdf_file.name} ({len(text):,} chars)")
        except Exception as exc:
            logger.error(f"Skipping {pdf_file.name}: {exc}")

    logger.info(f"Loaded {len(contracts)}/{len(pdf_files)} PDFs")
    return contracts


def _clean_id(title):
    """Convert CUAD contract title to a safe identifier."""
    stem = re.sub(r"\.(pdf|txt)$", "", title, flags=re.IGNORECASE)
    clean = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return clean[:100]


def _keyword_matches(question, keywords):
    q = question.lower()
    return any(kw in q for kw in keywords)


CUAD_CLAUSE_KEYWORDS = {
    "termination_clause":     ["terminat"],
    "confidentiality_clause": ["confidential"],
    "liability_clause":       ["liabilit", "indemnif", "cap on liability"],
}

CUAD_CACHE_DIR  = Path.home() / ".cache" / "legal_contract_intelligence"
CUAD_CACHE_PATH = CUAD_CACHE_DIR / "CUAD_v1.json"

# Direct download URL - CUAD_v1.json hosted on HuggingFace (theatticusproject/cuad repo)
CUAD_JSON_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad"
    "/resolve/main/CUAD_v1/CUAD_v1.json"
)


def _download_cuad_json():
    """Download CUAD_v1.json directly from HuggingFace (theatticusproject/cuad repo)."""
    CUAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            sys.stdout.write(f"\r  Downloading CUAD_v1.json: {pct}%  ")
            sys.stdout.flush()

    logger.info(f"Downloading CUAD_v1.json (~40 MB) from HuggingFace...")
    try:
        urllib.request.urlretrieve(CUAD_JSON_URL, CUAD_CACHE_PATH, reporthook=_progress)
        print()
        logger.info(f"Cached at {CUAD_CACHE_PATH}")
    except Exception as exc:
        if CUAD_CACHE_PATH.exists():
            CUAD_CACHE_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed: {exc}\n"
            f"Manually download CUAD_v1.json from:\n"
            f"  https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1\n"
            f"and place it at: {CUAD_CACHE_PATH}"
        ) from exc


def _load_from_json_cache(n_contracts):
    """Load from cached CUAD_v1.json (SQuAD format)."""
    logger.info(f"Using cached CUAD dataset: {CUAD_CACHE_PATH}")
    logger.info("Parsing CUAD_v1.json...")
    with open(CUAD_CACHE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    contracts = []
    for item in data.get("data", []):
        if len(contracts) >= n_contracts:
            break
        title = item.get("title", f"contract_{len(contracts)}")
        paragraphs = item.get("paragraphs", [])
        if not paragraphs:
            continue
        full_text = "\n\n".join(p.get("context", "") for p in paragraphs if p.get("context"))
        gt = {"termination_clause": [], "confidentiality_clause": [], "liability_clause": []}
        for para in paragraphs:
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                for clause_type, keywords in CUAD_CLAUSE_KEYWORDS.items():
                    if _keyword_matches(question, keywords) and answers:
                        gt[clause_type].extend(answers)
        contracts.append({
            "id": _clean_id(title),
            "text": full_text,
            "source": "cuad_v1_json",
            "ground_truth_answers": gt,
        })
    return contracts


def load_from_huggingface(n_contracts=DEFAULT_N_CONTRACTS):
    """Load unique contracts from CUAD dataset.

    Priority:
      1. Cached CUAD_v1.json at ~/.cache/legal_contract_intelligence/CUAD_v1.json
      2. HuggingFace datasets library (requires datasets<3.0.0 with trust_remote_code)

    To manually supply the data: download CUAD_v1.zip from
      https://github.com/TheAtticusProject/cuad
    extract CUAD_v1.json and place it at the cache path above.
    """
    # Priority 1: use manually cached JSON
    if CUAD_CACHE_PATH.exists():
        try:
            contracts = _load_from_json_cache(n_contracts)
            logger.info(f"Loaded {len(contracts)} contracts from cache")
            return contracts
        except Exception as exc:
            logger.warning(f"Cache load failed ({exc}), falling back to HuggingFace")

    # Priority 2: direct download of CUAD_v1.json from HuggingFace
    logger.info("No cached data found. Downloading CUAD_v1.json...")
    _download_cuad_json()
    contracts = _load_from_json_cache(n_contracts)
    logger.info(f"Loaded {len(contracts)} unique contracts from CUAD")
    return contracts


def load_contracts(n_contracts=DEFAULT_N_CONTRACTS, pdf_dir=None):
    """Unified entry point: load from local PDFs or CUAD dataset."""
    if pdf_dir and os.path.isdir(pdf_dir):
        logger.info(f"Source: local PDF directory -> {pdf_dir}")
        return load_from_pdf_directory(pdf_dir, n_contracts)
    logger.info("Source: CUAD dataset")
    return load_from_huggingface(n_contracts)