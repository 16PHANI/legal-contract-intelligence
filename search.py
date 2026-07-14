import logging
from typing import Dict, List, Optional

from config import SENTINEL_FAILED, SENTINEL_NOT_FOUND, SENTINEL_SUM_FAILED

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]
SKIP_VALUES = {SENTINEL_NOT_FOUND, SENTINEL_FAILED, SENTINEL_SUM_FAILED, ""}


class ClauseSearchEngine:
    """Semantic search over extracted legal clauses using sentence-transformers + ChromaDB.

    Embeds all non-sentinel clauses with all-MiniLM-L6-v2 (runs locally, no API cost)
    and indexes them in ChromaDB with cosine similarity for retrieval.
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers chromadb")

        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self._embedder = SentenceTransformer(EMBED_MODEL)
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name="contract_clauses",
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_count = 0

    def index(self, results):
        """Embed and index all non-sentinel clauses from pipeline results."""
        documents, embeddings_list, metadatas, ids = [], [], [], []

        for row in results:
            contract_id = row.get("contract_id", "unknown")
            for clause_type in CLAUSE_TYPES:
                text = row.get(clause_type, "").strip()
                if not text or text in SKIP_VALUES:
                    continue
                doc_id = f"{contract_id}__{clause_type}"
                emb = self._embedder.encode(text, show_progress_bar=False,
                                            normalize_embeddings=True).tolist()
                documents.append(text)
                embeddings_list.append(emb)
                metadatas.append({"contract_id": contract_id, "clause_type": clause_type})
                ids.append(doc_id)

        if not documents:
            logger.warning("No valid clauses to index")
            return 0

        self._collection.add(documents=documents, embeddings=embeddings_list,
                             metadatas=metadatas, ids=ids)
        self._indexed_count = len(documents)
        logger.info(f"Indexed {self._indexed_count} clauses")
        return self._indexed_count

    def search(self, query, top_k=5, clause_type=None):
        """Find clauses semantically similar to query.

        Returns list of dicts: contract_id, clause_type, clause_text, similarity_score.
        """
        if self._indexed_count == 0:
            raise RuntimeError("Index empty. Call engine.index(results) first.")
        if clause_type and clause_type not in CLAUSE_TYPES:
            raise ValueError(f"Invalid clause_type. Choose from: {CLAUSE_TYPES}")

        query_emb = self._embedder.encode(query, show_progress_bar=False,
                                          normalize_embeddings=True).tolist()
        where_filter = {"clause_type": clause_type} if clause_type else None
        n_results = min(top_k, self._indexed_count)

        raw = self._collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            similarity = round(1.0 - float(dist) / 2.0, 4)
            output.append({
                "contract_id": meta["contract_id"],
                "clause_type": meta["clause_type"],
                "clause_text": doc,
                "similarity_score": similarity,
            })
        return output