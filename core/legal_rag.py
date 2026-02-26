"""
core/legal_rag.py

Local RAG layer using sentence-transformers.

Design principles:
- Fully offline.
- No API keys.
- No billing.
- Deterministic retrieval.
- Clean architecture for portfolio use.

We embed legal chunks once at initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import re
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    metadata: Dict[str, str]
    content: str
    embedding: np.ndarray


class LegalRAG:

    def __init__(
        self,
        file_path: str,
        cache_path: str = "data/vector_store/article_39_embeddings.json",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.file_path = file_path
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)
        self.chunks: List[LegalChunk] = []

        self._load_chunks()

    # ---------------------------------------------------------
    # Structured file parsing
    # ---------------------------------------------------------

    def _parse_chunks(self, raw_text: str) -> List[Dict]:
        """
        Parse structured legal file into chunk dictionaries.

        Relies on explicit [CHUNK_ID: ...] markers.
        """
        pattern = r"\[CHUNK_ID: (.*?)\](.*?)((?=\[CHUNK_ID:)|\Z)"
        matches = re.findall(pattern, raw_text, re.DOTALL)

        parsed = []

        for chunk_id, body, _ in matches:
            metadata = {}
            content_lines = []

            for line in body.strip().split("\n"):
                if line.startswith("[ARTICLE:"):
                    metadata["ARTICLE"] = line.replace("[ARTICLE:", "").replace("]", "").strip()
                elif line.startswith("[SECTION:"):
                    metadata["SECTION"] = line.replace("[SECTION:", "").replace("]", "").strip()
                elif line.startswith("[TOPICS:"):
                    metadata["TOPICS"] = line.replace("[TOPICS:", "").replace("]", "").strip()
                else:
                    content_lines.append(line)

            content = "\n".join(content_lines).strip()

            parsed.append({
                "chunk_id": chunk_id.strip(),
                "metadata": metadata,
                "content": content
            })

        return parsed

    # ---------------------------------------------------------
    # Embedding logic
    # ---------------------------------------------------------

    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding using local sentence-transformers model.
        """
        return self.model.encode(text).tolist()

    # ---------------------------------------------------------
    # Load + cache embeddings
    # ---------------------------------------------------------

    def _load_chunks(self) -> None:

        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        parsed_chunks = self._parse_chunks(raw_text)

        # Load cache if exists
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
            except json.JSONDecodeError:
                cached_data = {}
        else:
            cached_data = {}

        updated_cache = False

        for item in parsed_chunks:
            chunk_id = item["chunk_id"]

            if chunk_id in cached_data:
                embedding = cached_data[chunk_id]
            else:
                print(f"Embedding chunk locally: {chunk_id}")
                embedding = self._embed_text(item["content"])
                cached_data[chunk_id] = embedding
                updated_cache = True

            self.chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    metadata=item["metadata"],
                    content=item["content"],
                    embedding=np.array(embedding)
                )
            )

        # Save cache only if new embeddings were created
        if updated_cache:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_data, f)

    # ---------------------------------------------------------
    # Similarity + retrieval
    # ---------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def retrieve(self, query: str, k: int = 3) -> List[LegalChunk]:
        """
        Retrieve top-k most relevant chunks.
        """
        query_embedding = np.array(self._embed_text(query))

        scored = []

        for chunk in self.chunks:
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored[:k]]