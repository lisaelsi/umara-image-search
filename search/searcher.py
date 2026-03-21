"""
FAISS-backed vector store for image embeddings.

On-disk layout (under VECTOR_STORE_DIR):
  index.faiss   — FAISS IndexIDMap wrapping a flat inner-product index
  meta.json     — { image_id: { filename, relative_path, file_size, ... } }
  id_map.json   — { str(int64_id): image_id }   (inverse of _id_int)

ID scheme:
  Each image gets a stable int64 derived from the first 8 bytes of its
  MD5 hash. Collision probability across 3 000 images is ~10^-13 — safe.

Why FAISS over ChromaDB here:
  - Zero heavy native dependencies (no onnxruntime, no server)
  - Flat inner-product search with L2-normalised vectors = cosine similarity
  - Reads/writes are file-level; trivially portable between environments
  - In production: swap for pgvector (already in your infra) or Milvus
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config

VECTOR_STORE_DIR = Path(config.CHROMA_PERSIST_DIR)  # reuse the env var name
INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
META_FILE = VECTOR_STORE_DIR / "meta.json"
ID_MAP_FILE = VECTOR_STORE_DIR / "id_map.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    image_id: str
    filename: str
    relative_path: str
    score: float           # cosine similarity ∈ [-1, 1], higher is better
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Store state (module-level, loaded once)
# ---------------------------------------------------------------------------

class _Store:
    """Lazily loaded, write-through vector store."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._meta: dict[str, dict] = {}      # image_id → metadata dict
        self._id_map: dict[int, str] = {}     # int64 → image_id

    # -- persistence ---------------------------------------------------------

    def _ensure_dir(self) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        self._ensure_dir()
        if INDEX_FILE.exists():
            self._index = faiss.read_index(str(INDEX_FILE))
        # else: index created on first upsert when we know the dimension
        if META_FILE.exists():
            self._meta = json.loads(META_FILE.read_text())
        if ID_MAP_FILE.exists():
            self._id_map = {int(k): v for k, v in json.loads(ID_MAP_FILE.read_text()).items()}

    def _save(self) -> None:
        self._ensure_dir()
        if self._index is not None:
            faiss.write_index(self._index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(self._meta, ensure_ascii=False))
        ID_MAP_FILE.write_text(json.dumps({str(k): v for k, v in self._id_map.items()}))

    # -- index lifecycle -----------------------------------------------------

    def _get_index(self, dim: int) -> faiss.Index:
        if self._index is None:
            flat = faiss.IndexFlatIP(dim)        # inner product on L2-normalised = cosine
            self._index = faiss.IndexIDMap(flat)
        return self._index

    # -- public API ----------------------------------------------------------

    @staticmethod
    def _stable_id(image_id: str) -> int:
        """Derive a stable int64 from the image_id string."""
        digest = hashlib.md5(image_id.encode()).digest()
        return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF

    def upsert(
        self,
        image_id: str,
        embedding: list[float],
        filename: str,
        metadata: dict[str, Any],
        description: str = "",
    ) -> None:
        if self._index is None and not INDEX_FILE.exists():
            pass  # fresh store — index created below
        elif self._index is None:
            self._load()

        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)                 # ensures cosine similarity

        dim = vec.shape[1]
        idx = self._get_index(dim)

        int_id = self._stable_id(image_id)

        # Remove existing entry if re-indexing the same image
        if int_id in self._id_map:
            idx.remove_ids(np.array([int_id], dtype=np.int64))

        idx.add_with_ids(vec, np.array([int_id], dtype=np.int64))
        self._id_map[int_id] = image_id
        self._meta[image_id] = {
            **metadata,
            "_description": description,
            "_filename": filename,
        }
        self._save()

    def exists(self, image_id: str) -> bool:
        if self._index is None:
            self._load()
        int_id = self._stable_id(image_id)
        return int_id in self._id_map

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if self._index is None:
            self._load()
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(top_k, self._index.ntotal)
        scores, int_ids = self._index.search(vec, k)

        results: list[SearchResult] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            image_id = self._id_map.get(int(int_id), "")
            meta = self._meta.get(image_id, {})
            results.append(
                SearchResult(
                    image_id=image_id,
                    filename=meta.get("filename", meta.get("_filename", "")),
                    relative_path=meta.get("relative_path", ""),
                    score=round(float(score), 4),
                    metadata={k: v for k, v in meta.items() if not k.startswith("_")},
                    description=meta.get("_description", ""),
                )
            )
        return results

    def count(self) -> int:
        if self._index is None:
            self._load()
        return self._index.ntotal if self._index else 0


_store = _Store()


# ---------------------------------------------------------------------------
# Public interface (used by backfill.py and api/app.py)
# ---------------------------------------------------------------------------

def upsert(
    image_id: str,
    embedding: list[float],
    filename: str,
    metadata: dict[str, Any],
    description: str = "",
) -> None:
    _store.upsert(image_id, embedding, filename, metadata, description)


def exists(image_id: str) -> bool:
    return _store.exists(image_id)


def search(query_embedding: list[float], top_k: int = config.TOP_K) -> list[SearchResult]:
    return _store.search(query_embedding, top_k)


def count() -> int:
    return _store.count()
