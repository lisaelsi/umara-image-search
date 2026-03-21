"""
FastAPI search API.

Endpoints:
  GET  /health              — liveness + indexed image count
  GET  /search?q=...        — text → image search
  POST /search/similar      — image → image similarity search (bonus)
"""
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

import config
from embedding.embedder import embed_image, embed_query
from search.searcher import SearchResult, count, search

app = FastAPI(
    title="Image Search API",
    description="Semantic image search powered by Gemini multimodal embeddings",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ImageMatch(BaseModel):
    image_id: str
    filename: str
    relative_path: str
    score: float
    description: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[ImageMatch]
    total_indexed: int


def _to_match(r: SearchResult) -> ImageMatch:
    return ImageMatch(
        image_id=r.image_id,
        filename=r.filename,
        relative_path=r.relative_path,
        score=r.score,
        description=r.description,
        metadata=r.metadata,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_images": count()}


@app.get("/search", response_model=SearchResponse)
def text_search(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(config.TOP_K, ge=1, le=50, description="Number of results"),
) -> SearchResponse:
    """
    Search by text. Examples:
      /search?q=red sneakers
      /search?q=minimalist white ceramic mug
      /search?q=outdoor jacket waterproof
    """
    query_embedding = embed_query(q)
    results = search(query_embedding, top_k=top_k)
    return SearchResponse(
        query=q,
        results=[_to_match(r) for r in results],
        total_indexed=count(),
    )


@app.post("/search/similar", response_model=SearchResponse)
async def image_similarity(
    file: UploadFile = File(..., description="Image file to find similar images for"),
    top_k: int = Query(config.TOP_K, ge=1, le=50),
) -> SearchResponse:
    """
    Upload an image and find visually similar images in the index.
    Useful for finding duplicates or variants of a product image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    # Write to temp file so embed_image() can open it with PIL
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        query_embedding = embed_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    results = search(query_embedding, top_k=top_k)
    return SearchResponse(
        query=f"[image: {file.filename}]",
        results=[_to_match(r) for r in results],
        total_indexed=count(),
    )
