"""
Generates embeddings and (optionally) text descriptions via the Gemini API.

Model choice: gemini-embedding-2-preview
  - Embeds images AND text in the same vector space (multimodal)
  - No intermediate description step needed for search to work
"""
import io
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

DESCRIPTION_PROMPT = """You are indexing e-commerce product images.
Describe this image in one dense paragraph covering:
product type, colors, materials, style, visible text/brand, intended use.
Be specific — this text will be stored as searchable metadata."""


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _resize_image(image_path: Path, max_px: int = config.IMAGE_MAX_PX) -> bytes:
    """
    Resize to max_px on the longest side and re-encode as JPEG.
    Keeps API costs low and stays well within Gemini's inline-data limit.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def embed_image(image_path: Path) -> list[float]:
    """
    Embed an image using gemini-embedding-2-preview.
    Returns a float vector in the same space as embed_query().
    """
    image_bytes = _resize_image(image_path)
    result = _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    )
    return list(result.embeddings[0].values)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def embed_query(text: str) -> list[float]:
    """
    Embed a text search query.
    Uses the same model as embed_image() so cosine similarity is meaningful.
    """
    result = _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
    )
    return list(result.embeddings[0].values)


# ---------------------------------------------------------------------------
# Optional: description generation
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def describe_image(image_path: Path) -> str:
    """
    Generate a human-readable text description via Gemini Flash.
    Only called when GENERATE_DESCRIPTIONS=true in config.
    Useful for displaying context in search results or building a
    text-based fallback index.
    """
    image_bytes = _resize_image(image_path)
    response = _client.models.generate_content(
        model=config.VISION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part(text=DESCRIPTION_PROMPT),
                ],
            )
        ],
    )
    return response.text.strip()
