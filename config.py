import os
from dotenv import load_dotenv

load_dotenv()

# Required
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# Models
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
VISION_MODEL: str = os.getenv("VISION_MODEL", "gemini-2.0-flash")

# ChromaDB
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "image_search")

# Ingestion
IMAGES_DIR: str = os.getenv("IMAGES_DIR", "./images")
IMAGE_MAX_PX: int = int(os.getenv("IMAGE_MAX_PX", "1024"))

# Search
TOP_K: int = int(os.getenv("TOP_K", "10"))

# Rate limiting (requests per minute — adjust to your Gemini quota)
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

# Set to "true" to generate text descriptions in addition to embeddings
GENERATE_DESCRIPTIONS: bool = os.getenv("GENERATE_DESCRIPTIONS", "false").lower() == "true"
