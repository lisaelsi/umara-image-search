"""
Scans a directory for supported images and yields ImageRecord objects.
"""
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    relative_path: str  # used as the stable ID in the vector store
    filename: str
    file_size: int       # bytes
    mime_type: str


def scan_images(images_dir: str) -> Generator[ImageRecord, None, None]:
    """Recursively yield all supported images under images_dir."""
    base = Path(images_dir).resolve()

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        mime_type, _ = mimetypes.guess_type(str(path))

        yield ImageRecord(
            path=path,
            relative_path=str(path.relative_to(base)),
            filename=path.name,
            file_size=path.stat().st_size,
            mime_type=mime_type or "image/jpeg",
        )
