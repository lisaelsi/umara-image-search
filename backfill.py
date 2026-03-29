#!/usr/bin/env python3
"""
Backfill script — indexes all images in IMAGES_DIR into the vector store.

Usage:
    uv run python backfill.py
    uv run python backfill.py --images-dir /path/to/images

Idempotent: already-indexed images are skipped.
In production this runs as a Kubernetes Job (one-shot, restartable).
"""
import argparse
import sys
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

import config
from embedding.embedder import describe_image, embed_image
from ingestion.loader import scan_images
from search.searcher import count, exists, upsert

console = Console()

# Seconds to wait between API calls to respect RATE_LIMIT_RPM.
# Each image triggers 1 embed call (+ 1 vision call if descriptions enabled).
_CALLS_PER_IMAGE = 2 if config.GENERATE_DESCRIPTIONS else 1
_MIN_DELAY = 60.0 / config.RATE_LIMIT_RPM / _CALLS_PER_IMAGE


def process(record) -> bool:
    """
    Embed one image and upsert into the vector store.
    Returns True if newly indexed, False if skipped.
    """
    if exists(record.relative_path):
        return False

    description = ""
    if config.GENERATE_DESCRIPTIONS:
        description = describe_image(record.path)
        time.sleep(_MIN_DELAY)

    embedding = embed_image(record.path)
    time.sleep(_MIN_DELAY)

    upsert(
        image_id=record.relative_path,
        embedding=embedding,
        filename=record.filename,
        description=description,
        metadata={
            "filename": record.filename,
            "relative_path": record.relative_path,
            "file_size": record.file_size,
            "mime_type": record.mime_type,
        },
    )
    return True


def main(images_dir: str) -> None:
    console.print(f"\n[bold]Image Search — Backfill[/bold]")
    console.print(f"Directory : {images_dir}")
    console.print(f"Model     : {config.EMBEDDING_MODEL}")
    console.print(f"Rate limit: {config.RATE_LIMIT_RPM} RPM\n")

    records = list(scan_images(images_dir))
    if not records:
        console.print("[yellow]No supported images found (.jpg .png .webp).[/yellow]")
        sys.exit(0)

    already_indexed = count()
    console.print(f"Found [bold]{len(records)}[/bold] images. Already indexed: [bold]{already_indexed}[/bold]\n")

    processed = skipped = failed = 0
    failures: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(records))

        for record in records:
            progress.update(task, description=f"{record.filename[:50]}")
            try:
                if process(record):
                    processed += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                failures.append((record.filename, str(exc)))
                console.print(f"\n[red]  ✗ {record.filename}: {exc}[/red]")
            progress.advance(task)

    # Summary table
    table = Table(title="Backfill complete", show_header=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Newly indexed", f"[green]{processed}[/green]")
    table.add_row("Skipped (exists)", str(skipped))
    table.add_row("Failed", f"[red]{failed}[/red]" if failed else "0")
    table.add_row("Total in DB", str(count()))
    console.print(table)

    if failures:
        console.print("\n[red]Failed files:[/red]")
        for name, err in failures:
            console.print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill image embeddings into the vector store")
    parser.add_argument("--images-dir", default=config.IMAGES_DIR)
    args = parser.parse_args()
    main(args.images_dir)
