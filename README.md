# Umara Image Search — Prototype

AI-powered semantic search for an e-commerce media archive. Search images by what they **look like**, not what they're named.

Built with Python, Gemini, FAISS, and FastAPI.

> Built with assistance from Claude (Anthropic).

---

## The problem

The current archive of ~3000 product images relies on filename search. This causes:
- **Duplicates** — the same image uploaded under different names is treated as two images
- **Poor discoverability** — finding all images of "white ceramic mugs" requires knowing exact filenames
- **No visual search** — there's no way to find images similar to a given image

---

## How it works

The core idea: **turn images and text into numbers that can be compared**.

`gemini-embedding-2-preview` is a multimodal model that converts both images and text into vectors (lists of 3072 numbers) in the **same mathematical space**. This means a text query like `"red sneakers"` and a photo of red sneakers produce vectors that point in roughly the same direction — so we can rank images by how close they are to the query.

```
Indexing:
  image.jpg ──► resize to 1024px ──► Gemini API ──► [3072 floats] ──► saved to disk

Searching:
  "red sneakers" ──► Gemini API ──► [3072 floats] ──► cosine similarity ──► top N images
```

This also solves the duplicate problem: two images with different filenames but identical content will produce nearly identical vectors.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ingestion Layer                      │
│                                                         │
│  Backfill (one-shot)          New uploads (ongoing)     │
│  backfill.py CLI              RabbitMQ consumer         │
│  • Scan images folder         • Listens for             │
│  • Skip already indexed         image.uploaded events   │
│  • Embed → store              • Same embed → store      │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              Embedding Layer (Gemini API)               │
│                                                         │
│  gemini-embedding-2-preview                             │
│  • Image bytes  ──► 3072-dim vector                     │
│  • Query text   ──► 3072-dim vector  (same space)       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Vector Store (FAISS)                 │
│                                                         │
│  index.faiss  — the vectors                             │
│  meta.json    — filename, path, size per image          │
│  id_map.json  — maps FAISS integer IDs to image paths   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Search API (FastAPI)                  │
│                                                         │
│  GET  /search?q=...     text → image search             │
│  POST /search/similar   image → image similarity        │
│  GET  /health           indexed image count             │
└─────────────────────────────────────────────────────────┘
```

---

## Project structure

```
├── config.py              Environment variables and defaults
├── ingestion/
│   └── loader.py          Scans a folder and yields image file records
├── embedding/
│   └── embedder.py        Calls Gemini to generate vectors from images or text
├── search/
│   └── searcher.py        Saves and queries vectors using FAISS
├── api/
│   └── app.py             FastAPI — exposes /search and /search/similar
├── backfill.py            CLI script to index an entire image folder
└── .env.example           Config template
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd umara-image-search

uv sync

cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here
```

---

## Running

**Step 1 — Index your images (run once)**

```bash
uv run python backfill.py --images-dir /path/to/images
```

This calls Gemini for each image and saves the vectors to `./vector_store/`. Safe to re-run — already-indexed images are skipped.

**Step 2 — Start the API server**

```bash
uv run uvicorn api.app:app --reload
```

**Step 3 — Search**

In your browser:
```
http://localhost:8000/search?q=red sneakers
http://localhost:8000/search?q=minimalist white ceramic mug
```

Or open the interactive UI:
```
http://localhost:8000/docs
```

---

## Tech choices

| Component | Choice | Why |
|---|---|---|
| Embedding model | `gemini-embedding-2-preview` | Only Gemini API model that embeds images *and* text in the same vector space — no intermediate description step needed |
| Vector store | FAISS | No server, no heavy dependencies, fast exact search. Sufficient for 3000 images |
| API framework | FastAPI | Lightweight, async, automatic docs |
| Image preprocessing | Pillow (resize to ≤1024px) | Reduces API cost 10–100× without affecting search quality |

---

## Cost estimate

| Operation | Cost |
|---|---|
| Embed one image | ~$0.00004 |
| Backfill all 3000 images | ~$0.12 (one-time) |
| 1000 new images/month | ~$0.04/month |
| Search queries | Negligible |

---

## What would change in production

| Prototype | Production |
|---|---|
| FAISS files on disk | pgvector on PostgreSQL |
| Manual backfill script | Kubernetes one-shot Job |
| No new-image handling | RabbitMQ consumer auto-indexes on upload |
| Local image files | BunnyCDN |
