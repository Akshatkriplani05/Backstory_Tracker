"""Embed novel passages once and reuse them."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backstory_checker.chunking import passages_from_text
from backstory_checker.config import CACHE_DIR, CHUNK_WORDS, EMBED_MODEL, NOVEL_DIR, OVERLAP_WORDS


def _slug(book_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", book_name.lower()).strip("_")
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise ValueError(f"Unexpected book slug: {book_name}")
    return slug


def _cache_path(slug: str, suffix: str) -> Path:
    root = CACHE_DIR.resolve()
    path = (root / f"{slug}{suffix}").resolve()
    if path.parent != root:
        raise ValueError(f"Cache path escaped the cache directory: {path}")
    return path


def _read_novel(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


@dataclass
class BookIndex:
    book_name: str
    chunks: list[str]
    embeddings: np.ndarray


def load_or_build(book_name: str, novel_path: Path, embedder) -> BookIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(book_name)
    novel_path = novel_path.resolve()
    if novel_path.parent != NOVEL_DIR.resolve():
        raise ValueError(f"Novel path is outside the novels folder: {novel_path}")
    meta_path = _cache_path(slug, "_meta.json")
    chunks_path = _cache_path(slug, "_chunks.json")
    emb_path = _cache_path(slug, "_embeddings.npy")
    expected = {
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "embed_model": EMBED_MODEL,
        "source_bytes": novel_path.stat().st_size,
    }

    if meta_path.exists() and chunks_path.exists() and emb_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == expected:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            embeddings = np.load(emb_path)
            if len(chunks) == len(embeddings):
                print(f"Loaded cached passages for {book_name}: {len(chunks)}")
                return BookIndex(book_name, chunks, embeddings)

    chunks = passages_from_text(_read_novel(novel_path), CHUNK_WORDS, OVERLAP_WORDS)
    if len(chunks) < 10:
        raise RuntimeError(f"{book_name} produced only {len(chunks)} passages")
    print(f"Embedding {len(chunks)} passages from {book_name}")
    embeddings = np.asarray(
        embedder.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps(expected), encoding="utf-8")
    return BookIndex(book_name, chunks, embeddings)
