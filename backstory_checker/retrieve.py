"""Pick the novel passages closest in meaning to a backstory."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backstory_checker.config import MODEL_PASSAGE_WORDS, TOP_K
from backstory_checker.index import BookIndex


def top_passages(
    backstory: str,
    index: BookIndex,
    embedder,
    k: int = TOP_K,
) -> tuple[list[str], list[float]]:
    """Return the top-k passages and their cosine similarities.

    The query is the backstory passage alone. Character names and captions
    are not part of the comparison.
    """
    query = embedder.encode([backstory], convert_to_numpy=True, show_progress_bar=False)
    scores = cosine_similarity(query, index.embeddings)[0]
    order = np.argsort(scores)[::-1][:k]
    passages = [index.chunks[int(i)] for i in order]
    similarities = [float(scores[int(i)]) for i in order]
    return passages, similarities


def model_evidence(passages: list[str], words_each: int = MODEL_PASSAGE_WORDS) -> str:
    """Keep a short head of every retrieved passage so all five fit."""
    snippets = [" ".join(passage.split()[:words_each]) for passage in passages]
    return "\n\n".join(snippets)


def attach_evidence(rows: list[dict], indexes: dict[str, BookIndex], embedder) -> list[dict]:
    """Add the top passages for every labeled backstory."""
    grouped: dict[str, list[int]] = {}
    for row_index, row in enumerate(rows):
        grouped.setdefault(row["book_name"], []).append(row_index)

    for book_name, row_indexes in grouped.items():
        index = indexes[book_name]
        queries = embedder.encode(
            [rows[i]["content"] for i in row_indexes],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = cosine_similarity(queries, index.embeddings)
        for local, row_index in enumerate(row_indexes):
            order = np.argsort(scores[local])[::-1][:TOP_K]
            passages = [index.chunks[int(i)] for i in order]
            rows[row_index]["backstory"] = rows[row_index]["content"]
            rows[row_index]["passages"] = passages
            rows[row_index]["scores"] = [float(scores[local][int(i)]) for i in order]
            rows[row_index]["evidence"] = model_evidence(passages)
    return rows
