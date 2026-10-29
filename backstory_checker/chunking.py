"""Split a novel into overlapping passages."""

from __future__ import annotations

import re

_START = re.compile(
    r"\*\*\*[^\n]*START OF (THIS |THE )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)
_END = re.compile(
    r"\*\*\*[^\n]*END OF (THIS |THE )?PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
_CHAPTER = re.compile(r"(?m)^[ \t]*(CHAPTER|Chapter)\s+(\d+|[IVXLC]+)\b")
_ILLUSTRATION = re.compile(r"\[Illustration[^\]]*\]", re.IGNORECASE)
_BOILERPLATE = ("project gutenberg", "gutenberg literary archive", "www.gutenberg.org")


def strip_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = _START.search(text)
    if start:
        text = text[start.end() :]
    end = _END.search(text)
    if end:
        text = text[: end.start()]

    chapters = list(_CHAPTER.finditer(text))
    if not chapters:
        return text
    # Contents headings sit a title apart. The story starts at the first
    # heading that is followed by a full chapter, so chapter 1 stays in.
    story_at = 0
    for index in range(len(chapters) - 1):
        gap = chapters[index + 1].start() - chapters[index].end()
        if gap >= 200:
            story_at = index
            break
    return text[chapters[story_at].start() :]


def passages_from_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Cut the novel into overlapping passages of about `target_words` words."""
    cleaned = _ILLUSTRATION.sub(" ", strip_gutenberg(text))
    words = cleaned.split()
    step = max(1, target_words - overlap_words)
    passages: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + target_words]
        if len(window) < max(12, target_words // 5) and passages:
            break
        passage = " ".join(window)
        if any(phrase in passage.lower() for phrase in _BOILERPLATE):
            continue
        passages.append(passage)
        if start + target_words >= len(words):
            break
    return passages
