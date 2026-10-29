"""Download the public-domain novels used as evidence."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from backstory_checker.config import BOOKS, NOVEL_DIR, USER_AGENT

_ALLOWED_HOSTS = {"www.gutenberg.org", "gutenberg.org"}


class _StayOnGutenberg(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urlparse(newurl).hostname
        if host not in _ALLOWED_HOSTS:
            raise urllib.error.URLError(f"Refusing redirect off Project Gutenberg: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str) -> bytes:
    host = urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Refusing download from {host}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_StayOnGutenberg)
    with opener.open(request, timeout=180) as response:
        return response.read()


def _decode(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _is_novel(text: str, snippets: list[str]) -> bool:
    stripped = text.lstrip()
    if stripped[:1] == "<" and stripped[:15].lower().startswith(("<!doctype", "<html")):
        return False
    if len(text) < 80_000:
        return False
    return all(snippet in text for snippet in snippets)


def ensure_novels() -> dict[str, Path]:
    """Save one plain-text file per book under data/novels."""
    NOVEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    notes: list[str] = []

    for book_name, spec in BOOKS.items():
        dest = NOVEL_DIR / spec["filename"]
        if dest.exists() and _is_novel(dest.read_text(encoding="utf-8", errors="replace"), spec["snippets"]):
            print(f"Novel already present: {dest.name}")
            paths[book_name] = dest
            notes.append(f"{book_name}: {spec['source']} -> {dest.name}")
            continue

        errors: list[str] = []
        saved = False
        for url in spec["urls"]:
            print(f"Downloading {book_name} from {url}")
            try:
                text = _decode(_fetch(url))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url} ({exc})")
                continue
            if not _is_novel(text, spec["snippets"]):
                errors.append(f"{url} (file was not the expected novel text)")
                continue
            dest.write_text(text, encoding="utf-8")
            paths[book_name] = dest
            notes.append(f"{book_name}: {spec['source']} -> {dest.name}")
            saved = True
            break
        if not saved:
            joined = "; ".join(errors)
            raise RuntimeError(f"Could not download {book_name}. {joined}")

    (NOVEL_DIR / "SOURCES.txt").write_text(
        "Public-domain novels from Project Gutenberg.\n" + "\n".join(notes) + "\n",
        encoding="utf-8",
    )
    return paths
