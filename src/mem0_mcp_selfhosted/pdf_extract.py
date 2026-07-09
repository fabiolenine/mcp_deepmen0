"""PDF text extraction via poppler-utils (v0.5a — digital PDFs only).

poppler (pdftotext/pdfinfo) is already a system dependency of the host and is
battle-tested for digital PDFs, so no Python PDF library is added. One single
``pdftotext`` run extracts the whole document; pages come back separated by
form feeds (``\\f``). A per-page ``-f/-l`` fallback covers documents where the
single pass fails.

Per-page classification keeps mixed documents useful: pages with a real text
layer are ingested, scanned/garbage pages are skipped and reported. A document
whose pages are ALL scanned raises ``FullyScannedPdf`` — OCR arrives in v0.5b
(VLM transcription of rasterized pages); rejecting loudly beats silently
storing nothing.

All failures here are poison (bad document, not sick infra) — the exceptions
derive from ValueError so the worker's classifier sends them to dead-letter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass

# below this many characters a page has no usable text layer
_MIN_PAGE_CHARS = 20
# above this ratio of replacement/control garbage the "text" is not text
_MAX_GARBAGE_RATIO = 0.30

_DEHYPHENATE_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])-\n([a-zà-öø-ÿ])")


class PdfExtractError(ValueError):
    """Base: unextractable document (poison — never retried)."""


class PopplerMissing(PdfExtractError):
    pass


class EncryptedPdf(PdfExtractError):
    pass


class FullyScannedPdf(PdfExtractError):
    pass


class ExtractionTimeout(PdfExtractError):
    pass


@dataclass
class PageText:
    number: int  # 1-based
    text: str
    has_text: bool


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise PopplerMissing(f"{binary} not found — install poppler-utils")
    return path


def pdf_info(path: str, timeout_s: float = 30) -> dict:
    """{"pages": int, "encrypted": bool, "title": str|None} via pdfinfo."""
    _require("pdfinfo")
    try:
        proc = subprocess.run(
            ["pdfinfo", path], capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise ExtractionTimeout(f"pdfinfo timed out after {timeout_s}s") from None
    if proc.returncode != 0:
        raise PdfExtractError(f"pdfinfo failed: {(proc.stderr or '').strip()[:200]}")
    info: dict = {"pages": 0, "encrypted": False, "title": None}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "Pages":
            try:
                info["pages"] = int(value)
            except ValueError:
                pass
        elif key == "Encrypted":
            info["encrypted"] = value.lower().startswith("yes")
        elif key == "Title" and value:
            info["title"] = value
    return info


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # rejoin words hyphenated across line breaks ("informa-\nção" -> "informação");
    # requiring a lowercase continuation avoids gluing legitimate hyphens
    return _DEHYPHENATE_RE.sub(r"\1\2", text)


def _classify(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _MIN_PAGE_CHARS:
        return False
    garbage = sum(
        1 for ch in stripped
        if ch == "�" or (unicodedata.category(ch) == "Cc" and ch not in "\n\t")
    )
    return (garbage / len(stripped)) <= _MAX_GARBAGE_RATIO


def _run_pdftotext(args: list[str], timeout_s: float) -> str:
    try:
        proc = subprocess.run(
            args, capture_output=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise ExtractionTimeout(f"pdftotext timed out after {timeout_s}s") from None
    if proc.returncode != 0:
        raise PdfExtractError(f"pdftotext failed: {(proc.stderr or b'').decode('utf-8', 'replace').strip()[:200]}")
    return proc.stdout.decode("utf-8", "replace")


def rasterize_pages(path: str, page_numbers: list[int], dpi: int = 150,
                    timeout_s: float = 120) -> dict[int, bytes]:
    """Render the given 1-based pages to PNG bytes via pdftoppm (v0.5b OCR).

    Used to turn scanned pages into images the VLM can transcribe. Returns
    ``{page_number: png_bytes}``; pages that fail to render are omitted (the
    caller treats a missing page as still-unextractable).
    """
    _require("pdftoppm")
    out: dict[int, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="mem0-raster-") as tmpdir:
        for num in page_numbers:
            root = os.path.join(tmpdir, f"p{num}")
            try:
                # poppler's pdftoppm does not reliably write PNG to stdout, so
                # render to a temp file (-singlefile drops the page-number suffix)
                proc = subprocess.run(
                    ["pdftoppm", "-png", "-r", str(dpi), "-f", str(num), "-l", str(num),
                     "-singlefile", path, root],
                    capture_output=True, timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                raise ExtractionTimeout(f"pdftoppm timed out after {timeout_s}s") from None
            png = root + ".png"
            if proc.returncode == 0 and os.path.exists(png):
                with open(png, "rb") as fh:
                    out[num] = fh.read()
    return out


def extract_pages(path: str, timeout_s: float = 120) -> list[PageText]:
    """Extract and classify every page. Raises FullyScannedPdf when no page
    has a usable text layer."""
    _require("pdftotext")
    info = pdf_info(path, timeout_s=min(timeout_s, 30))
    if info["encrypted"]:
        raise EncryptedPdf("encrypted PDF — decrypt it before ingesting")

    pages: list[str]
    try:
        # single pass over the whole document; \f separates pages
        raw = _run_pdftotext(
            ["pdftotext", "-layout", "-enc", "UTF-8", path, "-"], timeout_s,
        )
        pages = raw.split("\f")
        if pages and pages[-1].strip() == "":
            pages.pop()  # trailing form feed
    except ExtractionTimeout:
        raise
    except PdfExtractError:
        # fallback: page by page (some malformed documents fail whole-file)
        expected = info["pages"] or 1
        pages = []
        for num in range(1, expected + 1):
            try:
                pages.append(_run_pdftotext(
                    ["pdftotext", "-layout", "-enc", "UTF-8",
                     "-f", str(num), "-l", str(num), path, "-"],
                    timeout_s,
                ).rstrip("\f"))
            except PdfExtractError:
                pages.append("")

    result = []
    for i, text in enumerate(pages, start=1):
        normalized = _normalize(text)
        result.append(PageText(number=i, text=normalized, has_text=_classify(normalized)))

    if result and not any(p.has_text for p in result):
        raise FullyScannedPdf(
            "no page has a text layer (scanned PDF?) — OCR lands in v0.5b; "
            "only digital PDFs are ingestible today"
        )
    if not result:
        raise PdfExtractError("pdftotext produced no pages")
    return result
