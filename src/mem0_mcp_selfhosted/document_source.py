"""Document source resolution and content-addressed spooling (v0.5a).

v0.5a accepts documents by ``file_path`` only (Claude Code runs on the same
host; MCP tool-call base64 is model-generated token by token, so it cannot
carry real files — remote clients arrive in a later phase). The path is
validated against an allowlist and the bytes are copied into a spool directory
named by content hash:

    <spool_dir>/<sha256>.pdf

Content addressing gives three properties for free: re-submitting the same
file is idempotent on disk (atomic rename over an existing file is a no-op);
a crash between spool and enqueue leaves a valid reusable file, not junk; and
garbage collection is reference-counting — a file is deleted when no queue row
references its hash (see ``spool_gc``).

All validation failures raise typed exceptions that the worker/tool classify
as poison (bad input, not sick infra).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path

from mem0_mcp_selfhosted.env import env

# magic bytes -> (content_type, canonical extension)
_MAGIC_TYPES: tuple[tuple[bytes, str, str], ...] = (
    (b"%PDF-", "application/pdf", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
)


class DocumentSourceError(ValueError):
    """Base: bad document input (poison — never retried)."""


class SourcePathForbidden(DocumentSourceError):
    pass


class SourceNotFound(DocumentSourceError):
    pass


class DocumentTooLarge(DocumentSourceError):
    pass


class UnsupportedDocumentType(DocumentSourceError):
    pass


def _path_allowlist() -> list[Path]:
    raw = env("MEM0_DOC_PATH_ALLOWLIST") or str(Path.home())
    return [Path(p).resolve() for p in raw.split(":") if p.strip()]


def _max_bytes() -> int:
    return int(env("MEM0_DOC_MAX_BYTES", str(25 * 1024 * 1024)))


def spool_dir() -> Path:
    return Path(env("MEM0_DOC_SPOOL_DIR") or (Path.home() / ".mem0" / "ingest_spool"))


def sniff_content_type(head: bytes) -> tuple[str, str]:
    """(content_type, extension) from magic bytes; raises on unknown."""
    for magic, ctype, ext in _MAGIC_TYPES:
        if head.startswith(magic):
            return ctype, ext
    raise UnsupportedDocumentType(
        "unrecognized file type (expected a PDF, PNG or JPEG)"
    )


def resolve_and_spool(file_path: str) -> dict:
    """Validate a local path and copy it into the spool, content-addressed.

    Returns {"spool_path", "doc_sha256", "size_bytes", "content_type",
    "filename"}. Raises DocumentSourceError subclasses on any bad input.
    """
    candidate = Path(file_path).expanduser()
    try:
        real = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SourceNotFound(f"file not found: {file_path}") from None

    allowed = _path_allowlist()
    if not any(real == base or base in real.parents for base in allowed):
        raise SourcePathForbidden(
            f"path outside the allowed roots ({':'.join(str(b) for b in allowed)}): {file_path}"
        )
    if not real.is_file():
        raise SourceNotFound(f"not a regular file: {file_path}")

    size = real.stat().st_size
    limit = _max_bytes()
    if size > limit:
        raise DocumentTooLarge(f"file is {size} bytes; cap is {limit} (MEM0_DOC_MAX_BYTES)")
    if size == 0:
        raise UnsupportedDocumentType(f"empty file: {file_path}")

    sha = hashlib.sha256()
    head = b""
    with open(real, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            if not head:
                head = block[:16]
            sha.update(block)
    content_type, ext = sniff_content_type(head)
    digest = sha.hexdigest()

    target_dir = spool_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}.{ext}"
    if not target.exists():
        tmp = target_dir / f".{uuid.uuid4().hex}.tmp"
        shutil.copyfile(real, tmp)
        os.replace(tmp, target)  # atomic; concurrent same-content copies converge

    return {
        "spool_path": str(target),
        "doc_sha256": digest,
        "size_bytes": size,
        "content_type": content_type,
        "filename": os.path.basename(real),
    }


def spool_gc(referenced_hashes: set[str], directory: Path | None = None) -> int:
    """Delete spool files whose hash no queue row references; returns count.

    Also sweeps orphaned ``.tmp`` files older than one hour (a crash between
    copy and rename). Never raises — gc is bookkeeping.
    """
    directory = directory or spool_dir()
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    cutoff = time.time() - 3600
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            if entry.suffix == ".tmp":
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
                continue
            if entry.stem not in referenced_hashes:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed
