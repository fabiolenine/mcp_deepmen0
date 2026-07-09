"""Unit tests for document source resolution and content-addressed spool."""

from __future__ import annotations

import os

import pytest

from mem0_mcp_selfhosted.document_source import (
    DocumentTooLarge,
    SourceNotFound,
    SourcePathForbidden,
    UnsupportedDocumentType,
    resolve_and_spool,
    spool_gc,
)

PDF_BYTES = b"%PDF-1.4 fake minimal body for spool tests\n%%EOF\n"


@pytest.fixture
def docs_env(monkeypatch, tmp_path):
    """Isolated allowlist root + spool dir."""
    root = tmp_path / "allowed"
    root.mkdir()
    spool = tmp_path / "spool"
    monkeypatch.setenv("MEM0_DOC_PATH_ALLOWLIST", str(root))
    monkeypatch.setenv("MEM0_DOC_SPOOL_DIR", str(spool))
    return root, spool


def _write_pdf(root, name="doc.pdf", body=PDF_BYTES):
    p = root / name
    p.write_bytes(body)
    return p


class TestResolveAndSpool:
    def test_happy_path_content_addressed(self, docs_env):
        root, spool = docs_env
        src = _write_pdf(root)
        info = resolve_and_spool(str(src))
        assert info["content_type"] == "application/pdf"
        assert info["filename"] == "doc.pdf"
        assert info["size_bytes"] == len(PDF_BYTES)
        assert info["spool_path"] == str(spool / f"{info['doc_sha256']}.pdf")
        assert open(info["spool_path"], "rb").read() == PDF_BYTES

    def test_same_content_converges_different_name_same_spool(self, docs_env):
        root, spool = docs_env
        a = resolve_and_spool(str(_write_pdf(root, "a.pdf")))
        b = resolve_and_spool(str(_write_pdf(root, "b.pdf")))
        assert a["doc_sha256"] == b["doc_sha256"]
        assert a["spool_path"] == b["spool_path"]
        assert len(list(spool.iterdir())) == 1  # one file, no .tmp leftovers

    def test_path_outside_allowlist_forbidden(self, docs_env, tmp_path):
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(PDF_BYTES)
        with pytest.raises(SourcePathForbidden):
            resolve_and_spool(str(outside))

    def test_symlink_escaping_allowlist_forbidden(self, docs_env, tmp_path):
        root, _ = docs_env
        secret = tmp_path / "secret.pdf"
        secret.write_bytes(PDF_BYTES)
        link = root / "innocent.pdf"
        os.symlink(secret, link)
        with pytest.raises(SourcePathForbidden):  # realpath resolves the escape
            resolve_and_spool(str(link))

    def test_missing_and_directory_rejected(self, docs_env):
        root, _ = docs_env
        with pytest.raises(SourceNotFound):
            resolve_and_spool(str(root / "ghost.pdf"))
        sub = root / "subdir"
        sub.mkdir()
        with pytest.raises(SourceNotFound):
            resolve_and_spool(str(sub))

    def test_size_cap_and_empty(self, docs_env, monkeypatch):
        root, _ = docs_env
        monkeypatch.setenv("MEM0_DOC_MAX_BYTES", "10")
        with pytest.raises(DocumentTooLarge):
            resolve_and_spool(str(_write_pdf(root)))
        monkeypatch.setenv("MEM0_DOC_MAX_BYTES", str(1024 * 1024))
        empty = root / "empty.pdf"
        empty.write_bytes(b"")
        with pytest.raises(UnsupportedDocumentType):
            resolve_and_spool(str(empty))

    def test_non_pdf_magic_rejected(self, docs_env):
        root, _ = docs_env
        fake = root / "notes.pdf"
        fake.write_bytes(b"apenas texto com nome de pdf")
        with pytest.raises(UnsupportedDocumentType):
            resolve_and_spool(str(fake))


class TestSpoolGc:
    def test_unreferenced_files_removed_referenced_kept(self, docs_env):
        root, spool = docs_env
        kept = resolve_and_spool(str(_write_pdf(root, "kept.pdf", PDF_BYTES)))
        gone = resolve_and_spool(str(_write_pdf(root, "gone.pdf", PDF_BYTES + b"x")))
        removed = spool_gc({kept["doc_sha256"]})
        assert removed == 1
        assert os.path.exists(kept["spool_path"])
        assert not os.path.exists(gone["spool_path"])

    def test_stale_tmp_swept_fresh_tmp_kept(self, docs_env):
        _, spool = docs_env
        spool.mkdir(parents=True, exist_ok=True)
        stale = spool / ".dead.tmp"
        stale.write_bytes(b"x")
        os.utime(stale, (0, 0))  # ancient
        fresh = spool / ".alive.tmp"
        fresh.write_bytes(b"x")
        assert spool_gc(set()) == 1
        assert not stale.exists() and fresh.exists()

    def test_missing_dir_is_noop(self, docs_env, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_DOC_SPOOL_DIR", str(tmp_path / "nao-existe"))
        assert spool_gc(set()) == 0
