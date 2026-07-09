"""Unit tests for the add_document MCP tool (v0.5a)."""

from __future__ import annotations

import json
import shutil
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod
from tests.pdf_builder import build_pdf

poppler = pytest.mark.skipif(
    shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None,
    reason="poppler-utils not installed",
)

PAGE = "A decisão do projeto Hermes foi aprovada em 2026 com a porta 8081."


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Server with mocked Memory, isolated queue/spool, worker off."""
    monkeypatch.setenv("MEM0_ASYNC_INGEST", "true")
    monkeypatch.setenv("MEM0_QUEUE_WORKER", "false")
    monkeypatch.setenv("MEM0_QUEUE_DB_PATH", str(tmp_path / "q.db"))
    server_mod._ingest_queue = None
    server_mod._ingest_worker = None
    original_memory = server_mod.memory
    server_mod.memory = MagicMock()

    srv = server_mod._create_server()
    yield srv

    server_mod.memory = original_memory
    server_mod._ingest_queue = None
    server_mod._ingest_worker = None


def _tool(srv, name="add_document"):
    return srv._tool_manager._tools[name].fn


@pytest.fixture
def pdf_path(tmp_path):
    p = tmp_path / "relatorio.pdf"
    p.write_bytes(build_pdf([PAGE, PAGE.replace("Hermes", "Atlas")]))
    return str(p)


@poppler
class TestAddDocument:
    def test_queued_envelope_and_job_shape(self, server, pdf_path):
        parsed = json.loads(_tool(server)(file_path=pdf_path, user_id="alice"))
        assert parsed["status"] == "queued"
        assert parsed["task_id"].startswith("tsk_")
        assert parsed["source_doc"] == "relatorio.pdf"
        assert parsed["pages"] == 2
        assert parsed["chunks_estimate"] >= 1
        assert parsed["estimated_wait_s"] > 0

        queue, _ = server_mod._get_ingest()
        job = queue.claim_next()
        assert job["kind"] == "document"
        assert job["user_id"] == "alice"
        assert job["params"]["doc_sha256"]
        assert job["params"]["enable_graph"] is False  # document default
        assert job["messages"][0]["content"].startswith("[document sha256=")

    def test_custom_filename_becomes_source_doc(self, server, pdf_path):
        parsed = json.loads(_tool(server)(file_path=pdf_path, filename="Contrato 2026.pdf"))
        assert parsed["source_doc"] == "Contrato 2026.pdf"

    def test_path_outside_allowlist_errors(self, server, monkeypatch, tmp_path, pdf_path):
        monkeypatch.setenv("MEM0_DOC_PATH_ALLOWLIST", str(tmp_path / "elsewhere"))
        parsed = json.loads(_tool(server)(file_path=pdf_path))
        assert "error" in parsed

    def test_non_pdf_errors(self, server, tmp_path):
        txt = tmp_path / "notas.pdf"
        txt.write_text("só texto, sem magic de PDF")
        parsed = json.loads(_tool(server)(file_path=str(txt)))
        assert "error" in parsed

    def test_pages_over_cap_rejected(self, server, pdf_path, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_MAX_PAGES", "1")
        parsed = json.loads(_tool(server)(file_path=pdf_path))
        assert "error" in parsed
        assert "MEM0_DOC_MAX_PAGES" in parsed.get("detail", "")

    def test_kill_switch(self, server, pdf_path, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_ENABLED", "false")
        parsed = json.loads(_tool(server)(file_path=pdf_path))
        assert "error" in parsed

    def test_already_ingested_and_force(self, server, pdf_path):
        fn = _tool(server)
        first = json.loads(fn(file_path=pdf_path))
        queue, _ = server_mod._get_ingest()
        queue.claim_next()
        queue.mark_done(first["task_id"], {"memory_ids": ["m1"], "chunks_total": 3})

        again = json.loads(fn(file_path=pdf_path))
        assert again["status"] == "already_ingested"
        assert again["task_id"] == first["task_id"]
        assert again["result"]["memory_ids"] == ["m1"]

        forced = json.loads(fn(file_path=pdf_path, force=True))
        assert forced["status"] == "queued"
        assert forced["task_id"] != first["task_id"]

    def test_active_duplicate_returns_same_task(self, server, pdf_path):
        fn = _tool(server)
        first = json.loads(fn(file_path=pdf_path))
        second = json.loads(fn(file_path=pdf_path))
        assert second["status"] == "queued"
        assert second["task_id"] == first["task_id"]
        assert second["duplicate"] is True

    def test_estimated_wait_accounts_for_document_chunks(self, server, pdf_path, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_EST_CHUNK_S", "35")
        monkeypatch.setenv("MEM0_QUEUE_EST_ADD_S", "40")
        json.loads(_tool(server)(file_path=pdf_path))  # 2 pages -> 4 chunks estimated
        add_fn = _tool(server, "add_memory")
        parsed = json.loads(add_fn(text="um fato conversacional"))
        # 1 conversation × 40 + 4 estimated chunks × 35 = 180
        assert parsed["estimated_wait_s"] == 40 + 4 * 35

    def test_scope_separates_document_dedup(self, server, pdf_path):
        fn = _tool(server)
        a = json.loads(fn(file_path=pdf_path, user_id="alice"))
        b = json.loads(fn(file_path=pdf_path, user_id="bob"))
        assert a["task_id"] != b["task_id"]  # same bytes, different scope


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class TestAddImage:
    @pytest.fixture
    def png_path(self, tmp_path):
        p = tmp_path / "foto.png"
        p.write_bytes(PNG_BYTES)
        return str(p)

    def test_image_needs_vision(self, server, png_path, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "false")
        parsed = json.loads(_tool(server)(file_path=png_path))
        assert "error" in parsed
        assert "vision" in parsed.get("detail", "").lower()

    def test_image_queued_with_vision(self, server, png_path, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
        monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
        parsed = json.loads(_tool(server)(file_path=png_path, user_id="alice"))
        assert parsed["status"] == "queued"
        assert parsed["content_type"] == "image/png"
        assert parsed["pages"] == 1 and parsed["chunks_estimate"] == 1

        queue, _ = server_mod._get_ingest()
        job = queue.claim_next()
        assert job["kind"] == "document"
        assert job["params"]["content_type"] == "image/png"
