"""Unit tests for the worker's document branch (v0.5a) — no live infrastructure.

Real poppler runs over PDFs built in-code (tests/pdf_builder.py); the Memory
is the same FakeMemory stub the v0.4 worker tests use.
"""

from __future__ import annotations

import shutil

import pytest

import mem0_mcp_selfhosted.ingest_worker as worker_mod
from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.ingest_worker import DOC_EXTRACTION_INSTRUCTIONS, IngestWorker
from tests.pdf_builder import build_pdf

poppler = pytest.mark.skipif(
    shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None,
    reason="poppler-utils not installed",
)

PAGE = (
    "A decisão de arquitetura do projeto Hermes foi aprovada em 2026.\n"
    "O gateway de pagamentos usa retry com jitter de 200ms.\n"
    "A porta padrão do serviço é a 8081 e o vector store usa a 6333."
)


class FakeMemory:
    def __init__(self, raises_on_call: int | None = None):
        self.calls = []
        self.raises_on_call = raises_on_call

    def add(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.raises_on_call is not None and len(self.calls) == self.raises_on_call:
            raise ConnectionError("ollama down mid-document")
        return {"results": [{"id": f"mem-{len(self.calls)}", "memory": "fato", "event": "ADD"}]}


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _make_worker(queue, mem, **overrides):
    defaults = dict(max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)
    defaults.update(overrides)
    return IngestWorker(queue, lambda: mem, **defaults)


def _enqueue_document(queue, spool_path, sha="abc123", filename="relatorio.pdf",
                      user_id="alice", params_extra=None):
    params = {
        "spool_path": str(spool_path), "doc_sha256": sha, "filename": filename,
    }
    params.update(params_extra or {})
    return queue.enqueue(
        user_id=user_id, agent_id=None, run_id=None,
        messages=[{"role": "user", "content": f"[document sha256={sha}]"}],
        params=params, kind="document",
    )


@pytest.fixture
def doc_pdf(tmp_path):
    p = tmp_path / "spool" / "abc123.pdf"
    p.parent.mkdir()
    p.write_bytes(build_pdf([PAGE, PAGE.replace("Hermes", "Atlas")]))
    return p


@poppler
class TestProcessDocument:
    def test_chunks_become_adds_with_provenance(self, queue, doc_pdf):
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        res = _enqueue_document(queue, doc_pdf)
        job = queue.claim_next()
        worker._process(job)

        assert len(mem.calls) >= 1
        messages, kwargs = mem.calls[0]
        assert "Trecho do documento 'relatorio.pdf'" in messages[0]["content"]
        assert "Hermes" in messages[0]["content"]
        assert kwargs["prompt"] == DOC_EXTRACTION_INSTRUCTIONS
        assert kwargs["infer"] is True
        assert kwargs["user_id"] == "alice"
        md = kwargs["metadata"]
        assert md["source_doc"] == "relatorio.pdf"
        assert md["doc_sha256"] == "abc123"
        assert md["created_at"] == job["submitted_at"]  # canonical fact time
        assert md["task_id"] == res["task_id"]
        assert md["chunk_index"] == 0 and md["chunks_total"] == len(mem.calls)
        assert md["page_start"] >= 1 and md["page_end"] >= md["page_start"]

        status = queue.task_status(res["task_id"])
        assert status["status"] == "done"
        result = status["result"]
        assert result["memory_ids"] == [f"mem-{i+1}" for i in range(len(mem.calls))]
        assert result["pages"] == 2 and result["skipped_pages"] == []
        assert result["chunks_total"] == len(mem.calls)

    def test_progress_heartbeats_during_processing(self, queue, doc_pdf, monkeypatch):
        # force tiny chunks so the document yields several progress writes
        monkeypatch.setenv("MEM0_DOC_CHUNK_CHARS", "120")
        monkeypatch.setenv("MEM0_DOC_CHUNK_OVERLAP", "0")
        progress_seen = []
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        original = queue.update_progress

        def spy(task_id, progress):
            progress_seen.append(dict(progress))
            original(task_id, progress)

        monkeypatch.setattr(queue, "update_progress", spy)
        _enqueue_document(queue, doc_pdf)
        worker._process(queue.claim_next())
        dones = [p.get("chunks_done") for p in progress_seen]
        assert dones[0] == 0  # announced total before the first chunk
        assert dones[-1] == progress_seen[-1]["chunks_total"] >= 2
        assert dones == sorted(dones)

    def test_mixed_document_reports_skipped_pages(self, queue, tmp_path):
        p = tmp_path / "mixed.pdf"
        p.write_bytes(build_pdf([PAGE, None, PAGE]))
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        res = _enqueue_document(queue, p, sha="mix1")
        worker._process(queue.claim_next())
        result = queue.task_status(res["task_id"])["result"]
        assert result["pages"] == 3
        assert result["skipped_pages"] == [2]

    def test_fully_scanned_is_poison_dead(self, queue, tmp_path):
        p = tmp_path / "scanned.pdf"
        p.write_bytes(build_pdf([None, None]))
        worker = _make_worker(queue, FakeMemory())
        res = _enqueue_document(queue, p, sha="scan1")
        worker._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "dead"
        assert "v0.5b" in status["last_error"]

    def test_missing_spool_file_is_poison(self, queue, tmp_path):
        worker = _make_worker(queue, FakeMemory())
        res = _enqueue_document(queue, tmp_path / "ghost.pdf", sha="gone")
        worker._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "dead"
        assert "spool file missing" in status["last_error"]

    def test_midway_infra_failure_is_retryable_with_progress(self, queue, doc_pdf, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_CHUNK_CHARS", "120")
        mem = FakeMemory(raises_on_call=2)  # second chunk hits a dead Ollama
        worker = _make_worker(queue, mem)
        res = _enqueue_document(queue, doc_pdf)
        worker._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "failed_retryable"
        assert "ConnectionError" in status["last_error"]
        assert status["result"]["chunks_done"] == 1  # partial progress preserved

    def test_graph_defaults_off_for_documents(self, queue, doc_pdf):
        seen = {}

        def fake_call_with_graph(mem, enable_graph, default, fn):
            seen["enable_graph"], seen["default"] = enable_graph, default
            return fn()

        mem = FakeMemory()
        worker = _make_worker(queue, mem, call_with_graph=fake_call_with_graph)
        _enqueue_document(queue, doc_pdf)
        worker._process(queue.claim_next())
        assert seen["enable_graph"] is False and seen["default"] is False

    def test_interleave_drains_conversation_between_chunks(self, queue, doc_pdf, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_CHUNK_CHARS", "120")  # several chunks
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        doc = _enqueue_document(queue, doc_pdf)
        job = queue.claim_next()  # document claimed first (FIFO)
        conv = queue.enqueue(
            user_id="alice", agent_id=None, run_id=None,
            messages=[{"role": "user", "content": "fato conversacional urgente"}], params={},
        )
        worker._process(job)
        # both finished; the conversation was processed by the interleave
        assert queue.task_status(conv["task_id"])["status"] == "done"
        assert queue.task_status(doc["task_id"])["status"] == "done"
        conv_calls = [c for c in mem.calls if "urgente" in c[0][0]["content"]]
        assert len(conv_calls) == 1
        assert conv_calls[0][1].get("prompt") is None  # conversation path untouched
        # the conversation did NOT wait for the last chunk
        first_conv_pos = next(i for i, c in enumerate(mem.calls) if "urgente" in c[0][0]["content"])
        assert first_conv_pos < len(mem.calls) - 1

    def test_interleave_disabled_keeps_conversation_waiting(self, queue, doc_pdf, monkeypatch):
        monkeypatch.setenv("MEM0_DOC_CHUNK_CHARS", "120")
        monkeypatch.setenv("MEM0_DOC_INTERLEAVE", "false")
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        _enqueue_document(queue, doc_pdf)
        job = queue.claim_next()
        conv = queue.enqueue(
            user_id="alice", agent_id=None, run_id=None,
            messages=[{"role": "user", "content": "espera na fila"}], params={},
        )
        worker._process(job)
        assert queue.task_status(conv["task_id"])["status"] == "pending"

    def test_purge_scopes_by_created_at_sparing_updated_memories(self, queue, doc_pdf, monkeypatch):
        # R14: the purge must combine task_id AND created_at==submitted_at so a
        # pre-existing memory stamped by an UPDATE (which keeps its original
        # created_at) survives a retry purge.
        captured = {}

        class FakeQdrant:
            def delete(self, collection_name, points_selector):
                captured["filter"] = points_selector.filter

        class MemWithStore(FakeMemory):
            def __init__(self):
                super().__init__()
                self.vector_store = type("VS", (), {
                    "client": FakeQdrant(), "collection_name": "test_coll",
                })()

        mem = MemWithStore()
        worker = _make_worker(queue, mem)
        _enqueue_document(queue, doc_pdf, sha="r14")
        job = queue.claim_next()
        worker._process(job)
        conditions = captured["filter"].must
        keys = {c.key for c in conditions}
        assert keys == {"task_id", "created_at"}
        from datetime import datetime

        created = next(c for c in conditions if c.key == "created_at")
        submitted = datetime.fromisoformat(job["submitted_at"])
        assert created.range.gte == submitted  # qdrant-client parses the ISO string
        assert created.range.lte == submitted
