"""Worker vision path (v0.5b): scanned-page OCR and standalone images.

Real poppler rasterizes PDFs built in-code; the VLM (image_extract) and the
Memory are mocked. Verifies strict two-phase model management and provenance.
"""

from __future__ import annotations

import shutil

import pytest

import mem0_mcp_selfhosted.ingest_worker as worker_mod
from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.ingest_worker import IngestWorker
from tests.pdf_builder import build_pdf

poppler = pytest.mark.skipif(
    shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None,
    reason="poppler-utils not installed",
)

PAGE = "A decisão do projeto Hermes foi aprovada em 2026 com a porta 8081."
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeMemory:
    def __init__(self):
        self.calls = []

    def add(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"results": [{"id": f"mem-{len(self.calls)}", "memory": "fato", "event": "ADD"}]}


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _worker(queue, mem, **kw):
    d = dict(max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)
    d.update(kw)
    return IngestWorker(queue, lambda: mem, **d)


@pytest.fixture
def vision_on(monkeypatch):
    monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
    monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setenv("MEM0_LLM_MODEL", "llama3.1:8b:latest")


@pytest.fixture
def spy_vision(monkeypatch):
    """Record VLM transcriptions and the model-swap ordering.

    rasterize_pages is mocked to fake PNGs — turning a blank test page into a
    real raster is poppler's job (covered in test_pdf_extract); here we test
    the worker's orchestration and two-phase ordering.
    """
    events = []
    monkeypatch.setattr(worker_mod, "prepare_vision", lambda: events.append("prepare"))
    monkeypatch.setattr(worker_mod, "release_vision", lambda: events.append("release"))
    monkeypatch.setattr(worker_mod, "rasterize_pages",
                        lambda path, nums, **kw: {n: PNG for n in nums})

    def fake_transcribe(image, timeout_s=None):
        events.append("transcribe")
        return "Texto transcrito da página escaneada: fato OCR com número 42."

    monkeypatch.setattr(worker_mod, "transcribe_image", fake_transcribe)
    return events


def _enqueue(queue, spool, content_type, sha="v05b", filename="scan.pdf"):
    return queue.enqueue(
        user_id="alice", agent_id=None, run_id=None,
        messages=[{"role": "user", "content": f"[document sha256={sha}]"}],
        params={"spool_path": str(spool), "doc_sha256": sha, "filename": filename,
                "content_type": content_type},
        kind="document",
    )


@poppler
class TestScannedPdfOcr:
    def test_fully_scanned_pdf_ocrs_all_pages(self, queue, tmp_path, vision_on, spy_vision):
        p = tmp_path / "scanned.pdf"
        p.write_bytes(build_pdf([None, None]))  # no text layer
        mem = FakeMemory()
        res = _enqueue(queue, p, "application/pdf", filename="scan.pdf")
        _worker(queue, mem)._process(queue.claim_next())

        status = queue.task_status(res["task_id"])
        assert status["status"] == "done"
        result = status["result"]
        assert result["ocr_pages"] == [1, 2]
        assert result["skipped_pages"] == []
        assert "OCR" in mem.calls[0][0][0]["content"]
        # strict two-phase: prepare once, transcribe per page, release once,
        # THEN the add-loop — release must come before the first mem.add
        assert spy_vision == ["prepare", "transcribe", "transcribe", "release"]

    def test_mixed_pdf_ocrs_only_scanned_pages(self, queue, tmp_path, vision_on, spy_vision):
        p = tmp_path / "mixed.pdf"
        p.write_bytes(build_pdf([PAGE, None, PAGE]))  # page 2 scanned
        mem = FakeMemory()
        res = _enqueue(queue, p, "application/pdf")
        _worker(queue, mem)._process(queue.claim_next())
        result = queue.task_status(res["task_id"])["result"]
        assert result["ocr_pages"] == [2]  # only the scanned page
        assert result["skipped_pages"] == []
        assert spy_vision.count("transcribe") == 1

    def test_scanned_pdf_without_vision_is_poison(self, queue, tmp_path, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "false")
        p = tmp_path / "scanned.pdf"
        p.write_bytes(build_pdf([None, None]))
        res = _enqueue(queue, p, "application/pdf")
        _worker(queue, FakeMemory())._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "dead"
        assert "v0.5b" in status["last_error"]

    def test_ocr_failure_skips_page_not_job(self, queue, tmp_path, vision_on, monkeypatch):
        p = tmp_path / "mixed.pdf"
        p.write_bytes(build_pdf([PAGE, None, PAGE]))  # digital pages survive
        monkeypatch.setattr(worker_mod, "prepare_vision", lambda: None)
        monkeypatch.setattr(worker_mod, "release_vision", lambda: None)
        monkeypatch.setattr(worker_mod, "rasterize_pages",
                            lambda path, nums, **kw: {n: PNG for n in nums})
        monkeypatch.setattr(worker_mod, "transcribe_image",
                            lambda img, timeout_s=None: (_ for _ in ()).throw(ValueError("OCR broke")))
        mem = FakeMemory()
        res = _enqueue(queue, p, "application/pdf")
        _worker(queue, mem)._process(queue.claim_next())
        result = queue.task_status(res["task_id"])["result"]
        assert queue.task_status(res["task_id"])["status"] == "done"
        assert result["skipped_pages"] == [2]  # OCR failed -> skipped, digital pages ingested
        assert result["ocr_pages"] == []


class TestStandaloneImage:
    def test_image_transcribed_and_ingested(self, queue, tmp_path, vision_on, spy_vision):
        img = tmp_path / "foto.png"
        img.write_bytes(PNG)
        mem = FakeMemory()
        res = _enqueue(queue, img, "image/png", filename="foto.png")
        _worker(queue, mem)._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "done"
        result = status["result"]
        assert result["pages"] == 1 and result["ocr_pages"] == [1]
        md = mem.calls[0][1]["metadata"]
        assert md["source_doc"] == "foto.png"
        assert md["page_start"] == 1
        assert spy_vision == ["prepare", "transcribe", "release"]

    def test_image_without_vision_is_poison(self, queue, tmp_path, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "false")
        img = tmp_path / "foto.png"
        img.write_bytes(PNG)
        res = _enqueue(queue, img, "image/png")
        _worker(queue, FakeMemory())._process(queue.claim_next())
        status = queue.task_status(res["task_id"])
        assert status["status"] == "dead"
        assert "vision" in status["last_error"].lower()
