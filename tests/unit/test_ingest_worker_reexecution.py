"""Item #8 — instrumentação de RE-EXECUÇÃO de documento + contrato do ingest_failed.

O sinal-gatilho do item #8 é o evento `doc_reexecution` com `chunks_done_before_start>0`
(chunks já-feitos prestes a serem REPROCESSADOS). Cobre retry-por-exceção E o caso que uma
métrica baseada em `attempts` PERDERIA: crash + recover_orphans (volta com attempts=0).
"""
from __future__ import annotations

import pytest

import mem0_mcp_selfhosted.ingest_worker as iw
from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.ingest_worker import IngestWorker


class FakeMemory:
    def add(self, *a, **k):
        return {"results": []}


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _worker(queue, mem):
    return IngestWorker(queue, lambda: mem, max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)


def _capture(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(iw, "_observe", lambda e: events.append(e))
    return events


def _enqueue_doc(queue, sha):
    return queue.enqueue(
        user_id="alice", agent_id=None, run_id=None,
        messages=[{"role": "user", "content": f"[document sha256={sha}]"}],
        params={"spool_path": f"/tmp/{sha}.pdf", "doc_sha256": sha, "filename": f"{sha}.pdf"},
        kind="document",
    )["task_id"]


def _reexec(events):
    return [e for e in events if e.get("event") == "doc_reexecution"]


# --- doc_reexecution ------------------------------------------------------- #

def test_crash_orphan_reexecution_detected_despite_attempts_0(queue, monkeypatch):
    """O caso que uma métrica por `attempts` PERDERIA: crash após k chunks + recover_orphans
    -> re-claim com attempts=0. Tem que emitir doc_reexecution com k>0."""
    events = _capture(monkeypatch)
    worker = _worker(queue, FakeMemory())
    monkeypatch.setattr(worker, "_process_document", lambda *a, **k: None)  # isola o _process

    tid = _enqueue_doc(queue, "crash1")
    queue.update_progress(tid, {"chunks_done": 2, "chunks_total": 5})
    queue.claim_next()                       # -> processing (attempts=0)
    assert queue.recover_orphans() == 1      # crash: processing -> pending, attempts intacto
    job = queue.claim_next()                 # re-claim; attempts=0, result.chunks_done=2
    worker._process(job)

    evs = _reexec(events)
    assert len(evs) == 1
    assert evs[0]["chunks_done_before_start"] == 2
    assert evs[0]["attempts"] == 0           # <- crash-orphan: detectado mesmo com attempts=0
    assert evs[0]["kind"] == "document"
    assert evs[0]["chunks_total"] == 5


def test_retryable_reexecution_emits_with_progress(queue, monkeypatch):
    events = _capture(monkeypatch)
    worker = _worker(queue, FakeMemory())
    monkeypatch.setattr(worker, "_process_document", lambda *a, **k: None)

    tid = _enqueue_doc(queue, "retry1")
    queue.update_progress(tid, {"chunks_done": 1, "chunks_total": 3})
    job = queue.claim_next()
    queue.mark_failed(tid, "boom", retryable=True, max_attempts=3, backoff_base_s=0.0)  # attempts=1
    job = queue.claim_next()                 # re-claim (backoff 0)
    worker._process(job)

    evs = _reexec(events)
    assert len(evs) == 1
    assert evs[0]["chunks_done_before_start"] == 1
    assert evs[0]["attempts"] == 1


def test_first_attempt_does_not_emit(queue, monkeypatch):
    events = _capture(monkeypatch)
    worker = _worker(queue, FakeMemory())
    monkeypatch.setattr(worker, "_process_document", lambda *a, **k: None)

    _enqueue_doc(queue, "fresh1")
    job = queue.claim_next()                 # attempts=0, sem result prévio
    worker._process(job)
    assert _reexec(events) == []             # primeira execução limpa -> nada


def test_conversation_reexecution_not_flagged_as_doc(queue, monkeypatch):
    events = _capture(monkeypatch)
    worker = _worker(queue, FakeMemory())
    monkeypatch.setattr(worker, "_process_conversation", lambda *a, **k: None)

    tid = queue.enqueue(user_id="alice", agent_id=None, run_id=None,
                        messages=[{"role": "user", "content": "oi"}], params={}, kind="conversation")["task_id"]
    queue.claim_next()
    queue.mark_failed(tid, "boom", retryable=True, max_attempts=3, backoff_base_s=0.0)
    job = queue.claim_next()
    worker._process(job)
    assert _reexec(events) == []             # doc_reexecution é só p/ kind=document


# --- contrato do ingest_failed --------------------------------------------- #

def test_memory_init_failure_carries_kind_and_retryable(queue, monkeypatch):
    """O path de falha de init do Memory precisa carregar kind+retryable (senão um doc atrasado
    por infra fora some da query kind=document AND retryable=true)."""
    events = _capture(monkeypatch)
    worker = IngestWorker(queue, lambda: None, max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)
    tid = _enqueue_doc(queue, "infra1")
    job = queue.claim_next()
    worker._process(job)

    failed = [e for e in events if e.get("event") == "ingest_failed"]
    assert len(failed) == 1
    assert failed[0]["kind"] == "document"       # <- contrato consertado
    assert failed[0]["retryable"] is True
    assert failed[0]["error"] == "memory_init"
