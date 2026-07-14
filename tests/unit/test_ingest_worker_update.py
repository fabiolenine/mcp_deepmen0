"""Unit tests for async ``update_memory`` jobs (kind='update') in the worker.

Mirrors test_ingest_worker.py: a real IngestQueue + a fake Memory whose only job
is to record ``update`` calls. Locks the _process_update contract: it calls
mem.update(memory_id, data=text, metadata~task_id), marks done with the UPDATE
event, and classifies errors (poison -> dead, infra -> retryable).
"""

from __future__ import annotations

import pytest

from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.ingest_worker import IngestWorker


class FakeUpdateMemory:
    def __init__(self, raises=None):
        self.raises = raises
        self.calls = []

    def update(self, memory_id, data=None, metadata=None):
        self.calls.append((memory_id, data, metadata))
        if self.raises is not None:
            raise self.raises
        return {"message": "Memory updated successfully!"}


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _make_worker(queue, mem, **overrides):
    defaults = dict(max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)
    defaults.update(overrides)
    return IngestWorker(queue, lambda: mem, **defaults)


def _enqueue_update(queue, memory_id="uuid-1", text="new text"):
    queue.enqueue(
        user_id="alice", agent_id=None, run_id=None,
        messages=[{"role": "user", "content": f"[update memory_id={memory_id}]\n{text}"}],
        params={"memory_id": memory_id, "text": text},
        kind="update",
    )
    return queue.claim_next()


class TestProcessUpdate:
    def test_applies_update_and_marks_done(self, queue):
        mem = FakeUpdateMemory()
        worker = _make_worker(queue, mem)
        job = _enqueue_update(queue, memory_id="uuid-1", text="new text")

        worker._process(job)

        assert len(mem.calls) == 1
        mid, data, metadata = mem.calls[0]
        assert mid == "uuid-1"
        assert data == "new text"
        # task_id rides for provenance; created_at is NOT stamped (update preserves it)
        assert metadata["task_id"] == job["task_id"]
        assert "created_at" not in metadata

        status = queue.task_status(job["task_id"])
        assert status["status"] == "done"
        assert status["result"]["memory_ids"] == ["uuid-1"]
        assert status["result"]["events"][0]["event"] == "UPDATE"

    def test_missing_params_is_poison(self, queue):
        queue.enqueue(
            user_id="alice", agent_id=None, run_id=None,
            messages=[{"role": "user", "content": "[update ...]"}],
            params={}, kind="update",
        )
        job = queue.claim_next()
        mem = FakeUpdateMemory()
        worker = _make_worker(queue, mem)

        worker._process(job)

        assert mem.calls == []
        assert queue.task_status(job["task_id"])["status"] == "dead"

    def test_not_found_goes_dead(self, queue):
        mem = FakeUpdateMemory(raises=ValueError("memory not found"))
        worker = _make_worker(queue, mem)
        job = _enqueue_update(queue)
        worker._process(job)
        assert queue.task_status(job["task_id"])["status"] == "dead"

    def test_infra_error_is_retryable(self, queue):
        mem = FakeUpdateMemory(raises=ConnectionError("qdrant unreachable"))
        worker = _make_worker(queue, mem)
        job = _enqueue_update(queue)
        worker._process(job)
        assert queue.task_status(job["task_id"])["status"] == "failed_retryable"
