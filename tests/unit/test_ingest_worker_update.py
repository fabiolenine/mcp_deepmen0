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
    def __init__(self, raises=None, returns=None):
        self.raises = raises
        # DeepMem0 v0.7: the fork's update returns {"message", "id", "old_id"};
        # a legacy/in-place update returns id == old_id (or just a message).
        self.returns = returns or {"message": "Memory updated successfully!"}
        self.calls = []

    def update(self, memory_id, data=None, metadata=None):
        self.calls.append((memory_id, data, metadata))
        if self.raises is not None:
            raise self.raises
        return self.returns


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
        # DeepMem0 v0.7: task_id rides for provenance AND created_at is stamped with
        # submitted_at (canonical record-time for the new version; inert on the legacy
        # in-place path, where the fork preserves the original created_at anyway).
        assert metadata["task_id"] == job["task_id"]
        assert metadata["created_at"] == job["submitted_at"]

        status = queue.task_status(job["task_id"])
        assert status["status"] == "done"
        # legacy/in-place return (id == old_id): stays a plain UPDATE event.
        assert status["result"]["memory_ids"] == ["uuid-1"]
        assert status["result"]["events"][0]["event"] == "UPDATE"

    def test_versioned_update_surfaces_new_id(self, queue):
        # DeepMem0 v0.7: the fork mints a new version and returns its id; the worker
        # must report the NEW current id and a SUPERSEDED(old)+ADD(new) event pair.
        mem = FakeUpdateMemory(returns={"message": "ok", "id": "uuid-2", "old_id": "uuid-1"})
        worker = _make_worker(queue, mem)
        job = _enqueue_update(queue, memory_id="uuid-1", text="new text")

        worker._process(job)

        result = queue.task_status(job["task_id"])["result"]
        assert result["memory_ids"] == ["uuid-2"]
        events = {(e["event"], e.get("id")) for e in result["events"]}
        assert ("SUPERSEDED", "uuid-1") in events
        assert ("ADD", "uuid-2") in events
        superseded = next(e for e in result["events"] if e["event"] == "SUPERSEDED")
        assert superseded["superseded_by"] == "uuid-2"

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


class TestCrashRecovery:
    """DeepMem0 v0.7.2 (§E): a crash AFTER mem.update but BEFORE mark_done leaves the
    job 'processing'; recover_orphans re-queues it (no attempt bump) and the reprocess
    runs purge-on-retry scoped by task_id AND created_at==submitted_at (so v2 is deleted
    and the fork re-versions cleanly, while an UPDATEd pre-existing memory is spared)."""

    def test_crash_before_mark_done_recovers_and_purges_on_retry(self, queue, monkeypatch):
        import mem0_mcp_selfhosted.ingest_worker as iw

        mem = FakeUpdateMemory(returns={"message": "ok", "id": "uuid-2", "old_id": "uuid-1"})
        worker = _make_worker(queue, mem)
        job = _enqueue_update(queue, memory_id="uuid-1", text="new text")

        # CRASH: the job was claimed (processing) and the worker died before mark_done.
        assert queue.task_status(job["task_id"])["status"] == "processing"

        # Boot recovery flips processing -> pending WITHOUT bumping attempts.
        assert queue.recover_orphans() == 1
        assert queue.task_status(job["task_id"])["status"] == "pending"

        # Spy the module-level purge to prove it runs with the right scope on retry.
        purges = []
        monkeypatch.setattr(iw, "_purge_task_points",
                            lambda mem_, tid, created_at=None: purges.append((tid, created_at)))

        job2 = queue.claim_next()
        assert job2["task_id"] == job["task_id"]  # same job, re-claimed
        worker._process(job2)

        # purge-on-retry scoped by task_id AND created_at==submitted_at (spares UPDATEd
        # pre-existing memories; deletes only the half-written v2 this task created).
        assert purges == [(job["task_id"], job["submitted_at"])]
        assert len(mem.calls) == 1  # update re-applied
        st = queue.task_status(job["task_id"])
        assert st["status"] == "done"
        assert st["result"]["memory_ids"] == ["uuid-2"]

    def test_recover_orphans_does_not_bump_attempts(self, queue):
        mem = FakeUpdateMemory()
        _make_worker(queue, mem)
        job = _enqueue_update(queue)  # claimed -> processing
        queue.recover_orphans()
        # re-claimable immediately at the original submitted_at, no attempt penalty
        again = queue.claim_next()
        assert again["task_id"] == job["task_id"]
        assert again.get("attempts", 0) == 0
