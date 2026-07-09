"""Unit tests for the ingest worker's job processing (no live infrastructure).

The worker loop itself is a thin claim/sleep shell; the behavior worth locking
is in _process(): submitted_at -> created_at injection, task_id provenance,
error classification (poison vs retryable), purge-on-retry, and result shape.
"""

from __future__ import annotations

import pytest

import mem0_mcp_selfhosted.ingest_worker as worker_mod
from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.ingest_worker import IngestWorker, _is_retryable


class FakeMemory:
    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {
            "results": [{"id": "mem-1", "memory": "fact", "event": "ADD"}]
        }
        self.raises = raises
        self.calls = []

    def add(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _make_worker(queue, mem, **overrides):
    defaults = dict(max_attempts=3, backoff_base_s=0.0, poll_interval_s=0.01)
    defaults.update(overrides)
    return IngestWorker(queue, lambda: mem, **defaults)


def _enqueue_and_claim(queue, content="fact", params=None):
    queue.enqueue(
        user_id="alice", agent_id=None, run_id=None,
        messages=[{"role": "user", "content": content}], params=params or {},
    )
    return queue.claim_next()


class TestErrorClassification:
    def test_infra_errors_are_retryable(self):
        assert _is_retryable(ConnectionError("refused")) is True
        assert _is_retryable(TimeoutError("slow")) is True
        assert _is_retryable(RuntimeError("weird")) is True

    def test_payload_errors_are_poison(self):
        assert _is_retryable(ValueError("bad")) is False
        assert _is_retryable(TypeError("bad")) is False
        assert _is_retryable(KeyError("bad")) is False


class TestProcess:
    def test_success_injects_canonical_time_and_provenance(self, queue):
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        job = _enqueue_and_claim(queue, params={"metadata": {"source": "chat"}})

        worker._process(job)

        messages, kwargs = mem.calls[0]
        assert messages == [{"role": "user", "content": "fact"}]
        assert kwargs["infer"] is True
        assert kwargs["user_id"] == "alice"
        # submitted_at is the canonical fact time; task_id rides for provenance
        assert kwargs["metadata"]["created_at"] == job["submitted_at"]
        assert kwargs["metadata"]["task_id"] == job["task_id"]
        assert kwargs["metadata"]["source"] == "chat"

        status = queue.task_status(job["task_id"])
        assert status["status"] == "done"
        assert status["result"]["memory_ids"] == ["mem-1"]
        assert status["result"]["events"][0]["event"] == "ADD"

    def test_empty_extraction_is_done_with_reason(self, queue):
        mem = FakeMemory(result={"results": []})
        worker = _make_worker(queue, mem)
        job = _enqueue_and_claim(queue)
        worker._process(job)
        status = queue.task_status(job["task_id"])
        assert status["status"] == "done"
        assert status["result"]["reason"] == "no_new_facts"

    def test_poison_error_goes_dead(self, queue):
        mem = FakeMemory(raises=ValueError("malformed"))
        worker = _make_worker(queue, mem)
        job = _enqueue_and_claim(queue)
        worker._process(job)
        status = queue.task_status(job["task_id"])
        assert status["status"] == "dead"
        assert "ValueError" in status["last_error"]

    def test_retryable_error_backs_off_then_dies_at_cap(self, queue):
        mem = FakeMemory(raises=ConnectionError("ollama down"))
        worker = _make_worker(queue, mem, max_attempts=2)
        job = _enqueue_and_claim(queue)

        worker._process(job)
        assert queue.task_status(job["task_id"])["status"] == "failed_retryable"

        job2 = queue.claim_next()  # backoff 0 -> immediately due
        worker._process(job2)
        assert queue.task_status(job["task_id"])["status"] == "dead"

    def test_memory_unavailable_is_retryable(self, queue):
        worker = IngestWorker(queue, lambda: None, max_attempts=3, backoff_base_s=0.0)
        job = _enqueue_and_claim(queue)
        worker._process(job)
        status = queue.task_status(job["task_id"])
        assert status["status"] == "failed_retryable"
        assert "not initialized" in status["last_error"]

    def test_every_attempt_purges_prior_points(self, queue, monkeypatch):
        # Purge must be unconditional: recover_orphans() resets a crashed job
        # WITHOUT bumping attempts, so a first-attempt crash would otherwise
        # reprocess on top of its own orphaned points (mass duplication for
        # multi-chunk documents).
        purged = []
        monkeypatch.setattr(
            worker_mod, "_purge_task_points",
            lambda mem, tid, created_at=None: purged.append((tid, created_at)),
        )
        mem = FakeMemory()
        worker = _make_worker(queue, mem)

        job = _enqueue_and_claim(queue)
        worker._process(job)
        # even the first attempt cleans up, scoped by the job's canonical time
        assert purged == [(job["task_id"], job["submitted_at"])]

        queue2_job = _enqueue_and_claim(queue, content="other")
        queue.mark_failed(queue2_job["task_id"], "boom", retryable=True, max_attempts=3, backoff_base_s=0)
        retry_job = queue.claim_next()
        worker._process(retry_job)
        assert purged[-1][0] == retry_job["task_id"]

    def test_crash_orphan_reprocess_purges(self, queue, monkeypatch):
        # Simulated crash mid-first-attempt: job stuck in processing, service
        # restarts, recover_orphans() requeues it with attempts still 0 — the
        # reprocess MUST purge the partial points of the dead attempt.
        purged = []
        monkeypatch.setattr(
            worker_mod, "_purge_task_points",
            lambda mem, tid, created_at=None: purged.append(tid),
        )
        mem = FakeMemory()
        worker = _make_worker(queue, mem)

        job = _enqueue_and_claim(queue)  # claimed, then "crash" (never processed)
        assert queue.recover_orphans() == 1
        requeued = queue.claim_next()
        assert requeued["task_id"] == job["task_id"]
        assert requeued["attempts"] == 0  # recovery does not bump attempts
        worker._process(requeued)
        assert purged == [job["task_id"]]
        assert queue.task_status(job["task_id"])["status"] == "done"

    def test_graph_toggle_rides_with_the_job(self, queue):
        seen = {}

        def fake_call_with_graph(mem, enable_graph, default, fn):
            seen["enable_graph"] = enable_graph
            return fn()

        mem = FakeMemory()
        worker = _make_worker(queue, mem, call_with_graph=fake_call_with_graph)
        job = _enqueue_and_claim(queue, params={"enable_graph": True})
        worker._process(job)
        assert seen["enable_graph"] is True
        assert queue.task_status(job["task_id"])["status"] == "done"


class TestLifecycle:
    def test_start_is_idempotent_and_stop_joins(self, queue):
        worker = _make_worker(queue, FakeMemory())
        worker.start()
        first_thread = worker._thread
        worker.start()
        assert worker._thread is first_thread
        assert worker.is_alive()
        worker.stop()
        first_thread.join(timeout=2)
        assert not worker.is_alive()

    def test_worker_drains_queue_end_to_end(self, queue):
        mem = FakeMemory()
        worker = _make_worker(queue, mem)
        res = queue.enqueue(
            user_id="alice", agent_id=None, run_id=None,
            messages=[{"role": "user", "content": "drain me"}], params={},
        )
        worker.start()
        worker.notify()
        deadline = 50
        import time
        for _ in range(deadline):
            if queue.task_status(res["task_id"])["status"] == "done":
                break
            time.sleep(0.05)
        worker.stop()
        assert queue.task_status(res["task_id"])["status"] == "done"
