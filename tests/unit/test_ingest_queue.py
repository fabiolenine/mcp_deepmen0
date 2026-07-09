"""Unit tests for the durable SQLite ingest queue (no live infrastructure)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mem0_mcp_selfhosted.ingest_queue import IngestQueue, idempotency_key

MSGS = [{"role": "user", "content": "o embedder agora é bge-m3"}]


@pytest.fixture
def queue(tmp_path):
    return IngestQueue(tmp_path / "q.db")


def _enqueue(queue, content="fact", user_id="alice"):
    return queue.enqueue(
        user_id=user_id, agent_id=None, run_id=None,
        messages=[{"role": "user", "content": content}], params={},
    )


class TestEnqueue:
    def test_returns_task_id_and_depth(self, queue):
        res = _enqueue(queue)
        assert res["task_id"].startswith("tsk_")
        assert res["duplicate"] is False
        assert res["queue_depth"] == 1
        assert res["submitted_at"] <= datetime.now(timezone.utc).isoformat()

    def test_active_duplicate_returns_same_task(self, queue):
        first = _enqueue(queue)
        second = _enqueue(queue)
        assert second["task_id"] == first["task_id"]
        assert second["duplicate"] is True
        assert second["queue_depth"] == 1  # no ghost job

    def test_terminal_job_does_not_block_resubmission(self, queue):
        # re-adding the same fact after completion is reinforcement, not retry
        first = _enqueue(queue)
        queue.claim_next()
        queue.mark_done(first["task_id"], {"memory_ids": ["m1"]})
        second = _enqueue(queue)
        assert second["task_id"] != first["task_id"]
        assert second["duplicate"] is False

    def test_idempotency_key_is_scope_sensitive(self):
        k1 = idempotency_key("alice", None, None, MSGS)
        k2 = idempotency_key("bob", None, None, MSGS)
        k3 = idempotency_key("alice", "agent-1", None, MSGS)
        assert len({k1, k2, k3}) == 3


class TestClaimAndComplete:
    def test_fifo_by_submitted_at(self, queue):
        a = _enqueue(queue, "first")
        b = _enqueue(queue, "second")
        assert queue.claim_next()["task_id"] == a["task_id"]
        assert queue.claim_next()["task_id"] == b["task_id"]
        assert queue.claim_next() is None

    def test_claim_marks_processing_and_decodes_payload(self, queue):
        _enqueue(queue, "hello")
        job = queue.claim_next()
        assert job["status"] == "processing"
        assert job["messages"] == [{"role": "user", "content": "hello"}]
        assert job["params"] == {}
        assert queue.task_status(job["task_id"])["status"] == "processing"

    def test_mark_done_stores_result(self, queue):
        res = _enqueue(queue)
        queue.claim_next()
        queue.mark_done(res["task_id"], {"memory_ids": ["m1", "m2"]})
        status = queue.task_status(res["task_id"])
        assert status["status"] == "done"
        assert status["result"]["memory_ids"] == ["m1", "m2"]
        assert "finished_at" in status

    def test_unknown_task_status_is_none(self, queue):
        assert queue.task_status("tsk_missing") is None


class TestFailureModel:
    def test_retryable_backs_off_then_redispatches(self, queue):
        res = _enqueue(queue)
        queue.claim_next()
        status = queue.mark_failed(
            res["task_id"], "ollama timeout", retryable=True, max_attempts=4, backoff_base_s=60,
        )
        assert status == "failed_retryable"
        assert queue.claim_next() is None  # not due yet
        wakeup = queue.next_wakeup_in_s()
        assert wakeup is not None and 0 < wakeup <= 60

    def test_zero_backoff_is_immediately_claimable(self, queue):
        res = _enqueue(queue)
        queue.claim_next()
        queue.mark_failed(res["task_id"], "flaky", retryable=True, max_attempts=4, backoff_base_s=0)
        job = queue.claim_next()
        assert job is not None and job["attempts"] == 1

    def test_poison_goes_straight_to_dead(self, queue):
        res = _enqueue(queue)
        queue.claim_next()
        status = queue.mark_failed(
            res["task_id"], "bad payload", retryable=False, max_attempts=4, backoff_base_s=0,
        )
        assert status == "dead"
        assert queue.claim_next() is None
        assert queue.task_status(res["task_id"])["last_error"] == "bad payload"

    def test_max_attempts_caps_retries(self, queue):
        res = _enqueue(queue)
        for attempt in range(3):
            job = queue.claim_next()
            assert job is not None, f"attempt {attempt} should be dispatchable"
            queue.mark_failed(res["task_id"], "still down", retryable=True, max_attempts=3, backoff_base_s=0)
        assert queue.task_status(res["task_id"])["status"] == "dead"
        assert queue.task_status(res["task_id"])["attempts"] == 3
        assert queue.claim_next() is None

    def test_recover_orphans_requeues_processing(self, queue):
        res = _enqueue(queue)
        queue.claim_next()
        assert queue.claim_next() is None  # stuck in processing (simulated crash)
        assert queue.recover_orphans() == 1
        job = queue.claim_next()
        assert job["task_id"] == res["task_id"]


class TestGarbageCollection:
    def _finish_and_age(self, queue, task_id, status, age_s):
        """Force a terminal row's finished_at into the past."""
        import sqlite3
        from datetime import timedelta

        aged = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
        with sqlite3.connect(queue.db_path) as conn:
            conn.execute(
                "UPDATE ingest_queue SET status = ?, finished_at = ? WHERE task_id = ?",
                (status, aged, task_id),
            )

    def test_prunes_old_done_keeps_recent(self, queue):
        old = _enqueue(queue, "old")
        recent = _enqueue(queue, "recent")
        self._finish_and_age(queue, old["task_id"], "done", age_s=10 * 24 * 3600)
        self._finish_and_age(queue, recent["task_id"], "done", age_s=60)
        purged = queue.gc(done_retention_s=7 * 24 * 3600, dead_retention_s=30 * 24 * 3600)
        assert purged["done_purged"] == 1
        assert queue.task_status(old["task_id"]) is None
        assert queue.task_status(recent["task_id"])["status"] == "done"

    def test_dead_lingers_longer_than_done(self, queue):
        # both 10 days old: done is pruned, dead (inspect/requeue material) stays
        done = _enqueue(queue, "done-job")
        dead = _enqueue(queue, "dead-job")
        self._finish_and_age(queue, done["task_id"], "done", age_s=10 * 24 * 3600)
        self._finish_and_age(queue, dead["task_id"], "dead", age_s=10 * 24 * 3600)
        purged = queue.gc(done_retention_s=7 * 24 * 3600, dead_retention_s=30 * 24 * 3600)
        assert purged == {"done_purged": 1, "dead_purged": 0}
        assert queue.task_status(dead["task_id"])["status"] == "dead"

    def test_old_dead_is_eventually_pruned(self, queue):
        dead = _enqueue(queue, "ancient")
        self._finish_and_age(queue, dead["task_id"], "dead", age_s=40 * 24 * 3600)
        purged = queue.gc()
        assert purged["dead_purged"] == 1
        assert queue.task_status(dead["task_id"]) is None

    def test_retention_zero_keeps_forever(self, queue):
        done = _enqueue(queue, "keep-me")
        self._finish_and_age(queue, done["task_id"], "done", age_s=400 * 24 * 3600)
        purged = queue.gc(done_retention_s=0, dead_retention_s=0)
        assert purged == {"done_purged": 0, "dead_purged": 0}
        assert queue.task_status(done["task_id"])["status"] == "done"

    def test_active_jobs_are_never_touched(self, queue):
        _enqueue(queue, "pending-job")
        claimed = _enqueue(queue, "processing-job")
        queue.claim_next()
        purged = queue.gc(done_retention_s=1, dead_retention_s=1)
        assert purged == {"done_purged": 0, "dead_purged": 0}
        assert queue.depth() == 2
        assert claimed["task_id"]


class TestKindAndProgress:
    def test_migrates_v04_schema_adding_kind(self, tmp_path):
        # database created by the v0.4 schema (no kind column) must be usable
        import sqlite3
        db = tmp_path / "legacy.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ingest_queue ("
                "task_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL, "
                "user_id TEXT, agent_id TEXT, run_id TEXT, payload TEXT NOT NULL, "
                "params TEXT, submitted_at TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, "
                "next_attempt_at TEXT, started_at TEXT, finished_at TEXT, "
                "last_error TEXT, result TEXT)"
            )
            conn.execute(
                "INSERT INTO ingest_queue (task_id, idempotency_key, payload, submitted_at) "
                "VALUES ('tsk_old', 'k', '[]', '2026-07-01T00:00:00+00:00')"
            )
        queue = IngestQueue(db)  # runs the additive migration
        job = queue.claim_next()
        assert job["task_id"] == "tsk_old"
        assert job["kind"] == "conversation"  # backfilled default

    def test_claim_filtered_by_kind(self, queue):
        queue.enqueue(user_id="u", agent_id=None, run_id=None,
                      messages=[{"role": "user", "content": "[document sha256=abc]"}],
                      params={"doc_sha256": "abc"}, kind="document")
        conv = _enqueue(queue, "conversa")
        job = queue.claim_next(only_kind="conversation")
        assert job["task_id"] == conv["task_id"]  # skipped the older document
        assert queue.claim_next(only_kind="conversation") is None
        doc = queue.claim_next()  # unfiltered still reaches the document
        assert doc["kind"] == "document"

    def test_update_progress_merges_and_heartbeats(self, queue):
        res = _enqueue(queue, "doc-job")
        queue.claim_next()
        queue.update_progress(res["task_id"], {"chunks_done": 1, "chunks_total": 5})
        queue.update_progress(res["task_id"], {"chunks_done": 2})
        status = queue.task_status(res["task_id"])
        assert status["result"]["chunks_done"] == 2
        assert status["result"]["chunks_total"] == 5  # merge preserves prior keys
        assert "last_progress_at" in status["result"]
        assert status["status"] == "processing"  # progress never changes status

    def test_mark_done_after_progress_is_terminal(self, queue):
        res = _enqueue(queue, "doc-job-2")
        queue.claim_next()
        queue.update_progress(res["task_id"], {"chunks_done": 3, "chunks_total": 3})
        queue.mark_done(res["task_id"], {"memory_ids": ["m1"], "chunks_total": 3})
        status = queue.task_status(res["task_id"])
        assert status["status"] == "done"
        assert status["result"]["memory_ids"] == ["m1"]

    def test_latest_done_finds_terminal_document(self, queue):
        from mem0_mcp_selfhosted.ingest_queue import idempotency_key
        msgs = [{"role": "user", "content": "[document sha256=xyz]"}]
        res = queue.enqueue(user_id="u", agent_id=None, run_id=None,
                            messages=msgs, params={"doc_sha256": "xyz"}, kind="document")
        key = idempotency_key("u", None, None, msgs)
        assert queue.latest_done(key) is None  # still pending
        queue.claim_next()
        queue.mark_done(res["task_id"], {"memory_ids": ["m1", "m2"]})
        done = queue.latest_done(key)
        assert done["task_id"] == res["task_id"]
        assert done["result"]["memory_ids"] == ["m1", "m2"]

    def test_referenced_doc_hashes(self, queue):
        queue.enqueue(user_id="u", agent_id=None, run_id=None,
                      messages=[{"role": "user", "content": "[document sha256=aaa]"}],
                      params={"doc_sha256": "aaa"}, kind="document")
        done = queue.enqueue(user_id="u", agent_id=None, run_id=None,
                             messages=[{"role": "user", "content": "[document sha256=bbb]"}],
                             params={"doc_sha256": "bbb"}, kind="document")
        _enqueue(queue, "conversa sem doc")
        queue.claim_next(only_kind="document")
        queue.mark_done(done["task_id"])
        # done rows still reference their file (until gc prunes the row)
        assert queue.referenced_doc_hashes() == {"aaa", "bbb"}

    def test_queue_status_reports_kind_and_heartbeat(self, queue):
        queue.enqueue(user_id="u", agent_id=None, run_id=None,
                      messages=[{"role": "user", "content": "[document sha256=ccc]"}],
                      params={"doc_sha256": "ccc"}, kind="document")
        _enqueue(queue, "conversa")
        doc = queue.claim_next(only_kind="document")
        queue.update_progress(doc["task_id"], {"chunks_done": 4, "chunks_total": 10})
        status = queue.queue_status()
        assert status["depth_by_kind"] == {"document": 1, "conversation": 1}
        job = status["active_jobs"][0]
        assert job["kind"] == "document"
        assert job["chunks_done"] == 4 and job["chunks_total"] == 10
        assert job["heartbeat_age_s"] >= 0


class TestIntrospection:
    def test_pending_for_scope_counts_only_that_user(self, queue):
        _enqueue(queue, "a", user_id="alice")
        _enqueue(queue, "b", user_id="alice")
        _enqueue(queue, "c", user_id="bob")
        assert queue.pending_for_scope("alice") == 2
        assert queue.pending_for_scope("bob") == 1
        assert queue.pending_for_scope("carol") == 0

    def test_queue_status_shape(self, queue):
        done = _enqueue(queue, "x")
        queue.claim_next()
        queue.mark_done(done["task_id"])
        _enqueue(queue, "y")
        _enqueue(queue, "z")
        queue.claim_next()  # y -> processing; z stays pending
        status = queue.queue_status()
        assert status["done"] == 1
        assert status["pending"] == 1
        assert status["processing"] == 1
        assert status["depth"] == 2
        assert status["oldest_pending_age_s"] >= 0
