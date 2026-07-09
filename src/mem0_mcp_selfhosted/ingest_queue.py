"""Durable SQLite ingest queue for asynchronous add_memory.

LLM extraction dominates add latency (~26-37s on local Ollama), so the
``infer=true`` path is decoupled from the MCP ack: the tool enqueues and
returns a ``task_id`` immediately; a single serial worker (ingest_worker.py)
drains the queue. Design invariants:

- ``submitted_at`` is the canonical fact time. The worker injects it as the
  memory's ``created_at``, so record-time semantics (as_of anchors,
  supersession direction) survive the queue delay.
- Idempotency: an identical payload with an ACTIVE job (pending/processing/
  failed_retryable) returns the existing task_id instead of a duplicate job —
  a client that lost the ack can safely resubmit. Terminal jobs (done/dead)
  do NOT block resubmission: re-adding the same fact later is a legitimate
  reinforcement signal (DeepMem0 v0.2 T1), not a retry.
- Failure model: retryable errors back off exponentially
  (base * 2^attempts) up to a max-attempts cap, then the job goes ``dead``
  (poison message) — inspectable, never blocking the queue. Orphaned
  ``processing`` jobs from a crash are reset to ``pending`` at worker boot.
- Garbage collection: terminal rows are audit trail, not queue state. ``gc()``
  prunes ``done`` jobs after MEM0_QUEUE_DONE_RETENTION (default 7 days) and
  ``dead`` jobs after MEM0_QUEUE_DEAD_RETENTION (default 30 days — they are
  the inspect/requeue material, so they linger longer); a retention <= 0
  keeps that class forever. The worker runs gc opportunistically on idle.

WAL + busy_timeout let the MCP server threads (enqueue/status reads) and the
worker (claims/updates) share the file without lock wars. Connections are
opened per operation — cheap at queue rates and safe across threads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("pending", "processing", "failed_retryable")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_queue (
  task_id          TEXT PRIMARY KEY,
  idempotency_key  TEXT NOT NULL,
  user_id          TEXT,
  agent_id         TEXT,
  run_id           TEXT,
  payload          TEXT NOT NULL,
  params           TEXT,
  kind             TEXT NOT NULL DEFAULT 'conversation',
  submitted_at     TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending',
  attempts         INTEGER NOT NULL DEFAULT 0,
  next_attempt_at  TEXT,
  started_at       TEXT,
  finished_at      TEXT,
  last_error       TEXT,
  result           TEXT
);
CREATE INDEX IF NOT EXISTS idx_iq_dispatch
  ON ingest_queue(status, next_attempt_at, submitted_at);
CREATE INDEX IF NOT EXISTS idx_iq_idem
  ON ingest_queue(idempotency_key, status);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def idempotency_key(user_id: str | None, agent_id: str | None, run_id: str | None, messages: Any) -> str:
    """Stable key over scope + normalized payload (client-retry dedup)."""
    canonical = json.dumps(
        {"u": user_id, "a": agent_id, "r": run_id, "m": messages},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IngestQueue:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # must precede table creation for incremental_vacuum (gc) to work;
            # no-op on an existing database, which is harmless at queue scale
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.executescript(_SCHEMA)
            # additive migration for databases created by the v0.4 schema;
            # v0.4 code keeps working on the migrated file (column has default)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(ingest_queue)")}
            if "kind" not in cols:
                conn.execute(
                    "ALTER TABLE ingest_queue ADD COLUMN kind TEXT NOT NULL DEFAULT 'conversation'"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc(
        self,
        *,
        done_retention_s: float = 7 * 24 * 3600,
        dead_retention_s: float = 30 * 24 * 3600,
    ) -> dict[str, int]:
        """Prune terminal rows past their retention; retention <= 0 keeps forever.

        Cutoffs compare against ``finished_at`` (set on done and dead). Freed
        pages are handed back to the OS via incremental_vacuum — at queue
        rates a full VACUUM would be overkill.
        """
        now = _utcnow()
        purged = {"done_purged": 0, "dead_purged": 0}
        with self._connect() as conn:
            if done_retention_s > 0:
                cur = conn.execute(
                    "DELETE FROM ingest_queue WHERE status = 'done' AND finished_at < ?",
                    (_iso(now - timedelta(seconds=done_retention_s)),),
                )
                purged["done_purged"] = cur.rowcount
            if dead_retention_s > 0:
                cur = conn.execute(
                    "DELETE FROM ingest_queue WHERE status = 'dead' AND finished_at < ?",
                    (_iso(now - timedelta(seconds=dead_retention_s)),),
                )
                purged["dead_purged"] = cur.rowcount
            if purged["done_purged"] or purged["dead_purged"]:
                try:
                    conn.execute("PRAGMA incremental_vacuum")
                except sqlite3.Error:
                    pass
        return purged

    # ------------------------------------------------------------------
    # Producer side (MCP tool)
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        messages: list[dict],
        params: dict | None = None,
        kind: str = "conversation",
    ) -> dict[str, Any]:
        """Insert a job (or return the active duplicate). Never blocks on the LLM."""
        key = idempotency_key(user_id, agent_id, run_id, messages)
        now = _iso(_utcnow())
        with self._connect() as conn:
            dup = conn.execute(
                f"SELECT task_id, submitted_at FROM ingest_queue "
                f"WHERE idempotency_key = ? AND status IN {ACTIVE_STATUSES} "
                f"ORDER BY submitted_at LIMIT 1",
                (key,),
            ).fetchone()
            if dup is not None:
                depth = self._depth(conn)
                return {
                    "task_id": dup["task_id"],
                    "submitted_at": dup["submitted_at"],
                    "duplicate": True,
                    "queue_depth": depth,
                }
            task_id = "tsk_" + uuid.uuid4().hex[:10]
            conn.execute(
                "INSERT INTO ingest_queue "
                "(task_id, idempotency_key, user_id, agent_id, run_id, payload, params, kind, submitted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id, key, user_id, agent_id, run_id,
                    json.dumps(messages, ensure_ascii=False),
                    json.dumps(params or {}, ensure_ascii=False),
                    kind,
                    now,
                ),
            )
            depth = self._depth(conn)
        return {"task_id": task_id, "submitted_at": now, "duplicate": False, "queue_depth": depth}

    # ------------------------------------------------------------------
    # Consumer side (worker)
    # ------------------------------------------------------------------

    def claim_next(self, only_kind: str | None = None) -> dict[str, Any] | None:
        """Atomically claim the oldest dispatchable job (FIFO by submitted_at).

        ``only_kind`` restricts the claim to one job kind — the document
        interleave uses it to drain conversation adds between chunks without
        ever claiming another document.
        """
        now = _iso(_utcnow())
        kind_clause = "AND kind = ? " if only_kind else ""
        args: tuple = (now, only_kind) if only_kind else (now,)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM ingest_queue "
                "WHERE status IN ('pending', 'failed_retryable') "
                "  AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                f"{kind_clause}"
                "ORDER BY submitted_at LIMIT 1",
                args,
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute(
                "UPDATE ingest_queue SET status = 'processing', started_at = ? WHERE task_id = ?",
                (now, row["task_id"]),
            )
            conn.execute("COMMIT")
        job = dict(row)
        job["status"] = "processing"
        job["started_at"] = now
        job["messages"] = json.loads(job.pop("payload"))
        job["params"] = json.loads(job["params"] or "{}")
        return job

    def update_progress(self, task_id: str, progress: dict) -> None:
        """Merge partial progress into ``result`` while a long job runs.

        Adds ``last_progress_at`` (the worker heartbeat): dashboards tell a
        long-but-alive document job from a stuck worker by this age, never by
        the job's total age. Does not change ``status``.
        """
        now = _iso(_utcnow())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM ingest_queue WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return
            try:
                current = json.loads(row["result"]) if row["result"] else {}
            except (json.JSONDecodeError, TypeError):
                current = {}
            current.update(progress)
            current["last_progress_at"] = now
            conn.execute(
                "UPDATE ingest_queue SET result = ? WHERE task_id = ?",
                (json.dumps(current, ensure_ascii=False), task_id),
            )

    def latest_done(self, key: str) -> dict[str, Any] | None:
        """Most recent 'done' row for an idempotency key (document dedup).

        Re-adding the same conversational fact is a reinforcement signal, but
        re-ingesting a whole document is a 15-20 min accident — the caller uses
        this to answer ``already_ingested`` until the done row is gc-pruned.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_id, submitted_at, finished_at, result FROM ingest_queue "
                "WHERE idempotency_key = ? AND status = 'done' "
                "ORDER BY finished_at DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        if out.get("result"):
            try:
                out["result"] = json.loads(out["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        return out

    def pending_document_chunks(self) -> int:
        """Chunks still to process across active document jobs (wait estimates).

        Uses each job's submit-time ``chunks_estimate`` minus the progress
        already heartbeated; a document with no estimate counts as one chunk.
        """
        total = 0
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT params, result FROM ingest_queue "
                f"WHERE kind != 'conversation' AND status IN {ACTIVE_STATUSES}"
            ).fetchall()
        for row in rows:
            est = 0
            try:
                est = int((json.loads(row["params"] or "{}") or {}).get("chunks_estimate") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            done = 0
            if row["result"]:
                try:
                    done = int((json.loads(row["result"]) or {}).get("chunks_done") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            total += max(est - done, 1 if est == 0 else 0)
        return total

    def referenced_doc_hashes(self) -> set[str]:
        """doc_sha256 values referenced by any row still in the table.

        The spool gc deletes files whose hash appears in no row — a spool file
        lives exactly as long as some job (active or terminal, pre-prune)
        might still need or explain it.
        """
        hashes: set[str] = set()
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT params FROM ingest_queue WHERE kind != 'conversation' AND params IS NOT NULL"
            ):
                try:
                    sha = (json.loads(row["params"]) or {}).get("doc_sha256")
                    if sha:
                        hashes.add(sha)
                except (json.JSONDecodeError, TypeError):
                    continue
        return hashes

    def mark_done(self, task_id: str, result: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingest_queue SET status = 'done', finished_at = ?, last_error = NULL, result = ? "
                "WHERE task_id = ?",
                (_iso(_utcnow()), json.dumps(result or {}, ensure_ascii=False), task_id),
            )

    def mark_failed(
        self,
        task_id: str,
        error: str,
        *,
        retryable: bool,
        max_attempts: int,
        backoff_base_s: float,
    ) -> str:
        """Record a failure; returns the resulting status ('failed_retryable' or 'dead')."""
        now = _utcnow()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM ingest_queue WHERE task_id = ?", (task_id,)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            if not retryable or attempts >= max_attempts:
                status, next_at = "dead", None
            else:
                status = "failed_retryable"
                next_at = _iso(now + timedelta(seconds=backoff_base_s * (2 ** (attempts - 1))))
            conn.execute(
                "UPDATE ingest_queue SET status = ?, attempts = ?, next_attempt_at = ?, "
                "last_error = ?, finished_at = ? WHERE task_id = ?",
                (status, attempts, next_at, error[:2000], _iso(now) if status == "dead" else None, task_id),
            )
        return status

    def recover_orphans(self) -> int:
        """Jobs stuck in 'processing' after a crash/restart go back to the line."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE ingest_queue SET status = 'pending', started_at = NULL, "
                "last_error = 'reset after worker restart' WHERE status = 'processing'"
            )
            return cur.rowcount

    def next_wakeup_in_s(self) -> float | None:
        """Seconds until the earliest scheduled retry, if any retry is waiting."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(next_attempt_at) AS nxt FROM ingest_queue "
                "WHERE status = 'failed_retryable' AND next_attempt_at IS NOT NULL"
            ).fetchone()
        if not row or not row["nxt"]:
            return None
        try:
            delta = (datetime.fromisoformat(row["nxt"]) - _utcnow()).total_seconds()
            return max(delta, 0.0)
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------
    # Introspection (status tools / read-your-writes signal)
    # ------------------------------------------------------------------

    def _depth(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM ingest_queue WHERE status IN {ACTIVE_STATUSES}"
        ).fetchone()
        return int(row["n"])

    def depth(self) -> int:
        with self._connect() as conn:
            return self._depth(conn)

    def pending_for_scope(self, user_id: str | None) -> int:
        """Active jobs for a user — the search-time read-your-writes signal."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM ingest_queue "
                f"WHERE status IN {ACTIVE_STATUSES} AND user_id = ?",
                (user_id,),
            ).fetchone()
            return int(row["n"])

    def task_status(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_id, status, submitted_at, attempts, started_at, finished_at, "
                "last_error, result, next_attempt_at FROM ingest_queue WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        if out.get("result"):
            try:
                out["result"] = json.loads(out["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        return {k: v for k, v in out.items() if v is not None}

    def queue_status(self) -> dict[str, Any]:
        with self._connect() as conn:
            counts = {
                r["status"]: r["n"]
                for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM ingest_queue GROUP BY status"
                ).fetchall()
            }
            oldest = conn.execute(
                "SELECT MIN(submitted_at) AS oldest FROM ingest_queue WHERE status = 'pending'"
            ).fetchone()
            kind_rows = conn.execute(
                f"SELECT kind, COUNT(*) AS n FROM ingest_queue "
                f"WHERE status IN {ACTIVE_STATUSES} GROUP BY kind"
            ).fetchall()
            active_jobs = conn.execute(
                "SELECT task_id, kind, started_at, result FROM ingest_queue "
                "WHERE status = 'processing' ORDER BY started_at"
            ).fetchall()
        status: dict[str, Any] = {
            "depth": sum(counts.get(s, 0) for s in ACTIVE_STATUSES),
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "failed_retryable": counts.get("failed_retryable", 0),
            "dead": counts.get("dead", 0),
            "done": counts.get("done", 0),
            "depth_by_kind": {r["kind"]: r["n"] for r in kind_rows},
        }
        if oldest and oldest["oldest"]:
            try:
                age = (_utcnow() - datetime.fromisoformat(oldest["oldest"])).total_seconds()
                status["oldest_pending_age_s"] = round(max(age, 0.0), 1)
            except ValueError:
                pass
        jobs = []
        for row in active_jobs:
            job: dict[str, Any] = {"task_id": row["task_id"], "kind": row["kind"]}
            heartbeat = row["started_at"]
            if row["result"]:
                try:
                    progress = json.loads(row["result"])
                    heartbeat = progress.get("last_progress_at") or heartbeat
                    for k in ("chunks_done", "chunks_total"):
                        if k in progress:
                            job[k] = progress[k]
                except (json.JSONDecodeError, TypeError):
                    pass
            if heartbeat:
                try:
                    hb_age = (_utcnow() - datetime.fromisoformat(heartbeat)).total_seconds()
                    # stuck-worker signal: alert on THIS growing, not on job age
                    # (a 30-chunk document is legitimately old but heartbeats
                    # every chunk)
                    job["heartbeat_age_s"] = round(max(hb_age, 0.0), 1)
                except ValueError:
                    pass
            jobs.append(job)
        if jobs:
            status["active_jobs"] = jobs
        return status
