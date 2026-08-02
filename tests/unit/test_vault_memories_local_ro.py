"""Leitores SQLite read-only: fila de ingestão e histórico.

As fixtures usam o DDL REAL dos bancos de produção (extraído de
``~/.mem0/*.db``), inclusive a coluna ``kind`` que entrou por ``ALTER TABLE`` e
por isso fica no fim da tabela. Um schema aproximado aprovaria consulta que
quebra em produção.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import anyio
import pytest

from mem0_mcp_selfhosted.vault.memories.local_ro import (
    HistoryReader,
    QueueReader,
    job_view,
)

QUEUE_DDL = """
CREATE TABLE ingest_queue (
  task_id          TEXT PRIMARY KEY,
  idempotency_key  TEXT NOT NULL,
  user_id          TEXT,
  agent_id         TEXT,
  run_id           TEXT,
  payload          TEXT NOT NULL,
  params           TEXT,
  submitted_at     TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending',
  attempts         INTEGER NOT NULL DEFAULT 0,
  next_attempt_at  TEXT,
  started_at       TEXT,
  finished_at      TEXT,
  last_error       TEXT,
  result           TEXT
, kind TEXT NOT NULL DEFAULT 'conversation')
"""

HISTORY_DDL = """
CREATE TABLE history (
  id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT, new_memory TEXT,
  event TEXT, created_at DATETIME, updated_at DATETIME, is_deleted INTEGER,
  actor_id TEXT, role TEXT
)
"""

INTENTS_DDL = """
CREATE TABLE delete_intents (
  op_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, scope TEXT,
  before_image TEXT, created_at DATETIME, updated_at DATETIME
)
"""


def _ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


@pytest.fixture
def queue_db(tmp_path):
    path = tmp_path / "ingest_queue.db"
    conn = sqlite3.connect(path)
    conn.execute(QUEUE_DDL)
    rows = [
        ("tsk_pend", "k1", "ana", None, None, "[]", None, _ago(300), "pending", 0,
         None, None, None, None, None, "conversation"),
        ("tsk_proc", "k2", "ana", None, None, "[]", None, _ago(120), "processing", 1,
         None, _ago(100), None, None,
         json.dumps({"chunks_done": 3, "chunks_total": 10, "last_progress_at": _ago(20),
                     "source_doc": "manual.pdf"}), "document"),
        ("tsk_dead", "k3", "ana", None, None, "[]", None, _ago(900), "dead", 4,
         None, None, _ago(800), "boom: payload inválido", None, "conversation"),
        ("tsk_done", "k4", "ana", None, None, "[]", None, _ago(1200), "done", 1,
         None, None, _ago(1100), None,
         json.dumps({"memory_ids": ["m1", "m2"]}), "conversation"),
    ]
    conn.executemany(
        "INSERT INTO ingest_queue VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def history_db(tmp_path):
    path = tmp_path / "history.db"
    conn = sqlite3.connect(path)
    conn.execute(HISTORY_DDL)
    conn.execute(INTENTS_DDL)
    conn.executemany(
        "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("h1", "mem-1", None, "primeiro texto", "ADD", _ago(500), _ago(500), 0,
             "Maria", "user"),
            ("h2", "mem-1", "primeiro texto", "texto novo", "UPDATE", _ago(300),
             _ago(300), 0, None, "user"),
            ("h3", "outra-mem", None, "de outra", "ADD", _ago(100), _ago(100), 0, None, None),
        ],
    )
    conn.execute(
        "INSERT INTO delete_intents VALUES (?,?,?,?,?,?)",
        ("op-1", "mem-1", "user_id=ana", "{}", _ago(50), _ago(50)),
    )
    conn.commit()
    conn.close()
    return path


class TestQueueReader:
    def test_summary_counts_by_status(self, queue_db):
        summary = anyio.run(QueueReader(queue_db).summary)
        assert summary["by_status"] == {"pending": 1, "processing": 1, "dead": 1, "done": 1}

    def test_depth_counts_only_active_jobs(self, queue_db):
        """Fila = o que ainda tem trabalho. `done`/`dead` não são profundidade."""
        assert anyio.run(QueueReader(queue_db).summary)["depth"] == 2

    def test_oldest_pending_age(self, queue_db):
        age = anyio.run(QueueReader(queue_db).summary)["oldest_pending_age_s"]
        assert 250 < age < 400

    def test_jobs_returns_active_with_progress(self, queue_db):
        jobs = anyio.run(lambda: QueueReader(queue_db).jobs())
        doc = next(j for j in jobs if j["task_id"] == "tsk_proc")
        assert doc["chunks_done"] == 3 and doc["chunks_total"] == 10
        assert doc["source_doc"] == "manual.pdf"
        assert doc["kind"] == "document"

    def test_heartbeat_age_is_from_progress_not_submission(self, queue_db):
        """Job velho com heartbeat fresco NÃO está travado — um PDF demora."""
        jobs = anyio.run(lambda: QueueReader(queue_db).jobs())
        doc = next(j for j in jobs if j["task_id"] == "tsk_proc")
        assert doc["heartbeat_age_s"] < doc["age_s"]
        assert doc["heartbeat_age_s"] < 60

    def test_dead_letters_carry_the_error(self, queue_db):
        dead = anyio.run(lambda: QueueReader(queue_db).jobs(statuses=("dead",)))
        assert dead[0]["last_error"].startswith("boom")
        assert dead[0]["attempts"] == 4

    def test_job_by_id(self, queue_db):
        job = anyio.run(lambda: QueueReader(queue_db).job("tsk_done"))
        assert job["memory_ids"] == ["m1", "m2"]
        assert anyio.run(lambda: QueueReader(queue_db).job("nao-existe")) is None

    def test_reader_cannot_write(self, queue_db):
        """`query_only` é o que impede uma UI de leitura de tocar a fila."""
        import sqlite3 as s

        conn = s.connect(f"file:{queue_db}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        with pytest.raises(s.OperationalError):
            conn.execute("DELETE FROM ingest_queue")
        conn.close()


class TestJobView:
    def test_malformed_result_json_does_not_raise(self):
        view = job_view({"task_id": "t", "result": "{isso não é json"})
        assert view["chunks_done"] is None and view["memory_ids"] == []

    def test_result_that_is_not_an_object_is_ignored(self):
        assert job_view({"task_id": "t", "result": "[1,2,3]"})["chunks_total"] is None

    def test_missing_kind_defaults_to_conversation(self):
        assert job_view({"task_id": "t"})["kind"] == "conversation"


class TestHistoryReader:
    def test_returns_only_that_memorys_events_in_order(self, history_db):
        events = anyio.run(lambda: HistoryReader(history_db).for_memory("mem-1"))
        assert [e["event"] for e in events] == ["ADD", "UPDATE"]
        assert all(e["memory_id"] == "mem-1" for e in events)

    def test_exposes_actor_id(self, history_db):
        events = anyio.run(lambda: HistoryReader(history_db).for_memory("mem-1"))
        assert events[0]["actor_id"] == "Maria"

    def test_unknown_memory_yields_empty(self, history_db):
        assert anyio.run(lambda: HistoryReader(history_db).for_memory("nada")) == []

    def test_delete_intent_is_found(self, history_db):
        intent = anyio.run(lambda: HistoryReader(history_db).delete_intent("mem-1"))
        assert intent["op_id"] == "op-1"

    def test_no_delete_intent_returns_none(self, history_db):
        assert anyio.run(lambda: HistoryReader(history_db).delete_intent("outra-mem")) is None
