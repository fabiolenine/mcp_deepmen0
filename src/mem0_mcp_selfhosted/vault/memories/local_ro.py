"""Leitores dos bancos SQLite locais (fila de ingestão e histórico) — read-only.

Abrir em ``mode=ro`` + ``PRAGMA query_only=ON`` não é zelo: o worker de ingestão
escreve nesses arquivos o tempo todo, em WAL. Uma conexão de leitura comum já
seria segura para ler, mas ``query_only`` garante que nenhum caminho desta UI
possa escrever numa fila de produção nem que alguém acrescente um ``UPDATE``
por engano no futuro. É o mesmo padrão do ``scripts/infra_poller.py``.

Conexão por operação (não pooled): são consultas curtas e esparsas, e uma
conexão viva por horas dentro de um processo de UI só acumularia risco de
segurar snapshot antigo do WAL.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio

ACTIVE_STATUSES = ("pending", "processing", "failed_retryable")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _rows(path: Path, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    conn = _connect(path)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _age_s(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        moment = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - moment).total_seconds())


class QueueReader:
    """Estado da fila de ingestão (``~/.mem0/ingest_queue.db``)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    async def summary(self) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            by_status = {
                r["status"]: r["n"]
                for r in _rows(
                    self.path, "SELECT status, COUNT(*) AS n FROM ingest_queue GROUP BY status"
                )
            }
            by_kind = [
                dict(r)
                for r in _rows(
                    self.path,
                    "SELECT kind, status, COUNT(*) AS n FROM ingest_queue "
                    "WHERE status IN (?,?,?) GROUP BY kind, status",
                    ACTIVE_STATUSES,
                )
            ]
            oldest = _rows(
                self.path,
                "SELECT MIN(submitted_at) AS ts FROM ingest_queue WHERE status = 'pending'",
            )
            return {
                "by_status": by_status,
                "depth": sum(by_status.get(s, 0) for s in ACTIVE_STATUSES),
                "by_kind": by_kind,
                "oldest_pending_age_s": _age_s(oldest[0]["ts"] if oldest else None),
            }

        return await anyio.to_thread.run_sync(_read)

    async def jobs(self, statuses: tuple[str, ...] = ACTIVE_STATUSES, limit: int = 50) -> list[dict]:
        placeholders = ",".join("?" * len(statuses))

        def _read() -> list[dict[str, Any]]:
            return _rows(
                self.path,
                f"SELECT task_id, kind, status, user_id, submitted_at, started_at, "  # noqa: S608
                f"finished_at, attempts, last_error, result, next_attempt_at "
                f"FROM ingest_queue WHERE status IN ({placeholders}) "
                f"ORDER BY submitted_at DESC LIMIT ?",
                (*statuses, limit),
            )

        return [job_view(r) for r in await anyio.to_thread.run_sync(_read)]

    async def job(self, task_id: str) -> dict[str, Any] | None:
        def _read() -> list[dict[str, Any]]:
            return _rows(
                self.path, "SELECT * FROM ingest_queue WHERE task_id = ? LIMIT 1", (task_id,)
            )

        rows = await anyio.to_thread.run_sync(_read)
        return job_view(rows[0]) if rows else None


def job_view(row: dict[str, Any]) -> dict[str, Any]:
    """Linha da fila com o progresso extraído do JSON de ``result``.

    ``chunks_done``/``chunks_total`` e ``last_progress_at`` vivem dentro do
    ``result``, não em colunas — e o que indica job travado é a idade do
    HEARTBEAT, não a idade do job: um PDF de 50 páginas leva dezenas de minutos
    legitimamente, e alarmar pela idade do job daria falso positivo sempre.
    """
    result: dict[str, Any] = {}
    raw = row.get("result")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                result = parsed
        except (ValueError, TypeError):
            result = {}
    return {
        "task_id": row.get("task_id"),
        "kind": row.get("kind") or "conversation",
        "status": row.get("status"),
        "user_id": row.get("user_id"),
        "submitted_at": row.get("submitted_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "attempts": row.get("attempts") or 0,
        "next_attempt_at": row.get("next_attempt_at"),
        "last_error": row.get("last_error"),
        "chunks_done": result.get("chunks_done"),
        "chunks_total": result.get("chunks_total"),
        "source_doc": result.get("source_doc"),
        "memory_ids": result.get("memory_ids") or [],
        "heartbeat_age_s": _age_s(result.get("last_progress_at")),
        "age_s": _age_s(row.get("submitted_at")),
    }


class HistoryReader:
    """Histórico de eventos por memória (``~/.mem0/history.db``)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    async def for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            return _rows(
                self.path,
                "SELECT id, memory_id, old_memory, new_memory, event, created_at, "
                "updated_at, is_deleted, actor_id, role FROM history "
                "WHERE memory_id = ? ORDER BY created_at ASC, id ASC",
                (memory_id,),
            )

        return await anyio.to_thread.run_sync(_read)

    async def delete_intent(self, memory_id: str) -> dict[str, Any] | None:
        """Intenção de delete registrada (o journal durável da v0.7.2)."""

        def _read() -> list[dict[str, Any]]:
            return _rows(
                self.path,
                "SELECT op_id, memory_id, scope, created_at, updated_at "
                "FROM delete_intents WHERE memory_id = ? ORDER BY created_at DESC LIMIT 1",
                (memory_id,),
            )

        rows = await anyio.to_thread.run_sync(_read)
        return rows[0] if rows else None
