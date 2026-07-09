"""Shared test fixtures for mem0-mcp-selfhosted."""

import os

import pytest


@pytest.fixture(autouse=True)
def suppress_telemetry():
    """Ensure telemetry is always disabled in tests."""
    os.environ["MEM0_TELEMETRY"] = "false"


@pytest.fixture(autouse=True)
def _sync_ingest_by_default(monkeypatch, tmp_path):
    """Legacy tests exercise the synchronous add path; async-ingest tests
    opt back in explicitly. The worker never autostarts under pytest and the
    queue DB is always isolated in tmp so tests never touch a real queue."""
    monkeypatch.setenv("MEM0_ASYNC_INGEST", "false")
    monkeypatch.setenv("MEM0_QUEUE_WORKER", "false")
    monkeypatch.setenv("MEM0_QUEUE_DB_PATH", str(tmp_path / "ingest_queue.db"))
    monkeypatch.setenv("MEM0_QUEUE_WARMUP", "false")
    monkeypatch.setenv("MEM0_OBSERVE_URL", "")
    monkeypatch.setenv("MEM0_DOC_SPOOL_DIR", str(tmp_path / "ingest_spool"))
    monkeypatch.setenv("MEM0_DOC_PATH_ALLOWLIST", str(tmp_path))

    import mem0_mcp_selfhosted.server as server_mod

    server_mod._ingest_queue = None
    server_mod._ingest_worker = None
    yield
    server_mod._ingest_queue = None
    server_mod._ingest_worker = None
