"""Pins _estimate_wait_s defaults + arithmetic.

The recalibrated defaults (ADD 180 / CHUNK 120 / UPDATE 200,  jul/2026) are
UX contract: they must rarely UNDERSHOOT (an undershoot makes MCP clients retry —
that nearly caused a duplicate add once). The old suite only tested values via
explicit setenv, so a bad default would have shipped silently. These lock the
defaults when the envs are ABSENT and check the kind-aware sum.
"""
from __future__ import annotations

import pytest

import mem0_mcp_selfhosted.server as server_mod


class FakeQueue:
    """Minimal stand-in for IngestQueue's read surface used by _estimate_wait_s."""

    def __init__(self, conversations=0, updates=0, doc_chunks=0):
        self._by_kind = {"conversation": conversations, "update": updates}
        self._doc_chunks = doc_chunks

    def queue_status(self):
        return {"depth_by_kind": dict(self._by_kind)}

    def pending_document_chunks(self):
        return self._doc_chunks

    def depth(self):
        return sum(self._by_kind.values()) + self._doc_chunks


@pytest.fixture(autouse=True)
def _clear_estimate_envs(monkeypatch):
    for k in ("MEM0_QUEUE_EST_ADD_S", "MEM0_DOC_EST_CHUNK_S", "MEM0_QUEUE_EST_UPDATE_S"):
        monkeypatch.delenv(k, raising=False)


def test_empty_queue_is_zero():
    assert server_mod._estimate_wait_s(FakeQueue()) == 0


def test_default_add_is_180():
    assert server_mod._estimate_wait_s(FakeQueue(conversations=1)) == 180


def test_default_update_is_200():
    assert server_mod._estimate_wait_s(FakeQueue(updates=1)) == 200


def test_default_chunk_is_120():
    assert server_mod._estimate_wait_s(FakeQueue(doc_chunks=1)) == 120


def test_mixed_queue_sums_kind_aware():
    # 2 conversas × 180 + 1 update × 200 + 3 chunks × 120 = 920
    q = FakeQueue(conversations=2, updates=1, doc_chunks=3)
    assert server_mod._estimate_wait_s(q) == 2 * 180 + 200 + 3 * 120


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("MEM0_QUEUE_EST_ADD_S", "40")
    monkeypatch.setenv("MEM0_DOC_EST_CHUNK_S", "35")
    assert server_mod._estimate_wait_s(FakeQueue(conversations=1, doc_chunks=2)) == 40 + 2 * 35
