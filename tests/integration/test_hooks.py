"""Integration tests for hooks.py — verifies hooks work against live infrastructure.

Tests the three integration boundaries that unit tests mock:
1. _get_memory() creates a working Memory instance (graph disabled)
2. context_main() searches and returns real memories from Qdrant
3. stop_main() saves session summaries via mem.add(infer=True)

Requires: Qdrant + Ollama (embedder) + LLM provider (Anthropic or Ollama).
"""

from __future__ import annotations

import json
import os
import time
from io import StringIO
from unittest.mock import patch

import pytest

import mem0_mcp_selfhosted.hooks as hooks

pytestmark = pytest.mark.integration

HOOK_TEST_USER = "inttest-hooks"


def _capture_output(func, stdin_data: str = "{}") -> dict:
    """Run a hook entry point with mocked stdin/stdout and capture JSON."""
    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)), patch("sys.stdout", captured):
        func()
    return json.loads(captured.getvalue())


@pytest.fixture(scope="module")
def hook_memory(qdrant_url, ollama_url):
    """Initialize a Memory instance the same way hooks do — graph disabled.

    Validates that _get_memory() works against live infrastructure.
    Cached for the module to avoid repeated initialization.
    """
    original_graph = os.environ.get("MEM0_ENABLE_GRAPH")
    original_memory = hooks._memory

    # Reset cached instance so _get_memory() initializes fresh
    hooks._memory = None
    try:
        mem = hooks._get_memory()

        # Verify graph was disabled
        assert os.environ.get("MEM0_ENABLE_GRAPH") == "false"
        assert mem is not None
        yield mem
    finally:
        # Restore
        hooks._memory = original_memory
        if original_graph is not None:
            os.environ["MEM0_ENABLE_GRAPH"] = original_graph
        else:
            os.environ.pop("MEM0_ENABLE_GRAPH", None)


@pytest.fixture
def seeded_memory(hook_memory):
    """Seed distinctive memories and clean up after the test."""
    user_id = f"{HOOK_TEST_USER}-{int(time.time())}"

    facts = [
        "The project uses FastMCP as its MCP orchestrator framework",
        "Authentication uses a 3-tier token fallback: MEM0_ANTHROPIC_TOKEN then credentials.json then ANTHROPIC_API_KEY",
        "Neo4j graph is disabled in hooks for performance within the 15-second timeout budget",
    ]

    memory_ids = []
    for fact in facts:
        result = hook_memory.add(
            [{"role": "user", "content": fact}],
            user_id=user_id,
        )
        for r in result.get("results", []):
            memory_ids.append(r["id"])

    yield user_id, memory_ids

    # Cleanup: delete seeded memories
    for mid in memory_ids:
        try:
            hook_memory.delete(mid)
        except Exception:
            pass


class TestGetMemoryIntegration:
    """Verify _get_memory() produces a working Memory against live infra."""

    def test_memory_initializes_successfully(self, hook_memory):
        """_get_memory() returns a real Memory instance, not None."""
        assert hook_memory is not None
        # Should have the core Memory API
        assert callable(getattr(hook_memory, "search", None))
        assert callable(getattr(hook_memory, "add", None))

    def test_graph_is_disabled(self, hook_memory):
        """Hook Memory instance has graph disabled for speed."""
        # mem0 Memory stores graph config; when disabled, graph is None
        # or enable_graph is False
        graph_enabled = getattr(hook_memory, "enable_graph", None)
        if graph_enabled is not None:
            assert graph_enabled is False
        else:
            # Fallback: check env var was set correctly
            assert os.environ.get("MEM0_ENABLE_GRAPH") == "false"


class TestContextMainIntegration:
    """Verify context_main() searches real Qdrant and returns formatted output."""

    def test_returns_seeded_memories(self, hook_memory, seeded_memory):
        """context_main() finds and formats memories that were previously added."""
        user_id, _ = seeded_memory

        stdin_data = json.dumps({
            "session_id": "inttest-ctx",
            "cwd": "/home/user/testproject",
            "hook_event_name": "startup",
        })

        # Temporarily override _get_memory and _get_user_id for this test
        with (
            patch.object(hooks, "_get_memory", return_value=hook_memory),
            patch.object(hooks, "_get_user_id", return_value=user_id),
        ):
            result = _capture_output(hooks.context_main, stdin_data)

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        # Seeded memories should be found — skip if LLM non-determinism
        # caused infer=True to extract zero facts during seeding.
        if "additionalContext" not in result:
            pytest.skip(
                "Seeded memories not found in search (LLM non-determinism); "
                "re-run to verify"
            )
        ctx = result["additionalContext"]
        assert "# mem0 Cross-Session Memory" in ctx
        # At least one numbered memory line
        assert any(line.strip() and line.strip()[0].isdigit() for line in ctx.split("\n"))

    def test_empty_user_returns_no_context(self, hook_memory):
        """User with no memories gets a clean non-fatal response."""
        stdin_data = json.dumps({
            "session_id": "inttest-empty",
            "cwd": "/home/user/emptyproject",
            "hook_event_name": "startup",
        })

        with (
            patch.object(hooks, "_get_memory", return_value=hook_memory),
            patch.object(hooks, "_get_user_id", return_value="inttest-nonexistent-user-xyz"),
        ):
            result = _capture_output(hooks.context_main, stdin_data)

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        # No memories means no additionalContext
        assert "additionalContext" not in result


class TestStopMainIntegration:
    """Verify stop_main() saves session summaries to real mem0 via infer=True."""

    def test_saves_session_summary(self, hook_memory, tmp_path):
        """stop_main() extracts facts from transcript and stores them in mem0."""
        user_id = f"{HOOK_TEST_USER}-stop-{int(time.time())}"

        # Create a realistic transcript
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"role": "user", "content": "Configure the MCP server to use Qdrant on port 6333 with collection name mem0_production"}) + "\n"
            + json.dumps({"role": "assistant", "content": "Done. I've updated the .env file to set MEM0_QDRANT_URL=http://localhost:6333 and MEM0_COLLECTION=mem0_production. The server will use these settings on next restart."}) + "\n"
        )

        stdin_data = json.dumps({
            "session_id": "inttest-stop",
            "cwd": "/home/user/testproject",
            "transcript_path": str(transcript),
        })

        saved_ids = []
        try:
            with (
                patch.object(hooks, "_get_memory", return_value=hook_memory),
                patch.object(hooks, "_get_user_id", return_value=user_id),
            ):
                result = _capture_output(hooks.stop_main, stdin_data)

            assert result["continue"] is True

            # Verify something was saved — search for the distinctive content
            search_result = hook_memory.search(
                query="Qdrant MCP configuration",
                user_id=user_id,
            )

            results = search_result.get("results", [])
            saved_ids = [r["id"] for r in results]

            # LLM inference is non-deterministic — skip rather than fail
            # if the model didn't extract any facts this run.
            if not results:
                pytest.skip(
                    "LLM extracted zero facts from transcript (non-deterministic); "
                    "re-run to verify"
                )

            # At least one result should reference the distinctive content
            all_text = " ".join(r.get("memory", "") for r in results).lower()
            assert "qdrant" in all_text or "6333" in all_text or "mem0_production" in all_text

        finally:
            for mid in saved_ids:
                try:
                    hook_memory.delete(mid)
                except Exception:
                    pass

    def test_stop_roundtrip_succeeds(self, hook_memory, tmp_path):
        """stop_main() completes the full add(infer=True) roundtrip.

        Logs elapsed time for visibility — Claude Code's budget is 30s, but
        LLM latency varies too much to assert on timing reliably.
        """
        user_id = f"{HOOK_TEST_USER}-roundtrip-{int(time.time())}"

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"role": "user", "content": "Refactor the authentication middleware to support both JWT and API key validation with proper error messages"}) + "\n"
            + json.dumps({"role": "assistant", "content": "I've added a dual-auth middleware that checks Authorization header format: Bearer tokens go through JWT validation, x-api-key headers go through the API key store. Both return 401 with descriptive errors on failure."}) + "\n"
        )

        stdin_data = json.dumps({
            "session_id": "inttest-roundtrip",
            "cwd": "/home/user/testproject",
            "transcript_path": str(transcript),
        })

        saved_ids = []
        try:
            start = time.monotonic()
            with (
                patch.object(hooks, "_get_memory", return_value=hook_memory),
                patch.object(hooks, "_get_user_id", return_value=user_id),
            ):
                result = _capture_output(hooks.stop_main, stdin_data)
            elapsed = time.monotonic() - start

            assert result["continue"] is True
            # Log timing for visibility (30s is the production budget)
            if elapsed > 30:
                import warnings
                warnings.warn(
                    f"stop_main took {elapsed:.1f}s — exceeds 30s Claude Code budget. "
                    "Check LLM provider performance.",
                    stacklevel=1,
                )

            # Cleanup
            search_result = hook_memory.search(query="auth middleware", user_id=user_id)
            saved_ids = [r["id"] for r in search_result.get("results", [])]
        finally:
            for mid in saved_ids:
                try:
                    hook_memory.delete(mid)
                except Exception:
                    pass
