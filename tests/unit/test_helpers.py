"""Tests for helpers.py — error wrapper, call_with_graph, bulk delete, user_id, sanitizer, Gemini patch."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted.helpers import (
    _make_enhanced_sanitizer,
    _mem0_call,
    call_with_graph,
    get_default_user_id,
    patch_gemini_parse_response,
    safe_bulk_delete,
)


class TestMem0Call:
    def test_success_returns_json(self):
        result = _mem0_call(lambda: {"status": "ok"})
        parsed = json.loads(result)
        assert parsed == {"status": "ok"}

    def test_memory_error_caught(self):
        """MemoryError subclass returns structured error JSON."""
        # Create a mock MemoryError-like exception
        class FakeMemoryError(Exception):
            pass

        FakeMemoryError.__name__ = "MemoryError"

        exc = FakeMemoryError("something failed")
        exc.error_code = "VALIDATION_ERROR"
        exc.details = "missing field"
        exc.suggestion = "add user_id"

        # Patch the MRO check
        def _raise():
            raise exc

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_generic_exception_caught(self):
        """Generic Exception returns type name and detail."""
        def _raise():
            raise ValueError("bad input")

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert parsed["error"] == "ValueError"
        assert parsed["detail"] == "bad input"

    def test_ensure_ascii_false(self):
        """Non-ASCII characters preserved in output."""
        result = _mem0_call(lambda: {"text": "Alice prefiere TypeScript"})
        assert "prefiere" in result


class TestCallWithGraph:
    def test_sets_enable_graph_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, True, False, _check)
        assert result == "ok"

    def test_sets_enable_graph_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, False, True, _check)
        assert result == "ok"

    def test_uses_default_when_none(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, None, True, _check)
        assert result == "ok"

    def test_graph_none_forces_false(self):
        """If memory.graph is None, enable_graph stays False regardless."""
        memory = MagicMock()
        memory.graph = None

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, True, True, _check)
        assert result == "ok"

    def test_none_memory_raises_runtime_error(self):
        """call_with_graph raises RuntimeError when memory is None."""
        with pytest.raises(RuntimeError, match="Memory not initialized"):
            call_with_graph(None, False, False, lambda: "ok")


def _pt(pid):
    """Minimal stand-in for a Qdrant point."""
    m = MagicMock()
    m.id = pid
    return m


class TestSafeBulkDelete:
    @staticmethod
    def _draining_store(memory, ids):
        """A store that actually drains as delete() is called.

        A store mock that keeps returning the same rows is not a realistic
        stand-in for a paginated delete — it cannot tell "drained" from
        "looping".
        """
        remaining = list(ids)

        def _list(filters=None, top_k=100):
            return ([_pt(i) for i in remaining[:top_k]],)

        def _delete(mid):
            remaining.remove(mid)
            return {"message": "Memory deleted successfully!"}

        memory.vector_store.list.side_effect = _list
        memory.delete.side_effect = _delete
        return lambda: remaining

    def test_iterates_and_deletes(self):
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None
        self._draining_store(memory, ["id-1", "id-2"])

        r = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert r.deleted == 2
        assert r.vector_scope_drained is True
        assert r.failed_ids == []
        assert memory.delete.call_count == 2
        memory.delete.assert_any_call("id-1")
        memory.delete.assert_any_call("id-2")

    def test_drains_beyond_one_page(self):
        """The actual production defect: list() defaults to top_k=100, so a
        250-memory scope reported count=100 and looked successful."""
        memory = MagicMock()
        memory.graph = None
        ids = [f"id-{i}" for i in range(250)]
        left = self._draining_store(memory, ids)

        r = safe_bulk_delete(memory, {"user_id": "u"}, page_size=100)

        assert r.deleted == 250
        assert r.vector_scope_drained is True
        assert left() == []

    def test_top_k_is_always_explicit(self):
        """Relying on the store default WAS the bug. Pin it."""
        memory = MagicMock()
        memory.graph = None
        seen = []

        def _list(filters=None, top_k=None):
            seen.append(top_k)
            return ([],)

        memory.vector_store.list.side_effect = _list
        safe_bulk_delete(memory, {"user_id": "u"})
        assert seen and all(k is not None for k in seen), seen

    def test_partial_version_chain_delete_counts_as_failure(self):
        """memory.delete() deletes a whole version chain and RETURNS a dict on
        partial failure instead of raising — a non-raising call is not proof of
        success."""
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: (
            [_pt("id-1")],) if not memory.delete.called else ([_pt("id-1")],)
        memory.delete.return_value = {"message": "Memory partially deleted; retry",
                                      "deleted": ["v1"], "remaining": ["id-1"]}

        r = safe_bulk_delete(memory, {"user_id": "u"})

        assert r.failed_ids == ["id-1"]
        assert r.vector_scope_drained is False

    def test_undeletable_row_terminates_and_reports(self):
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([_pt("id-1")],)
        memory.delete.side_effect = RuntimeError("nope")

        r = safe_bulk_delete(memory, {"user_id": "u"}, max_pages=50)

        assert r.deleted == 0
        assert r.failed_ids == ["id-1"]
        assert r.vector_scope_drained is False
        assert memory.delete.call_count == 1, "must not retry the same id forever"

    def test_graph_cleanup_skipped_when_scope_not_drained(self):
        """Cleaning the graph after a PARTIAL delete would remove graph data for
        memories that are still alive."""
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([_pt("id-1")],)
        memory.delete.side_effect = RuntimeError("nope")

        r = safe_bulk_delete(memory, {"user_id": "u"}, graph_enabled=True)

        assert r.vector_scope_drained is False
        memory.graph.delete_all.assert_not_called()

    def test_graph_cleanup_when_graph_enabled_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=True)

        memory.graph.delete_all.assert_called_once_with({"user_id": "testuser"})

    def test_no_graph_cleanup_when_graph_enabled_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=False)

        memory.graph.delete_all.assert_not_called()

    def test_no_graph_cleanup_default(self):
        """Default graph_enabled=False skips graph cleanup."""
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {"user_id": "testuser"})

        memory.graph.delete_all.assert_not_called()


class TestSafeBulkDeleteNormalizesScope:
    """Defesa em profundidade: escopo não normalizado aqui não vira erro — vira
    um delete que não casa nada e responde ``deleted: 0`` como sucesso.

    MEDIDO no corpus de produção: um ``user_id`` real casa 1159 pontos; o
    MESMO com um espaço à esquerda casa 0. O filtro do store é casamento EXATO.
    """

    @pytest.mark.parametrize("key", ["user_id", "agent_id", "run_id", "actor_id"])
    def test_padded_scope_is_trimmed_before_it_reaches_the_store(self, key):
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {key: " alice "})

        assert memory.vector_store.list.call_args.kwargs["filters"] == {key: "alice"}

    def test_integer_scope_is_coerced(self):
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {"user_id": 42})

        assert memory.vector_store.list.call_args.kwargs["filters"] == {"user_id": "42"}

    @pytest.mark.parametrize("key", ["user_id", "agent_id", "run_id", "actor_id"])
    @pytest.mark.parametrize("bad", ["a b", "   ", 3.5, True])
    def test_invalid_scope_raises_instead_of_deleting_nothing_quietly(self, key, bad):
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        with pytest.raises(ValueError, match=key):
            safe_bulk_delete(memory, {key: bad})

        memory.vector_store.list.assert_not_called()

    def test_non_scope_filter_keys_are_left_alone(self):
        """CONTROLE NEGATIVO: normalizar chave livre quebraria filtro de
        metadata legítimo — só as chaves de ESCOPO passam pela regra."""
        memory = MagicMock()
        memory.graph = None
        memory.vector_store.list.side_effect = lambda filters=None, top_k=100: ([],)

        safe_bulk_delete(memory, {"user_id": "u", "project": "a b c", "importance": 0.8})

        assert memory.vector_store.list.call_args.kwargs["filters"] == {
            "user_id": "u",
            "project": "a b c",
            "importance": 0.8,
        }

    def test_valid_scope_still_deletes(self):
        """CONTROLE NEGATIVO: a normalização não pode impedir o caminho feliz."""
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None
        TestSafeBulkDelete._draining_store(memory, ["id-1", "id-2"])

        r = safe_bulk_delete(memory, {"user_id": " testuser "})

        assert r.deleted == 2
        assert r.vector_scope_drained is True


class TestGetDefaultUserId:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert get_default_user_id() == "user"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "bob")
        assert get_default_user_id() == "bob"


class TestEnhancedSanitizer:
    """Tests for the enhanced relationship name sanitizer."""

    @pytest.fixture()
    def sanitize(self):
        """Create enhanced sanitizer wrapping a passthrough original."""
        # Simulate mem0ai's original: returns input unchanged (our tests
        # focus on what the wrapper adds, not on the original's char_map).
        return _make_enhanced_sanitizer(lambda r: r)

    def test_leading_digit_gets_prefix(self, sanitize):
        """Neo4j types can't start with digits — should get rel_ prefix."""
        assert sanitize("3tier_fallback") == "rel_3tier_fallback"

    def test_leading_digit_with_hyphen(self, sanitize):
        """The exact error case: '3-tier_oat_token_fallback'."""
        result = sanitize("3-tier_oat_token_fallback")
        assert result == "rel_3_tier_oat_token_fallback"
        assert result[0].isalpha()

    def test_hyphens_replaced(self, sanitize):
        """Hyphens should become underscores."""
        assert sanitize("has-authored") == "has_authored"

    def test_multiple_hyphens(self, sanitize):
        """Multiple hyphens in a row collapse to single underscore."""
        assert sanitize("is--related--to") == "is_related_to"

    def test_spaces_replaced(self, sanitize):
        """Spaces should become underscores."""
        assert sanitize("author of") == "author_of"

    def test_mixed_special_chars(self, sanitize):
        """Mix of problematic characters."""
        assert sanitize("has.authored-by!user") == "has_authored_by_user"

    def test_already_valid(self, sanitize):
        """Valid relationship type passes through unchanged."""
        assert sanitize("WORKS_FOR") == "WORKS_FOR"

    def test_already_valid_lowercase(self, sanitize):
        """Valid lowercase type passes through unchanged."""
        assert sanitize("prefers") == "prefers"

    def test_empty_string_fallback(self, sanitize):
        """Empty string gets fallback name."""
        assert sanitize("") == "related_to"

    def test_only_special_chars_fallback(self, sanitize):
        """String of only special chars gets fallback name."""
        assert sanitize("---") == "related_to"

    def test_consecutive_underscores_collapsed(self, sanitize):
        """Multiple underscores collapse to one."""
        assert sanitize("foo___bar") == "foo_bar"

    def test_leading_trailing_underscores_stripped(self, sanitize):
        """Leading/trailing underscores are stripped."""
        assert sanitize("_foo_bar_") == "foo_bar"

    def test_pure_digits(self, sanitize):
        """Pure numeric string gets prefix."""
        assert sanitize("123") == "rel_123"

    def test_unicode_stripped(self, sanitize):
        """Non-ASCII characters become underscores then get collapsed/stripped."""
        result = sanitize("関係_type")
        # 関係 → __ → stripped, leaving just "type"
        assert result == "type"

    def test_wraps_original_function(self):
        """Enhanced sanitizer calls the original function first."""
        call_log = []

        def mock_original(r):
            call_log.append(r)
            return r.replace("&", "_ampersand_")

        enhanced = _make_enhanced_sanitizer(mock_original)
        result = enhanced("has&uses")
        assert call_log == ["has&uses"]
        assert result == "has_ampersand_uses"

    def test_valid_neo4j_pattern(self, sanitize):
        """All outputs must match Neo4j's identifier pattern."""
        import re

        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        test_cases = [
            "3-tier_oat_token_fallback",
            "has-authored",
            "is professor of",
            "123",
            "---",
            "",
            "WORKS_FOR",
            "has.dot.notation",
            "with spaces and-hyphens",
            "5th_element",
        ]
        for case in test_cases:
            result = sanitize(case)
            assert pattern.match(result), f"'{case}' → '{result}' is not a valid Neo4j type"


class TestPatchGeminiParseResponse:
    """Tests for the Gemini null content guard monkey-patch."""

    def test_null_content_returns_empty_string(self):
        """When Gemini returns candidate with content=None, return empty string."""
        # Create a mock GeminiLLM class with a _parse_response method
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original result")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            # Apply the patch
            patch_gemini_parse_response()

            # Verify the method was replaced
            patched_method = mock_gemini_cls._parse_response
            assert patched_method is not original_parse

            # Test with null content
            response = MagicMock()
            candidate = MagicMock()
            candidate.content = None
            response.candidates = [candidate]

            result = patched_method(MagicMock(), response)
            assert result == ""

    def test_normal_response_delegates_to_original(self):
        """Normal responses with valid content delegate to original method."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="parsed content")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            # Test with valid content
            response = MagicMock()
            candidate = MagicMock()
            candidate.content = MagicMock()
            candidate.content.parts = [MagicMock()]
            response.candidates = [candidate]

            instance = MagicMock()
            patched_method(instance, response)
            original_parse.assert_called_once_with(instance, response)

    def test_empty_candidates_returns_empty_string(self):
        """Response with empty candidates list returns empty string."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            response = MagicMock()
            response.candidates = []

            result = patched_method(MagicMock(), response)
            assert result == ""
