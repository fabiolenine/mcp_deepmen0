"""Tests for llm_anthropic.py — parse_response, schema detection, extract_json, token refresh."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from mem0_mcp_selfhosted.llm_anthropic import (
    FACT_RETRIEVAL_SCHEMA,
    MEMORY_UPDATE_SCHEMA,
    OAT_HEADERS,
    AnthropicOATLLM,
    extract_json,
)


class TestExtractJson:
    def test_clean_json(self):
        assert extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_code_fenced_json(self):
        text = '```json\n{"facts": ["a", "b"]}\n```'
        assert extract_json(text) == '{"facts": ["a", "b"]}'

    def test_code_fenced_no_lang(self):
        text = '```\n{"facts": ["a"]}\n```'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_text_prefixed_json(self):
        text = 'Here is the result:\n{"facts": ["a"]}'
        result = extract_json(text)
        assert result.startswith('{"facts"')

    def test_text_prefixed_array(self):
        text = "The output is: [1, 2, 3]"
        result = extract_json(text)
        assert result.startswith("[1,")

    def test_empty_string(self):
        assert extract_json("") == ""

    def test_unclosed_code_fence(self):
        text = '```json\n{"key": "value"}'
        result = extract_json(text)
        assert '{"key": "value"}' in result


class TestParseResponse:
    def _make_text_block(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def _make_tool_use_block(self, name: str, input_data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.input = input_data
        return block

    def _make_response(self, blocks):
        response = MagicMock()
        response.content = blocks
        return response

    def test_single_tool_use(self):
        response = self._make_response([
            self._make_tool_use_block("extract_entities", {"entities": ["Alice"]})
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": ["Alice"]}}
        ]
        assert result["content"] == ""

    def test_mixed_text_and_tool(self):
        response = self._make_response([
            self._make_text_block("Analyzing..."),
            self._make_tool_use_block("extract_entities", {"entities": ["Alice"]}),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["content"] == "Analyzing..."
        assert len(result["tool_calls"]) == 1

    def test_text_only_when_tools_provided(self):
        response = self._make_response([
            self._make_text_block("No entities found."),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["content"] == "No entities found."
        assert result["tool_calls"] == []

    def test_multiple_tool_use_blocks(self):
        response = self._make_response([
            self._make_tool_use_block("tool_a", {"a": 1}),
            self._make_tool_use_block("tool_b", {"b": 2}),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert len(result["tool_calls"]) == 2


class TestSchemaDetection:
    def test_fact_extraction_has_system_message(self):
        """System message present → FACT_RETRIEVAL_SCHEMA."""
        llm = MagicMock(spec=AnthropicOATLLM)
        messages = [
            {"role": "system", "content": "Extract facts from the following..."},
            {"role": "user", "content": "Alice prefers TypeScript"},
        ]
        result = AnthropicOATLLM._select_schema(llm, messages)
        assert result == FACT_RETRIEVAL_SCHEMA

    def test_memory_update_no_system_message(self):
        """No system message → MEMORY_UPDATE_SCHEMA."""
        llm = MagicMock(spec=AnthropicOATLLM)
        messages = [
            {"role": "user", "content": "Update memory..."},
        ]
        result = AnthropicOATLLM._select_schema(llm, messages)
        assert result == MEMORY_UPDATE_SCHEMA


# --- Helpers for token refresh tests ---

OAT_TOKEN = "sk-ant-oat01-test-token-old"
OAT_TOKEN_NEW = "sk-ant-oat01-test-token-new"
API_KEY = "sk-ant-api03-test-key"


def _make_auth_error():
    """Create an anthropic.AuthenticationError with minimal mocks."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.headers = httpx.Headers({})
    mock_response.is_closed = True
    mock_response.is_stream_consumed = True
    return anthropic.AuthenticationError(
        message="authentication_error", response=mock_response, body=None
    )


def _make_rate_limit_error():
    """Create an anthropic.RateLimitError with minimal mocks."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = httpx.Headers({"retry-after": "1"})
    mock_response.is_closed = True
    mock_response.is_stream_consumed = True
    return anthropic.RateLimitError(
        message="rate_limit_error", response=mock_response, body=None
    )


def _make_api_response(text="ok", stop_reason="end_turn"):
    """Create a mock anthropic.types.Message."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = stop_reason
    return response


def _make_llm(token=OAT_TOKEN):
    """Create an AnthropicOATLLM with mocked internals (no real API client).

    Mocks read_credentials_full to prevent reading real ~/.claude/.credentials.json,
    so _refresh_token and _expires_at start as None.
    """
    with patch("mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=token):
        with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
            with patch("anthropic.Anthropic"):
                from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

                config = AnthropicOATConfig(model="claude-sonnet-4-20250514", auth_token=token)
                llm = AnthropicOATLLM(config=config)
    return llm


class TestCallApiTokenRefresh:
    """Tests for _call_api auth retry logic."""

    def test_retry_succeeds_with_new_token(self):
        """Step 1 piggyback: _call_api retries when resolve_token returns a different token."""
        llm = _make_llm(OAT_TOKEN)
        success_response = _make_api_response("retried ok")

        llm.client.messages.create = MagicMock(
            side_effect=[_make_auth_error(), success_response]
        )

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN_NEW
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
                with patch.object(llm, "_build_client") as mock_build:
                    result = llm._call_api({"model": "test"})

        mock_build.assert_called_once_with(OAT_TOKEN_NEW)
        assert result is success_response

    def test_no_retry_when_token_unchanged(self):
        """All 3 steps fail: same token, no refresh token, wait-and-retry same token."""
        llm = _make_llm(OAT_TOKEN)
        llm.client.messages.create = MagicMock(side_effect=_make_auth_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
                with pytest.raises(anthropic.AuthenticationError):
                    llm._call_api({"model": "test"})

        # Only one call — no retry (all 3 steps failed to get a new token)
        assert llm.client.messages.create.call_count == 1

    def test_no_retry_when_resolve_returns_none(self):
        """All 3 steps fail: None token, no refresh token, wait-and-retry None."""
        llm = _make_llm(OAT_TOKEN)
        llm.client.messages.create = MagicMock(side_effect=_make_auth_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=None
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
                with pytest.raises(anthropic.AuthenticationError):
                    llm._call_api({"model": "test"})

        assert llm.client.messages.create.call_count == 1

    def test_no_retry_for_non_oat_token(self):
        """4.4: _call_api does NOT retry for standard API keys."""
        llm = _make_llm(API_KEY)
        llm.client.messages.create = MagicMock(side_effect=_make_auth_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token"
        ) as mock_resolve:
            with pytest.raises(anthropic.AuthenticationError):
                llm._call_api({"model": "test"})

        # resolve_token should never be called for non-OAT tokens
        mock_resolve.assert_not_called()
        assert llm.client.messages.create.call_count == 1

    def test_non_auth_error_propagates(self):
        """4.5: _call_api does NOT catch non-auth errors like RateLimitError."""
        llm = _make_llm(OAT_TOKEN)
        llm.client.messages.create = MagicMock(side_effect=_make_rate_limit_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token"
        ) as mock_resolve:
            with pytest.raises(anthropic.RateLimitError):
                llm._call_api({"model": "test"})

        mock_resolve.assert_not_called()

    def test_no_infinite_retry_loops(self):
        """Only one retry — if retry also fails with AuthError, it propagates."""
        llm = _make_llm(OAT_TOKEN)
        llm.client.messages.create = MagicMock(
            side_effect=[_make_auth_error(), _make_auth_error()]
        )

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN_NEW
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
                with patch.object(llm, "_build_client"):
                    with pytest.raises(anthropic.AuthenticationError):
                        llm._call_api({"model": "test"})

        # Exactly 2 calls: original + one retry
        assert llm.client.messages.create.call_count == 2


class TestBuildClient:
    """Tests for _build_client client construction."""

    def test_oat_token_uses_auth_token_and_headers(self):
        """4.7: OAT token → auth_token kwarg + OAT headers."""
        llm = _make_llm(OAT_TOKEN)

        with patch("anthropic.Anthropic") as mock_cls:
            llm._build_client(OAT_TOKEN)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["auth_token"] == OAT_TOKEN
        assert call_kwargs["default_headers"] == OAT_HEADERS
        assert "api_key" not in call_kwargs

    def test_api_key_uses_api_key_no_headers(self):
        """4.8: Standard API key → api_key kwarg, no OAT headers."""
        llm = _make_llm(API_KEY)

        with patch("anthropic.Anthropic") as mock_cls:
            llm._build_client(API_KEY)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == API_KEY
        assert "auth_token" not in call_kwargs
        assert "default_headers" not in call_kwargs

    def test_current_token_updated(self):
        """4.9: _current_token is updated after _build_client."""
        llm = _make_llm(OAT_TOKEN)
        assert llm._current_token == OAT_TOKEN

        with patch("anthropic.Anthropic"):
            llm._build_client(OAT_TOKEN_NEW)

        assert llm._current_token == OAT_TOKEN_NEW


def _make_internal_server_error(status_code=500):
    """Create an anthropic.InternalServerError (or APIStatusError) with minimal mocks."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.headers = httpx.Headers({})
    mock_response.is_closed = True
    mock_response.is_stream_consumed = True
    return anthropic.InternalServerError(
        message="internal_server_error", response=mock_response, body=None
    )


def _make_api_status_error(status_code):
    """Create an anthropic.APIStatusError with a specific status code."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.headers = httpx.Headers({})
    mock_response.is_closed = True
    mock_response.is_stream_consumed = True
    return anthropic.APIStatusError(
        message=f"error_{status_code}", response=mock_response, body=None
    )


class TestTransientRetry:
    """Tests for _call_with_transient_retry backoff logic."""

    def test_retries_on_500_then_succeeds(self):
        """Transient 500 on first attempt, succeeds on retry."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_internal_server_error(500), success]
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep") as mock_sleep:
            result = llm._call_api({"model": "test"})

        assert result is success
        assert llm.client.messages.create.call_count == 2
        mock_sleep.assert_called_once_with(1)

    def test_retries_on_502_then_succeeds(self):
        """Transient 502 on first attempt, succeeds on retry."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_api_status_error(502), success]
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
            result = llm._call_api({"model": "test"})

        assert result is success
        assert llm.client.messages.create.call_count == 2

    def test_retries_on_529_then_succeeds(self):
        """Transient 529 (overloaded) on first attempt, succeeds on retry."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_api_status_error(529), success]
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
            result = llm._call_api({"model": "test"})

        assert result is success

    def test_exhausts_all_retries_then_raises(self):
        """All retries fail with 500 — raises after max retries."""
        llm = _make_llm(API_KEY)
        llm.client.messages.create = MagicMock(
            side_effect=[
                _make_internal_server_error(500),
                _make_internal_server_error(500),
                _make_internal_server_error(500),
            ]
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
            with pytest.raises(anthropic.InternalServerError):
                llm._call_api({"model": "test"})

        assert llm.client.messages.create.call_count == 3  # 1 original + 2 retries

    def test_backoff_delays(self):
        """Verify exponential backoff: 1s, 2s."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[
                _make_internal_server_error(500),
                _make_internal_server_error(500),
                success,
            ]
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep") as mock_sleep:
            result = llm._call_api({"model": "test"})

        assert result is success
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    def test_no_retry_on_400(self):
        """Non-retryable 400 raises immediately without retry."""
        llm = _make_llm(API_KEY)
        llm.client.messages.create = MagicMock(
            side_effect=_make_api_status_error(400)
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep") as mock_sleep:
            with pytest.raises(anthropic.APIStatusError):
                llm._call_api({"model": "test"})

        assert llm.client.messages.create.call_count == 1
        mock_sleep.assert_not_called()

    def test_no_retry_on_404(self):
        """Non-retryable 404 raises immediately without retry."""
        llm = _make_llm(API_KEY)
        llm.client.messages.create = MagicMock(
            side_effect=_make_api_status_error(404)
        )

        with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep") as mock_sleep:
            with pytest.raises(anthropic.APIStatusError):
                llm._call_api({"model": "test"})

        assert llm.client.messages.create.call_count == 1
        mock_sleep.assert_not_called()

    def test_oat_refresh_then_transient_retry(self):
        """OAT 401 refresh succeeds, then transient 500 retried on refreshed client."""
        llm = _make_llm(OAT_TOKEN)
        success = _make_api_response("ok")

        # First call: 401 (auth error) → refresh → second call: 500 → retry → success
        call_count = {"n": 0}
        def _side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_auth_error()
            if call_count["n"] == 2:
                raise _make_internal_server_error(500)
            return success

        llm.client.messages.create = MagicMock(side_effect=_side_effect)

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN_NEW
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
                with patch.object(llm, "_build_client"):
                    with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
                        result = llm._call_api({"model": "test"})

        assert result is success
        assert call_count["n"] == 3


# --- Tests for the 3-step defensive auth retry strategy ---

REFRESH_TOKEN = "sk-ant-ort01-test-refresh"
REFRESH_TOKEN_NEW = "sk-ant-ort01-test-refresh-new"


class TestThreeStepAuthRetry:
    """Tests for the piggyback → self-refresh → wait-and-retry strategy."""

    def test_step1_piggyback_success(self):
        """Step 1: credentials file has new token → piggyback success."""
        llm = _make_llm(OAT_TOKEN)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_auth_error(), success]
        )

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN_NEW
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
                with patch.object(llm, "_build_client") as mock_build:
                    result = llm._call_api({"model": "test"})

        mock_build.assert_called_once_with(OAT_TOKEN_NEW)
        assert result is success

    def test_step2_self_refresh_success(self):
        """Step 2: piggyback fails (same token), OAuth self-refresh succeeds."""
        llm = _make_llm(OAT_TOKEN)
        llm._refresh_token = REFRESH_TOKEN
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_auth_error(), success]
        )

        oauth_result = {
            "access_token": OAT_TOKEN_NEW,
            "refresh_token": REFRESH_TOKEN_NEW,
            "expires_in": 28800,
        }

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN
        ):
            with patch(
                "mem0_mcp_selfhosted.llm_anthropic.refresh_oat_token", return_value=oauth_result
            ):
                with patch.object(llm, "_build_client") as mock_build:
                    result = llm._call_api({"model": "test"})

        mock_build.assert_called_once_with(OAT_TOKEN_NEW)
        assert result is success
        # In-memory state updated
        assert llm._refresh_token == REFRESH_TOKEN_NEW
        assert llm._expires_at is not None

    def test_step3_wait_and_retry_success(self):
        """Step 3: piggyback + self-refresh fail, wait-and-retry finds new token."""
        llm = _make_llm(OAT_TOKEN)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(
            side_effect=[_make_auth_error(), success]
        )

        # resolve_token returns same token first (Step 1), then new token (Step 3)
        resolve_calls = {"n": 0}
        def _resolve_side_effect():
            resolve_calls["n"] += 1
            if resolve_calls["n"] <= 1:
                return OAT_TOKEN  # Step 1: same token
            return OAT_TOKEN_NEW  # Step 3: new token after wait

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token",
            side_effect=_resolve_side_effect,
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.read_credentials_full", return_value=None):
                with patch.object(llm, "_build_client") as mock_build:
                    with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep") as mock_sleep:
                        result = llm._call_api({"model": "test"})

        mock_sleep.assert_called_once_with(2)
        mock_build.assert_called_once_with(OAT_TOKEN_NEW)
        assert result is success

    def test_all_steps_exhausted_raises(self):
        """All 3 steps fail → re-raise original AuthenticationError."""
        llm = _make_llm(OAT_TOKEN)
        llm.client.messages.create = MagicMock(side_effect=_make_auth_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN
        ):
            with patch("mem0_mcp_selfhosted.llm_anthropic.time.sleep"):
                with pytest.raises(anthropic.AuthenticationError):
                    llm._call_api({"model": "test"})

        assert llm.client.messages.create.call_count == 1

    def test_non_oat_token_skips_all_refresh(self):
        """API key tokens skip all refresh logic entirely."""
        llm = _make_llm(API_KEY)
        llm.client.messages.create = MagicMock(side_effect=_make_auth_error())

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token"
        ) as mock_resolve:
            with patch(
                "mem0_mcp_selfhosted.llm_anthropic.refresh_oat_token"
            ) as mock_refresh:
                with pytest.raises(anthropic.AuthenticationError):
                    llm._call_api({"model": "test"})

        mock_resolve.assert_not_called()
        mock_refresh.assert_not_called()
        assert llm.client.messages.create.call_count == 1


class TestProactiveRefresh:
    """Tests for pre-call proactive token refresh."""

    def test_proactive_refresh_triggered_when_expiring_soon(self):
        """Token expiring soon → proactive refresh attempted before API call."""
        llm = _make_llm(OAT_TOKEN)
        llm._expires_at = int(__import__("time").time() * 1000) + (60 * 1000)  # 1min left
        llm._refresh_token = REFRESH_TOKEN
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(return_value=success)

        oauth_result = {
            "access_token": OAT_TOKEN_NEW,
            "refresh_token": REFRESH_TOKEN_NEW,
            "expires_in": 28800,
        }

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.resolve_token", return_value=OAT_TOKEN
        ):
            with patch(
                "mem0_mcp_selfhosted.llm_anthropic.refresh_oat_token", return_value=oauth_result
            ):
                with patch.object(llm, "_build_client") as mock_build:
                    result = llm._call_api({"model": "test"})

        # Proactive refresh called _build_client before the API call
        mock_build.assert_called_once_with(OAT_TOKEN_NEW)
        assert result is success
        # API call succeeded on first try (no AuthenticationError)
        assert llm.client.messages.create.call_count == 1

    def test_proactive_refresh_skipped_when_ample_time(self):
        """Token has ample time → no proactive refresh, direct API call."""
        llm = _make_llm(OAT_TOKEN)
        llm._expires_at = int(__import__("time").time() * 1000) + (4 * 3600 * 1000)  # 4hrs left
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(return_value=success)

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.refresh_oat_token"
        ) as mock_refresh:
            result = llm._call_api({"model": "test"})

        mock_refresh.assert_not_called()
        assert result is success

    def test_proactive_refresh_skipped_for_api_key(self):
        """Non-OAT token → proactive refresh check skipped entirely."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(return_value=success)

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.is_token_expiring_soon"
        ) as mock_expiry:
            result = llm._call_api({"model": "test"})

        mock_expiry.assert_not_called()
        assert result is success

    def test_proactive_refresh_skipped_for_none_token(self):
        """None token → proactive refresh returns immediately (no TypeError)."""
        llm = _make_llm(API_KEY)
        llm._current_token = None  # Simulate no token available
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(return_value=success)

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.is_oat_token"
        ) as mock_is_oat:
            result = llm._call_api({"model": "test"})

        mock_is_oat.assert_not_called()
        assert result is success

    def test_proactive_refresh_calls_is_oat_for_valid_token(self):
        """Valid string token → is_oat_token is called normally."""
        llm = _make_llm(OAT_TOKEN)
        llm._expires_at = int(__import__("time").time() * 1000) + (4 * 3600 * 1000)  # 4hrs left
        success = _make_api_response("ok")
        llm.client.messages.create = MagicMock(return_value=success)

        with patch(
            "mem0_mcp_selfhosted.llm_anthropic.is_oat_token", return_value=True
        ) as mock_is_oat:
            with patch(
                "mem0_mcp_selfhosted.llm_anthropic.is_token_expiring_soon", return_value=False
            ):
                llm._call_api({"model": "test"})

        mock_is_oat.assert_called_once_with(OAT_TOKEN)


class TestGenerateResponseEmptyContent:
    """Tests for empty response.content guard in generate_response."""

    def test_structured_output_empty_content_returns_empty_string(self):
        """Structured output path returns '' when response.content is []."""
        llm = _make_llm(API_KEY)
        empty_response = MagicMock()
        empty_response.content = []
        llm.client.messages.create = MagicMock(return_value=empty_response)

        result = llm.generate_response(
            messages=[{"role": "user", "content": "test"}],
            response_format="json",
        )

        assert result == ""

    def test_plain_text_empty_content_returns_empty_string(self):
        """Plain text path returns '' when response.content is []."""
        llm = _make_llm(API_KEY)
        empty_response = MagicMock()
        empty_response.content = []
        llm.client.messages.create = MagicMock(return_value=empty_response)

        result = llm.generate_response(
            messages=[{"role": "user", "content": "test"}],
        )

        assert result == ""

    def test_normal_content_still_works(self):
        """Non-empty content returns text normally (regression check)."""
        llm = _make_llm(API_KEY)
        success = _make_api_response("hello world")
        llm.client.messages.create = MagicMock(return_value=success)

        result = llm.generate_response(
            messages=[{"role": "user", "content": "test"}],
        )

        assert result == "hello world"
