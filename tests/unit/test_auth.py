"""Tests for auth.py — token resolution, OAT detection, OAuth refresh, expiry checking."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mem0_mcp_selfhosted.auth import (
    _CREDENTIALS_PATH,
    _OAUTH_CLIENT_ID,
    _OAUTH_TOKEN_URL,
    is_oat_token,
    is_token_expiring_soon,
    read_credentials_full,
    refresh_oat_token,
    resolve_token,
)


class TestIsOatToken:
    def test_oat_token_detected(self):
        assert is_oat_token("sk-ant-oat01-abc123") is True

    def test_api_key_not_oat(self):
        assert is_oat_token("sk-ant-api03-xyz789") is False

    def test_empty_string(self):
        assert is_oat_token("") is False

    def test_partial_match(self):
        assert is_oat_token("sk-ant-oat") is True


class TestResolveToken:
    """Test the prioritized fallback chain."""

    def test_priority_1_env_var(self, monkeypatch):
        """MEM0_ANTHROPIC_TOKEN takes highest priority."""
        monkeypatch.setenv("MEM0_ANTHROPIC_TOKEN", "sk-ant-oat01-from-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-use")
        assert resolve_token() == "sk-ant-oat01-from-env"

    def test_priority_2_credentials_file(self, monkeypatch, tmp_path):
        """Falls back to credentials file when env var not set."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        creds = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-from-creds"}}
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(creds))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-oat01-from-creds"

    def test_priority_3_api_key(self, monkeypatch):
        """Falls back to ANTHROPIC_API_KEY when others unavailable."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() == "sk-ant-api03-fallback"

    def test_no_token_returns_none(self, monkeypatch):
        """Returns None when no auth is available."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() is None

    def test_missing_credentials_file_silent(self, monkeypatch):
        """Missing credentials file doesn't log error, just skips."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            result = resolve_token()
            assert result is None

    def test_malformed_json_credentials(self, monkeypatch, tmp_path):
        """Malformed JSON in credentials file warns and continues."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text("{bad json")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-api03-fallback"

    def test_missing_access_token_key(self, monkeypatch, tmp_path):
        """Missing accessToken key warns and continues."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps({"claudeAiOauth": {"noToken": True}}))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-api03-fallback"

    def test_whitespace_only_token_falls_through(self, monkeypatch):
        """Whitespace-only MEM0_ANTHROPIC_TOKEN falls through to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("MEM0_ANTHROPIC_TOKEN", "  \n")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-real")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() == "sk-ant-api03-real"

    def test_all_whitespace_sources_returns_none(self, monkeypatch):
        """All auth sources whitespace-only results in None."""
        monkeypatch.setenv("MEM0_ANTHROPIC_TOKEN", "  ")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  \n")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() is None


# --- Tests for read_credentials_full() ---

_FULL_CREDS = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-test-access",
        "refreshToken": "sk-ant-ort01-test-refresh",
        "expiresAt": 1771821600097,
        "scopes": ["user:inference", "user:profile"],
    }
}


class TestReadCredentialsFull:
    def test_valid_file_with_all_fields(self, tmp_path):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(_FULL_CREDS))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            result = read_credentials_full()

        assert result is not None
        assert result["access_token"] == "sk-ant-oat01-test-access"
        assert result["refresh_token"] == "sk-ant-ort01-test-refresh"
        assert result["expires_at"] == 1771821600097
        assert result["scopes"] == ["user:inference", "user:profile"]

    def test_missing_file(self):
        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert read_credentials_full() is None

    def test_malformed_json(self, tmp_path):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text("{bad json")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert read_credentials_full() is None

    def test_missing_claude_ai_oauth_key(self, tmp_path):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps({"otherKey": {}}))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert read_credentials_full() is None

    def test_missing_refresh_token(self, tmp_path):
        creds = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-test", "expiresAt": 123}}
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(creds))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert read_credentials_full() is None


# --- Tests for refresh_oat_token() ---


class TestRefreshOatToken:
    def test_successful_200_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "sk-ant-oat01-new-access",
            "refresh_token": "sk-ant-ort01-new-refresh",
            "expires_in": 28800,
        }

        with patch("mem0_mcp_selfhosted.auth.httpx.post", return_value=mock_response) as mock_post:
            result = refresh_oat_token("sk-ant-ort01-old-refresh")

        assert result is not None
        assert result["access_token"] == "sk-ant-oat01-new-access"
        assert result["refresh_token"] == "sk-ant-ort01-new-refresh"
        assert result["expires_in"] == 28800

        # Verify correct request parameters
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["grant_type"] == "refresh_token"
        assert call_kwargs[1]["json"]["refresh_token"] == "sk-ant-ort01-old-refresh"
        assert call_kwargs[1]["json"]["client_id"] == _OAUTH_CLIENT_ID
        assert call_kwargs[1]["headers"]["Content-Type"] == "application/json"
        assert call_kwargs[1]["timeout"] == 10.0

    def test_400_returns_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch("mem0_mcp_selfhosted.auth.httpx.post", return_value=mock_response):
            assert refresh_oat_token("sk-ant-ort01-consumed") is None

    def test_401_returns_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("mem0_mcp_selfhosted.auth.httpx.post", return_value=mock_response):
            assert refresh_oat_token("sk-ant-ort01-invalid") is None

    def test_network_timeout_returns_none(self):
        with patch(
            "mem0_mcp_selfhosted.auth.httpx.post",
            side_effect=httpx.TimeoutException("Connection timed out"),
        ):
            assert refresh_oat_token("sk-ant-ort01-test") is None

    def test_network_error_returns_none(self):
        with patch(
            "mem0_mcp_selfhosted.auth.httpx.post",
            side_effect=httpx.RequestError("DNS resolution failed"),
        ):
            assert refresh_oat_token("sk-ant-ort01-test") is None

    def test_unexpected_status_returns_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("mem0_mcp_selfhosted.auth.httpx.post", return_value=mock_response):
            assert refresh_oat_token("sk-ant-ort01-test") is None


# --- Tests for is_token_expiring_soon() ---


class TestIsTokenExpiringSoon:
    def test_within_threshold_returns_true(self):
        # Expires in 10 minutes (600s), threshold is 30 minutes (1800s)
        expires_at_ms = int(time.time() * 1000) + (600 * 1000)
        assert is_token_expiring_soon(expires_at_ms, threshold_seconds=1800) is True

    def test_ample_time_returns_false(self):
        # Expires in 2 hours, threshold is 30 minutes
        expires_at_ms = int(time.time() * 1000) + (7200 * 1000)
        assert is_token_expiring_soon(expires_at_ms, threshold_seconds=1800) is False

    def test_already_expired_returns_true(self):
        # Expired 1 hour ago
        expires_at_ms = int(time.time() * 1000) - (3600 * 1000)
        assert is_token_expiring_soon(expires_at_ms) is True

    def test_none_input_returns_false(self):
        assert is_token_expiring_soon(None) is False

    def test_custom_threshold(self):
        # Expires in 45 minutes, default threshold (30min) → False
        expires_at_ms = int(time.time() * 1000) + (2700 * 1000)
        assert is_token_expiring_soon(expires_at_ms, threshold_seconds=1800) is False

        # Same expiry, threshold 1 hour → True
        assert is_token_expiring_soon(expires_at_ms, threshold_seconds=3600) is True
