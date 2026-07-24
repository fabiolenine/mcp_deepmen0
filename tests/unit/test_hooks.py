"""Tests for hooks.py — Claude Code session hooks."""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted import hooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_output(func, stdin_data: str = "{}") -> dict:
    """Run a hook entry point with mocked stdin and capture stdout JSON."""
    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)), patch("sys.stdout", captured):
        func()
    return json.loads(captured.getvalue())


# ---------------------------------------------------------------------------
# 6.1  _get_user_id
# ---------------------------------------------------------------------------


class TestGetUserId:
    def test_returns_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "alice")
        assert hooks._get_user_id() == "alice"

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert hooks._get_user_id() == "user"

    def test_dotenv_loaded_before_get_user_id(self):
        """load_dotenv() runs at module init, so .env values are visible."""
        # Verify the module-level load_dotenv() import exists —
        # this guards against regression of the bug where _get_user_id()
        # was called before load_dotenv() in context_main().
        import inspect
        source = inspect.getsource(hooks)
        # load_dotenv() should be called at module level, not just inside a function
        lines = source.split("\n")
        found_module_level_call = False
        for line in lines:
            stripped = line.strip()
            # Skip comments and function/class definitions
            if stripped.startswith("#") or stripped.startswith("def ") or stripped.startswith("class "):
                continue
            if "load_dotenv()" in stripped and not line.startswith("    "):
                found_module_level_call = True
                break
        assert found_module_level_call, "load_dotenv() must be called at module level"


# ---------------------------------------------------------------------------
# 6.2  _get_memory
# ---------------------------------------------------------------------------


class TestGetMemory:
    def test_caching_returns_same_instance(self):
        """_get_memory() returns the cached instance on repeated calls."""
        sentinel = MagicMock(name="Memory")
        with patch.object(hooks, "_memory", sentinel):
            assert hooks._get_memory() is sentinel

    def test_graph_disabled_in_env(self, monkeypatch):
        """_get_memory() sets MEM0_ENABLE_GRAPH=false and caches result."""
        calls = []

        def fake_build_config():
            calls.append(os.environ.get("MEM0_ENABLE_GRAPH"))
            return {}, [], None

        fake_mem = MagicMock(name="FreshMemory")

        # monkeypatch auto-restores _memory after the test
        monkeypatch.setattr(hooks, "_memory", None)
        with (
            patch("mem0_mcp_selfhosted.config.build_config", fake_build_config),
            patch("mem0_mcp_selfhosted.server.register_providers"),
            patch("mem0.Memory.from_config", return_value=fake_mem),
        ):
            result = hooks._get_memory()

        assert calls == ["false"]
        assert result is fake_mem
        # Verify the result was cached in the module global
        assert hooks._memory is fake_mem


# ---------------------------------------------------------------------------
# 6.3  context_main
# ---------------------------------------------------------------------------


class TestContextMain:
    def _make_stdin(self, **overrides):
        data = {
            "session_id": "sess-1",
            "cwd": "/home/user/myproject",
            "hook_event_name": "startup",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_memories_found(self):
        """When search returns memories, additionalContext is included."""
        fake_results = {
            "results": [
                {"id": "m1", "memory": "Uses TypeScript with strict mode"},
                {"id": "m2", "memory": "Prefers pytest for testing"},
            ]
        }

        mock_mem = MagicMock()
        mock_mem.search.return_value = fake_results

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        assert "additionalContext" in result
        assert "TypeScript" in result["additionalContext"]
        assert "pytest" in result["additionalContext"]
        assert "# mem0 Cross-Session Memory" in result["additionalContext"]

    def test_no_memories_omits_additional_context(self):
        """When no memories found, additionalContext is absent."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        assert "additionalContext" not in result

    def test_deduplication_across_queries(self):
        """Duplicate memory IDs across queries are deduplicated."""
        mock_mem = MagicMock()
        # Both queries return overlapping results
        mock_mem.search.side_effect = [
            {"results": [
                {"id": "m1", "memory": "fact one"},
                {"id": "m2", "memory": "fact two"},
            ]},
            {"results": [
                {"id": "m2", "memory": "fact two"},  # duplicate
                {"id": "m3", "memory": "fact three"},
            ]},
        ]

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        ctx = result["additionalContext"]
        # m2 should appear only once
        assert ctx.count("fact two") == 1
        assert "fact one" in ctx
        assert "fact three" in ctx

    def test_results_as_list_format(self):
        """Handle mem0 search returning a plain list (not dict with 'results')."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = [
            {"id": "m1", "memory": "plain list result"},
        ]

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert "plain list result" in result["additionalContext"]

    def test_exception_returns_nonfatal(self):
        """Any exception produces a non-fatal response."""
        with patch.object(hooks, "_get_memory", side_effect=RuntimeError("boom")):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result == {"continue": True, "suppressOutput": True}

    def test_max_memories_cap(self):
        """Results are capped at _MAX_MEMORIES."""
        mock_mem = MagicMock()
        many = [{"id": f"m{i}", "memory": f"fact {i}"} for i in range(30)]
        mock_mem.search.return_value = {"results": many}

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        lines = [l for l in result["additionalContext"].split("\n") if l and l[0].isdigit()]
        assert len(lines) == hooks._MAX_MEMORIES

    def test_empty_cwd_uses_project_fallback(self):
        """Empty cwd falls back to 'project' in search queries."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": [
            {"id": "m1", "memory": "some fact"},
        ]}

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin(cwd=""))

        # Verify search was called with 'project' fallback
        first_query = mock_mem.search.call_args_list[0].kwargs["query"]
        assert "project" in first_query
        assert "additionalContext" in result


# ---------------------------------------------------------------------------
# _extract_content edge cases
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_plain_string(self):
        assert hooks._extract_content("hello world") == "hello world"

    def test_content_blocks_text_only(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert hooks._extract_content(content) == "hello world"

    def test_mixed_block_types_filters_non_text(self):
        """Non-text blocks (tool_use, tool_result) are silently ignored."""
        content = [
            {"type": "tool_use", "id": "t1", "name": "Read"},
            {"type": "text", "text": "the actual response"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "file data"},
        ]
        assert hooks._extract_content(content) == "the actual response"

    def test_text_block_missing_text_key(self):
        """A block with type=text but no 'text' key returns empty string for that part."""
        content = [{"type": "text"}, {"type": "text", "text": "ok"}]
        assert hooks._extract_content(content) == " ok"

    def test_none_content(self):
        assert hooks._extract_content(None) == ""

    def test_integer_content(self):
        assert hooks._extract_content(42) == ""

    def test_empty_list(self):
        assert hooks._extract_content([]) == ""


# ---------------------------------------------------------------------------
# _read_recent_messages edge cases
# ---------------------------------------------------------------------------


class TestReadRecentMessages:
    def test_malformed_jsonl_lines_skipped(self, tmp_path):
        """Corrupted lines in transcript are silently skipped."""
        p = tmp_path / "transcript.jsonl"
        p.write_text(
            '{"role": "user", "content": "first valid message is long enough"}\n'
            'THIS IS NOT JSON\n'
            '{"role": "assistant", "content": "second valid response text"}\n'
            '{ALSO BROKEN\n'
        )
        result = hooks._read_recent_messages(str(p))
        assert len(result) == 2
        assert result[0] == ("user", "first valid message is long enough")
        assert result[1] == ("assistant", "second valid response text")

    def test_empty_file_returns_empty(self, tmp_path):
        """Empty transcript returns empty list."""
        p = tmp_path / "transcript.jsonl"
        p.write_text("")
        assert hooks._read_recent_messages(str(p)) == []

    def test_returns_recent_window(self, tmp_path):
        """Returns the last _RECENT_WINDOW messages in chronological order."""
        p = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "first user msg"}),
            json.dumps({"role": "assistant", "content": "first assistant msg"}),
            json.dumps({"role": "user", "content": "second user msg"}),
            json.dumps({"role": "assistant", "content": "second assistant msg"}),
        ]
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        # 4 messages fits within _RECENT_WINDOW (6), so all are returned
        assert len(result) == 4
        assert result[-1] == ("assistant", "second assistant msg")
        assert result[-2] == ("user", "second user msg")

    def test_window_truncates_old_messages(self, tmp_path):
        """Transcripts longer than _RECENT_WINDOW are truncated to recent end."""
        p = tmp_path / "transcript.jsonl"
        lines = []
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            lines.append(json.dumps({"role": role, "content": f"msg {i}"}))
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        assert len(result) == hooks._RECENT_WINDOW
        # Should contain messages 4-9 (the last 6)
        assert result[0] == ("user", "msg 4")
        assert result[-1] == ("assistant", "msg 9")

    def test_skips_non_user_assistant_roles(self, tmp_path):
        """tool_use, tool_result, system roles are excluded from the window."""
        p = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "user request"}),
            json.dumps({"role": "tool_use", "content": "tool call"}),
            json.dumps({"role": "tool_result", "content": "tool output"}),
            json.dumps({"role": "system", "content": "system prompt"}),
            json.dumps({"role": "assistant", "content": "assistant response"}),
        ]
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        assert len(result) == 2
        assert result[0] == ("user", "user request")
        assert result[1] == ("assistant", "assistant response")


# ---------------------------------------------------------------------------
# 6.4  stop_main
# ---------------------------------------------------------------------------


class TestStopMain:
    def _make_transcript(self, tmp_path, messages):
        """Write a JSONL transcript file and return its path."""
        p = tmp_path / "transcript.jsonl"
        lines = [json.dumps(m) for m in messages]
        p.write_text("\n".join(lines))
        return str(p)

    def _make_stdin(self, tmp_path=None, transcript_path="", **overrides):
        data = {
            "session_id": "sess-1",
            "cwd": "/home/user/myproject",
            "transcript_path": transcript_path,
        }
        data.update(overrides)
        return json.dumps(data)

    def test_normal_transcript_saves_to_mem0(self, tmp_path):
        """Normal session with meaningful messages saves to mem0."""
        transcript = self._make_transcript(tmp_path, [
            {"role": "user", "content": "Please refactor the authentication module to use JWT tokens instead of sessions"},
            {"role": "assistant", "content": "I've refactored the auth module. The key changes are: replaced express-session with jsonwebtoken, added token refresh endpoint, and updated all middleware to validate JWT headers."},
        ])

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result["continue"] is True
        mock_mem.add.assert_called_once()
        call_kwargs = mock_mem.add.call_args
        assert call_kwargs.kwargs["infer"] is True
        assert call_kwargs.kwargs["metadata"]["source"] == "session-stop-hook"
        assert call_kwargs.kwargs["metadata"]["session_id"] == "sess-1"
        # Summary includes both user and assistant exchanges
        summary = call_kwargs.kwargs["messages"][0]["content"]
        assert "[User]:" in summary
        assert "[Assistant]:" in summary
        assert "refactor" in summary.lower()

    def test_content_blocks_format(self, tmp_path):
        """Handles Claude Code's content block format."""
        transcript = self._make_transcript(tmp_path, [
            {"role": "user", "content": [{"type": "text", "text": "Implement a caching layer for the database queries with TTL support"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done. Added Redis-backed cache with configurable TTL per query type. Default is 5 minutes."}]},
        ])

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=transcript),
            )

        mock_mem.add.assert_called_once()
        messages = mock_mem.add.call_args.kwargs["messages"]
        summary_text = messages[0]["content"]
        # Summary should contain both user and assistant content from blocks
        assert "[User]:" in summary_text
        assert "[Assistant]:" in summary_text
        assert "caching layer" in summary_text.lower() or "Implement" in summary_text

    def test_short_session_skipped(self, tmp_path):
        """Short sessions (both messages below threshold) are skipped."""
        transcript = self._make_transcript(tmp_path, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ])

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_missing_transcript_skipped(self):
        """Missing transcript_path produces non-fatal response."""
        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=""),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_nonexistent_transcript_file_skipped(self):
        """Transcript path pointing to non-existent file is handled."""
        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path="/tmp/nonexistent_transcript.jsonl"),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_stop_hook_active_guard(self):
        """stop_hook_active=true exits immediately without reading transcript."""
        mock_mem = MagicMock()

        stdin_data = json.dumps({
            "session_id": "sess-1",
            "cwd": "/home/user/proj",
            "transcript_path": "/some/path",
            "stop_hook_active": True,
        })

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.stop_main, stdin_data)

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_exception_returns_nonfatal(self):
        """Any exception during stop produces a non-fatal response."""
        with patch.object(hooks, "_get_memory", side_effect=RuntimeError("boom")):
            result = _capture_output(
                hooks.stop_main,
                json.dumps({
                    "session_id": "s",
                    "cwd": "/x",
                    "transcript_path": "/nonexistent",
                }),
            )

        assert result == {"continue": True, "suppressOutput": True}

    def test_multi_exchange_captures_session_arc(self, tmp_path):
        """Multiple exchanges are included in the summary for richer context."""
        transcript = self._make_transcript(tmp_path, [
            {"role": "user", "content": "Let's add authentication to the API using JWT tokens"},
            {"role": "assistant", "content": "I'll set up JWT authentication. First, I'll install jsonwebtoken and create the middleware."},
            {"role": "user", "content": "Good. Now add refresh token rotation for security"},
            {"role": "assistant", "content": "Added refresh token rotation. Tokens are stored in Redis with a 7-day TTL and single-use enforcement."},
        ])

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=transcript),
            )

        mock_mem.add.assert_called_once()
        summary = mock_mem.add.call_args.kwargs["messages"][0]["content"]
        # Both exchanges should be present — not just the last one
        assert "JWT" in summary
        assert "refresh token" in summary.lower()
        assert "Redis" in summary

    def test_mem_add_raises_returns_nonfatal(self, tmp_path):
        """Exception during mem.add() is caught and produces non-fatal response."""
        transcript = self._make_transcript(tmp_path, [
            {"role": "user", "content": "Please refactor the authentication module to use JWT tokens instead of sessions"},
            {"role": "assistant", "content": "I've refactored the auth module. Replaced express-session with jsonwebtoken and added refresh endpoint."},
        ])

        mock_mem = MagicMock()
        mock_mem.add.side_effect = RuntimeError("LLM timeout")

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.stop_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result == {"continue": True, "suppressOutput": True}
        mock_mem.add.assert_called_once()


# ---------------------------------------------------------------------------
# 6.5  install_main
# ---------------------------------------------------------------------------


class TestInstallMain:
    def test_fresh_install(self, tmp_path):
        """Fresh install creates settings.json with both hook entries in nested format."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings_path = project_dir / ".claude" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings

        # SessionStart: matcher group with nested hooks array
        assert len(settings["hooks"]["SessionStart"]) == 1
        ss_group = settings["hooks"]["SessionStart"][0]
        assert ss_group["matcher"] == "startup|compact"
        assert len(ss_group["hooks"]) == 1
        assert ss_group["hooks"][0]["type"] == "command"
        assert ss_group["hooks"][0]["command"] == "mem0-hook-context"
        assert ss_group["hooks"][0]["timeout"] == 15000

        # Stop: matcher group with nested hooks array
        assert len(settings["hooks"]["Stop"]) == 1
        stop_group = settings["hooks"]["Stop"][0]
        assert len(stop_group["hooks"]) == 1
        assert stop_group["hooks"][0]["type"] == "command"
        assert stop_group["hooks"][0]["command"] == "mem0-hook-stop"
        assert stop_group["hooks"][0]["timeout"] == 30000

    def test_idempotent_reinstall(self, tmp_path, capsys):
        """Running install twice doesn't create duplicate entries."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["Stop"]) == 1

        captured = capsys.readouterr()
        assert "Already installed" in captured.out

    def test_preserves_existing_settings(self, tmp_path):
        """Existing settings (permissions, etc.) are preserved."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "permissions": {"allow": ["Read", "Write"]},
            "mcpServers": {"mem0": {"command": "mem0-mcp-selfhosted"}},
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert settings["permissions"] == {"allow": ["Read", "Write"]}
        assert settings["mcpServers"] == {"mem0": {"command": "mem0-mcp-selfhosted"}}
        assert "hooks" in settings

    def test_global_install(self, tmp_path, monkeypatch):
        """--global installs to ~/.claude/settings.json."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        with patch("sys.argv", ["mem0-install-hooks", "--global"]):
            hooks.install_main()

        settings_path = fake_home / ".claude" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        assert "SessionStart" in settings["hooks"]
        assert "Stop" in settings["hooks"]

    def test_default_project_dir_uses_cwd(self, tmp_path, monkeypatch):
        """Without --project-dir, install uses CWD."""
        monkeypatch.chdir(tmp_path)

        with patch("sys.argv", ["mem0-install-hooks"]):
            hooks.install_main()

        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "SessionStart" in settings["hooks"]

    def test_corrupt_settings_json_exits_with_error(self, tmp_path, capsys):
        """Invalid JSON in existing settings.json produces user-friendly error."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text("{broken json")

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                hooks.install_main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid JSON" in captured.err

    def test_existing_hooks_with_different_commands_not_matched(self, tmp_path):
        """Hooks with different commands don't prevent mem0 hooks from being added."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command", "command": "other-hook", "timeout": 5000}]}],
                "Stop": [{"hooks": [{"type": "command", "command": "another-stop-hook", "timeout": 10000}]}],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        # Both the original and new matcher groups should be present
        assert len(settings["hooks"]["SessionStart"]) == 2
        assert len(settings["hooks"]["Stop"]) == 2
        # Extract commands from nested hooks arrays
        commands = [
            handler["command"]
            for group in settings["hooks"]["SessionStart"]
            for handler in group.get("hooks", [])
        ]
        assert "other-hook" in commands
        assert "mem0-hook-context" in commands

    def test_fresh_install_output_messages(self, tmp_path, capsys):
        """Fresh install prints 'Installed:' for both hooks and the settings path."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        captured = capsys.readouterr()
        assert "Installed: SessionStart (mem0-hook-context)" in captured.out
        assert "Installed: Stop (mem0-hook-stop)" in captured.out
        assert "Settings:" in captured.out

    def test_malformed_hooks_structure_is_repaired(self, tmp_path):
        """Handles settings where 'hooks' or event keys are wrong types."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        # hooks is null, SessionStart is a string — both invalid types
        existing = {"hooks": None}
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert isinstance(settings["hooks"], dict)
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["Stop"]) == 1

    def test_nonexistent_project_dir_exits_with_error(self, tmp_path, capsys):
        """--project-dir pointing to nonexistent path exits with error."""
        fake_dir = tmp_path / "does_not_exist"

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(fake_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                hooks.install_main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_legacy_flat_format_migrated_on_reinstall(self, tmp_path, capsys):
        """Old flat-format hooks are migrated to nested format without duplicates."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        # Old flat format from previous package version
        existing = {
            "hooks": {
                "SessionStart": [{"command": "mem0-hook-context", "matcher": "startup|compact", "timeout": 15000}],
                "Stop": [{"command": "mem0-hook-stop", "timeout": 30000}],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())

        # Should have exactly 1 entry per event (migrated, not duplicated)
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["Stop"]) == 1

        # Migrated to nested format
        ss = settings["hooks"]["SessionStart"][0]
        assert ss["matcher"] == "startup|compact"
        assert ss["hooks"][0]["type"] == "command"
        assert ss["hooks"][0]["command"] == "mem0-hook-context"
        assert ss["hooks"][0]["timeout"] == 15000

        stop = settings["hooks"]["Stop"][0]
        assert stop["hooks"][0]["type"] == "command"
        assert stop["hooks"][0]["command"] == "mem0-hook-stop"
        assert stop["hooks"][0]["timeout"] == 30000

        captured = capsys.readouterr()
        assert "Already installed" in captured.out

    def test_legacy_mixed_with_other_hooks_preserved(self, tmp_path):
        """Migration preserves non-mem0 hooks alongside legacy mem0 hooks."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "hooks": {
                "SessionStart": [
                    {"command": "other-hook", "timeout": 5000},
                    {"command": "mem0-hook-context", "matcher": "startup|compact", "timeout": 15000},
                ],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        # Both hooks migrated, no duplicates for mem0-hook-context
        assert len(settings["hooks"]["SessionStart"]) == 2
        commands = [
            handler["command"]
            for group in settings["hooks"]["SessionStart"]
            for handler in group.get("hooks", [])
        ]
        assert "other-hook" in commands
        assert "mem0-hook-context" in commands

        # Stop hook auto-installed since it wasn't in the original settings
        assert len(settings["hooks"]["Stop"]) == 1
        assert settings["hooks"]["Stop"][0]["hooks"][0]["command"] == "mem0-hook-stop"
