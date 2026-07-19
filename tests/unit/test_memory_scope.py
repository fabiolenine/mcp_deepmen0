"""memory_scope Passo 2 — campo PASSIVO (routing bloqueado até o Passo 4).

Trava o contrato v1 no MCP: enum de 4 valores ou null/ausente; evidência em
NÍVEIS (nunca decimal); proveniência default carimbada (version=1, source=manual)
sem sobrescrever o que o caller mandou; whitelist expõe os campos; add_memory
rejeita valores fora do contrato ANTES de enfileirar/gravar.
"""

from __future__ import annotations

import json

import pytest

import mem0_mcp_selfhosted.server as server_mod
from mem0_mcp_selfhosted.server import _validate_scope_metadata


class TestValidateScopeMetadata:
    def test_valid_scope_stamps_provenance_defaults(self):
        md = {"memory_scope": "user_fact"}
        assert _validate_scope_metadata(md) is None
        assert md["memory_scope_version"] == 1
        assert md["memory_scope_source"] == "manual"

    def test_caller_provenance_not_overwritten(self):
        md = {"memory_scope": "system_meta", "memory_scope_version": 1,
              "memory_scope_source": "backfill_rule"}
        assert _validate_scope_metadata(md) is None
        assert md["memory_scope_source"] == "backfill_rule"

    def test_invalid_scope_rejected(self):
        err = _validate_scope_metadata({"memory_scope": "unscoped"})
        assert err and "memory_scope inválido" in err

    def test_null_scope_means_absence(self):
        md = {"memory_scope": None, "other": 1}
        assert _validate_scope_metadata(md) is None
        assert "memory_scope" not in md  # abstention = chave ausente, não 5º valor
        assert "memory_scope_version" not in md

    def test_evidence_levels_only(self):
        assert _validate_scope_metadata({"memory_scope_evidence": "decisive"}) is None
        err = _validate_scope_metadata({"memory_scope_evidence": 0.87})
        assert err and "memory_scope_evidence inválido" in err

    def test_absent_metadata_ok(self):
        assert _validate_scope_metadata(None) is None
        assert _validate_scope_metadata({}) is None


class TestWhitelistExposure:
    def test_scope_fields_in_search_whitelist(self):
        # a whitelist é literal dentro de search_memories (em _register_tools);
        # validar por fonte evita depender de infra viva
        import inspect
        src = inspect.getsource(server_mod._register_tools)
        for field in ("memory_scope", "memory_scope_version", "memory_scope_source",
                      "memory_scope_evidence", "memory_scope_reason"):
            assert f'"{field}"' in src, f"{field} ausente da _metadata_whitelist"


def test_add_memory_rejects_invalid_scope(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    mem = MagicMock()
    original = server_mod.memory
    server_mod.memory = mem
    try:
        srv = server_mod._create_server()
        tool = srv._tool_manager._tools["add_memory"].fn
        out = json.loads(tool(text="x", user_id="alice",
                              metadata={"memory_scope": "meta"}))
        assert "error" in out and "memory_scope inválido" in out["error"]
        mem.add.assert_not_called()
    finally:
        server_mod.memory = original
