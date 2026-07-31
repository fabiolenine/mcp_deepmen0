"""O opt-out de reforço na busca (T3) — a defesa do instrumento de medição.

Com T3 ligado, toda busca reforça o que devolve. O golden set roda 37 buscas no
corpus VIVO com alvos conhecidos: sem opt-out, cada rodada do gate reforçaria os
próprios alvos e a métrica inflaria sozinha, destruindo em silêncio o único
instrumento que este projeto usa para decidir.

Todo teste negativo aqui vem com o CONTROLE POSITIVO ao lado (com `true` o
parâmetro realmente chega ao core) — senão "não reforçou" seria indistinguível de
"o teste não exercita nada".
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def server_with_mock():
    original = server_mod.memory
    mem = MagicMock()
    mem.graph = None
    mem.enable_graph = False
    mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.9}]}
    server_mod.memory = mem
    srv = server_mod._create_server()
    yield srv, mem
    server_mod.memory = original


def _tool(srv, name="search_memories"):
    return srv._tool_manager._tools[name].fn


class TestReinforceForwarding:
    def test_false_is_forwarded_exactly(self, server_with_mock):
        srv, mem = server_with_mock
        _tool(srv)(query="q", reinforce=False)
        assert mem.search.call_args.kwargs["reinforce"] is False

    def test_true_is_forwarded_exactly(self, server_with_mock):
        """Controle positivo: sem ele, o teste acima passaria mesmo se o
        parâmetro fosse descartado no caminho."""
        srv, mem = server_with_mock
        _tool(srv)(query="q", reinforce=True)
        assert mem.search.call_args.kwargs["reinforce"] is True

    def test_omitted_defers_to_server_config(self, server_with_mock):
        srv, mem = server_with_mock
        _tool(srv)(query="q")
        assert "reinforce" not in mem.search.call_args.kwargs

    def test_exposed_in_the_tool_schema(self, server_with_mock):
        """O harness só consegue se proteger se o parâmetro existir na tool."""
        srv, _ = server_with_mock
        tool = srv._tool_manager._tools["search_memories"]
        schema = tool.parameters if isinstance(tool.parameters, dict) else tool.parameters.schema()
        assert "reinforce" in schema["properties"]

    def test_results_are_unaffected(self, server_with_mock):
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(query="q", reinforce=False))
        assert parsed["results"][0]["id"] == "mem-1"


class TestDynamicsEnvMapping:
    """tie_band NÃO era mapeado: no drop-in viraria configuração ignorada em
    silêncio — a mesma classe do `Environment=` sem aspas de 19/07."""

    def _dynamics(self, monkeypatch, **envs):
        for k in ("MEM0_DYNAMICS_ENABLED", "MEM0_DYNAMICS_WEIGHT", "MEM0_DYNAMICS_TIE_BAND",
                  "MEM0_REINFORCE_ON_SEARCH", "MEM0_REINFORCE_TOP_N",
                  "MEM0_REINFORCE_ON_SEARCH_WINDOW", "MEM0_REINFORCE_ON_SIMILAR",
                  "MEM0_REINFORCE_SIMILARITY_THRESHOLD"):
            monkeypatch.delenv(k, raising=False)
        for k, v in envs.items():
            monkeypatch.setenv(k, v)
        from mem0_mcp_selfhosted.config import build_config
        cfg, _providers, _split = build_config()
        return cfg.get("dynamics", {})

    def test_tie_band_reaches_the_config(self, monkeypatch):
        assert self._dynamics(monkeypatch, MEM0_DYNAMICS_TIE_BAND="0.01")["tie_band"] == 0.01

    def test_shadow_mode_zeroes_the_fusion_weight(self, monkeypatch):
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_ON_SEARCH="true",
                             MEM0_DYNAMICS_WEIGHT="0")
        assert dyn["reinforce_on_search"] is True
        assert dyn["weight"] == 0.0

    def test_top_n_and_exposure_window(self, monkeypatch):
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_TOP_N="5",
                             MEM0_REINFORCE_ON_SEARCH_WINDOW="3600")
        assert dyn["reinforce_top_n"] == 5
        assert dyn["reinforce_on_search_window"] == 3600

    def test_no_envs_means_no_dynamics_block(self, monkeypatch):
        """Sem env, valem os defaults do fork — o MCP não deve inventar config."""
        assert self._dynamics(monkeypatch) == {}

    def test_similar_flag_false_is_forwarded(self, monkeypatch):
        """v0.8 (t1s): desligar por env TEM que chegar como False, não sumir."""
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_ON_SIMILAR="false")
        assert dyn["reinforce_on_similar"] is False

    def test_similar_threshold_zero_reaches_the_config(self, monkeypatch):
        """Regressão do /critic-plan #4: env "0" é uma STRING truthy e TEM que
        chegar como 0.0 (0 = gatilho desligado no fork). Se este teste quebrar,
        o kill switch por threshold morreu em silêncio."""
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_SIMILARITY_THRESHOLD="0")
        assert dyn["reinforce_similarity_threshold"] == 0.0

    def test_similar_threshold_custom_value(self, monkeypatch):
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_ON_SIMILAR="true",
                             MEM0_REINFORCE_SIMILARITY_THRESHOLD="0.97")
        assert dyn["reinforce_on_similar"] is True
        assert dyn["reinforce_similarity_threshold"] == 0.97

    def test_similar_empty_env_is_ignored_not_crash(self, monkeypatch):
        """Env vazia (classe docker-compose) não pode virar float('') no boot."""
        dyn = self._dynamics(monkeypatch, MEM0_REINFORCE_SIMILARITY_THRESHOLD="")
        assert "reinforce_similarity_threshold" not in dyn

    def test_historical_recall_env_reaches_temporality(self, monkeypatch):
        """v0.10: kill switch da recordação histórica."""
        monkeypatch.delenv("MEM0_HISTORICAL_RECALL", raising=False)
        from mem0_mcp_selfhosted.config import build_config

        monkeypatch.setenv("MEM0_HISTORICAL_RECALL", "false")
        cfg, _p, _s = build_config()
        assert cfg["temporality"]["historical_recall"] is False
        monkeypatch.setenv("MEM0_HISTORICAL_RECALL", "true")
        cfg, _p, _s = build_config()
        assert cfg["temporality"]["historical_recall"] is True

    def test_version_inherits_dynamics_env_reaches_temporality(self, monkeypatch):
        """v0.9: sem o mapeamento, o env no drop-in seria config silenciosamente
        ignorada — a mesma classe do tie_band ausente."""
        for k in ("MEM0_VERSION_INHERITS_DYNAMICS",):
            monkeypatch.delenv(k, raising=False)
        from mem0_mcp_selfhosted.config import build_config

        monkeypatch.setenv("MEM0_VERSION_INHERITS_DYNAMICS", "false")
        cfg, _p, _s = build_config()
        assert cfg["temporality"]["version_inherits_dynamics"] is False
        monkeypatch.setenv("MEM0_VERSION_INHERITS_DYNAMICS", "true")
        cfg, _p, _s = build_config()
        assert cfg["temporality"]["version_inherits_dynamics"] is True
