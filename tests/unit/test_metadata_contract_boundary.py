"""Fronteira de escrita: contrato dos campos tipados de metadata (26/07/2026).

Por que estes testes existem: 17 memórias entraram no corpus com
`importance='high'` e `domain='deepmem0'` porque `add_memory` gravava a metadata
do caller sem validar contra o contrato que o resto do sistema assume. O defeito
só apareceu quando o Open WebUI tentou RECUPERAR memória — a comparação str vs
float derrubou a busca inteira. Aqui a rejeição acontece no único ponto em que
ainda dá para devolver um erro acionável a quem escreveu: o submit.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")
    monkeypatch.delenv("MEM0_METADATA_CONTRACT", raising=False)


@pytest.fixture
def server_with_mock():
    original_memory = server_mod.memory
    mem = MagicMock()
    mem.graph = None
    mem.enable_graph = False
    mem.add.return_value = {"results": [{"id": "mem-1", "memory": "fato", "event": "ADD"}]}
    server_mod.memory = mem
    srv = server_mod._create_server()
    yield srv, mem
    server_mod.memory = original_memory


def _tool(srv, name="add_memory"):
    return srv._tool_manager._tools[name].fn


INVALID = [
    ({"importance": "high"}, "importance"),          # o caso real do incidente
    ({"importance": True}, "importance"),            # bool passa por isinstance(int)
    ({"importance": 1.5}, "importance"),             # fora da faixa
    ({"importance": float("nan")}, "importance"),    # NaN
    ({"confidence": "alta"}, "confidence"),
    ({"domain": "deepmem0"}, "domain"),              # nome de projeto no slot errado
    ({"memory_type": "semantica"}, "memory_type"),
    ({"tags": "deploy"}, "tags"),                    # string crua em vez de lista
    ({"tags": ["ok", 3]}, "tags"),
]


class TestBoundaryRejects:
    @pytest.mark.parametrize("metadata,field", INVALID)
    def test_add_memory_rejects_and_stores_nothing(self, server_with_mock, metadata, field):
        srv, mem = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata=metadata, infer=False))
        assert field in parsed["error"]
        mem.add.assert_not_called()  # nada persiste

    @pytest.mark.parametrize("metadata,field", [
        ({"domain": []}, "domain"),                 # unhashable: `in frozenset` levantava TypeError
        ({"domain": {"a": 1}}, "domain"),
        ({"memory_type": []}, "memory_type"),
        ({"importance": 10 ** 400}, "importance"),  # float(int gigante) levantava OverflowError
        ({"tags": ["\x00"]}, "tags"),
        ({"tags": ["​"]}, "tags"),
        ({"tags": ["x" * 65]}, "tags"),
    ])
    def test_hostile_input_returns_error_never_raises(self, server_with_mock, metadata, field):
        """A tool tem que devolver {"error": ...} — validador que EXPLODE com
        entrada hostil é o mesmo defeito que o contrato existe para consertar."""
        srv, mem = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata=metadata, infer=False))
        assert field in parsed["error"]
        mem.add.assert_not_called()

    def test_domain_error_points_to_the_project_field(self, server_with_mock):
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata={"domain": "deepmem0"}, infer=False))
        assert "project" in parsed["error"]

    def test_async_path_rejects_before_enqueue(self, server_with_mock, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_ASYNC_INGEST", "true")
        monkeypatch.setenv("MEM0_QUEUE_DB_PATH", str(tmp_path / "q.db"))
        server_mod._ingest_queue = None
        server_mod._ingest_worker = None
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata={"importance": "high"}))
        assert "importance" in parsed["error"]
        queue, _ = server_mod._get_ingest()
        assert queue.claim_next() is None  # não entrou na fila
        server_mod._ingest_queue = None
        server_mod._ingest_worker = None

    def test_add_document_rejects_before_touching_the_file(self, server_with_mock):
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv, "add_document")(
            file_path="/caminho/inexistente.pdf", metadata={"importance": "high"}))
        assert "importance" in parsed["error"]  # metadata falha antes do arquivo


class TestBoundaryAccepts:
    def test_valid_typed_metadata_passes_through(self, server_with_mock):
        srv, mem = server_with_mock
        parsed = json.loads(_tool(srv)(
            text="fato",
            metadata={"importance": 0.8, "confidence": 0.9, "domain": "ai",
                      "memory_type": "episodic", "tags": ["deploy", "roadmap"]},
            infer=False))
        assert parsed["status"] == "stored"
        assert mem.add.call_args.kwargs["metadata"]["importance"] == 0.8

    def test_free_form_keys_still_pass(self, server_with_mock):
        """NÃO-REGRESSÃO: 940 pontos do corpus usam campos livres — restringir a
        metadata desconhecida quebraria o padrão de escrita de quase tudo."""
        srv, mem = server_with_mock
        free = {"project": "DeepMem0", "subcategory": "rag", "topic": "x",
                "attributed_to": "user", "session_date": "2026-07-26"}
        parsed = json.loads(_tool(srv)(text="fato", metadata=dict(free), infer=False))
        assert parsed["status"] == "stored"
        assert mem.add.call_args.kwargs["metadata"] == free

    def test_none_is_abstention_not_error(self, server_with_mock):
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata={"importance": None}, infer=False))
        assert parsed["status"] == "stored"

    def test_scope_contract_still_enforced(self, server_with_mock):
        """Não-regressão do contrato memory_scope v1 (Passo 2)."""
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(
            text="fato", metadata={"memory_scope": "inexistente"}, infer=False))
        assert "memory_scope" in parsed["error"]


class TestKillSwitch:
    def test_warn_accepts_and_logs(self, server_with_mock, monkeypatch, caplog):
        monkeypatch.setenv("MEM0_METADATA_CONTRACT", "warn")
        srv, mem = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata={"importance": "high"}, infer=False))
        assert parsed["status"] == "stored"
        assert any("metadata_contract" in r.message for r in caplog.records)

    def test_off_disables_the_boundary(self, server_with_mock, monkeypatch):
        monkeypatch.setenv("MEM0_METADATA_CONTRACT", "off")
        srv, _ = server_with_mock
        parsed = json.loads(_tool(srv)(text="fato", metadata={"importance": "high"}, infer=False))
        assert parsed["status"] == "stored"

    def test_typo_in_env_raises_instead_of_silently_disabling(self, server_with_mock, monkeypatch):
        monkeypatch.setenv("MEM0_METADATA_CONTRACT", "enforce_pls")
        srv, _ = server_with_mock
        with pytest.raises(ValueError):
            _tool(srv)(text="fato", metadata={"importance": 0.5}, infer=False)
