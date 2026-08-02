"""Fontes de leitura: cache com TTL, leitor do Qdrant e a guarda de só-leitura."""

import ast
import inspect
from pathlib import Path

import anyio
import pytest

from mem0_mcp_selfhosted.vault.memories import qdrant_read, sources as src
from mem0_mcp_selfhosted.vault.memories.cache import TTLCache
from mem0_mcp_selfhosted.vault.memories.model import Cursor
from mem0_mcp_selfhosted.vault.memories.qdrant_read import QdrantReader

from tests.fakes.fake_qdrant import FakeQdrant, point


# ------------------------------------------------------------------- cache


class TestTTLCache:
    def test_builds_once_while_valid(self):
        calls = []

        async def build():
            calls.append(1)
            return "v"

        async def scenario():
            cache = TTLCache()
            assert await cache.get_or_build("k", 60, build) == "v"
            assert await cache.get_or_build("k", 60, build) == "v"

        anyio.run(scenario)
        assert len(calls) == 1

    def test_rebuilds_after_expiry(self, monkeypatch):
        now = {"t": 1000.0}
        monkeypatch.setattr(
            "mem0_mcp_selfhosted.vault.memories.cache.time.monotonic", lambda: now["t"]
        )
        calls = []

        async def build():
            calls.append(1)
            return len(calls)

        async def scenario():
            cache = TTLCache()
            await cache.get_or_build("k", 1, build)
            now["t"] += 10  # passou do TTL
            await cache.get_or_build("k", 1, build)

        anyio.run(scenario)
        assert len(calls) == 2

    def test_concurrent_misses_build_once(self):
        calls = []

        async def build():
            await anyio.sleep(0.01)
            calls.append(1)
            return "v"

        async def scenario():
            cache = TTLCache()
            async with anyio.create_task_group() as tg:
                for _ in range(5):
                    tg.start_soon(cache.get_or_build, "k", 60, build)

        anyio.run(scenario)
        assert len(calls) == 1

    def test_invalidate(self):
        async def scenario():
            cache = TTLCache()
            await cache.get_or_build("k", 60, lambda: _const("a"))
            cache.invalidate("k")
            assert cache.peek("k") is None

        anyio.run(scenario)


async def _const(value):
    return value


# ------------------------------------------------------ invariante de escrita


class TestReadOnlyGuard:
    """O leitor não pode ganhar um verbo de escrita sem alguém decidir.

    Guarda no espírito do `test_vault_no_extra`: o arquivo é lido e reprovado se
    chamar um método mutador do cliente. Um `delete` que entrasse aqui por
    conveniência apagaria corpus de produção a partir de uma tela de leitura.
    """

    FORBIDDEN = {
        "upsert", "delete", "set_payload", "overwrite_payload", "delete_payload",
        "clear_payload", "update_vectors", "delete_vectors", "create_collection",
        "delete_collection", "recreate_collection", "create_payload_index",
        "delete_payload_index", "update_collection", "batch_update_points",
        "create_snapshot", "delete_snapshot", "recover_snapshot",
    }

    def test_reader_calls_no_mutating_method(self):
        source = Path(inspect.getfile(qdrant_read)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not (called & self.FORBIDDEN), f"verbo de escrita no leitor: {called & self.FORBIDDEN}"


# ------------------------------------------------------------------ leitor


@pytest.fixture
def reader():
    client = FakeQdrant(
        {
            "col": [
                point("a", "2026-07-23T10:00:00+00:00", data="alfa", domain="ai"),
                point("b", "2026-07-23T09:00:00+00:00", data="beta", domain="infra"),
                point("c", "2026-07-23T09:00:00+00:00", data="gama", domain="ai"),
                point("z", "2026-07-23T08:00:00+00:00", data="outro", user_id="alguem"),
            ]
        }
    )
    return QdrantReader(client, "col", "col_entities", {"user_id": "ana"})


class TestQdrantReader:
    def test_list_respects_scope(self, reader):
        rows, _ = anyio.run(
            lambda: reader.list_memories(filters=None, cursor=Cursor(), page_size=10)
        )
        # "z" é de outro user_id. A ordem entre b e c (mesmo instante) não é
        # promessa da API, então o que se afirma é o conjunto.
        assert {r["id"] for r in rows} == {"a", "b", "c"}

    def test_list_orders_newest_first(self, reader):
        rows, _ = anyio.run(
            lambda: reader.list_memories(filters=None, cursor=Cursor(), page_size=10)
        )
        assert rows[0]["id"] == "a"

    def test_pagination_does_not_repeat_across_a_tie(self, reader):
        """b e c compartilham created_at: a página 2 não pode repeti-los."""

        async def scenario():
            page1, cur = await reader.list_memories(
                filters=None, cursor=Cursor(), page_size=2
            )
            page2, _ = await reader.list_memories(filters=None, cursor=cur, page_size=2)
            return page1, page2

        page1, page2 = anyio.run(scenario)
        assert page1[0]["id"] == "a"  # o mais novo vem primeiro
        entregues = [r["id"] for r in page1] + [r["id"] for r in page2]
        assert entregues[0] == "a"
        assert sorted(entregues) == ["a", "b", "c"]  # nada repetido, nada perdido

    def test_full_walk_yields_every_point_exactly_once(self):
        """Um instante com mais pontos que a página: nada some, nada repete."""
        ts = "2026-07-23T03:31:28.904335+00:00"
        client = FakeQdrant(
            {"col": [point(f"p{i:02d}", ts, data=f"chunk {i}") for i in range(30)]}
        )
        reader = QdrantReader(client, "col", "col_e", {"user_id": "ana"})

        async def walk():
            seen, cursor = [], Cursor()
            for _ in range(20):  # trava anti-loop: 30 itens em páginas de 7
                rows, cursor = await reader.list_memories(
                    filters=None, cursor=cursor, page_size=7
                )
                if not rows:
                    break
                seen.extend(r["id"] for r in rows)
                if cursor is None:
                    break
            return seen

        seen = anyio.run(walk)
        assert len(seen) == 30
        assert len(set(seen)) == 30

    def test_filter_by_domain(self, reader):
        rows, _ = anyio.run(
            lambda: reader.list_memories(
                filters={"domain": "ai"}, cursor=Cursor(), page_size=10
            )
        )
        assert [r["id"] for r in rows] == ["a", "c"]

    def test_count_applies_scope(self, reader):
        assert anyio.run(reader.count) == 3

    def test_facets_are_scoped(self, reader):
        facets = anyio.run(reader.facets)
        assert facets["domain"] == [
            {"value": "ai", "count": 2},
            {"value": "infra", "count": 1},
        ]

    def test_get_memory_returns_full_payload(self, reader):
        payload = anyio.run(lambda: reader.get_memory("a"))
        assert payload["data"] == "alfa" and payload["_id"] == "a"

    def test_get_memory_outside_scope_is_invisible(self, reader):
        """Id de outro escopo colado na URL não pode virar leitura."""
        assert anyio.run(lambda: reader.get_memory("z")) is None

    def test_get_memory_missing_returns_none(self, reader):
        assert anyio.run(lambda: reader.get_memory("nao-existe")) is None

    def test_get_memories_by_ids(self, reader):
        rows = anyio.run(lambda: reader.get_memories(["a", "b"]))
        assert {r["id"] for r in rows} == {"a", "b"}
        assert anyio.run(lambda: reader.get_memories([])) == []


# ------------------------------------------------------------------ fábrica


class TestBuildSources:
    def test_missing_api_key_degrades_instead_of_raising(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_QDRANT_API_KEY", raising=False)
        monkeypatch.setenv("VAULT_QDRANT_ENV_FILE", str(tmp_path / "ausente.env"))
        sources = src.build_sources()
        assert sources.qdrant is None
        assert "MEM0_QDRANT_API_KEY" in sources.qdrant_error
        with pytest.raises(src.SourceUnavailable):
            sources.require_qdrant()

    def test_api_key_read_from_env_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_QDRANT_API_KEY", raising=False)
        env_file = tmp_path / ".qdrant.env"
        env_file.write_text("# comentário\nMEM0_QDRANT_API_KEY=segredo\n", encoding="utf-8")
        monkeypatch.setenv("VAULT_QDRANT_ENV_FILE", str(env_file))
        assert src._qdrant_api_key() == "segredo"

    def test_env_wins_over_file(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".qdrant.env"
        env_file.write_text("MEM0_QDRANT_API_KEY=do-arquivo\n", encoding="utf-8")
        monkeypatch.setenv("VAULT_QDRANT_ENV_FILE", str(env_file))
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", "do-ambiente")
        assert src._qdrant_api_key() == "do-ambiente"

    def test_entity_collection_derives_from_collection(self, monkeypatch):
        monkeypatch.setenv("VAULT_QDRANT_COLLECTION", "minha_col")
        monkeypatch.delenv("VAULT_QDRANT_ENTITY_COLLECTION", raising=False)
        assert src.build_sources().entity_collection == "minha_col_entities"

    def test_scope_defaults_to_user_id(self, monkeypatch):
        monkeypatch.delenv("VAULT_MEMORY_USER_ID", raising=False)
        monkeypatch.setenv("MEM0_USER_ID", "alguem")
        assert src.build_sources().scope == {"user_id": "alguem"}

    def test_scope_uses_the_same_default_as_the_mcp_tools(self, monkeypatch):
        """A UI não pode enxergar um escopo diferente do que os clientes escrevem.

        E o pacote não pode sair de fábrica apontando para o corpus de uma
        instalação concreta: o escopo real vive no `.vault.env` de cada host.
        """
        from mem0_mcp_selfhosted.helpers import get_default_user_id

        monkeypatch.delenv("VAULT_MEMORY_USER_ID", raising=False)
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert src.build_sources().scope == {"user_id": get_default_user_id()}

    def test_vault_scope_env_wins_over_the_mcp_default(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "do-mcp")
        monkeypatch.setenv("VAULT_MEMORY_USER_ID", "so-da-ui")
        assert src.build_sources().scope == {"user_id": "so-da-ui"}

    def test_default_collection_is_the_packages_own(self):
        """Nome de collection de uma instalação concreta não é default de pacote."""
        assert src.DEFAULT_COLLECTION == "mem0_mcp_selfhosted"

    def test_missing_local_dbs_are_reported_as_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VAULT_INGEST_DB", str(tmp_path / "nao-existe.db"))
        monkeypatch.setenv("VAULT_HISTORY_DB", str(tmp_path / "nao-existe2.db"))
        sources = src.build_sources()
        assert sources.queue_db is None and sources.history_db is None
