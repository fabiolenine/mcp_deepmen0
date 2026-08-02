"""Entidades: filtro do índice em memória e as duas telas."""

import pytest
from starlette.testclient import TestClient

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault import web
from mem0_mcp_selfhosted.vault.memories import model
from mem0_mcp_selfhosted.vault.memories.qdrant_read import QdrantReader
from mem0_mcp_selfhosted.vault.memories.sources import Sources

from tests.fakes.fake_qdrant import FakeQdrant, FakePoint, point

ADMIN_EMAIL = "ana.souza@acme.com.br"
ADMIN_PASSWORD = "uma senha longa o suficiente"
SCOPE = {"user_id": "ana"}

MEM_A = "aaaaaaaa-0000-4000-8000-00000000000a"
MEM_B = "bbbbbbbb-0000-4000-8000-00000000000b"
ENT_DEEPMEM0 = "11111111-1111-5111-8111-111111111111"
ENT_FASE = "22222222-2222-5222-8222-222222222222"
ENT_ALHEIA = "33333333-3333-5333-8333-333333333333"


def _entity(pid, data, normalized, kind, links, user_id="ana"):
    return FakePoint(pid, {
        "data": data, "data_normalized": normalized, "entity_type": kind,
        "linked_memory_ids": links, "user_id": user_id,
        **{f"lnk_{i}": True for i in links},
    })


@pytest.fixture(autouse=True)
def _clean_health_cache():
    web._health_cache.update(at=0.0, value=None)
    yield
    web._health_cache.update(at=0.0, value=None)


@pytest.fixture
def store(tmp_path):
    s = vs.VaultStore(tmp_path / "vault.db")
    s.create_user(
        email=ADMIN_EMAIL, display_name="Ana", is_admin=True,
        password_hash=sec.hash_password(ADMIN_PASSWORD),
    )
    return s


def _sources(error=""):
    client = FakeQdrant({
        "col": [
            point(MEM_A, "2026-07-23T10:00:00+00:00", data="memória sobre o DeepMem0"),
            point(MEM_B, "2026-07-23T09:00:00+00:00", data="outra memória"),
        ],
        "col_entities": [
            _entity(ENT_DEEPMEM0, "DeepMem0", "deepmem0", "PROPER", [MEM_A, MEM_B]),
            _entity(ENT_FASE, "FASE", "fase", "COMPOUND", [MEM_A]),
            _entity(ENT_ALHEIA, "Alheia", "alheia", "PROPER", [], user_id="outro"),
        ],
    })
    reader = None if error else QdrantReader(client, "col", "col_entities", SCOPE)
    return Sources(
        collection="col", entity_collection="col_entities", scope=SCOPE,
        qdrant=reader, qdrant_error=error,
    )


def _client(store, error=""):
    app = web.create_app(store.db_path, secret_key="k", sources=_sources(error))
    client = TestClient(app)
    page = client.get("/login")
    csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
    client.post(
        "/login",
        data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "csrf": csrf},
        follow_redirects=False,
    )
    return client


# ------------------------------------------------------------------ filtro


ROWS = [
    {"data": "DeepMem0", "data_normalized": "deepmem0", "entity_type": "PROPER", "link_count": 12},
    {"data": "FASE", "data_normalized": "fase", "entity_type": "COMPOUND", "link_count": 3},
    {"data": "Fase 7", "data_normalized": "fase 7", "entity_type": "COMPOUND", "link_count": 9},
]


class TestFilterEntities:
    def test_empty_query_returns_everything(self):
        assert len(model.filter_entities(ROWS)) == 3

    def test_substring_is_case_insensitive_via_normalized_identity(self):
        """Procurar "fase" tem de achar FASE e Fase 7 — é a mesma normalização
        que o store usa para não duplicar linhas."""
        got = model.filter_entities(ROWS, query="FASE")
        assert {r["data"] for r in got} == {"FASE", "Fase 7"}

    def test_sorted_by_link_count_desc(self):
        got = model.filter_entities(ROWS)
        assert [r["link_count"] for r in got] == [12, 9, 3]

    def test_filter_by_kind(self):
        got = model.filter_entities(ROWS, entity_type="PROPER")
        assert [r["data"] for r in got] == ["DeepMem0"]

    def test_query_and_kind_compose(self):
        assert model.filter_entities(ROWS, query="fase", entity_type="PROPER") == []

    def test_no_match_is_empty(self):
        assert model.filter_entities(ROWS, query="não existe") == []

    def test_kinds_are_counted(self):
        kinds = model.entity_kinds(ROWS)
        assert kinds[0] == {"value": "COMPOUND", "count": 2}


class TestEntityRaw:
    def test_link_keys_are_omitted(self):
        """Numa entidade com 116 vínculos, as chaves lnk_ encobririam o payload."""
        raw = model.entity_raw({"data": "X", "lnk_a": True, "lnk_b": True, "outro": 1})
        assert raw == {"outro": 1}


# -------------------------------------------------------------------- telas


class TestEntitiesScreen:
    def test_requires_login(self, store):
        app = web.create_app(store.db_path, secret_key="k", sources=_sources())
        assert TestClient(app).get("/entities", follow_redirects=False).status_code == 303
        assert (
            TestClient(app).get(f"/entities/{ENT_DEEPMEM0}", follow_redirects=False).status_code
            == 303
        )

    def test_lists_scoped_entities_only(self, store):
        body = _client(store).get("/entities").text
        assert "DeepMem0" in body and "FASE" in body
        assert "Alheia" not in body

    def test_search_filters(self, store):
        body = _client(store).get("/entities?q=deep").text
        assert "DeepMem0" in body and ">FASE<" not in body

    def test_kind_chips_are_rendered(self, store):
        body = _client(store).get("/entities").text
        assert "PROPER" in body and "COMPOUND" in body

    def test_detail_lists_linked_memories(self, store):
        body = _client(store).get(f"/entities/{ENT_DEEPMEM0}").text
        assert "memória sobre o DeepMem0" in body
        assert f'href="/memories/{MEM_A}"' in body

    def test_detail_shows_normalized_identity(self, store):
        assert "deepmem0" in _client(store).get(f"/entities/{ENT_DEEPMEM0}").text

    def test_dangling_links_are_reported(self, store):
        """Vínculo para memória que não existe é o invariante que o
        check_corpus mede; a tela mostra em vez de omitir."""
        sources = _sources()
        sources.qdrant._c.collections["col_entities"].append(
            _entity("44444444-4444-5444-8444-444444444444", "Órfã", "órfã", "PROPER",
                    ["99999999-9999-4999-8999-999999999999"])
        )
        app = web.create_app(store.db_path, secret_key="k", sources=sources)
        client = TestClient(app)
        page = client.get("/login")
        csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
        client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
                                    "csrf": csrf}, follow_redirects=False)
        body = client.get("/entities/44444444-4444-5444-8444-444444444444").text
        assert "apontando para memórias que não existem" in body

    def test_entity_of_another_scope_is_not_found(self, store):
        assert _client(store).get(f"/entities/{ENT_ALHEIA}").status_code == 404

    def test_malformed_id_is_404(self, store):
        assert _client(store).get("/entities/nao-e-uuid").status_code == 404

    def test_degraded_source_renders_card(self, store):
        response = _client(store, error="Qdrant fora do ar").get("/entities")
        assert response.status_code == 200 and "Qdrant fora do ar" in response.text

    def test_nav_links_to_entities(self, store):
        assert 'href="/entities"' in _client(store).get("/").text
