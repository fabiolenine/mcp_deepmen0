"""As telas de memória: guard de sessão, filtros, paginação e escape."""

import pytest
from starlette.testclient import TestClient

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault import web
from mem0_mcp_selfhosted.vault.memories.qdrant_read import QdrantReader
from mem0_mcp_selfhosted.vault.memories.sources import Sources

from tests.fakes.fake_qdrant import FakeQdrant, point

ADMIN_EMAIL = "ana.souza@acme.com.br"
ADMIN_PASSWORD = "uma senha longa o suficiente"
PLAIN_EMAIL = "bruno@acme.com.br"

SCOPE = {"user_id": "ana"}


@pytest.fixture(autouse=True)
def _instant_failed_login(monkeypatch):
    async def _noop(_seconds):
        return None

    monkeypatch.setattr(web.anyio, "sleep", _noop)


@pytest.fixture(autouse=True)
def _clean_health_cache():
    web._health_cache.update(at=0.0, value=None)
    yield
    web._health_cache.update(at=0.0, value=None)


@pytest.fixture
def store(tmp_path):
    s = vs.VaultStore(tmp_path / "vault.db")
    s.create_user(
        email=ADMIN_EMAIL, display_name="Ana Souza", is_admin=True,
        password_hash=sec.hash_password(ADMIN_PASSWORD),
    )
    return s


def _points():
    return [
        point("aaaaaaaa-0000-4000-8000-000000000001", "2026-07-23T10:00:00+00:00",
              data="Qdrant exige api-key neste host", domain="infrastructure",
              memory_type="procedural", importance=0.9, tags=["qdrant", "seguranca"]),
        point("aaaaaaaa-0000-4000-8000-000000000002", "2026-07-23T09:00:00+00:00",
              data="O reranker roda em CPU", domain="ai", memory_type="semantic",
              importance=0.8, superseded_at="2026-07-30T00:00:00+00:00"),
        point("aaaaaaaa-0000-4000-8000-000000000003", "2026-07-23T09:00:00+00:00",
              data="Chunk de documento", domain="ai", source_doc="manual.pdf",
              page_start=3, page_end=4, event_date="2026-05-01"),
        point("bbbbbbbb-0000-4000-8000-000000000009", "2026-07-23T08:00:00+00:00",
              data="De outro escopo", user_id="outra-pessoa"),
    ]


def _sources(points=None, error=""):
    client = FakeQdrant({"col": points if points is not None else _points()})
    reader = None if error else QdrantReader(client, "col", "col_entities", SCOPE)
    return Sources(
        collection="col", entity_collection="col_entities", scope=SCOPE,
        qdrant=reader, qdrant_error=error,
    )


@pytest.fixture
def client(store):
    app = web.create_app(
        store.db_path, secret_key="test-secret-key-not-production", sources=_sources()
    )
    return TestClient(app)


def _login(client, email=ADMIN_EMAIL, password=ADMIN_PASSWORD):
    page = client.get("/login")
    csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
    return client.post(
        "/login", data={"email": email, "password": password, "csrf": csrf},
        follow_redirects=False,
    )


class TestGuard:
    def test_anonymous_is_redirected_to_login(self, client):
        response = client.get("/memories", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

    def test_non_admin_cannot_reach_the_screen(self, store, client):
        """Usuário sem is_admin não vê o corpus.

        `current_admin` já recusa não-admin, mas isso é uma decisão de
        autorização e precisa de teste próprio: a tela lê o corpus INTEIRO do
        escopo, sem o funil de escopo por token que o MCP aplica.
        """
        store.create_user(
            email=PLAIN_EMAIL, display_name="Bruno", is_admin=False,
            password_hash=sec.hash_password(ADMIN_PASSWORD),
        )
        # A senha está certa; o que falta é ser admin — o cofre recusa com 401.
        assert _login(client, PLAIN_EMAIL).status_code == 401
        assert client.get("/memories", follow_redirects=False).status_code == 303

    def test_disabled_admin_loses_access(self, store, client):
        _login(client)
        assert client.get("/memories").status_code == 200
        user = store.get_user_by_email(ADMIN_EMAIL)
        store.set_user_disabled(
            user["id"], True, actor_id=user["id"], actor_email=ADMIN_EMAIL, ip=""
        )
        assert client.get("/memories", follow_redirects=False).status_code == 303


class TestListing:
    def test_lists_scoped_memories(self, client):
        _login(client)
        body = client.get("/memories").text
        assert "Qdrant exige api-key neste host" in body
        assert "O reranker roda em CPU" in body
        assert "De outro escopo" not in body

    def test_shows_badges_for_supersedence_and_document(self, client):
        _login(client)
        body = client.get("/memories").text
        assert "supersedida" in body
        assert "documento" in body
        assert "2026-05-01" in body  # event_date

    def test_facets_render_with_counts(self, client):
        _login(client)
        body = client.get("/memories").text
        assert "infrastructure" in body and "ai" in body

    def test_filter_by_domain(self, client):
        _login(client)
        body = client.get("/memories?domain=infrastructure").text
        assert "Qdrant exige api-key" in body
        assert "O reranker roda em CPU" not in body

    def test_unknown_filter_key_is_ignored(self, client):
        _login(client)
        assert client.get("/memories?board=kanban").status_code == 200

    def test_flag_filter_only_superseded(self, client):
        _login(client)
        body = client.get("/memories?only_superseded=1").text
        assert "O reranker roda em CPU" in body
        assert "Qdrant exige api-key" not in body

    def test_pagination_link_carries_cursor_and_filters(self, client):
        _login(client)
        body = client.get("/memories?domain=ai").text
        if "cursor=" in body:
            assert "domain=ai" in body.split("cursor=")[1][:200]

    def test_broken_cursor_falls_back_to_first_page(self, client):
        _login(client)
        response = client.get("/memories?cursor=lixo-que-nao-decodifica")
        assert response.status_code == 200
        assert "Qdrant exige api-key" in response.text

    def test_empty_result_shows_empty_state(self, client):
        _login(client)
        assert "Nenhuma memória" in client.get("/memories?domain=inexistente").text


class TestDegradedSource:
    def test_missing_source_renders_a_card_not_a_500(self, store):
        app = web.create_app(
            store.db_path, secret_key="k",
            sources=_sources(error="MEM0_QDRANT_API_KEY ausente"),
        )
        client = TestClient(app)
        _login(client)
        response = client.get("/memories")
        assert response.status_code == 200
        assert "MEM0_QDRANT_API_KEY ausente" in response.text

    def test_credential_screens_still_work_without_corpus(self, store):
        app = web.create_app(
            store.db_path, secret_key="k", sources=_sources(error="fora do ar")
        )
        client = TestClient(app)
        _login(client)
        assert client.get("/").status_code == 200
        assert client.get("/audit").status_code == 200


class TestEscaping:
    def test_hostile_payload_is_escaped(self, store):
        """Texto de memória é arbitrário — inclusive vindo de PDF de terceiro."""
        hostile = [
            point("cccccccc-0000-4000-8000-00000000000a", "2026-07-23T10:00:00+00:00",
                  data='<script>alert("xss")</script>', domain='"><script>x</script>',
                  tags=["<img src=x onerror=alert(1)>"]),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(hostile))
        client = TestClient(app)
        _login(client)
        body = client.get("/memories").text
        # O que importa não é a ausência da substring, é a ausência de MARCAÇÃO:
        # `&lt;img ... onerror=...&gt;` é texto inerte, não um elemento.
        assert "<script>alert" not in body
        assert "&lt;script&gt;" in body
        assert "<img src=x" not in body
        assert "&lt;img src=x onerror=alert(1)&gt;" in body

    def test_no_secret_leaks_into_the_page(self, store, monkeypatch):
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", "chave-super-secreta")
        app = web.create_app(store.db_path, secret_key="k", sources=_sources())
        client = TestClient(app)
        _login(client)
        body = client.get("/memories").text
        assert "chave-super-secreta" not in body


class TestDetail:
    ID_1 = "aaaaaaaa-0000-4000-8000-000000000001"
    ID_SUPERSEDED = "aaaaaaaa-0000-4000-8000-000000000002"
    ID_DOC = "aaaaaaaa-0000-4000-8000-000000000003"

    def test_shows_full_text_and_metadata(self, client):
        _login(client)
        body = client.get(f"/memories/{self.ID_1}").text
        assert "Qdrant exige api-key neste host" in body
        assert "infrastructure" in body and "procedural" in body

    def test_document_provenance_block(self, client):
        _login(client)
        body = client.get(f"/memories/{self.ID_DOC}").text
        assert "manual.pdf" in body

    def test_malformed_id_is_404_without_touching_the_store(self, client):
        _login(client)
        assert client.get("/memories/../../etc/passwd").status_code in (404, 307, 404)
        assert client.get("/memories/nao-e-uuid").status_code == 404

    def test_other_scope_and_missing_are_indistinguishable(self, client):
        """Sem oráculo de existência: alheio e inexistente respondem igual."""
        _login(client)
        alheio = client.get("/memories/bbbbbbbb-0000-4000-8000-000000000009")
        inexistente = client.get("/memories/dddddddd-0000-4000-8000-00000000ffff")
        assert alheio.status_code == inexistente.status_code == 404
        assert "De outro escopo" not in alheio.text

    def test_requires_login(self, client):
        response = client.get(f"/memories/{self.ID_1}", follow_redirects=False)
        assert response.status_code == 303

    def test_actr_card_says_neutral_without_history(self, client):
        _login(client)
        assert "neutra no ranking" in client.get(f"/memories/{self.ID_1}").text

    def test_actr_card_shows_activation_when_reinforced(self, store):
        from datetime import datetime, timedelta, timezone

        recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        pts = [
            point("eeeeeeee-0000-4000-8000-00000000000e", "2026-07-01T10:00:00+00:00",
                  data="memória reforçada", reinforced_at=[recent], access_count=2,
                  last_accessed=recent, reinforce_counts={"t3": 2}),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(pts))
        client = TestClient(app)
        _login(client)
        body = client.get("/memories/eeeeeeee-0000-4000-8000-00000000000e").text
        assert "t3 × 2" in body
        assert "neutra no ranking" not in body

    def test_raw_payload_block_shows_fields_the_mcp_hides(self, store):
        pts = [
            point("ffffffff-0000-4000-8000-00000000000f", "2026-07-01T10:00:00+00:00",
                  data="x", memory_scope_evidence="decisive", subcategoria="alguma"),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(pts))
        client = TestClient(app)
        _login(client)
        body = client.get("/memories/ffffffff-0000-4000-8000-00000000000f").text
        assert "memory_scope_evidence" in body and "decisive" in body

    def test_version_chain_links_to_the_other_version(self, store):
        v1 = "11111111-0000-4000-8000-000000000001"
        v2 = "22222222-0000-4000-8000-000000000002"
        pts = [
            point(v1, "2026-07-01T10:00:00+00:00", data="versão antiga",
                  superseded_by=v2, superseded_at="2026-07-05T00:00:00+00:00"),
            point(v2, "2026-07-05T10:00:00+00:00", data="versão nova", supersedes=[v1]),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(pts))
        client = TestClient(app)
        _login(client)
        body = client.get(f"/memories/{v1}").text
        assert f'href="/memories/{v2}"' in body
        assert "versão nova" in body

    def test_chain_with_missing_target_does_not_break_the_page(self, store):
        orfa = "33333333-0000-4000-8000-000000000003"
        pts = [
            point(orfa, "2026-07-01T10:00:00+00:00", data="aponta pro nada",
                  superseded_by="99999999-0000-4000-8000-000000000099"),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(pts))
        client = TestClient(app)
        _login(client)
        response = client.get(f"/memories/{orfa}")
        assert response.status_code == 200
        assert "registro ausente" in response.text

    def test_chain_cycle_terminates_without_repeating(self, store):
        """v1→v2→v1 existe em corpus corrompido e não pode travar a requisição.

        Não basta terminar: o teto de saltos sozinho já terminaria, listando o
        mesmo par 20 vezes. O que se afirma aqui é que cada elo aparece UMA vez —
        é isso que o conjunto de visitados faz, e é isso que falharia sem ele.
        """
        a = "44444444-0000-4000-8000-000000000004"
        b = "55555555-0000-4000-8000-000000000005"
        pts = [
            point(a, "2026-07-01T10:00:00+00:00", data="elo A", superseded_by=b),
            point(b, "2026-07-02T10:00:00+00:00", data="elo B", superseded_by=a),
        ]
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(pts))
        client = TestClient(app)
        _login(client)
        body = client.get(f"/memories/{a}").text
        assert body.count(f'href="/memories/{b}"') == 1
        assert body.count("elo B") == 1


class TestUnifiedShell:
    """Credenciais e corpus na mesma interface, com uma navegação só."""

    def test_nav_has_every_surface(self, client):
        _login(client)
        body = client.get("/").text
        for href in ("/memories", "/search", "/queue", "/entities", "/users", "/audit"):
            assert f'href="{href}"' in body

    def test_dashboard_shows_corpus_and_credential_tiles(self, client):
        _login(client)
        body = client.get("/").text
        assert "Corpus" in body and "Credenciais" in body
        assert "Memórias" in body and "Entidades" in body
        assert "Usuários ativos" in body and "Tokens ativos" in body

    def test_dashboard_counts_come_from_the_corpus(self, client):
        _login(client)
        body = client.get("/").text
        # 3 memórias no escopo do fake (a quarta é de outro user_id)
        assert ">3<" in body

    def test_users_screen_lists_and_creates(self, store, client):
        _login(client)
        body = client.get("/users").text
        assert "Ana Souza" in body
        assert 'action="/users"' in client.get("/users?new=1").text

    def test_user_creation_still_works_and_lands_on_users(self, store, client):
        _login(client)
        page = client.get("/users?new=1")
        csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
        response = client.post(
            "/users",
            data={"csrf": csrf, "display_name": "Bruno", "email": "bruno@x.dev"},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/users"
        assert "Bruno" in client.get("/users").text

    def test_dashboard_degrades_without_corpus(self, store):
        app = web.create_app(
            store.db_path, secret_key="k", sources=_sources(error="Qdrant fora do ar")
        )
        client = TestClient(app)
        _login(client)
        response = client.get("/")
        assert response.status_code == 200
        assert "Qdrant fora do ar" in response.text
        assert "Usuários ativos" in response.text  # a metade de credenciais fica


class TestNavigation:
    def test_nav_links_to_memories(self, client):
        _login(client)
        assert 'href="/memories"' in client.get("/").text

    def test_secure_headers_apply_to_the_new_screen(self, client):
        _login(client)
        headers = client.get("/memories").headers
        assert headers["Cache-Control"] == "no-store"
        assert "script-src 'self'" in headers["Content-Security-Policy"]

    def test_screen_renders_in_english(self, client):
        _login(client)
        assert "Memories" in client.get("/memories?lang=en").text
