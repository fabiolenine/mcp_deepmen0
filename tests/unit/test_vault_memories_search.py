"""Busca via MCP: montagem de parâmetros, parsing da resposta e a tela."""

from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest
from starlette.testclient import TestClient

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault import web
from mem0_mcp_selfhosted.vault.memories import model
from mem0_mcp_selfhosted.vault.memories.mcp_client import (
    McpError,
    McpSearchClient,
    describe_failure,
    parse_tool_result,
)
from mem0_mcp_selfhosted.vault.memories.sources import Sources

ADMIN_EMAIL = "ana.souza@acme.com.br"
ADMIN_PASSWORD = "uma senha longa o suficiente"
SCOPE = {"user_id": "ana"}


# --------------------------------------------------------- dublês do SDK MCP


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Result:
    content: list = field(default_factory=list)
    structuredContent: Any = None  # noqa: N815 — o nome vem do SDK
    isError: bool = False  # noqa: N815


class FakeMcp:
    """Registra o que foi pedido e devolve um envelope preparado."""

    def __init__(self, envelope=None, raises=None):
        self.envelope = envelope if envelope is not None else {"results": []}
        self.raises = raises
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return True

    async def search(self, params):
        self.calls.append(params)
        if self.raises:
            raise self.raises
        return self.envelope


def _sources(mcp=None, mcp_error=""):
    return Sources(
        collection="col", entity_collection="col_e", scope=SCOPE,
        mcp=mcp, mcp_error=mcp_error,
    )


# NÃO existe aqui o fixture que troca `web.anyio.sleep` por um no-op (usado nos
# outros arquivos do cofre para pular o atraso anti-timing do login falho):
# `web.anyio` É o módulo anyio, então aquele monkeypatch vale para o processo
# inteiro — inclusive para o `anyio.sleep` do dublê de MCP lento, que deixaria de
# esperar e faria o teste de timeout passar sem nunca haver timeout. Todos os
# logins deste arquivo são válidos, então o atraso nunca é cobrado.


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


def _client(store, mcp=None, mcp_error=""):
    app = web.create_app(store.db_path, secret_key="k", sources=_sources(mcp, mcp_error))
    client = TestClient(app)
    page = client.get("/login")
    csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
    client.post(
        "/login",
        data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "csrf": csrf},
        follow_redirects=False,
    )
    return client


# ------------------------------------------------------------------ params


class TestSearchParams:
    def test_scope_is_always_applied(self):
        args, _ = model.search_params({"query": "x"}, SCOPE)
        assert args["user_id"] == "ana"

    def test_numbers_are_cast(self):
        args, warns = model.search_params(
            {"query": "x", "limit": "5", "threshold": "0.3", "min_importance": "0.8"}, SCOPE
        )
        assert args["limit"] == 5 and args["threshold"] == 0.3
        assert args["min_importance"] == 0.8 and warns == []

    def test_invalid_number_is_warned_not_fatal(self):
        args, warns = model.search_params({"query": "x", "limit": "abc"}, SCOPE)
        assert "limit" not in args and warns == ["limit"]

    def test_attributed_to_becomes_a_structured_filter(self):
        """Não é parâmetro da tool; iria para o **kwargs e sumiria em silêncio."""
        args, _ = model.search_params({"query": "x", "attributed_to": "document"}, SCOPE)
        assert "attributed_to" not in args
        assert args["filters"] == {"attributed_to": "document"}

    def test_historical_requires_as_of(self):
        args, warns = model.search_params({"query": "x", "historical": "1"}, SCOPE)
        assert "historical" not in args and warns == ["historical"]

    def test_historical_with_as_of_is_accepted(self):
        args, warns = model.search_params(
            {"query": "x", "historical": "1", "as_of": "2026-07-01"}, SCOPE
        )
        assert args["historical"] is True and args["as_of"] == "2026-07-01"
        assert warns == []

    def test_temporal_fields_pass_through(self):
        args, _ = model.search_params(
            {"query": "x", "event_from": "2026-05", "event_to": "2026-06"}, SCOPE
        )
        assert args["event_from"] == "2026-05" and args["event_to"] == "2026-06"

    def test_blank_fields_are_omitted(self):
        args, _ = model.search_params({"query": "x", "domain": "  ", "as_of": ""}, SCOPE)
        assert "domain" not in args and "as_of" not in args


# ------------------------------------------------------------------ parsing


class TestParseToolResult:
    def test_reads_json_from_a_text_block(self):
        env = parse_tool_result(_Result(content=[_Block('{"results": [{"id": "a"}]}')]))
        assert env["results"] == [{"id": "a"}]

    def test_unwraps_the_result_envelope(self):
        """O servidor devolve {"result": "<json string>"} — dois níveis."""
        env = parse_tool_result(_Result(content=[_Block('{"result": "{\\"results\\": []}"}')]))
        assert env == {"results": []}

    def test_structured_content_wins_when_present(self):
        env = parse_tool_result(
            _Result(content=[_Block("ignorado")], structuredContent={"results": [{"id": "z"}]})
        )
        assert env["results"] == [{"id": "z"}]

    def test_is_error_raises_with_the_server_message(self):
        """isError não é exceção do SDK: sem checar, erro viraria 'zero resultados'."""
        with pytest.raises(McpError, match="401"):
            parse_tool_result(_Result(content=[_Block("401 Unauthorized")], isError=True))

    def test_empty_response_raises(self):
        with pytest.raises(McpError, match="vazia"):
            parse_tool_result(_Result(content=[]))

    def test_non_json_text_raises_with_context(self):
        with pytest.raises(McpError, match="não é JSON"):
            parse_tool_result(_Result(content=[_Block("<html>502 Bad Gateway</html>")]))

    def test_bare_list_is_wrapped(self):
        env = parse_tool_result(_Result(content=[_Block('[{"id": "a"}]')]))
        assert env["results"] == [{"id": "a"}]


class TestClientContract:
    def test_reinforce_is_always_false(self):
        """Navegar no console não pode contar como uso da memória.

        Se contasse, a ativação ACT-R exibida na tela de detalhe mediria o uso
        da própria UI — e o ranking de produção seria enviesado por quem abre a
        página.
        """
        sent = {}

        class _Spy(McpSearchClient):
            async def _call(self, payload):
                sent.update(payload)
                return {"results": []}

        client = _Spy("http://x/mcp", "tok")
        anyio.run(lambda: client.search({"query": "x", "reinforce": True}))
        assert sent["reinforce"] is False

    def test_none_and_blank_params_are_stripped(self):
        sent = {}

        class _Spy(McpSearchClient):
            async def _call(self, payload):
                sent.update(payload)
                return {"results": []}

        anyio.run(
            lambda: _Spy("http://x/mcp", "tok").search(
                {"query": "x", "domain": None, "as_of": ""}
            )
        )
        assert "domain" not in sent and "as_of" not in sent

    def test_timeout_becomes_a_readable_error(self):
        class _Slow(McpSearchClient):
            async def _call(self, payload):
                await anyio.sleep(5)
                return {}

        client = _Slow("http://x/mcp", "tok", timeout_s=0.05)
        with pytest.raises(McpError, match="sem resposta"):
            anyio.run(lambda: client.search({"query": "x"}))

    def test_unconfigured_client_is_reported(self):
        assert not McpSearchClient("http://x/mcp", "").configured
        assert McpSearchClient("http://x/mcp", "tok").configured


class TestDescribeFailure:
    """O SDK roda o transporte num task group: a causa real fica aninhada.

    Medido contra o servidor real: um token recusado chegava ao operador como
    "ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)" — que
    não diz o que houve nem o que fazer.
    """

    def _http_error(self, status: int):
        class _Resp:
            status_code = status

        class _HttpStatusError(Exception):
            response = _Resp()

        return _HttpStatusError(f"Client error '{status}'")

    def test_unwraps_exception_group_to_the_real_cause(self):
        group = ExceptionGroup("unhandled errors in a TaskGroup", [self._http_error(401)])
        message = describe_failure(group)
        assert "401" in message
        assert "TaskGroup" not in message
        assert "VAULT_MCP_TOKEN" in message

    def test_nested_groups_are_unwrapped(self):
        inner = ExceptionGroup("inner", [self._http_error(403)])
        outer = ExceptionGroup("outer", [inner])
        assert "403" in describe_failure(outer)

    def test_duplicate_causes_are_not_repeated(self):
        group = ExceptionGroup("g", [ValueError("mesmo erro"), ValueError("mesmo erro")])
        assert describe_failure(group).count("mesmo erro") == 1

    def test_connection_failure_says_so(self):
        assert "conectar" in describe_failure(ConnectionError("recusado"))

    def test_plain_exception_passes_through(self):
        assert describe_failure(ValueError("boom")) == "ValueError: boom"


# --------------------------------------------------------------------- tela


class TestSearchScreen:
    def test_requires_login(self, store):
        app = web.create_app(store.db_path, secret_key="k", sources=_sources(FakeMcp()))
        assert TestClient(app).get("/search", follow_redirects=False).status_code == 303
        assert (
            TestClient(app).get("/search/results?query=x", follow_redirects=False).status_code
            == 303
        )

    def test_form_renders_the_temporal_fields(self, store):
        body = _client(store, FakeMcp()).get("/search").text
        assert 'name="as_of"' in body and 'name="event_from"' in body
        assert 'name="historical"' in body

    def test_empty_query_does_not_call_the_mcp(self, store):
        mcp = FakeMcp()
        client = _client(store, mcp)
        client.get("/search/results?query=")
        assert mcp.calls == []

    def test_results_show_rerank_score_and_signals(self, store):
        mcp = FakeMcp({
            "results": [{
                "id": "aaaaaaaa-0000-4000-8000-000000000001",
                "memory": "O reranker roda em CPU",
                "score": 0.77, "rerank_score": 0.993,
                "superseded_penalty": 0.2, "created_at": "2026-07-01T00:00:00+00:00",
                "metadata": {"domain": "ai", "event_date": "2026-05-01"},
            }],
            "pending_ingest": 2,
            "event_anchor": {"from": "2026-05-01", "to": "2026-05-31"},
        })
        body = _client(store, mcp).get("/search/results?query=reranker").text
        assert "0.993" in body
        assert "O reranker roda em CPU" in body
        assert "Penalidade" in body
        assert "Âncora de evento" in body
        assert "Ingestão pendente" in body

    def test_result_links_to_the_detail(self, store):
        mcp = FakeMcp({"results": [{"id": "abc", "memory": "x"}]})
        assert 'href="/memories/abc"' in _client(store, mcp).get("/search/results?query=x").text

    def test_mcp_failure_renders_a_card_with_200(self, store):
        """O alvo é um fragmento: um 500 apagaria a tela em vez de explicar."""
        mcp = FakeMcp(raises=McpError("401 Unauthorized"))
        response = _client(store, mcp).get("/search/results?query=x")
        assert response.status_code == 200
        assert "401 Unauthorized" in response.text
        assert "corpus não foi alterado" in response.text

    def test_missing_token_is_explained_on_the_form(self, store):
        client = _client(store, None, mcp_error="VAULT_MCP_TOKEN ausente — emita um token")
        assert "VAULT_MCP_TOKEN ausente" in client.get("/search").text

    def test_invalid_number_is_reported_but_search_still_runs(self, store):
        mcp = FakeMcp()
        body = _client(store, mcp).get("/search/results?query=x&limit=abc").text
        assert len(mcp.calls) == 1
        assert "ignorados" in body

    def test_indicator_uses_htmx_attributes_not_inline_script(self, store):
        """A CSP proíbe script inline; o indicador tem de ser atributo + CSS."""
        body = _client(store, FakeMcp()).get("/search").text
        assert "hx-indicator" in body and "hx-disabled-elt" in body
        assert "<script>" not in body

    def test_no_token_leaks_into_the_page(self, store):
        mcp = FakeMcp()
        mcp_client = McpSearchClient("http://127.0.0.1:18081/mcp", "dm0_segredo_do_token")
        app = web.create_app(
            store.db_path, secret_key="k", sources=_sources(mcp_client)
        )
        client = TestClient(app)
        page = client.get("/login")
        csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
        client.post(
            "/login",
            data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "csrf": csrf},
            follow_redirects=False,
        )
        assert "dm0_segredo_do_token" not in client.get("/search").text
