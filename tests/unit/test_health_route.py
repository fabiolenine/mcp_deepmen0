"""Testes da ROTA /health.

POR QUE EXISTE: ao acrescentar `_provenance()` eu o inseri entre o decorador
`@mcp.custom_route("/health")` e a função `health`, então o decorador passou a
registrar `_provenance` e `health` ficou SEM ROTA. A suíte inteira — 964 testes —
passou, porque nenhum teste tocava a rota. Um endpoint de saúde sem teste de
rota é um endpoint que ninguém sabe se existe.

Estes testes cobrem o registro, a forma da resposta e o contrato de que
`_provenance` nunca derruba a sonda.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.graph = None
    mem.enable_graph = False
    mem.search.return_value = {"results": []}
    mem.vector_store.collection_name = "test_collection"
    mem.config.vector_store.provider = "qdrant"
    mem.config.language = "pt"
    mem.reranker = None
    return mem


@pytest.fixture
def server_with_mock(mock_memory):
    """Espelha a fixture de test_server.py — o /health precisa de um app real."""
    orig_mem = server_mod.memory
    orig_graph = server_mod._enable_graph_default
    server_mod.memory = mock_memory
    server_mod._enable_graph_default = False
    srv = server_mod._create_server()
    yield srv, mock_memory
    server_mod.memory = orig_mem
    server_mod._enable_graph_default = orig_graph


def _rotas(srv):
    """Caminhos registrados no app ASGI do FastMCP."""
    app = srv.streamable_http_app()
    return {getattr(r, "path", None) for r in app.routes}


def test_health_route_is_registered(server_with_mock):
    """O defeito literal: o decorador tinha ido para outra função."""
    srv, _mem = server_with_mock
    assert "/health" in _rotas(srv)


def test_health_handler_is_the_health_function(server_with_mock):
    """Registrar QUALQUER coisa em /health não basta — o handler tem que aceitar
    a assinatura Starlette (request) e devolver JSONResponse, não um dict."""
    srv, _mem = server_with_mock
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)
    assert endpoint is not None
    nome = getattr(endpoint, "__name__", "")
    assert nome == "health", f"/health está servindo {nome!r}, não health()"


@pytest.mark.anyio
async def test_health_returns_json_with_provenance(server_with_mock):
    from starlette.requests import Request

    srv, _mem = server_with_mock
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)

    req = Request({"type": "http", "method": "GET", "path": "/health",
                   "headers": [], "query_string": b""})
    resp = await endpoint(req)
    body = json.loads(bytes(resp.body).decode())

    assert resp.status_code in (200, 503)
    for chave in ("status", "auth_mode", "vault_db", "provenance"):
        assert chave in body, f"/health sem {chave!r}"
    prov = body["provenance"]
    # Os campos que existem para responder "o restart pegou a mudança?"
    for chave in ("pid", "started_at", "mem0_file", "fork_sha", "collection",
                  "entity_collection", "spacy_model"):
        assert chave in prov, f"provenance sem {chave!r}"


@pytest.mark.anyio
async def test_provenance_is_always_json_serializable(server_with_mock, monkeypatch):
    """Valor não-primitivo (ex.: um Mock que vazou de uma config) derrubava o
    /health com 500 no encoder. A sonda tem que ser a coisa mais robusta do
    serviço, não a mais frágil."""
    from starlette.requests import Request

    srv, _mem = server_with_mock
    vazando = MagicMock()
    vazando.vector_store.collection_name = MagicMock()      # não serializável
    monkeypatch.setattr(server_mod, "memory", vazando)
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)
    resp = await endpoint(Request({"type": "http", "method": "GET", "path": "/health",
                                   "headers": [], "query_string": b""}))
    json.loads(bytes(resp.body).decode())    # não pode levantar
    assert resp.status_code in (200, 503)


@pytest.mark.anyio
async def test_provenance_never_breaks_the_probe(server_with_mock, monkeypatch):
    """Sonda de saúde não pode virar 500 por causa de metadado. Cada campo falha
    para o próprio valor de erro."""
    from starlette.requests import Request

    srv, _mem = server_with_mock
    # Memory que explode ao ser inspecionado
    quebrado = MagicMock()
    type(quebrado).vector_store = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(server_mod, "memory", quebrado)
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)

    resp = await endpoint(Request({"type": "http", "method": "GET", "path": "/health",
                                   "headers": [], "query_string": b""}))
    body = json.loads(bytes(resp.body).decode())
    assert resp.status_code in (200, 503)
    assert "erro" in str(body["provenance"].get("collection", "")).lower()


@pytest.mark.anyio
async def test_provenance_does_not_force_memory_init(server_with_mock, monkeypatch):
    """A sonda NÃO pode disparar `_ensure_memory()`: inicializar Memory carrega
    Ollama e o reranker, dezenas de segundos. Health check que inicializa o
    sistema deixa de ser health check."""
    from starlette.requests import Request

    srv, _mem = server_with_mock
    chamou = []
    monkeypatch.setattr(server_mod, "_ensure_memory",
                        lambda: chamou.append(1) or MagicMock())
    monkeypatch.setattr(server_mod, "memory", None)
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)
    resp = await endpoint(Request({"type": "http", "method": "GET", "path": "/health",
                                   "headers": [], "query_string": b""}))
    prov = json.loads(bytes(resp.body).decode())["provenance"]
    assert chamou == [], "a sonda disparou a inicialização do Memory"
    assert prov.get("collection") == "não inicializado"


@pytest.mark.anyio
async def test_entity_collection_is_derived_not_guessed(server_with_mock):
    """A collection de entidades é DERIVADA de `_entity_collection_name`, não
    configurável. Presumir `<coll>_entities` foi o que quase me fez construir a
    v2 na collection errada — então o valor tem que vir da função do fork."""
    from starlette.requests import Request

    srv, _mem = server_with_mock
    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)
    resp = await endpoint(Request({"type": "http", "method": "GET", "path": "/health",
                                   "headers": [], "query_string": b""}))
    prov = json.loads(bytes(resp.body).decode())["provenance"]
    coll, ent = prov.get("collection"), prov.get("entity_collection")
    if isinstance(coll, str) and isinstance(ent, str) and not coll.startswith("erro"):
        assert ent.startswith(coll), f"{ent!r} não deriva de {coll!r}"


async def _health_body(srv):
    """Chama a rota /health e devolve (status_code, body)."""
    from starlette.requests import Request

    app = srv.streamable_http_app()
    rota = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    endpoint = getattr(rota, "endpoint", None) or getattr(rota, "app", None)
    req = Request({"type": "http", "method": "GET", "path": "/health",
                   "headers": [], "query_string": b""})
    resp = await endpoint(req)
    return resp.status_code, json.loads(bytes(resp.body).decode())


@pytest.mark.anyio
async def test_health_reports_the_configured_language_pipeline(server_with_mock, monkeypatch):
    """A sonda tem que responder QUAL pipeline o idioma configurado exige.

    A versão anterior chamava `get_nlp_full()` sem argumento, o que inspeciona o
    modelo do idioma DEFAULT: dizia `en_core_web_sm` num deployment português,
    que não é a pergunta.
    """
    monkeypatch.delenv("MEM0_LANGUAGE", raising=False)
    monkeypatch.setenv("MEM0_BM25_LANGUAGE", "portuguese")
    srv, _mem = server_with_mock

    _status, body = await _health_body(srv)

    ep = body["entity_pipeline"]
    assert ep["language"] == "pt"
    assert ep["model"] == "pt_core_news_sm"
    assert body["provenance"]["spacy_model"] == "pt_core_news_sm"


@pytest.mark.anyio
async def test_readiness_fails_when_the_pipeline_is_degraded(server_with_mock, monkeypatch):
    """O critério: idioma configurado SEM o modelo dele reprova a readiness.

    Sem isto o serviço sobe, a extração de entidade roda com o POS errado
    (verbo português volta PROPN) e grava lixo em silêncio.
    """
    import mem0.utils.spacy_models as sm

    monkeypatch.setattr(sm, "model_available", lambda language=None: False)
    monkeypatch.delenv("MEM0_LANGUAGE", raising=False)
    monkeypatch.setenv("MEM0_BM25_LANGUAGE", "portuguese")
    srv, _mem = server_with_mock

    status, body = await _health_body(srv)

    assert status == 503, "modelo ausente tem que REPROVAR a readiness"
    assert body["status"] == "degraded"
    assert body["entity_pipeline"]["degraded"] is True


@pytest.mark.anyio
async def test_readiness_fails_when_the_model_is_installed_but_will_not_load(
    server_with_mock, monkeypatch
):
    """Instalado != utilizável.

    `load_failed` era o único dos três estados de indisponibilidade que não
    entrava em `degraded`, então um modelo corrompido — instalado, presente,
    inutilizável — passava a readiness com 200.
    """
    import mem0.utils.spacy_models as sm

    monkeypatch.setattr(sm, "model_available", lambda language=None: True)
    monkeypatch.setattr(sm, "_load_failed_full", {"pt"})
    monkeypatch.delenv("MEM0_LANGUAGE", raising=False)
    monkeypatch.setenv("MEM0_BM25_LANGUAGE", "portuguese")
    srv, _mem = server_with_mock

    status, body = await _health_body(srv)

    assert status == 503
    assert body["entity_pipeline"]["installed"] is True
    assert body["entity_pipeline"]["load_failed"] is True


@pytest.mark.anyio
async def test_degraded_flag_survives_sanitization_as_a_bool(server_with_mock, monkeypatch):
    """A sanitização achatava todo não-primitivo em string.

    `entity_pipeline` é dict e a readiness LÊ `degraded` dele — achatado viraria
    a string "{'degraded': False...}", e `bool` de string não-vazia é True: a
    sonda passaria a reprovar SEMPRE, ou (pior) a decisão viraria texto.
    """
    monkeypatch.delenv("MEM0_LANGUAGE", raising=False)
    monkeypatch.setenv("MEM0_BM25_LANGUAGE", "portuguese")
    srv, _mem = server_with_mock

    _status, body = await _health_body(srv)

    assert isinstance(body["entity_pipeline"], dict)
    assert isinstance(body["entity_pipeline"]["degraded"], bool)


@pytest.mark.anyio
async def test_pipeline_check_that_explodes_still_degrades(server_with_mock, monkeypatch):
    """Se a checagem estourar, o silêncio NÃO pode virar 'ok'."""
    import mem0.utils.spacy_models as sm

    def _explode(language=None):
        raise RuntimeError("modelo corrompido")

    monkeypatch.setattr(sm, "entity_pipeline_status", _explode)
    srv, _mem = server_with_mock

    status, body = await _health_body(srv)

    assert status == 503
    assert body["entity_pipeline"]["degraded"] is True


@pytest.mark.anyio
async def test_absent_pipeline_field_fails_closed(server_with_mock, monkeypatch):
    """CHAVE AUSENTE é o caso que o default do `.get` existe para cobrir.

    A primeira versão deste teste patcheava `entity_pipeline_status` para
    estourar — mas o `except` grava `{"degraded": True}`, então a chave ficava
    PRESENTE e o default nunca era exercido: trocar `.get(..., True)` por
    `.get(..., False)` não reprovava nada. Guarda que não pode disparar.
    Aqui a chave é removida de verdade.
    """
    real = server_mod._relabel_disk_fields

    def _sem_pipeline(prov):
        out = dict(real(prov))
        out.pop("entity_pipeline", None)
        return out

    # `_provenance` é closure de `_register_health` e não dá para trocar pelo
    # módulo; `_relabel_disk_fields` é global e está no caminho, logo antes da
    # leitura de `degraded`.
    monkeypatch.setattr(server_mod, "_relabel_disk_fields", _sem_pipeline)
    srv, _mem = server_with_mock

    status, body = await _health_body(srv)

    assert status == 503, "campo ausente tem que contar como degradado"
    assert body["status"] == "degraded"


def test_configured_language_agrees_with_build_config(monkeypatch):
    """A sonda e o runtime têm que derivar o MESMO idioma.

    Duas derivações divergiriam, e uma sonda que discorda do runtime é pior que
    sonda nenhuma: ela afirma.
    """
    from mem0_mcp_selfhosted.config import build_config, configured_language

    for env_val, esperado in (("portuguese", "pt"), ("english", "en")):
        monkeypatch.delenv("MEM0_LANGUAGE", raising=False)
        monkeypatch.setenv("MEM0_BM25_LANGUAGE", env_val)
        cfg, _p, _s = build_config()
        assert configured_language() == esperado
        assert cfg.get("language") == esperado
