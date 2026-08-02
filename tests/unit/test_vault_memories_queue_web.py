"""A tela da fila de ingestão."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault import web
from mem0_mcp_selfhosted.vault.memories.sources import Sources

from tests.unit.test_vault_memories_local_ro import QUEUE_DDL

ADMIN_EMAIL = "ana.souza@acme.com.br"
ADMIN_PASSWORD = "uma senha longa o suficiente"


def _ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


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


@pytest.fixture
def queue_db(tmp_path):
    path = tmp_path / "q.db"
    conn = sqlite3.connect(path)
    conn.execute(QUEUE_DDL)
    conn.executemany(
        "INSERT INTO ingest_queue VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("tsk_doc", "k1", "ana", None, None, "[]", None, _ago(600), "processing", 0,
             None, _ago(590), None, None,
             json.dumps({"chunks_done": 4, "chunks_total": 11,
                         "last_progress_at": _ago(15), "source_doc": "LATAM.pdf"}),
             "document"),
            ("tsk_wait", "k2", "ana", None, None, "[]", None, _ago(90), "pending", 0,
             None, None, None, None, None, "conversation"),
            ("tsk_dead", "k3", "ana", None, None, "[]", None, _ago(9000), "dead", 4,
             None, None, _ago(8000), "PoisonPayload: documento sem texto extraível",
             None, "document"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _client(store, queue_db=None):
    sources = Sources(
        collection="col", entity_collection="col_e", scope={"user_id": "ana"},
        queue_db=queue_db,
    )
    app = web.create_app(store.db_path, secret_key="k", sources=sources)
    client = TestClient(app)
    page = client.get("/login")
    csrf = page.text.split('name="csrf" value="')[1].split('"')[0]
    client.post(
        "/login",
        data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "csrf": csrf},
        follow_redirects=False,
    )
    return client


class TestQueueScreen:
    def test_requires_login(self, store, queue_db):
        sources = Sources(collection="c", entity_collection="e", scope={}, queue_db=queue_db)
        app = web.create_app(store.db_path, secret_key="k", sources=sources)
        assert TestClient(app).get("/queue", follow_redirects=False).status_code == 303
        assert (
            TestClient(app).get("/queue/summary", follow_redirects=False).status_code == 303
        )

    def test_shows_depth_and_status_tiles(self, store, queue_db):
        body = _client(store, queue_db).get("/queue").text
        assert "Profundidade" in body
        assert "Descartados" in body

    def test_active_job_shows_chunk_progress(self, store, queue_db):
        body = _client(store, queue_db).get("/queue").text
        assert "4/11" in body
        assert "LATAM.pdf" in body
        assert "width: 36%" in body  # 4/11 arredondado

    def test_heartbeat_is_shown_not_job_age(self, store, queue_db):
        """Job submetido há 10 min com heartbeat de 15 s NÃO está travado."""
        body = _client(store, queue_db).get("/queue").text
        assert "sinal de vida há 15s" in body
        assert "stalled" not in body

    def test_stalled_job_is_marked(self, store, tmp_path):
        path = tmp_path / "stalled.db"
        conn = sqlite3.connect(path)
        conn.execute(QUEUE_DDL)
        conn.execute(
            "INSERT INTO ingest_queue VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("tsk_slow", "k", "ana", None, None, "[]", None, _ago(4000), "processing", 0,
             None, _ago(3900), None, None,
             json.dumps({"chunks_done": 1, "chunks_total": 9,
                         "last_progress_at": _ago(3000)}), "document"),
        )
        conn.commit()
        conn.close()
        assert "stalled" in _client(store, path).get("/queue").text

    def test_dead_letters_are_listed_with_the_error(self, store, queue_db):
        body = _client(store, queue_db).get("/queue").text
        assert "PoisonPayload" in body
        assert "documento sem texto extraível" in body

    def test_no_action_controls_are_offered(self, store, queue_db):
        """A v1 é somente leitura: nenhum formulário de retry/cancel."""
        body = _client(store, queue_db).get("/queue").text
        assert "<form" not in body.split('class="page"')[1]

    def test_polling_partial_renders_standalone(self, store, queue_db):
        response = _client(store, queue_db).get("/queue/summary")
        assert response.status_code == 200
        assert "Profundidade" in response.text
        assert "<html" not in response.text  # é fragmento, não página

    def test_polling_is_declared_with_htmx_attributes(self, store, queue_db):
        body = _client(store, queue_db).get("/queue").text
        assert 'hx-get="/queue/summary"' in body and 'hx-trigger="every 10s"' in body
        assert "<script>" not in body

    def test_missing_queue_db_renders_a_card(self, store):
        response = _client(store, None).get("/queue")
        assert response.status_code == 200
        assert "Fila indisponível" in response.text

    def test_partial_also_degrades_with_200(self, store):
        response = _client(store, None).get("/queue/summary")
        assert response.status_code == 200
        assert "Fila indisponível" in response.text

    def test_nav_links_to_queue(self, store, queue_db):
        assert 'href="/queue"' in _client(store, queue_db).get("/").text
