"""The vault UI: session, CSRF, the one-time reveal, and the admin flows."""

import re

import pytest
from starlette.testclient import TestClient

from mem0_mcp_selfhosted.vault import security as sec
from mem0_mcp_selfhosted.vault import store as vs
from mem0_mcp_selfhosted.vault import web

ADMIN_EMAIL = "ana.souza@acme.com.br"
ADMIN_PASSWORD = "uma senha longa o suficiente"
TOKEN_RE = re.compile(r"dm0_[A-Za-z0-9_-]{43}")


@pytest.fixture(autouse=True)
def _instant_failed_login(monkeypatch):
    """The 2 s anti-timing delay is asserted once; the rest of the suite skips it."""

    async def _noop(_seconds):
        return None

    monkeypatch.setattr(web.anyio, "sleep", _noop)


@pytest.fixture
def store(tmp_path):
    s = vs.VaultStore(tmp_path / "vault.db")
    s.create_user(
        email=ADMIN_EMAIL, display_name="Ana Souza", is_admin=True,
        password_hash=sec.hash_password(ADMIN_PASSWORD),
    )
    return s


@pytest.fixture(autouse=True)
def _clean_health_cache():
    """O modo de auth é cacheado por 20s; teste não pode herdar leitura de outro."""
    web._health_cache.update(at=0.0, value=None)
    yield
    web._health_cache.update(at=0.0, value=None)


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setenv("MEM0_REQUIRE_AUTH", "shadow")
    # injeta na SONDA, não no _mcp_health: o cache e a escolha da fonte da
    # verdade fazem parte do que está sob teste
    monkeypatch.setattr(web, "_probe_mcp_health", lambda: {"ok": True, "code": 200, "detail": {}})
    app = web.create_app(store.db_path, secret_key="test-secret-key-not-production")
    return TestClient(app, follow_redirects=False)


def csrf_of(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match, "no CSRF token in the rendered form"
    return match.group(1)


@pytest.fixture
def admin(client):
    """A logged-in admin client, plus its CSRF token."""
    page = client.get("/login")
    resp = client.post("/login", data={
        "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
    })
    assert resp.status_code == 303, resp.text
    dash = client.get("/")
    return {"client": client, "csrf": csrf_of(dash.text)}


# ---------------------------------------------------------------- session


class TestLogin:
    def test_unauthenticated_is_sent_to_login(self, client):
        for path in ("/", "/audit", "/users/1"):
            resp = client.get(path)
            assert resp.status_code == 303
            assert resp.headers["location"] == "/login"

    def test_login_page_renders_both_languages(self, client):
        assert "Entrar no cofre" in client.get("/login?lang=pt").text
        assert "Sign in to the vault" in client.get("/login?lang=en").text

    def test_successful_login_starts_a_session(self, client):
        page = client.get("/login")
        resp = client.post("/login", data={
            "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
        })
        assert resp.status_code == 303 and resp.headers["location"] == "/"
        assert "Ana Souza" in client.get("/").text

    def test_wrong_password_is_refused_and_audited(self, client, store):
        page = client.get("/login")
        resp = client.post("/login", data={
            "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": "wrong",
        })
        assert resp.status_code == 401
        assert client.get("/").status_code == 303  # still anonymous
        assert store.list_audit(limit=1)[0]["action"] == "login.failed"
        assert store.list_audit(limit=1)[0]["success"] == 0

    def test_unknown_email_is_refused_the_same_way(self, client, store):
        page = client.get("/login")
        resp = client.post("/login", data={
            "csrf": csrf_of(page.text), "email": "ghost@nowhere.io", "password": "x" * 12,
        })
        assert resp.status_code == 401
        assert "E-mail ou senha incorretos." in resp.text

    def test_failed_login_pays_the_constant_delay(self, client, monkeypatch):
        """Constant delay, and awaited — a blocking sleep here would freeze the
        whole UI worker and hand an attacker a one-request DoS."""
        slept = []

        async def _record(seconds):
            slept.append(seconds)

        monkeypatch.setattr(web.anyio, "sleep", _record)
        page = client.get("/login")
        client.post("/login", data={
            "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": "wrong",
        })
        assert slept == [web.FAILED_LOGIN_DELAY_S]

    def test_login_without_csrf_is_refused(self, client):
        client.get("/login")
        resp = client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        assert resp.status_code == 403

    def test_session_rotates_on_login(self, client):
        page = client.get("/login")
        before = csrf_of(page.text)
        client.post("/login", data={
            "csrf": before, "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
        })
        assert csrf_of(client.get("/").text) != before

    def test_non_admin_cannot_log_in(self, client, store):
        store.create_user(
            email="carlos@cliente.dev", password_hash=sec.hash_password(ADMIN_PASSWORD),
        )
        page = client.get("/login")
        resp = client.post("/login", data={
            "csrf": csrf_of(page.text), "email": "carlos@cliente.dev",
            "password": ADMIN_PASSWORD,
        })
        assert resp.status_code == 401

    def test_disabled_admin_loses_the_session(self, admin, store):
        store.set_user_disabled(store.get_user_by_email(ADMIN_EMAIL)["id"], True)
        assert admin["client"].get("/").status_code == 303

    def test_logout(self, admin):
        resp = admin["client"].post("/logout", data={"csrf": admin["csrf"]})
        assert resp.status_code == 303
        assert admin["client"].get("/").headers["location"] == "/login"


# ---------------------------------------------------------------- headers


class TestHardening:
    def test_authenticated_pages_are_never_cached(self, admin):
        for path in ("/", "/audit"):
            resp = admin["client"].get(path)
            assert resp.headers["cache-control"] == "no-store"
            assert resp.headers["x-frame-options"] == "DENY"
            assert resp.headers["x-content-type-options"] == "nosniff"

    def test_login_sets_httponly_session_cookie(self, client):
        page = client.get("/login")
        resp = client.post("/login", data={
            "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
        })
        cookie = resp.headers["set-cookie"].lower()
        assert "vault_session=" in cookie
        assert "httponly" in cookie
        assert "samesite=lax" in cookie

    def test_static_assets_are_served_locally(self, client):
        for asset in ("/static/vault.css", "/static/htmx.min.js"):
            assert client.get(asset).status_code == 200

    def test_no_external_script_is_referenced(self, admin):
        html = admin["client"].get("/").text
        assert "unpkg.com" not in html and "cdn.jsdelivr" not in html


# ---------------------------------------------------------------- users


class TestUsers:
    def test_create_user(self, admin, store):
        resp = admin["client"].post("/users", data={
            "csrf": admin["csrf"], "display_name": "Carlos Lima",
            "email": "Carlos.Lima@Cliente.dev",
        })
        assert resp.status_code == 303
        created = store.get_user_by_email("carlos.lima@cliente.dev")
        assert created["display_name"] == "Carlos Lima"
        assert created["is_admin"] == 0
        assert store.list_audit(limit=1)[0]["action"] == "user.create"

    def test_invalid_email_is_rejected_with_a_message(self, admin, store):
        resp = admin["client"].post("/users", data={
            "csrf": admin["csrf"], "display_name": "X", "email": "not-an-email",
        })
        assert resp.status_code == 303
        assert "error=" in resp.headers["location"]
        assert len(store.list_users()) == 1

    def test_duplicate_email_is_rejected(self, admin):
        resp = admin["client"].post("/users", data={
            "csrf": admin["csrf"], "display_name": "Dup", "email": ADMIN_EMAIL,
        })
        assert "error=" in resp.headers["location"]

    def test_create_user_requires_csrf(self, admin, store):
        resp = admin["client"].post("/users", data={
            "csrf": "forged", "display_name": "X", "email": "x@y.com",
        })
        assert resp.status_code == 303
        assert store.get_user_by_email("x@y.com") is None

    def test_disable_and_reenable(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev", display_name="Carlos")
        client = admin["client"]

        client.post(f"/users/{uid}/disable", data={"csrf": admin["csrf"]})
        assert store.get_user(uid)["disabled_at"] is not None
        assert "Desativado" in client.get(f"/users/{uid}").text

        client.post(f"/users/{uid}/enable", data={"csrf": admin["csrf"]})
        assert store.get_user(uid)["disabled_at"] is None

    def test_user_detail_lists_tokens(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev", display_name="Carlos")
        store.create_token(user_id=uid, token=sec.generate_token(), label="open-webui")
        html = admin["client"].get(f"/users/{uid}").text
        assert "open-webui" in html
        assert "Ativo" in html

    def test_unknown_user_goes_back_to_the_dashboard(self, admin):
        assert admin["client"].get("/users/9999").headers["location"] == "/"


# ---------------------------------------------------------------- tokens


class TestTokenLifecycle:
    @pytest.fixture
    def target(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev", display_name="Carlos")
        return uid

    def test_plaintext_is_shown_exactly_once(self, admin, store, target):
        client = admin["client"]
        resp = client.post(
            f"/users/{target}/tokens",
            data={"csrf": admin["csrf"], "label": "claude-code-local"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        found = TOKEN_RE.findall(resp.text)
        assert len(found) == 1, "the reveal must contain exactly one token"
        token = found[0]
        assert resp.headers["cache-control"] == "no-store"

        # every later page shows the prefix and never the token again
        detail = client.get(f"/users/{target}").text
        assert token not in detail
        assert token[:12] in detail
        assert token not in client.get("/audit").text
        assert store.verify_token(token)["status"] == vs.OK

    def test_token_is_never_placed_in_a_url(self, admin, target):
        resp = admin["client"].post(
            f"/users/{target}/tokens", data={"csrf": admin["csrf"], "label": "x"},
            headers={"HX-Request": "true"},
        )
        assert "location" not in resp.headers

    def test_no_js_fallback_renders_the_modal_inline(self, admin, target):
        resp = admin["client"].post(
            f"/users/{target}/tokens", data={"csrf": admin["csrf"], "label": "x"},
        )
        assert resp.status_code == 200
        assert len(TOKEN_RE.findall(resp.text)) == 1
        assert "Token gerado" in resp.text

    def test_bad_label_is_refused(self, admin, store, target):
        resp = admin["client"].post(
            f"/users/{target}/tokens", data={"csrf": admin["csrf"], "label": "x" * 61},
        )
        assert "error=" in resp.headers["location"]
        assert store.list_tokens(target) == []

    def test_disabled_user_cannot_get_a_token(self, admin, store, target):
        store.set_user_disabled(target, True)
        resp = admin["client"].post(
            f"/users/{target}/tokens", data={"csrf": admin["csrf"], "label": "x"},
        )
        assert "error=" in resp.headers["location"]
        assert store.list_tokens(target) == []

    def test_rotate_issues_a_successor_and_keeps_the_old_one_alive(self, admin, store, target):
        old = sec.generate_token()
        tid = store.create_token(user_id=target, token=old, label="open-webui")["id"]

        resp = admin["client"].post(
            f"/tokens/{tid}/rotate",
            data={"csrf": admin["csrf"], "user_id": target},
            headers={"HX-Request": "true"},
        )
        new = TOKEN_RE.findall(resp.text)[0]

        assert store.verify_token(new)["status"] == vs.OK
        assert store.verify_token(old)["status"] == vs.OK  # migration window
        assert "Expirando" in admin["client"].get(f"/users/{target}").text

    def test_revoke_kills_immediately(self, admin, store, target):
        token = sec.generate_token()
        tid = store.create_token(user_id=target, token=token)["id"]
        admin["client"].post(
            f"/tokens/{tid}/revoke", data={"csrf": admin["csrf"], "user_id": target}
        )
        assert store.verify_token(token)["status"] == vs.REVOKED

    def test_token_actions_require_csrf(self, admin, store, target):
        token = sec.generate_token()
        tid = store.create_token(user_id=target, token=token)["id"]
        admin["client"].post(
            f"/tokens/{tid}/revoke", data={"csrf": "forged", "user_id": target}
        )
        assert store.verify_token(token)["status"] == vs.OK

    def test_creation_requires_csrf(self, admin, store, target):
        admin["client"].post(
            f"/users/{target}/tokens", data={"csrf": "forged", "label": "x"}
        )
        assert store.list_tokens(target) == []


# ---------------------------------------------------------------- audit


class TestAudit:
    def test_lists_events_newest_first(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev")
        store.create_token(user_id=uid, token=sec.generate_token(), label="t")
        html = admin["client"].get("/audit").text
        assert html.index("token.create") < html.index("user.create")

    def test_pagination(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev")
        for _ in range(web.AUDIT_PAGE_SIZE + 5):
            store.create_token(user_id=uid, token=sec.generate_token())
        first = admin["client"].get("/audit").text
        assert "offset=40" in first
        assert admin["client"].get("/audit?offset=40").status_code == 200

    def test_malformed_offset_does_not_crash(self, admin):
        assert admin["client"].get("/audit?offset=abc").status_code == 200
        assert admin["client"].get("/audit?offset=-5").status_code == 200


# ---------------------------------------------------------------- misc


class TestMisc:
    def test_healthz_is_public(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "vault_db": "ok"}

    def test_language_cookie_persists(self, admin):
        client = admin["client"]
        assert "Overview" in client.get("/?lang=en").text
        assert "Overview" in client.get("/").text  # cookie remembered
        assert "Visão geral" in client.get("/?lang=pt").text

    def test_unknown_language_falls_back(self, admin):
        assert "Visão geral" in admin["client"].get("/?lang=xx").text

    def test_auth_mode_is_reported_from_the_environment(self, admin, monkeypatch):
        assert "shadow" in admin["client"].get("/").text
        monkeypatch.setenv("MEM0_REQUIRE_AUTH", "on")
        assert "Requisições sem token = 401" in admin["client"].get("/").text

    def test_invalid_auth_mode_is_shown_not_hidden(self, admin, monkeypatch):
        monkeypatch.setenv("MEM0_REQUIRE_AUTH", "true")
        assert "invalid" in admin["client"].get("/").text

    def test_missing_secret_key_refuses_to_start(self, store, monkeypatch):
        monkeypatch.delenv("VAULT_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VAULT_SECRET_KEY"):
            web.create_app(store.db_path)

    def test_login_page_explains_how_to_bootstrap_when_empty(self, tmp_path, monkeypatch):
        empty = vs.VaultStore(tmp_path / "empty.db")
        monkeypatch.setattr(web, "_probe_mcp_health", lambda: {"ok": False, "code": None, "detail": {}})
        app = web.create_app(empty.db_path, secret_key="k")
        html = TestClient(app).get("/login").text
        assert "bootstrap-admin" in html


class TestHumanize:
    @pytest.mark.parametrize("lang,expected", [("pt", "há"), ("en", "ago")])
    def test_relative_past(self, lang, expected):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat()
        assert expected in web.humanize(ts, lang)

    @pytest.mark.parametrize("lang,expected", [("pt", "em"), ("en", "in")])
    def test_relative_future(self, lang, expected):
        from datetime import datetime, timedelta, timezone

        ts = (datetime.now(timezone.utc) + timedelta(hours=18)).isoformat()
        assert web.humanize(ts, lang).startswith(expected)

    @pytest.mark.parametrize("value", [None, "", "not-a-date"])
    def test_missing_or_broken_is_a_dash(self, value):
        assert web.humanize(value, "pt") == "—"


class TestInjection:
    """Admin-supplied text is rendered in HTML; autoescape must hold."""

    def test_display_name_cannot_inject_script(self, admin, store):
        payload = '<script>alert("xss")</script>'
        admin["client"].post("/users", data={
            "csrf": admin["csrf"], "display_name": payload, "email": "evil@x.com",
        })
        html = admin["client"].get("/").text
        assert payload not in html
        assert "&lt;script&gt;" in html

    def test_token_label_cannot_inject_markup(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev")
        # the label validator rejects angle brackets outright...
        resp = admin["client"].post(
            f"/users/{uid}/tokens", data={"csrf": admin["csrf"], "label": "<b>x</b>"},
        )
        assert "error=" in resp.headers["location"]
        # ...and a label written straight into the store is still escaped
        store.create_token(user_id=uid, token=sec.generate_token(), label="<i>raw</i>")
        html = admin["client"].get(f"/users/{uid}").text
        assert "<i>raw</i>" not in html and "&lt;i&gt;" in html

    def test_error_query_parameter_is_escaped(self, admin):
        resp = admin["client"].get("/?error=<img src=x onerror=alert(1)>&new=1")
        assert "<img src=x" not in resp.text


class TestRotationRace:
    def test_second_rotation_of_the_same_token_is_not_a_500(self, admin, store):
        """Two tabs, same Rotate button: the loser gets the list, not a stack trace."""
        uid = store.create_user(email="carlos@cliente.dev")
        tid = store.create_token(user_id=uid, token=sec.generate_token())["id"]

        first = admin["client"].post(
            f"/tokens/{tid}/rotate", data={"csrf": admin["csrf"], "user_id": uid},
            headers={"HX-Request": "true"},
        )
        assert len(TOKEN_RE.findall(first.text)) == 1

        second = admin["client"].post(
            f"/tokens/{tid}/rotate", data={"csrf": admin["csrf"], "user_id": uid},
            headers={"HX-Request": "true"},
        )
        assert second.status_code == 303
        assert TOKEN_RE.findall(second.text) == []
        successors = [t for t in store.list_tokens(uid) if t["renewed_from"] == tid]
        assert len(successors) == 1


class TestNoPythonObjectsLeakIntoTheUi:
    """`{{ t.copy }}` used to render "<built-in method copy of dict object...>".

    Jinja resolves attributes before keys, so every string whose name collides
    with a dict method was a landmine. These tests fail on ANY Python repr that
    reaches the page, not just that one key.
    """

    LEAKS = ("<built-in method", "<bound method", "<function ", "object at 0x")

    def _assert_clean(self, html: str):
        for leak in self.LEAKS:
            assert leak not in html, f"Python object rendered into the page: {leak}"

    def test_token_modal_copy_button(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev")
        resp = admin["client"].post(
            f"/users/{uid}/tokens", data={"csrf": admin["csrf"], "label": "x"},
            headers={"HX-Request": "true"},
        )
        assert ">\n        Copiar\n      <" in resp.text or "Copiar" in resp.text
        self._assert_clean(resp.text)

    @pytest.mark.parametrize("lang", ["pt", "en"])
    def test_every_page_in_both_languages(self, admin, store, lang):
        uid = store.create_user(email="carlos@cliente.dev", display_name="Carlos")
        store.create_token(user_id=uid, token=sec.generate_token(), label="open-webui")
        client = admin["client"]
        for path in ("/", "/?new=1", f"/users/{uid}", "/audit"):
            self._assert_clean(client.get(f"{path}{'&' if '?' in path else '?'}lang={lang}").text)
        client.get("/logout")
        self._assert_clean(client.get(f"/login?lang={lang}").text)

    def test_dict_method_names_resolve_to_strings(self):
        """The whole class of bug, not just 'copy'."""
        from mem0_mcp_selfhosted.vault import i18n

        for lang in i18n.LANGS:
            table = i18n.strings(lang)
            for key in list(table):
                assert isinstance(getattr(table, key), str), f"{lang}.{key} is not a string"
            for shadowed in ("copy", "keys", "values", "items", "get", "update", "pop", "clear"):
                if shadowed in table:
                    assert isinstance(getattr(table, shadowed), str)

    def test_both_tables_have_the_same_keys(self):
        from mem0_mcp_selfhosted.vault import i18n

        assert set(i18n.strings("pt")) == set(i18n.strings("en"))


class TestCopyButton:
    """navigator.clipboard does not exist over plain http on a LAN address, and
    the old inline handler threw synchronously there — the button did nothing."""

    def test_the_modal_uses_the_declarative_hook(self, admin, store):
        uid = store.create_user(email="carlos@cliente.dev")
        html = admin["client"].post(
            f"/users/{uid}/tokens", data={"csrf": admin["csrf"], "label": "x"},
            headers={"HX-Request": "true"},
        ).text
        assert 'data-copy-target="vault-token"' in html
        assert "navigator.clipboard" not in html, "no inline clipboard call in the page"
        assert 'id="vault-token"' in html

    def test_the_fallback_script_is_served_and_self_contained(self, client):
        resp = client.get("/static/vault-copy.js")
        assert resp.status_code == 200
        body = resp.text
        assert "execCommand" in body, "legacy path required for non-secure contexts"
        assert "isSecureContext" in body
        assert "selectNode" in body, "last resort: leave it selected for Ctrl+C"
        for network in ("fetch(", "XMLHttpRequest", "WebSocket", "import("):
            assert network not in body, f"the copy helper must not talk to anything: {network}"

    def test_the_script_is_referenced_by_every_page(self, admin):
        assert "/static/vault-copy.js" in admin["client"].get("/").text

    def test_both_languages_have_the_manual_fallback_label(self):
        from mem0_mcp_selfhosted.vault import i18n

        assert i18n.strings("pt").copyManual
        assert i18n.strings("en").copyManual


class TestHttpsMode:
    """VAULT_HTTPS_ONLY: Secure cookie, and a cutover that kills old sessions."""

    def _app(self, store, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("VAULT_HTTPS_ONLY", raising=False)
        else:
            monkeypatch.setenv("VAULT_HTTPS_ONLY", value)
        return TestClient(
            web.create_app(store.db_path, secret_key="k"), follow_redirects=False,
            base_url="https://vault.test",
        )

    def _login(self, client):
        page = client.get("/login")
        return client.post("/login", data={
            "csrf": csrf_of(page.text), "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
        })

    def test_cookie_is_secure_when_https_is_on(self, store, monkeypatch):
        monkeypatch.setattr(web, "_probe_mcp_health", lambda: {"ok": True, "code": 200, "detail": {}})
        cookie = self._login(self._app(store, monkeypatch, "true")).headers["set-cookie"].lower()
        assert "secure" in cookie
        assert "vault_session_s=" in cookie

    def test_cookie_is_not_secure_over_plain_http(self, store, monkeypatch):
        monkeypatch.setattr(web, "_probe_mcp_health", lambda: {"ok": True, "code": 200, "detail": {}})
        cookie = self._login(self._app(store, monkeypatch, "false")).headers["set-cookie"].lower()
        assert "secure" not in cookie
        assert "vault_session=" in cookie

    def test_the_cutover_orphans_pre_tls_sessions(self, store, monkeypatch):
        """A cookie issued over http must not authenticate after the flip."""
        monkeypatch.setattr(web, "_probe_mcp_health", lambda: {"ok": True, "code": 200, "detail": {}})
        http_client = self._app(store, monkeypatch, "false")
        self._login(http_client)
        stolen = http_client.cookies.get("vault_session")
        assert stolen

        https_client = self._app(store, monkeypatch, "true")
        https_client.cookies.set("vault_session", stolen)
        assert https_client.get("/").status_code == 303  # name changed → anonymous

    @pytest.mark.parametrize("typo", ["yes please", "on", "1 ", "sim", "enabled"])
    def test_a_typo_raises_instead_of_silently_disabling_tls(self, store, monkeypatch, typo):
        monkeypatch.setenv("VAULT_HTTPS_ONLY", typo)
        if typo.strip() in ("1",):
            pytest.skip("'1' is a documented true value")
        with pytest.raises(ValueError, match="VAULT_HTTPS_ONLY"):
            web.create_app(store.db_path, secret_key="k")

    def test_unset_means_plain_http(self, store, monkeypatch):
        monkeypatch.delenv("VAULT_HTTPS_ONLY", raising=False)
        assert web._https_only() is False


class TestAuthModeSourceOfTruth:
    """O modo exibido vem de QUEM aplica a auth, não de uma cópia do env.

    Bug real (20/07/2026): o MCP já respondia 401 (`on`) e o cofre exibia
    "off", porque lia a própria variável de ambiente — que ninguém tinha
    setado nele. O indicador mais visível da UI mentia sobre a postura de
    segurança, e o painel de prontidão oferecia promover o que já estava
    promovido.
    """

    @staticmethod
    def _reports(monkeypatch, detail, ok=True):
        """Injeta a resposta do :8081 E invalida o cache de 20s.

        O login do fixture já fez requisições, então o cache está quente com
        a resposta anterior — sem invalidar, o teste mediria o passado.
        """
        monkeypatch.setattr(
            web, "_probe_mcp_health",
            lambda: {"ok": ok, "code": 200 if ok else None, "detail": detail},
        )
        web._health_cache.update(at=0.0, value=None)

    def test_mcp_report_wins_over_the_local_env(self, admin, monkeypatch):
        monkeypatch.setenv("MEM0_REQUIRE_AUTH", "off")  # cópia local desatualizada
        self._reports(monkeypatch, {"auth_mode": "on", "vault_db": "ok"})
        html = admin["client"].get("/").text
        assert "Requisições sem token = 401" in html, "a UI tem de refletir o 'on' real"
        assert "Prontidão para MEM0_REQUIRE_AUTH=on" not in html, (
            "não faz sentido oferecer promover o que já está promovido"
        )

    def test_falls_back_to_the_env_but_says_it_is_unconfirmed(self, admin, monkeypatch):
        monkeypatch.setenv("MEM0_REQUIRE_AUTH", "shadow")
        self._reports(monkeypatch, {}, ok=False)
        html = admin["client"].get("/").text
        assert "não confirmado" in html
        assert "Valida e registra" in html  # o valor do env ainda é exibido

    def test_a_confirmed_mode_is_not_flagged(self, admin, monkeypatch):
        self._reports(monkeypatch, {"auth_mode": "shadow", "vault_db": "ok"})
        assert "não confirmado" not in admin["client"].get("/").text

    def test_health_is_cached_between_renders(self, admin, monkeypatch):
        calls = []
        monkeypatch.setattr(web, "_probe_mcp_health", lambda: calls.append(1) or {
            "ok": True, "code": 200, "detail": {"auth_mode": "on"},
        })
        web._health_cache.update(at=0.0, value=None)
        for _ in range(4):
            admin["client"].get("/")
        assert len(calls) == 1, "o modo aparece em toda página; não pode custar uma chamada HTTP cada"


class TestSessionInvalidation:
    """Logout tem de matar a sessão no SERVIDOR, não só no navegador.

    A sessão é cookie assinado no cliente: sem uma época guardada no servidor,
    "sair" apenas apaga a cópia local e qualquer cópia roubada continua válida
    pelas 12h inteiras. Achado da revisão independente (20/07/2026).
    """

    def _steal(self, client):
        return client.cookies.get("vault_session") or client.cookies.get("vault_session_s")

    def test_a_stolen_cookie_dies_with_logout(self, admin, client):
        stolen = self._steal(admin["client"])
        assert stolen, "não consegui capturar o cookie de sessão"
        assert admin["client"].get("/").status_code == 200

        admin["client"].post("/logout", data={"csrf": admin["csrf"]})

        # o "atacante" tem a cópia do cookie e um cliente novo
        thief = TestClient(admin["client"].app, follow_redirects=False)
        thief.cookies.set("vault_session", stolen)
        assert thief.get("/").status_code == 303, "cookie roubado sobreviveu ao logout"

    def test_changing_the_password_kills_open_sessions(self, admin, store):
        assert admin["client"].get("/").status_code == 200
        uid = store.get_user_by_email(ADMIN_EMAIL)["id"]

        store.set_password_hash(uid, sec.hash_password("uma senha totalmente nova"))

        assert admin["client"].get("/").status_code == 303, (
            "trocar a senha tem de derrubar as sessões abertas"
        )

    def test_a_normal_session_survives_normal_use(self, admin):
        for _ in range(3):
            assert admin["client"].get("/").status_code == 200
            assert admin["client"].get("/audit").status_code == 200

    def test_the_epoch_advances_and_is_audited(self, admin, store):
        uid = store.get_user_by_email(ADMIN_EMAIL)["id"]
        before = store.get_user(uid)["session_epoch"]
        admin["client"].post("/logout", data={"csrf": admin["csrf"]})
        assert store.get_user(uid)["session_epoch"] == before + 1
        assert store.list_audit(limit=1)[0]["action"] == "session.invalidate"

    def test_logout_without_csrf_does_not_invalidate(self, admin, store):
        uid = store.get_user_by_email(ADMIN_EMAIL)["id"]
        before = store.get_user(uid)["session_epoch"]
        admin["client"].post("/logout", data={"csrf": "forjado"})
        assert store.get_user(uid)["session_epoch"] == before
        assert admin["client"].get("/").status_code == 200


class TestContentSecurityPolicy:
    """A UI não busca nada de fora — nem fonte, nem script.

    Uma interface de credenciais numa LAN não deve depender da internet nem
    contar a um terceiro quando o admin a abre. Com tudo local, a política
    pode ser fechada sem exceção.
    """

    def test_csp_is_present_and_closed_by_default(self, admin):
        csp = admin["client"].get("/").headers["content-security-policy"]
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'none'" in csp

    def test_scripts_may_only_come_from_us(self, admin):
        csp = admin["client"].get("/").headers["content-security-policy"]
        assert "script-src 'self'" in csp
        assert "unsafe-inline" not in csp.split("style-src")[0], (
            "script-src não pode aceitar inline — o clipboard virou arquivo justamente por isso"
        )

    @pytest.mark.parametrize("path", ["/", "/audit", "/login"])
    def test_no_page_references_an_external_host(self, admin, path):
        html = admin["client"].get(path).text
        for external in ("googleapis.com", "gstatic.com", "unpkg.com", "cdn.jsdelivr", "//fonts."):
            assert external not in html, f"{path} depende de {external}"

    def test_the_stylesheet_still_names_the_mockup_fonts_first(self):
        from mem0_mcp_selfhosted.vault import web

        css = (web.STATIC_DIR / "vault.css").read_text()
        # quem tiver as fontes do mockup instaladas continua vendo o desenho
        assert "Sora" in css and "Instrument Sans" in css and "JetBrains Mono" in css
        assert "@import" not in css, "CSS não pode buscar nada na rede"
        assert "url(http" not in css
