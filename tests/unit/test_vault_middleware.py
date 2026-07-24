"""Vault ASGI gate: mode semantics, fail-closed behavior, principal exposure."""

import logging
import secrets
from datetime import datetime, timedelta, timezone

import pytest

from mem0_mcp_selfhosted.vault import middleware as mw
from mem0_mcp_selfhosted.vault import store as vs


def _token() -> str:
    return vs.TOKEN_PREFIX + secrets.token_urlsafe(32)


class RecordingApp:
    """Downstream ASGI app: records what reached it."""

    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"downstream"})

    @property
    def called(self) -> bool:
        return bool(self.calls)


def make_scope(*, path="/mcp", headers=None, method="POST", type="http"):
    return {
        "type": type,
        "method": method,
        "path": path,
        "client": ("10.0.0.150", 51234),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or [])],
    }


async def call(middleware, scope):
    """Drive the middleware, returning (status, body)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (start["status"] if start else None), body, (start or {}).get("headers", [])


@pytest.fixture
def vault(tmp_path):
    store = vs.VaultStore(tmp_path / "vault.db")
    user_id = store.create_user(email="dev@example.com")
    token = _token()
    store.create_token(user_id=user_id, token=token, label="laptop")
    path = tmp_path / "vault.db"
    import shutil

    shutil.copy(path, str(path) + ".backup")  # pristine copy for recovery tests
    return {"store": store, "path": str(path), "token": token, "user_id": user_id}


@pytest.fixture
def app():
    return RecordingApp()


def gate(app, vault, mode):
    return mw.BearerTokenMiddleware(app, db_path=vault["path"], mode=mode)


# ---------------------------------------------------------------- modes


class TestModeOff:
    pytestmark = pytest.mark.asyncio

    async def test_delegates_without_opening_the_vault(self, app, tmp_path):
        gate_off = mw.BearerTokenMiddleware(
            app, db_path=str(tmp_path / "does-not-exist.db"), mode=mw.MODE_OFF
        )
        status, body, _ = await call(gate_off, make_scope())
        assert status == 200 and body == b"downstream"
        assert gate_off._store is None  # never even looked

    async def test_no_principal_is_published(self, app, vault):
        await call(gate(app, vault, mw.MODE_OFF),
                   make_scope(headers=[("authorization", f"Bearer {vault['token']}")]))
        assert app.calls[0].get("state", {}).get("vault_user") is None


class TestModeShadow:
    pytestmark = pytest.mark.asyncio

    async def test_valid_token_passes_and_is_touched(self, app, vault):
        await call(gate(app, vault, mw.MODE_SHADOW),
                   make_scope(headers=[("authorization", f"Bearer {vault['token']}")]))
        assert app.called
        tokens = vault["store"].list_tokens(vault["user_id"])
        assert tokens[0]["last_used_at"] is not None, "shadow must produce promotion evidence"

    async def test_unauthorized_passes_but_warns(self, app, vault, caplog):
        with caplog.at_level(logging.WARNING):
            status, body, _ = await call(gate(app, vault, mw.MODE_SHADOW), make_scope())
        assert status == 200 and body == b"downstream"
        assert "SHADOW" in caplog.text

    async def test_warnings_are_throttled(self, app, vault, caplog):
        g = gate(app, vault, mw.MODE_SHADOW)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                await call(g, make_scope())
        assert caplog.text.count("SHADOW") == 1

    async def test_missing_vault_passes_through(self, app, tmp_path, caplog):
        g = mw.BearerTokenMiddleware(
            app, db_path=str(tmp_path / "absent.db"), mode=mw.MODE_SHADOW
        )
        with caplog.at_level(logging.WARNING):
            status, _, _ = await call(g, make_scope())
        assert status == 200 and app.called
        assert "vault unavailable" in caplog.text


class TestModeOn:
    pytestmark = pytest.mark.asyncio

    async def test_valid_token_reaches_the_app(self, app, vault):
        status, body, _ = await call(
            gate(app, vault, mw.MODE_ON),
            make_scope(headers=[("authorization", f"Bearer {vault['token']}")]),
        )
        assert status == 200 and body == b"downstream"

    @pytest.mark.parametrize("headers,reason", [
        ([], "missing"),
        ([("authorization", "Bearer garbage")], vs.MALFORMED),
        ([("authorization", "Basic abc")], vs.MALFORMED),
    ])
    async def test_denies_bad_headers(self, app, vault, headers, reason):
        status, body, resp_headers = await call(
            gate(app, vault, mw.MODE_ON), make_scope(headers=headers)
        )
        assert status == 401
        assert not app.called, "the MCP app must never see an unauthorized request"
        assert reason.encode() in body
        assert (b"www-authenticate", b'Bearer realm="DeepMem0", error="invalid_token"') in resp_headers

    async def test_denies_unknown_token(self, app, vault):
        status, _, _ = await call(
            gate(app, vault, mw.MODE_ON),
            make_scope(headers=[("authorization", f"Bearer {_token()}")]),
        )
        assert status == 401 and not app.called

    async def test_denies_revoked_token(self, app, vault):
        tid = vault["store"].list_tokens(vault["user_id"])[0]["id"]
        vault["store"].revoke_token(tid)
        status, body, _ = await call(
            gate(app, vault, mw.MODE_ON),
            make_scope(headers=[("authorization", f"Bearer {vault['token']}")]),
        )
        assert status == 401 and b"revoked" in body

    async def test_denies_expired_token(self, app, vault):
        expired = _token()
        vault["store"].create_token(
            user_id=vault["user_id"], token=expired,
            expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        status, body, _ = await call(
            gate(app, vault, mw.MODE_ON),
            make_scope(headers=[("authorization", f"Bearer {expired}")]),
        )
        assert status == 401 and b"expired" in body

    async def test_denies_repeated_authorization_header(self, app, vault):
        scope = make_scope(headers=[
            ("authorization", f"Bearer {vault['token']}"),
            ("authorization", f"Bearer {vault['token']}"),
        ])
        status, _, _ = await call(gate(app, vault, mw.MODE_ON), scope)
        assert status == 401 and not app.called

    async def test_fails_closed_without_a_vault(self, app, tmp_path, caplog):
        g = mw.BearerTokenMiddleware(
            app, db_path=str(tmp_path / "absent.db"), mode=mw.MODE_ON
        )
        with caplog.at_level(logging.CRITICAL):
            status, body, _ = await call(
                g, make_scope(headers=[("authorization", f"Bearer {_token()}")])
            )
        assert status == 401 and b"vault_unavailable" in body
        assert not app.called
        assert "VAULT UNAVAILABLE" in caplog.text

    async def test_fails_closed_on_incompatible_schema(self, app, tmp_path):
        import sqlite3

        path = tmp_path / "future.db"
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {vs.SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        g = mw.BearerTokenMiddleware(app, db_path=str(path), mode=mw.MODE_ON)
        status, _, _ = await call(g, make_scope())
        assert status == 401 and not app.called


# ---------------------------------------------------------------- plumbing


class TestPassthrough:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("mode", [mw.MODE_OFF, mw.MODE_SHADOW, mw.MODE_ON])
    async def test_lifespan_is_never_intercepted(self, app, vault, mode):
        """Auth must not break startup/shutdown — the session manager lives there."""
        scope = {"type": "lifespan"}
        received = []

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            received.append(message)

        await gate(app, vault, mode)(scope, receive, send)
        assert app.calls == [scope]

    @pytest.mark.parametrize("path", ["/health", "/healthz"])
    async def test_health_is_exempt_in_on_mode(self, app, vault, path):
        status, _, _ = await call(
            gate(app, vault, mw.MODE_ON), make_scope(path=path, method="GET")
        )
        assert status == 200 and app.called


class TestPrincipalExposure:
    pytestmark = pytest.mark.asyncio

    async def test_published_on_scope_state(self, app, vault):
        await call(
            gate(app, vault, mw.MODE_ON),
            make_scope(headers=[("authorization", f"Bearer {vault['token']}")]),
        )
        principal = app.calls[0]["state"]["vault_user"]
        assert principal["email"] == "dev@example.com"
        assert principal["status"] == vs.OK

    async def test_contextvar_is_set_during_the_call_and_reset_after(self, vault):
        seen = {}

        class Peeking:
            async def __call__(self, scope, receive, send):
                seen["principal"] = mw.current_principal.get()
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b""})

        g = mw.BearerTokenMiddleware(Peeking(), db_path=vault["path"], mode=mw.MODE_ON)
        await call(g, make_scope(headers=[("authorization", f"Bearer {vault['token']}")]))
        assert seen["principal"]["email"] == "dev@example.com"
        assert mw.current_principal.get() is None, "principal must not leak across requests"


class TestNormalizeMode:
    @pytest.mark.parametrize("raw,expected", [
        ("", mw.MODE_OFF), ("  ", mw.MODE_OFF), ("off", mw.MODE_OFF),
        ("SHADOW", mw.MODE_SHADOW), (" on ", mw.MODE_ON),
    ])
    def test_accepts(self, raw, expected):
        assert mw.normalize_mode(raw) == expected

    @pytest.mark.parametrize("raw", ["true", "1", "yes", "enabled", "no"])
    def test_typo_is_a_hard_error_not_a_silent_downgrade(self, raw):
        with pytest.raises(ValueError):
            mw.normalize_mode(raw)


class TestRuntimeFailureAfterOpen:
    """The vault breaking mid-flight (deleted, corrupted, unreadable).

    The store is cached after the first request, so these paths are NOT the
    same as "missing at startup" — an independent review found shadow could
    500 here, which would break its one promise: never block.
    """

    pytestmark = pytest.mark.asyncio

    @staticmethod
    def _break(gate, vault):
        """Warm the cache, then corrupt the file underneath it."""
        gate._get_store()
        import pathlib

        pathlib.Path(vault["path"]).write_bytes(b"not a database at all")

    async def test_shadow_never_blocks_when_the_vault_breaks(self, app, vault, caplog):
        g = gate(app, vault, mw.MODE_SHADOW)
        self._break(g, vault)
        with caplog.at_level(logging.WARNING):
            status, body, _ = await call(
                g, make_scope(headers=[("authorization", f"Bearer {vault['token']}")])
            )
        assert status == 200 and body == b"downstream"
        assert "vault read failed" in caplog.text

    async def test_on_fails_closed_when_the_vault_breaks(self, app, vault, caplog):
        g = gate(app, vault, mw.MODE_ON)
        self._break(g, vault)
        with caplog.at_level(logging.CRITICAL):
            status, body, _ = await call(
                g, make_scope(headers=[("authorization", f"Bearer {vault['token']}")])
            )
        assert status == 401 and b"vault_unavailable" in body
        assert not app.called
        assert "VAULT READ FAILED" in caplog.text

    async def test_cached_store_is_dropped_so_recovery_needs_no_restart(self, app, vault):
        g = gate(app, vault, mw.MODE_SHADOW)
        self._break(g, vault)
        await call(g, make_scope(headers=[("authorization", f"Bearer {vault['token']}")]))
        assert g._store is None, "a broken store must not stay cached"

        # restore the file: the very next request authorizes again
        import shutil

        shutil.copy(vault["path"] + ".backup", vault["path"])
        g2 = gate(app, vault, mw.MODE_ON)
        status, _, _ = await call(
            g2, make_scope(headers=[("authorization", f"Bearer {vault['token']}")])
        )
        assert status == 200


class TestVaultFilePermissions:
    def test_database_is_owner_only(self, tmp_path):
        import stat

        path = tmp_path / "perm.db"
        path.touch(mode=0o644)
        vs.VaultStore(path)
        assert not (path.stat().st_mode & 0o077), "vault must not be world/group readable"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


class TestDurableDenialCounters:
    """Throttled logs cannot answer "silent client or denied client?" after a
    restart. These counters can, and they are what the promotion gate reads."""

    pytestmark = pytest.mark.asyncio

    async def test_denials_are_persisted_by_reason(self, app, vault):
        g = gate(app, vault, mw.MODE_SHADOW)
        await call(g, make_scope())                                    # missing (flushes)
        await call(g, make_scope(headers=[("authorization", "Bearer nope")]))  # malformed
        await call(g, make_scope(headers=[("authorization", f"Bearer {_token()}")]))  # unknown
        g._last_flush = 0.0  # the flush window elapses
        await call(g, make_scope())

        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        by_reason = {d["reason"]: d["count"] for d in vault["store"].denials_since(cutoff)}
        assert by_reason == {"missing": 2, vs.MALFORMED: 1, vs.UNKNOWN: 1}

    async def test_a_flood_costs_bounded_writes(self, app, vault):
        g = gate(app, vault, mw.MODE_ON)
        for _ in range(50):
            await call(g, make_scope())

        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        rows = vault["store"].denials_since(cutoff)
        # one flush happened (the first); the other 49 are buffered in memory
        assert rows[0]["count"] == 1
        assert g._denials["missing"] == 49

    async def test_telemetry_failure_never_fails_a_request(self, app, vault, monkeypatch):
        g = gate(app, vault, mw.MODE_SHADOW)

        def boom(*_a, **_kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr(vault["store"], "record_denials", boom)
        monkeypatch.setattr(g, "_get_store", lambda: vault["store"])
        status, _, _ = await call(g, make_scope())
        assert status == 200

    async def test_successful_use_is_counted_too(self, app, vault):
        g = gate(app, vault, mw.MODE_SHADOW)
        await call(g, make_scope(headers=[("authorization", f"Bearer {vault['token']}")]))
        assert vault["store"].list_tokens(vault["user_id"])[0]["use_count"] == 1


class TestBehindAReverseProxy:
    """The audit log and the promotion counters must name the REAL client.

    Behind Caddy every connection arrives from 127.0.0.1, so without proxy
    handling every denial would be attributed to the proxy — destroying exactly
    the evidence the shadow window exists to collect. uvicorn already does this
    (it rewrites scope["client"] before our gate runs); this pins the composition
    and, more importantly, pins that an UNTRUSTED peer cannot forge its identity.
    """

    pytestmark = pytest.mark.asyncio

    @staticmethod
    def behind_proxy(gate_app, trusted="127.0.0.1"):
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        return ProxyHeadersMiddleware(gate_app, trusted_hosts=trusted)

    @staticmethod
    def last_client(vault):
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        rows = vault["store"].denials_since(cutoff)
        return rows[0]["last_client"] if rows else None

    async def test_trusted_proxy_hands_over_the_real_client_ip(self, app, vault):
        stack = self.behind_proxy(gate(app, vault, mw.MODE_SHADOW))
        scope = make_scope(headers=[("x-forwarded-for", "10.0.0.77")])
        scope["client"] = ("127.0.0.1", 51000)  # Caddy on loopback
        await call(stack, scope)
        assert self.last_client(vault) == "10.0.0.77"

    async def test_an_untrusted_peer_cannot_forge_its_own_identity(self, app, vault):
        stack = self.behind_proxy(gate(app, vault, mw.MODE_SHADOW))
        scope = make_scope(headers=[("x-forwarded-for", "10.0.0.1")])
        scope["client"] = ("10.0.0.50", 51000)  # straight from the LAN
        await call(stack, scope)
        assert self.last_client(vault) == "10.0.0.50", "XFF from a stranger must be ignored"

    async def test_without_a_proxy_nothing_changes(self, app, vault):
        g = gate(app, vault, mw.MODE_SHADOW)
        scope = make_scope(headers=[("x-forwarded-for", "10.0.0.1")])
        scope["client"] = ("10.0.0.50", 51000)
        await call(g, scope)
        assert self.last_client(vault) == "10.0.0.50"

    async def test_the_gate_still_authorizes_normally_behind_the_proxy(self, app, vault):
        stack = self.behind_proxy(gate(app, vault, mw.MODE_ON))
        scope = make_scope(headers=[
            ("authorization", f"Bearer {vault['token']}"),
            ("x-forwarded-for", "10.0.0.77"),
        ])
        scope["client"] = ("127.0.0.1", 51000)
        status, body, _ = await call(stack, scope)
        assert status == 200 and body == b"downstream"
