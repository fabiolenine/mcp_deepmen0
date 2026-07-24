"""The authorization contract, proven over a real streamable-HTTP session.

This is the gate the vault plan demands before any UI work: it is not enough
that the middleware resolves a principal — the principal must still be
reachable when the SDK dispatches the tool call, which happens on a task of
its own. The test drives the actual MCP client against the actual
``streamable_http_app()`` (over an in-process ASGI transport, so there is no
port to flake) and asserts on tool *output*.

The observable used is deliberate: ``_effective_user_id`` runs before
``_ensure_memory``, so a scope conflict is reported without any live
infrastructure — and a run where the principal did NOT arrive produces a
different answer ("Memory not initialized"). The two outcomes cannot be
confused, which is what makes this a proof and not a smoke test.

Also pins the SDK surface we wrap (``streamable_http_app`` + lifespan): if a
future ``mcp`` release moves it, this fails instead of production.
"""

from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mem0_mcp_selfhosted.vault import middleware as mw
from mem0_mcp_selfhosted.vault import store as vs

pytestmark = [pytest.mark.contract, pytest.mark.asyncio]

BASE_URL = "http://vault.test/mcp"

ALICE_MEMORY = "11111111-1111-4111-8111-111111111111"
BOB_MEMORY = "22222222-2222-4222-8222-222222222222"


def _token() -> str:
    return vs.TOKEN_PREFIX + secrets.token_urlsafe(32)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A vault with two principals: unbound (today) and bound to 'alice'."""
    monkeypatch.setenv("MEM0_VAULT_DB_PATH", str(tmp_path / "vault.db"))
    monkeypatch.setenv("MEM0_USER_ID", "default-user")

    # Authorization is decided before Memory is needed, and this test must
    # never reach real infrastructure — pin the tool's post-auth answer so the
    # observable stays the same whatever other tests left in the module globals.
    from mem0_mcp_selfhosted import server as server_mod

    monkeypatch.setattr(server_mod, "_ensure_memory", lambda: None)

    store = vs.VaultStore(tmp_path / "vault.db")

    unbound_uid = store.create_user(email="unbound@example.com")
    bound_uid = store.create_user(email="alice@example.com", mem0_user_id="alice")

    tokens = {"unbound": _token(), "bound": _token(), "revoked": _token()}
    store.create_token(user_id=unbound_uid, token=tokens["unbound"])
    store.create_token(user_id=bound_uid, token=tokens["bound"])
    revoked_id = store.create_token(user_id=unbound_uid, token=tokens["revoked"])["id"]

    return {
        "store": store, "path": str(tmp_path / "vault.db"),
        "tokens": tokens, "revoked_id": revoked_id,
    }


def build_app(vault, mode):
    """The production wiring: FastMCP's ASGI app behind the vault gate."""
    from mem0_mcp_selfhosted import server as server_mod

    mcp = server_mod._create_server()
    return mw.BearerTokenMiddleware(
        mcp.streamable_http_app(), db_path=vault["path"], mode=mode
    )


def asgi_client(app, token):
    """An httpx client that speaks to the wrapped app in-process."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://vault.test", headers=headers, timeout=30,
    )


@asynccontextmanager
async def running(app):
    """Drive the ASGI lifespan by hand.

    ``httpx.ASGITransport`` speaks only the http scope, and the MCP session
    manager's task group is created in lifespan — without this the first
    request dies with "Task group is not initialized". Running it here is also
    the assertion that our wrapper delegates lifespan untouched.
    """
    to_app_send, to_app_recv = anyio.create_memory_object_stream(4)
    started, stopped = anyio.Event(), anyio.Event()

    async def receive():
        return await to_app_recv.receive()

    async def send(message):
        if message["type"].startswith("lifespan.startup"):
            started.set()
        elif message["type"].startswith("lifespan.shutdown"):
            stopped.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(app, {"type": "lifespan", "state": {}}, receive, send)
        await to_app_send.send({"type": "lifespan.startup"})
        await started.wait()
        try:
            yield
        finally:
            await to_app_send.send({"type": "lifespan.shutdown"})
            await stopped.wait()


@asynccontextmanager
async def mcp_session(app, token):
    async with running(app):
        async with asgi_client(app, token) as http_client:
            async with streamable_http_client(BASE_URL, http_client=http_client) as (
                read, write, _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session


def flatten(exc: BaseException) -> str:
    """Readable text for a (possibly nested) ExceptionGroup from anyio."""
    parts = [f"{type(exc).__name__}: {exc}"]
    for sub in getattr(exc, "exceptions", []):
        parts.append(flatten(sub))
    return " | ".join(parts)


async def call_tool(app, token, tool, args):
    """Full MCP session through the gate: initialize → initialized → call."""
    async with mcp_session(app, token) as session:
        result = await session.call_tool(tool, args)
        return json.loads(result.content[0].text)


class MemorySpy:
    """Minimal stand-in for mem0ai's Memory that records what it was asked.

    The negative side of the propagation proof (a scope conflict) is decisive
    on its own, but the positive side used to prove only "execution got past
    the authorization check". This records the ACTUAL scope handed to the
    backend, so "the bound identity reached mem0" becomes an assertion instead
    of an inference.
    """

    def __init__(self, records: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.records = records or {}
        self.enable_graph = False

    def _log(self, name, **kwargs):
        self.calls.append((name, kwargs))

    # id-addressed surface
    def get(self, memory_id):
        self._log("get", memory_id=memory_id)
        return self.records.get(memory_id)

    def history(self, memory_id):
        self._log("history", memory_id=memory_id)
        return [{"event": "ADD", "memory_id": memory_id}]

    def update(self, memory_id, data=None, **kwargs):
        self._log("update", memory_id=memory_id, data=data)
        return {"message": "updated"}

    def delete(self, memory_id):
        self._log("delete", memory_id=memory_id)
        return None

    # scope-addressed surface
    def get_all(self, **kwargs):
        self._log("get_all", **kwargs)
        return {"results": []}

    def search(self, **kwargs):
        self._log("search", **kwargs)
        return {"results": []}

    def add(self, *_args, **kwargs):
        self._log("add", **kwargs)
        return {"results": []}

    def __getattr__(self, name):
        def call(*_args, **kwargs):
            self._log(name, **kwargs)
            return {"results": []}

        return call

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def scopes_for(self, method: str) -> list[str | None]:
        return [kwargs.get("user_id") for name, kwargs in self.calls if name == method]


@pytest.fixture
def spy(monkeypatch):
    """Swap the lazy Memory for a recorder, for the whole tool pipeline."""
    from mem0_mcp_selfhosted import server as server_mod

    recorder = MemorySpy({
        ALICE_MEMORY: {"id": ALICE_MEMORY, "memory": "alice fact", "user_id": "alice"},
        BOB_MEMORY: {"id": BOB_MEMORY, "memory": "bob fact", "user_id": "bob"},
    })
    monkeypatch.setattr(server_mod, "_ensure_memory", lambda: recorder)
    return recorder


# ------------------------------------------------------------------ contract


class TestPrincipalPropagation:
    async def test_bound_token_overrides_the_client_scope(self, vault):
        """The proof: the principal survives the SDK's task hop.

        A token bound to 'alice' asking for 'bob' must be refused BY THE TOOL.
        If the principal had not propagated, the tool would have accepted
        'bob' and failed later on infrastructure instead.
        """
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "bob"})
        assert "error" in out
        assert "bound to user_id 'alice'" in out["error"]
        assert "cannot access 'bob'" in out["error"]

    async def test_mismatch_is_never_a_silent_redirect(self, vault):
        """Refused, not quietly served from alice's scope."""
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "bob"})
        assert "Memory not initialized" not in json.dumps(out)

    async def test_bound_token_agreeing_with_itself_proceeds(self, vault):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "alice"})
        # past authorization; only infrastructure stops it in this test env
        assert out.get("error") == "Memory not initialized"

    async def test_unbound_token_keeps_todays_behavior(self, vault):
        """Current phase: every token has mem0_user_id='' — zero behavior change."""
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["unbound"], "get_memories", {"user_id": "bob"})
        assert out.get("error") == "Memory not initialized"  # no scope conflict raised


class TestGateOverRealSessions:
    async def test_revoked_token_is_refused_on_the_next_request(self, vault):
        """Revocation semantics: new requests fail; MCP tool calls are new requests."""
        token = vault["tokens"]["revoked"]

        # a session manager runs once per app instance, so each session gets its own
        out = await call_tool(build_app(vault, mw.MODE_ON), token, "get_memories", {})
        assert out.get("error") == "Memory not initialized"  # still valid: authorized

        vault["store"].revoke_token(vault["revoked_id"])

        with pytest.raises(Exception) as exc:
            await call_tool(build_app(vault, mw.MODE_ON), token, "get_memories", {})
        assert "401" in flatten(exc.value)

    async def test_no_token_is_refused(self, vault):
        app = build_app(vault, mw.MODE_ON)
        with pytest.raises(Exception) as exc:
            await call_tool(app, None, "get_memories", {})
        assert "401" in flatten(exc.value)

    async def test_shadow_mode_lets_an_anonymous_session_through(self, vault):
        app = build_app(vault, mw.MODE_SHADOW)
        out = await call_tool(app, None, "get_memories", {})
        assert out.get("error") == "Memory not initialized"

    async def test_off_mode_is_untouched_behavior(self, vault):
        app = build_app(vault, mw.MODE_OFF)
        out = await call_tool(app, None, "get_memories", {})
        assert out.get("error") == "Memory not initialized"

    async def test_tools_list_survives_the_wrapper(self, vault):
        """initialize + tools/list round-trip through the gate (SDK surface pin)."""
        app = build_app(vault, mw.MODE_ON)
        async with mcp_session(app, vault["tokens"]["unbound"]) as session:
            info = await session.send_ping()  # session already initialized
            tools = await session.list_tools()
        assert info is not None
        assert "add_memory" in {t.name for t in tools.tools}


class TestPropagationMechanism:
    """Which of the two lookups actually carries the principal (plan item A4).

    The SDK dispatches tool calls on its own task, so the middleware's
    contextvar is not guaranteed to survive — the ASGI scope is. Pinning both
    behaviors here means a future SDK change shows up as a test failure and
    not as a silently unauthenticated tool call.
    """

    async def test_scope_state_alone_is_enough(self, vault, monkeypatch):
        """Fallback path: with the contextvar blinded, authorization still holds."""

        class BlindContextVar:
            """Simulates a contextvar that does not survive the task hop."""

            def get(self, *_a):
                return None

            def set(self, _value):
                return object()

            def reset(self, _token):
                pass

        monkeypatch.setattr(mw, "current_principal", BlindContextVar())
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "bob"})
        assert "bound to user_id 'alice'" in out["error"]


# ------------------------------------------------------- raw session plumbing


INIT_BODY = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "vault-test", "version": "1"},
    },
}
MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


async def raw_initialize(http_client, token):
    """POST initialize by hand and return the server's session id.

    The high-level client hides the session id, and these tests are precisely
    about what happens when a DIFFERENT credential presents it.
    """
    headers = dict(MCP_HEADERS)
    if token:
        headers["authorization"] = f"Bearer {token}"
    resp = await http_client.post(BASE_URL, json=INIT_BODY, headers=headers)
    assert resp.status_code == 200, resp.text
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "server did not open a session"
    await http_client.post(
        BASE_URL,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**headers, "mcp-session-id": session_id},
    )
    return session_id


async def raw_tool_call(http_client, token, session_id, tool="get_memories", args=None):
    headers = {**MCP_HEADERS, "mcp-session-id": session_id}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return await http_client.post(
        BASE_URL,
        json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": tool, "arguments": args or {}},
        },
        headers=headers,
    )


class TestSessionBinding:
    """A session belongs to the credential that opened it.

    Without ``scope["user"]`` the SDK records every session owner as ``None``,
    so any valid token could ride any leaked session id. Found by independent
    review; this pins the fix.
    """

    async def test_a_session_cannot_be_reused_by_another_token(self, vault):
        app = build_app(vault, mw.MODE_ON)
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, vault["tokens"]["unbound"])

                # the owner keeps working
                mine = await raw_tool_call(http, vault["tokens"]["unbound"], session_id)
                assert mine.status_code == 200

                # a DIFFERENT valid token presenting the same session id is
                # answered as if the session did not exist
                theirs = await raw_tool_call(http, vault["tokens"]["bound"], session_id)
                assert theirs.status_code == 404, theirs.text

    async def test_anonymous_cannot_ride_an_authenticated_session(self, vault):
        app = build_app(vault, mw.MODE_ON)
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, vault["tokens"]["unbound"])
                anon = await raw_tool_call(http, None, session_id)
                assert anon.status_code == 401  # stopped at the gate, before the session

    async def test_principal_identity_is_per_token_not_per_user(self, vault, monkeypatch):
        """Two tokens of the same user must still be distinct session owners."""
        second = vault["store"].create_token(
            user_id=vault["store"].get_user_by_email("unbound@example.com")["id"],
            token=(extra := vs.TOKEN_PREFIX + secrets.token_urlsafe(32)),
        )
        assert second["id"]
        app = build_app(vault, mw.MODE_ON)
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, vault["tokens"]["unbound"])
                other = await raw_tool_call(http, extra, session_id)
                assert other.status_code == 404


class TestRevocationInsideALiveSession:
    """Revoking mid-session: the next request must fail, not the one in flight."""

    async def test_next_call_on_an_established_session_is_refused(self, vault):
        app = build_app(vault, mw.MODE_ON)
        token = vault["tokens"]["revoked"]  # still valid at this point
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, token)
                before = await raw_tool_call(http, token, session_id)
                assert before.status_code == 200

                vault["store"].revoke_token(vault["revoked_id"])

                after = await raw_tool_call(http, token, session_id)
                assert after.status_code == 401, "revocation must apply to new requests"
                assert b"revoked" in after.content

    async def test_reopening_the_get_stream_after_revocation_is_refused(self, vault):
        """The standalone SSE stream is a request too — reconnect must 401."""
        app = build_app(vault, mw.MODE_ON)
        token = vault["tokens"]["revoked"]
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, token)
                vault["store"].revoke_token(vault["revoked_id"])
                resp = await http.get(
                    BASE_URL,
                    headers={
                        "accept": "text/event-stream",
                        "authorization": f"Bearer {token}",
                        "mcp-session-id": session_id,
                    },
                )
                assert resp.status_code == 401

    async def test_the_session_survives_for_the_owner_until_revoked(self, vault):
        app = build_app(vault, mw.MODE_ON)
        token = vault["tokens"]["unbound"]
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, token)
                for _ in range(3):
                    resp = await raw_tool_call(http, token, session_id)
                    assert resp.status_code == 200

    async def test_counterfactual_without_binding_the_session_is_reusable(self, vault, monkeypatch):
        """Why the binding exists — and proof the test above is not decorative.

        With ``scope["user"]`` unset (the pre-fix behavior, reproduced here by
        disabling the identity helper), the SDK records every session owner as
        None and a foreign token rides the session id successfully.
        """
        monkeypatch.setattr(mw, "_session_binding_broken", True)
        app = build_app(vault, mw.MODE_ON)
        async with running(app):
            async with asgi_client(app, None) as http:
                session_id = await raw_initialize(http, vault["tokens"]["unbound"])
                theirs = await raw_tool_call(http, vault["tokens"]["bound"], session_id)
                assert theirs.status_code == 200, (
                    "expected the vulnerable behavior without session binding"
                )


class TestScopeReachesTheBackend:
    """What the backend is actually asked for — not just what authorization allowed."""

    async def test_bound_token_forces_its_own_scope_on_the_backend(self, vault, spy):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {})
        assert "error" not in out
        assert spy.scopes_for("get_all") == ["alice"], (
            "the backend must be queried with the token's bound scope"
        )

    async def test_bound_token_asking_for_its_own_scope_is_passed_through(self, vault, spy):
        app = build_app(vault, mw.MODE_ON)
        await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "alice"})
        assert spy.scopes_for("get_all") == ["alice"]

    async def test_a_refused_call_never_reaches_the_backend(self, vault, spy):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], "get_memories", {"user_id": "bob"})
        assert "bound to user_id 'alice'" in out["error"]
        assert spy.calls == [], "a refused call must not touch mem0 at all"

    async def test_unbound_token_still_reaches_the_default_scope(self, vault, spy):
        """Today's tokens: the client's value wins, exactly as before the vault."""
        app = build_app(vault, mw.MODE_ON)
        await call_tool(app, vault["tokens"]["unbound"], "get_memories", {"user_id": "bob"})
        assert spy.scopes_for("get_all") == ["bob"]

    async def test_unbound_token_with_no_argument_uses_the_configured_default(self, vault, spy):
        app = build_app(vault, mw.MODE_ON)
        await call_tool(app, vault["tokens"]["unbound"], "get_memories", {})
        assert spy.scopes_for("get_all") == ["default-user"]  # MEM0_USER_ID


# =====================================================================
# The 15-tool authorization matrix
#
# The spike proved ONE tool. An independent review pointed out that nine
# others addressed records by id, enumerated globally, or reported on the
# whole server without ever consulting the principal — so a bound token
# would have been isolation on the front door and an open window at the
# back. This section walks every registered tool with an Alice-bound token
# reaching for Bob's data.
# =====================================================================


class TestPolicyCompleteness:
    async def test_every_registered_tool_has_a_declared_policy(self, vault):
        """Tool #16 cannot ship without an authorization decision."""
        from mem0_mcp_selfhosted import server as server_mod

        app = build_app(vault, mw.MODE_ON)
        async with mcp_session(app, vault["tokens"]["unbound"]) as session:
            tools = {t.name for t in (await session.list_tools()).tools}

        declared = set(server_mod.TOOL_SCOPE_POLICY)
        assert tools == declared, (
            f"undeclared: {sorted(tools - declared)}; stale: {sorted(declared - tools)}"
        )

    async def test_policies_are_from_the_known_vocabulary(self):
        from mem0_mcp_selfhosted import server as server_mod

        assert set(server_mod.TOOL_SCOPE_POLICY.values()) <= {
            "scope-arg", "record-owner", "filtered", "operator-only",
        }


class TestScopeArgTools:
    """Tools that take a user_id: a foreign scope is an explicit error."""

    @pytest.mark.parametrize("tool,args", [
        ("add_memory", {"text": "x", "user_id": "bob", "infer": False}),
        ("search_memories", {"query": "x", "user_id": "bob"}),
        ("get_memories", {"user_id": "bob"}),
        ("delete_all_memories", {"user_id": "bob"}),
        ("delete_entities", {"user_id": "bob"}),
    ])
    async def test_bound_token_cannot_reach_another_scope(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], tool, args)
        assert "bound to user_id 'alice'" in json.dumps(out), out
        assert spy.calls == [], f"{tool} touched the backend despite being refused"

    @pytest.mark.parametrize("tool,args", [
        ("search_memories", {"query": "x"}),
        ("get_memories", {}),
    ])
    async def test_omitting_the_scope_uses_the_binding(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        await call_tool(app, vault["tokens"]["bound"], tool, args)
        assert all(kw.get("user_id") == "alice" for _n, kw in spy.calls if "user_id" in kw)


class TestRecordOwnerTools:
    """Tools addressed by id: the record's owner decides."""

    @pytest.mark.parametrize("tool,args", [
        ("get_memory", {"memory_id": BOB_MEMORY}),
        ("memory_history", {"memory_id": BOB_MEMORY}),
        ("delete_memory", {"memory_id": BOB_MEMORY}),
        ("update_memory", {"memory_id": BOB_MEMORY, "text": "hijack"}),
    ])
    async def test_bound_token_cannot_touch_another_users_record(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], tool, args)

        assert "not found" in json.dumps(out), out
        mutations = [n for n in spy.names if n in ("update", "delete", "history")]
        assert mutations == [], f"{tool} reached the backend for a foreign record"

    @pytest.mark.parametrize("tool,args", [
        ("get_memory", {"memory_id": ALICE_MEMORY}),
        ("memory_history", {"memory_id": ALICE_MEMORY}),
        ("delete_memory", {"memory_id": ALICE_MEMORY}),
        ("update_memory", {"memory_id": ALICE_MEMORY, "text": "ok"}),
    ])
    async def test_bound_token_works_on_its_own_records(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], tool, args)
        assert "not accessible" not in json.dumps(out)
        assert "not found" not in json.dumps(out), out

    async def test_a_foreign_record_is_indistinguishable_from_a_missing_one(self, vault, spy):
        """No existence oracle: same answer for someone else's id and a bogus id."""
        foreign = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["bound"], "get_memory",
            {"memory_id": BOB_MEMORY},
        )
        bogus = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["bound"], "get_memory",
            {"memory_id": "33333333-3333-4333-8333-333333333333"},
        )
        assert foreign["error"].split(":")[0] == bogus["error"].split(":")[0]

    async def test_unbound_token_keeps_reaching_every_record(self, vault, spy):
        """Today's behavior must not change for tokens without a binding."""
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(
            app, vault["tokens"]["unbound"], "get_memory", {"memory_id": BOB_MEMORY}
        )
        assert out["user_id"] == "bob"


class TestTaskOwnership:
    async def test_task_status_of_another_users_job_is_hidden(self, vault, spy, monkeypatch):
        monkeypatch.setenv("MEM0_ASYNC_INGEST", "true")
        from mem0_mcp_selfhosted import server as server_mod

        queue, _worker = server_mod._get_ingest()
        job = queue.enqueue(
            user_id="bob", agent_id=None, run_id=None,
            messages=[{"role": "user", "content": "bob's private ingest"}],
        )

        hidden = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["bound"], "memory_task_status",
            {"task_id": job["task_id"]},
        )
        assert "unknown task_id" in json.dumps(hidden)

        seen = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["unbound"], "memory_task_status",
            {"task_id": job["task_id"]},
        )
        assert seen.get("task_id") == job["task_id"]


class TestOperatorOnlyTools:
    @pytest.mark.parametrize("tool,args", [
        ("memory_queue_status", {}),
        ("mcp_search_graph", {"query": "x"}),
        ("mcp_get_entity", {"name": "x"}),
    ])
    async def test_refused_to_a_bound_token(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["bound"], tool, args)
        assert "not available to a scope-bound token" in json.dumps(out), out

    @pytest.mark.parametrize("tool,args", [
        ("memory_queue_status", {}),
    ])
    async def test_allowed_for_an_unbound_token(self, vault, spy, tool, args):
        app = build_app(vault, mw.MODE_ON)
        out = await call_tool(app, vault["tokens"]["unbound"], tool, args)
        assert "not available" not in json.dumps(out)


class TestFilteredTools:
    async def test_list_entities_shows_only_the_bound_scope(self, vault, spy, monkeypatch):
        from mem0_mcp_selfhosted import server as server_mod

        monkeypatch.setattr(
            server_mod, "list_entities_facet",
            lambda _mem: {"users": [{"value": "alice", "count": 3},
                                    {"value": "bob", "count": 9}], "agents": []},
        )
        bound = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["bound"], "list_entities", {}
        )
        assert [u["value"] for u in bound["users"]] == ["alice"]

        unbound = await call_tool(
            build_app(vault, mw.MODE_ON), vault["tokens"]["unbound"], "list_entities", {}
        )
        assert {u["value"] for u in unbound["users"]} == {"alice", "bob"}
