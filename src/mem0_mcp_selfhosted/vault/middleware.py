"""Bearer-token gate in front of the MCP app (stdlib only, pure ASGI).

Three modes, switched by ``MEM0_REQUIRE_AUTH``:

- ``off``    — delegate without even opening the vault database (today's
               behavior; zero risk while the vault is being populated).
- ``shadow`` — verify and *record*, never block: valid tokens get their
               ``last_used_at`` touched (that is the positive evidence the
               rollout requires before flipping to ``on``) and anything
               unauthorized logs a throttled WARNING.
- ``on``     — unauthorized requests get 401 before they reach the MCP
               session manager.

Written as raw ASGI rather than Starlette's ``BaseHTTPMiddleware`` on purpose:
BaseHTTPMiddleware buffers the response body, which would break streamable-HTTP
SSE. ``lifespan`` and websocket scopes are delegated untouched.

Semantics worth knowing: revocation applies to *new requests*. An SSE stream
already open is not torn down, but every MCP tool call is a fresh POST, so a
revoked token stops working on the client's next call.
"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from mem0_mcp_selfhosted.vault import store as vault_store

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ON = "on"
VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_ON)

DEFAULT_EXEMPT_PATHS = ("/health", "/healthz")

#: Authenticated principal for the current request, when one was resolved.
#: Read by server.py's authorization contract.
current_principal: ContextVar[dict[str, Any] | None] = ContextVar(
    "vault_current_principal", default=None
)

_WARN_THROTTLE_S = 60.0
#: Denials are buffered in memory and flushed at most this often, so a flood of
#: unauthorized requests costs a bounded number of writes. At most one window's
#: worth of counts is lost if the process dies — the decision this feeds is
#: measured in days.
_DENIAL_FLUSH_S = 60.0

#: Set once if the SDK's auth types can't be built (version drift). Session
#: binding is hardening on top of authentication — losing it must never turn
#: into losing the gate, so we degrade loudly and keep verifying.
_session_binding_broken = False


def _session_identity(principal: dict[str, Any]) -> Any | None:
    """Wrap the principal in the type the MCP SDK compares sessions against.

    The SDK binds an MCP session to the credential that created it only when
    ``scope["user"]`` is an ``AuthenticatedUser``; with anything else every
    session owner is ``None`` and *any* valid token could reuse *any* leaked
    session id. Imported lazily so this module still loads (and the gate still
    works) if the SDK moves these types.

    The ``token`` field carries the PREFIX, never the plaintext: it exists to
    identify, and nothing downstream re-verifies it — no reason to spread the
    secret through the ASGI scope and into every traceback.
    """
    global _session_binding_broken
    if _session_binding_broken:
        return None
    try:
        from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
        from mcp.server.auth.provider import AccessToken

        return AuthenticatedUser(
            AccessToken(
                token=principal["prefix"],
                client_id=f"vault:{principal['token_id']}",
                scopes=[],
                subject=principal["mem0_user_id"] or principal["email"],
            )
        )
    except Exception as exc:  # noqa: BLE001
        _session_binding_broken = True
        logger.error(
            "MCP session binding unavailable (%s) — tokens still authorize, but a "
            "leaked session id could be reused by another token", exc,
        )
        return None


def normalize_mode(raw: str) -> str:
    """Parse MEM0_REQUIRE_AUTH. Unset/empty = off; a typo is a hard error.

    Failing fast at boot beats silently serving unauthenticated traffic
    because someone wrote ``MEM0_REQUIRE_AUTH=true``.
    """
    mode = (raw or "").strip().lower()
    if not mode:
        return MODE_OFF
    if mode not in VALID_MODES:
        raise ValueError(
            f"MEM0_REQUIRE_AUTH={raw!r} is not one of {'/'.join(VALID_MODES)}"
        )
    return mode


class BearerTokenMiddleware:
    def __init__(
        self,
        app: Callable,
        *,
        db_path: str | Path,
        mode: str = MODE_OFF,
        exempt_paths: tuple[str, ...] = DEFAULT_EXEMPT_PATHS,
    ):
        self.app = app
        self.db_path = str(db_path)
        self.mode = mode
        self.exempt_paths = exempt_paths
        self._store: vault_store.VaultStore | None = None
        self._store_error: str | None = None
        self._last_warn: dict[str, float] = {}
        self._denials: dict[str, int] = {}
        self._last_flush = 0.0

    # ------------------------------------------------------------------

    def _get_store(self) -> vault_store.VaultStore | None:
        """Open the vault lazily. None = unusable (missing/corrupt/incompatible)."""
        if self._store is not None:
            return self._store
        try:
            if not Path(self.db_path).exists():
                # Never create it here: an empty vault would authorize nobody
                # anyway, and the file belongs to the vault service.
                self._store_error = "missing"
                return None
            self._store = vault_store.VaultStore(self.db_path, create=False)
            self._store_error = None
        except vault_store.SchemaIncompatible as exc:
            self._store_error = f"schema_incompatible: {exc}"
            return None
        except Exception as exc:  # noqa: BLE001
            self._store_error = f"error: {exc}"
            return None
        return self._store

    def _warn(self, key: str, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_warn.get(key, 0.0) < _WARN_THROTTLE_S:
            return
        self._last_warn[key] = now
        logger.warning(msg, *args)

    def _note_denial(self, store, reason: str, path: str, client: str) -> None:
        """Durable count of unauthorized attempts (the promotion gate reads it).

        Telemetry must never be able to fail a request, so every error here is
        swallowed after a debug line.
        """
        self._denials[reason] = self._denials.get(reason, 0) + 1
        now = time.monotonic()
        if self._last_flush and now - self._last_flush < _DENIAL_FLUSH_S:
            return
        self._last_flush = now
        pending, self._denials = self._denials, {}
        try:
            store.record_denials(pending, path=path, client=client)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not persist denial counters: %s", exc)

    def _authorization_headers(self, scope: dict) -> list[str]:
        return [
            v.decode("latin-1")
            for k, v in scope.get("headers", [])
            if k == b"authorization"
        ]

    async def _deny(self, send: Callable[[dict], Awaitable[None]], reason: str) -> None:
        body = json.dumps({"error": "unauthorized", "reason": reason}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="DeepMem0", error="invalid_token"'),
                (b"cache-control", b"no-store"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    # ------------------------------------------------------------------

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # lifespan/websocket and the off switch: delegate, never touch the DB.
        if scope.get("type") != "http" or self.mode == MODE_OFF:
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in self.exempt_paths:
            return await self.app(scope, receive, send)

        store = self._get_store()
        if store is None:
            if self.mode == MODE_ON:
                # Fail closed, and make the operator's fix obvious in the log.
                logger.critical(
                    "VAULT UNAVAILABLE (%s) at %s — denying every request while "
                    "MEM0_REQUIRE_AUTH=on. Fix the vault database or set the "
                    "drop-in back to shadow.",
                    self._store_error, self.db_path,
                )
                return await self._deny(send, "vault_unavailable")
            self._warn(
                "store", "vault unavailable (%s) at %s — shadow mode, passing through",
                self._store_error, self.db_path,
            )
            return await self.app(scope, receive, send)

        headers = self._authorization_headers(scope)
        token = vault_store.parse_bearer(headers)
        if token is None:
            status = "missing" if not headers else vault_store.MALFORMED
            principal = None
        else:
            try:
                principal = store.verify_token(token)
                status = principal["status"]
            except Exception as exc:  # noqa: BLE001
                # The vault broke AFTER we opened it (deleted, corrupted,
                # permissions changed, lock never released). Opening it again
                # is the recovery path, so drop the cached handle.
                self._store, self._store_error = None, f"error: {exc}"
                if self.mode == MODE_ON:
                    logger.critical(
                        "VAULT READ FAILED (%s) at %s — denying while "
                        "MEM0_REQUIRE_AUTH=on", exc, self.db_path,
                    )
                    return await self._deny(send, "vault_unavailable")
                # shadow must NEVER block: a broken vault is an operator
                # problem, not the MCP client's.
                self._warn("verify", "vault read failed in shadow (%s) — passing through", exc)
                return await self.app(scope, receive, send)

        if status != vault_store.OK:
            client = scope.get("client") or ("?", 0)
            self._note_denial(store, status, path, str(client[0]))
            if self.mode == MODE_ON:
                logger.info(
                    "vault denied %s %s from %s: %s",
                    scope.get("method"), path, client[0], status,
                )
                return await self._deny(send, status)
            self._warn(
                f"unauth:{status}",
                "SHADOW: unauthorized MCP request (%s) %s %s from %s — would be "
                "401 once MEM0_REQUIRE_AUTH=on",
                status, scope.get("method"), path, client[0],
            )
            return await self.app(scope, receive, send)

        # Authorized: expose the principal both ways. The contextvar is the
        # ergonomic path; scope state survives task hops the SDK may make.
        state = scope.setdefault("state", {})
        state["vault_user"] = principal
        identity = _session_identity(principal)
        if identity is not None:
            # Ties the MCP session to THIS token: another token presenting the
            # same session id is answered as if the session did not exist.
            scope["user"] = identity
        reset_token = current_principal.set(principal)
        try:
            return await self.app(scope, receive, send)
        finally:
            current_principal.reset(reset_token)
