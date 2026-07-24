"""The MCP server must authorize requests without the ``[vault]`` extra.

The UI needs starlette/jinja2/itsdangerous/argon2; the gate must not. This is
a deployment invariant, not a style preference: the :8081 venv is expected to
carry the auth path with nothing new installed, and a stray import would only
surface as a crash on the box.
"""

import ast
import sys
from pathlib import Path

import pytest

from mem0_mcp_selfhosted.vault import middleware, store

VAULT_ONLY_DEPS = {"starlette", "jinja2", "itsdangerous", "argon2", "uvicorn"}

STDLIB = set(sys.stdlib_module_names)
ALLOWED_LOCAL = {"mem0_mcp_selfhosted"}
# middleware.py may touch the MCP SDK (it exists wherever this server runs) to
# bind sessions to credentials; store.py may not — it stays portable.
ALLOWED_PER_MODULE = {"middleware": {"mcp"}}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module", [store, middleware])
def test_only_stdlib_and_own_package(module):
    short = module.__name__.rsplit(".", 1)[-1]
    roots = imported_roots(Path(module.__file__))
    foreign = roots - STDLIB - ALLOWED_LOCAL - ALLOWED_PER_MODULE.get(short, set())
    assert not foreign, f"{module.__name__} imports non-stdlib {sorted(foreign)}"


def test_store_is_strictly_stdlib():
    """The portable half: no MCP SDK, no framework, nothing but Python."""
    assert not (imported_roots(Path(store.__file__)) - STDLIB - ALLOWED_LOCAL)


@pytest.mark.parametrize("module", [store, middleware])
def test_no_vault_extra_dependency_is_reachable(module):
    """Even indirectly: importing the gate must not pull the UI stack."""
    roots = imported_roots(Path(module.__file__))
    assert not (roots & VAULT_ONLY_DEPS)


def test_gate_works_when_the_extra_is_hidden(tmp_path, monkeypatch):
    """Import and verify a token with the [vault] deps blocked at import time."""
    import secrets

    for name in list(sys.modules):
        if name.split(".")[0] in VAULT_ONLY_DEPS:
            monkeypatch.delitem(sys.modules, name, raising=False)

    class Blocker:
        def find_module(self, fullname, path=None):
            if fullname.split(".")[0] in VAULT_ONLY_DEPS:
                raise ImportError(f"{fullname} is not installed (simulated)")
            return None

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    from mem0_mcp_selfhosted.vault import store as fresh_store

    db = tmp_path / "vault.db"
    s = fresh_store.VaultStore(db)
    uid = s.create_user(email="nodeps@example.com")
    token = fresh_store.TOKEN_PREFIX + secrets.token_urlsafe(32)
    s.create_token(user_id=uid, token=token)
    assert s.verify_token(token)["status"] == fresh_store.OK
    assert fresh_store.probe(db) == "ok"


def test_sse_transport_refuses_to_boot_with_auth_enabled(monkeypatch):
    """The legacy SSE app is not wrapped — a silent bypass must become a crash."""
    from mem0_mcp_selfhosted import server as server_mod

    monkeypatch.setenv("MEM0_TRANSPORT", "sse")
    monkeypatch.setenv("MEM0_REQUIRE_AUTH", "shadow")
    monkeypatch.setattr(server_mod, "_create_server", lambda: object())
    monkeypatch.setattr(server_mod, "_async_ingest_enabled", lambda: False)

    with pytest.raises(RuntimeError, match="sse"):
        server_mod.run_server()


def test_sse_transport_still_works_with_auth_off(monkeypatch):
    from mem0_mcp_selfhosted import server as server_mod

    class FakeServer:
        def __init__(self):
            self.ran = None

        def run(self, transport):
            self.ran = transport

    fake = FakeServer()
    monkeypatch.setenv("MEM0_TRANSPORT", "sse")
    monkeypatch.delenv("MEM0_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(server_mod, "_create_server", lambda: fake)
    monkeypatch.setattr(server_mod, "_async_ingest_enabled", lambda: False)

    server_mod.run_server()
    assert fake.ran == "sse"
