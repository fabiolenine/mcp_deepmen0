"""DeepMem0 Vault — user/token credential management for the MCP server.

Two processes share one SQLite file (``vault.db``):

- the MCP server (:8081) *reads* it through ``middleware.BearerTokenMiddleware``
  to authorize requests;
- the vault UI (:8080, ``web.py``) *writes* it — admin creates users, issues,
  rotates and revokes tokens.

``store`` and ``middleware`` import stdlib only, so the MCP server works
without the ``[vault]`` extra installed (enforced by a unit test). The web
layer (``security``/``web``/``main``) pulls the heavier deps and is only
imported by the vault service itself.
"""

from __future__ import annotations

__all__ = ["store", "middleware"]
