"""Guard de sessão compartilhado pelas telas do cofre.

Vive fora de ``web.py`` para que os mixins de rota (``memories/views.py``)
possam usá-lo sem importar ``web`` — que os importa de volta. É o mesmo
decorador de sempre; só mudou de arquivo.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import Response


def login_required(handler: Callable) -> Callable:
    """Redireciona para /login quem não é admin autenticado.

    ``current_admin`` recusa sessão sem uid, usuário desabilitado, usuário NÃO
    admin e sessão de época antiga — então este decorador é, na prática,
    admin-only. Toda tela nova entra por aqui.
    """

    @wraps(handler)
    async def wrapper(self: Any, request: Request) -> Response:
        if self.current_admin(request) is None:
            return self.redirect(request, "/login")
        return await handler(self, request)

    return wrapper
