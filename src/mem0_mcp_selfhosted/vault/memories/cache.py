"""Cache com TTL para agregações caras (facetas, contagens, índice de entidades).

INVARIANTE (testado): a chave NUNCA deriva do request. Todas as chaves vêm de um
conjunto fixo, declarado em código. Um cache cuja chave é texto do usuário cresce
sem limite e vira um vetor de memória — e a evicção que se acrescenta depois nunca
é tão simples quanto não ter deixado entrar. Filtro de texto (``?q=``) é aplicado
em Python SOBRE o valor cacheado, não como parte da chave.

O lock é por chave para que N requisições concorrentes na mesma chave fria façam
UMA construção, não N (o scroll completo do corpus é barato, mas não é grátis).
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

import anyio


class TTLCache:
    """Cache assíncrono chave→valor com expiração por tempo monotônico."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self._locks: dict[str, anyio.Lock] = {}

    def _lock_for(self, key: str) -> anyio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = anyio.Lock()
            self._locks[key] = lock
        return lock

    def peek(self, key: str) -> Any | None:
        """Valor ainda válido, ou None. Não constrói nada."""
        hit = self._values.get(key)
        if hit is None:
            return None
        expires_at, value = hit
        if expires_at < time.monotonic():
            return None
        return value

    async def get_or_build(
        self, key: str, ttl_s: float, builder: Callable[[], Awaitable[Any]]
    ) -> Any:
        cached = self.peek(key)
        if cached is not None:
            return cached
        async with self._lock_for(key):
            # Reconferir sob o lock: quem esperou pode ter ganhado o valor pronto.
            cached = self.peek(key)
            if cached is not None:
                return cached
            value = await builder()
            self._values[key] = (time.monotonic() + ttl_s, value)
            return value

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._values.clear()
        else:
            self._values.pop(key, None)
