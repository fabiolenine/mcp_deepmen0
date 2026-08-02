"""Fábrica das fontes de leitura da UI (Qdrant, fila, histórico, MCP).

Regra de indisponibilidade: **a UI nunca deixa de subir por falta de fonte**. Se
a api-key do Qdrant não estiver no ambiente, ou o serviço estiver fora, quem
falha é a TELA de memórias, com um card explicando o quê — as telas de usuários
e tokens, que são a função original do cofre, continuam funcionando. O oposto
(boot que aborta) transformaria uma feature de leitura em ponto único de falha
de um serviço de credenciais.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mem0_mcp_selfhosted.env import env
from mem0_mcp_selfhosted.vault.memories.cache import TTLCache
from mem0_mcp_selfhosted.vault.memories.qdrant_read import QdrantReader

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
#: O MESMO default do resto do pacote (`config.py`). Fixar aqui o nome da
#: collection de uma instalação concreta faria o pacote sair de fábrica
#: apontando para o corpus de outra pessoa.
DEFAULT_COLLECTION = "mem0_mcp_selfhosted"
#: O MCP escuta em 127.0.0.1:18081. A 8081 é o Caddy na bridge do Docker, que só
#: atende pelo IP da bridge — apontar para lá daqui daria conexão recusada.
DEFAULT_MCP_URL = "http://127.0.0.1:18081/mcp"

FACETS_TTL_S = 300.0
COUNT_TTL_S = 60.0
QUEUE_TTL_S = 5.0
ENTITY_INDEX_TTL_S = 300.0


class SourceUnavailable(RuntimeError):
    """Fonte de leitura ausente ou fora do ar — vira card na tela, nunca 500."""


def _qdrant_api_key() -> str | None:
    """A api-key do Qdrant: ambiente, com fallback no ``.qdrant.env`` do repo.

    Espelha ``scripts/qdrant_auth.py`` (que não é importável daqui — vive em
    ``scripts/``, fora do pacote). O drop-in de systemd põe a chave no ambiente
    do serviço; o fallback serve ao desenvolvimento local.
    """
    value = (os.environ.get("MEM0_QDRANT_API_KEY") or "").strip()
    if value:
        return value
    configured = env("VAULT_QDRANT_ENV_FILE")
    candidate = Path(configured) if configured else Path.home() / "mem0-stack" / ".qdrant.env"
    try:
        with open(candidate, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("MEM0_QDRANT_API_KEY="):
                    parsed = line.split("=", 1)[1]
                    if parsed:
                        return parsed
    except OSError:
        return None
    return None


def _scope() -> dict[str, str]:
    """Escopo de memória que a UI apresenta.

    Uma tela que somasse escopos misturaria corpora de projetos diferentes numa
    lista só; a UI mostra um, e o padrão é o MESMO que as tools do MCP usam
    (``get_default_user_id``) — a UI não pode enxergar um escopo diferente do
    que os clientes escrevem.
    """
    from mem0_mcp_selfhosted.helpers import get_default_user_id

    scope = {"user_id": env("VAULT_MEMORY_USER_ID") or get_default_user_id()}
    for key, var in (("agent_id", "VAULT_MEMORY_AGENT_ID"), ("run_id", "VAULT_MEMORY_RUN_ID")):
        value = env(var)
        if value:
            scope[key] = value
    return scope


@dataclass
class Sources:
    """Fontes de leitura da UI. Campos ausentes = superfície indisponível."""

    collection: str
    entity_collection: str
    scope: dict[str, str]
    qdrant: QdrantReader | None = None
    qdrant_error: str = ""
    queue_db: Path | None = None
    history_db: Path | None = None
    mcp: Any | None = None
    mcp_error: str = ""
    cache: TTLCache = field(default_factory=TTLCache)

    def require_qdrant(self) -> QdrantReader:
        if self.qdrant is None:
            raise SourceUnavailable(self.qdrant_error or "Qdrant indisponível")
        return self.qdrant

    def require_mcp(self) -> Any:
        if self.mcp is None:
            raise SourceUnavailable(self.mcp_error or "MCP indisponível")
        return self.mcp

    async def facets(self) -> dict[str, list[dict[str, Any]]]:
        reader = self.require_qdrant()
        return await self.cache.get_or_build("facets", FACETS_TTL_S, reader.facets)

    async def total(self) -> int:
        reader = self.require_qdrant()
        return await self.cache.get_or_build("count:all", COUNT_TTL_S, reader.count)


def build_sources() -> Sources:
    """Monta as fontes a partir do ambiente. Nunca levanta."""
    collection = env("VAULT_QDRANT_COLLECTION") or env("MEM0_COLLECTION") or DEFAULT_COLLECTION
    entity_collection = env("VAULT_QDRANT_ENTITY_COLLECTION") or f"{collection}_entities"
    scope = _scope()
    sources = Sources(
        collection=collection, entity_collection=entity_collection, scope=scope
    )

    try:
        from qdrant_client import QdrantClient

        api_key = _qdrant_api_key()
        if not api_key:
            raise SourceUnavailable(
                "MEM0_QDRANT_API_KEY ausente (o Qdrant deste host exige api-key) — "
                "instale o drop-in qdrant-auth.conf no deepmem0-vault.service"
            )
        url = env("VAULT_QDRANT_URL") or DEFAULT_QDRANT_URL
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        sources.qdrant = QdrantReader(client, collection, entity_collection, scope)
    except Exception as exc:  # noqa: BLE001 — indisponibilidade é informação, não crash
        sources.qdrant_error = str(exc)
        logger.warning("vault memories: Qdrant indisponível (%s)", exc)

    mcp_url = env("VAULT_MCP_URL") or DEFAULT_MCP_URL
    mcp_token = env("VAULT_MCP_TOKEN")
    if mcp_token:
        from mem0_mcp_selfhosted.vault.memories.mcp_client import McpSearchClient

        sources.mcp = McpSearchClient(mcp_url, mcp_token)
    else:
        # Sem token a busca não sobe, e isso é dito na tela. Um token do cofre é
        # o que torna a UI um cliente auditável como qualquer outro, em vez de
        # um caminho privilegiado por dentro.
        sources.mcp_error = (
            "VAULT_MCP_TOKEN ausente — emita um token do cofre "
            "(`deepmem0-vault issue-token --label vault-ui`) e ponha em .vault.env"
        )

    queue_db = Path(env("VAULT_INGEST_DB") or (Path.home() / ".mem0" / "ingest_queue.db"))
    sources.queue_db = queue_db if queue_db.exists() else None
    history_db = Path(env("VAULT_HISTORY_DB") or (Path.home() / ".mem0" / "history.db"))
    sources.history_db = history_db if history_db.exists() else None

    return sources
