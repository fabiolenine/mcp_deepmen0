"""Cliente da busca semântica, falando com o próprio servidor MCP.

Por que a busca NÃO lê o Qdrant direto, ao contrário do resto desta UI: o valor
está no pipeline, não no índice. Uma consulta de produção passa por denso
multilíngue + BM25 português + fusão + over-fetch + reranker cross-encoder +
ativação ACT-R + âncora de evento + penalidade de supersedência. Reimplementar
isso aqui produziria uma segunda ordenação, sempre um pouco diferente da que os
clientes reais recebem — e a UI existe justamente para mostrar o que eles veem.

Usa-se o SDK ``mcp`` e não um POST JSON-RPC à mão porque o transporte
streamable-http tem handshake (``initialize`` + ``notifications/initialized``),
id de sessão em header e resposta que pode chegar como SSE. Reimplementar isso
seria manter um segundo cliente do protocolo, que quebra em silêncio a cada
upgrade do servidor.

Sessão efêmera por busca: abre, chama, fecha. Uma consulta a cada vários
segundos não justifica gerenciar reconexão, expiração de sessão e invalidação —
e a UI é de um operador, não de um agente em laço.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anyio

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0


class McpError(RuntimeError):
    """Falha ao consultar o MCP — vira card na tela, com a mensagem original."""


class McpSearchClient:
    """Chama ``search_memories`` no servidor MCP com o token do cofre."""

    def __init__(self, url: str, token: str, *, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.url = url
        self._token = token
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.url and self._token)

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Executa a busca e devolve o envelope já desserializado.

        ``reinforce`` é forçado a False aqui, e não no chamador: navegar pelo
        console de um operador não é um re-encontro da memória. Deixar o padrão
        do servidor valer faria a própria UI reforçar o que ela exibe — e a
        ativação ACT-R que a tela de detalhe mostra passaria a medir o uso da
        UI, não o uso real.
        """
        payload = {k: v for k, v in params.items() if v is not None and v != ""}
        payload["reinforce"] = False

        try:
            with anyio.fail_after(self.timeout_s):
                return await self._call(payload)
        except TimeoutError as exc:
            raise McpError(
                f"a busca passou de {self.timeout_s:.0f}s sem resposta"
            ) from exc
        except McpError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise McpError(describe_failure(exc)) from exc

    async def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self._token}"}
        async with streamablehttp_client(
            self.url, headers=headers, timeout=self.timeout_s,
            sse_read_timeout=self.timeout_s,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("search_memories", payload)
        return parse_tool_result(result)


def describe_failure(exc: BaseException, _depth: int = 0) -> str:
    """Mensagem útil a partir da exceção que o SDK deixou escapar.

    Sem isto, um token recusado chega ao operador como "ExceptionGroup:
    unhandled errors in a TaskGroup (1 sub-exception)" — MEDIDO contra o
    servidor real. O SDK roda o transporte num task group, então a causa real
    (o 401) fica aninhada, e a mensagem de fora não diz nada sobre o que houve
    nem sobre o que fazer.
    """
    inner = getattr(exc, "exceptions", None)
    if inner and _depth < 5:
        # Um grupo pode conter vários; todos interessam, sem repetir texto.
        vistos: list[str] = []
        for sub in inner:
            texto = describe_failure(sub, _depth + 1)
            if texto not in vistos:
                vistos.append(texto)
        if vistos:
            return " · ".join(vistos)

    text = f"{type(exc).__name__}: {exc}".strip()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return f"401 — o servidor MCP recusou o token do cofre (verifique VAULT_MCP_TOKEN). {text}"
    if status == 403:
        return f"403 — token sem permissão para esta chamada. {text}"
    if isinstance(exc, ConnectionError) or "ConnectError" in type(exc).__name__:
        return f"não foi possível conectar ao servidor MCP. {text}"
    return text


def parse_tool_result(result: Any) -> dict[str, Any]:
    """Extrai o envelope de um ``CallToolResult``.

    O servidor devolve o JSON dentro de um bloco de texto, e sinaliza falha por
    ``isError`` — que NÃO é exceção: sem checar, um erro do servidor viraria
    "nenhum resultado", que é a leitura errada e silenciosa.
    """
    if getattr(result, "isError", False):
        raise McpError(_text_of(result) or "o servidor MCP recusou a chamada")

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        inner = structured.get("result", structured)
        return _as_envelope(inner)

    text = _text_of(result)
    if not text:
        raise McpError("resposta vazia do servidor MCP")
    return _as_envelope(text)


def _text_of(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _as_envelope(raw: Any) -> dict[str, Any]:
    """Normaliza para dicionário — o corpo pode vir como JSON aninhado em texto."""
    value = raw
    for _ in range(3):  # o servidor embrulha o JSON em string; 3 níveis bastam
        if isinstance(value, dict):
            if "result" in value and isinstance(value["result"], (str, dict, list)):
                value = value["result"]
                continue
            return value
        if isinstance(value, list):
            return {"results": value}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise McpError(f"resposta do MCP não é JSON: {value[:160]}") from exc
            continue
        break
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"results": value}
    raise McpError(f"formato inesperado na resposta do MCP: {type(value).__name__}")
