"""Handlers das telas de memória, como mixin sobre ``VaultWeb``.

Mixin em vez de módulo de rotas solto para herdar de graça o que a UI do cofre
já resolveu: render com contexto de i18n, headers endurecidos, sessão e o
decorador de login. As rotas só existem se o admin estiver logado — o mesmo
guard das telas de credencial, sem exceção.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import anyio
from starlette.requests import Request
from starlette.responses import Response

from mem0_mcp_selfhosted.vault.guards import login_required
from mem0_mcp_selfhosted.vault.memories import model
from mem0_mcp_selfhosted.vault.memories.sources import (
    ENTITY_INDEX_TTL_S,
    QUEUE_TTL_S,
    Sources,
    SourceUnavailable,
)

logger = logging.getLogger(__name__)

#: Tetos de renderização. Não são paginação: a lista de entidades é para
#: encontrar uma linha, não para varrer 6 mil. O total casado é sempre exibido,
#: para que o corte NUNCA se pareça com "só existem estas".
MAX_ENTITY_ROWS = 200
MAX_ENTITY_LINKS = 100


def _sources(request: Request) -> Sources:
    return request.app.state.sources


class MemoriesRoutes:
    """Telas de leitura do corpus. Espera a interface de ``VaultWeb``."""

    @login_required
    async def memories_list(self, request: Request) -> Response:
        sources = _sources(request)
        filters = model.clean_filters(request.query_params)
        cursor = model.Cursor.decode(request.query_params.get("cursor"))
        try:
            reader = sources.require_qdrant()
            rows, next_cursor = await reader.list_memories(
                filters=filters, cursor=cursor, page_size=model.DEFAULT_PAGE_SIZE
            )
            facets = await sources.facets()
            total = await sources.total()
            filtered = await reader.count(filters) if filters else total
            error = ""
        except SourceUnavailable as exc:
            rows, next_cursor, facets, total, filtered = [], None, {}, 0, 0
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — a tela informa, não estoura
            logger.exception("vault memories: falha ao listar")
            rows, next_cursor, facets, total, filtered = [], None, {}, 0, 0
            error = f"{type(exc).__name__}: {exc}"

        return self.render(
            request,
            "memories.html",
            nav="memories",
            rows=rows,
            facets=facets,
            facet_keys=model.FACET_ALLOWLIST,
            filters=filters,
            total=total,
            filtered=filtered,
            scope=sources.scope,
            source_error=error,
            next_cursor=next_cursor.encode() if next_cursor else "",
            has_prev=not cursor.is_first_page,
            base_query=model.filters_query(filters),
            page_size=model.DEFAULT_PAGE_SIZE,
            # Links de faceta: aplicar/remover um filtro preservando os outros.
            # O cursor NÃO é preservado de propósito — trocar o filtro muda o
            # conjunto, e uma posição do conjunto antigo não significa nada no novo.
            setf=lambda key, value: model.filters_query(filters, **{key: value}),
            drop=lambda key: model.filters_query(filters, **{key: None}),
        )

    @login_required
    async def search_page(self, request: Request) -> Response:
        """Formulário de busca. Os resultados chegam por HTMX em /search/results."""
        sources = _sources(request)
        return self.render(
            request,
            "search.html",
            nav="search",
            form=dict(request.query_params),
            scope=sources.scope,
            mcp_error=sources.mcp_error,
        )

    @login_required
    async def search_results(self, request: Request) -> Response:
        """Executa a busca pelo caminho de produção e renderiza o parcial.

        Sempre 200, mesmo em falha: o alvo é um fragmento dentro da página, e um
        500 aqui apagaria a tela do operador em vez de explicar o que houve.
        """
        sources = _sources(request)
        args, warnings = model.search_params(request.query_params, sources.scope)
        envelope: dict[str, Any] = {"results": []}
        error = ""
        elapsed_ms = 0

        if not (args.get("query") or "").strip():
            return self.render(
                request, "partials/search_results.html", ran=False, warnings=warnings,
                envelope=envelope, error="", elapsed_ms=0,
            )

        started = time.monotonic()
        try:
            client = sources.require_mcp()
            envelope = model.search_envelope(await client.search(args))
        except SourceUnavailable as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — inclusive McpError
            logger.warning("vault memories: busca falhou (%s)", exc)
            error = str(exc)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        return self.render(
            request, "partials/search_results.html", ran=True, warnings=warnings,
            envelope=envelope, error=error, elapsed_ms=elapsed_ms,
        )

    async def corpus_tiles(self, request: Request) -> dict[str, Any]:
        """Números do corpus para o painel. Nunca levanta: painel degrada em '—'.

        Sem decorador de login: é chamado de DENTRO de um handler que já o tem.
        """
        sources = _sources(request)
        tiles: dict[str, Any] = {
            "memories": None, "entities": None, "queue_depth": None,
            "queue_dead": None, "ok": False, "error": "",
        }
        try:
            tiles["memories"] = await sources.total()
            tiles["ok"] = True
        except Exception as exc:  # noqa: BLE001
            tiles["error"] = str(exc)

        try:
            reader = sources.require_qdrant()
            index = await sources.cache.get_or_build(
                "entity_index", ENTITY_INDEX_TTL_S, reader.entity_index
            )
            tiles["entities"] = len(index)
        except Exception:  # noqa: BLE001 — já reportado pelo tile de memórias
            pass

        try:
            state = await self._queue_state(sources)
            tiles["queue_depth"] = state["summary"].get("depth")
            tiles["queue_dead"] = state["summary"].get("by_status", {}).get("dead", 0)
        except Exception:  # noqa: BLE001
            pass
        return tiles

    @login_required
    async def entities_list(self, request: Request) -> Response:
        sources = _sources(request)
        query = (request.query_params.get("q") or "").strip()
        kind = (request.query_params.get("entity_type") or "").strip()
        try:
            reader = sources.require_qdrant()
            index = await sources.cache.get_or_build(
                "entity_index", ENTITY_INDEX_TTL_S, reader.entity_index
            )
            error = ""
        except SourceUnavailable as exc:
            index, error = [], str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vault memories: entidades indisponíveis (%s)", exc)
            index, error = [], str(exc)

        rows = model.filter_entities(index, query=query, entity_type=kind)
        return self.render(
            request,
            "entities.html",
            nav="entities",
            rows=rows[:MAX_ENTITY_ROWS],
            shown=min(len(rows), MAX_ENTITY_ROWS),
            matched=len(rows),
            total=len(index),
            kinds=model.entity_kinds(index),
            q=query,
            entity_type=kind,
            source_error=error,
        )

    @login_required
    async def entity_detail(self, request: Request) -> Response:
        sources = _sources(request)
        point_id = request.path_params["point_id"]
        if not model.is_point_id(point_id):
            return self.render(
                request, "entity.html", nav="entities", status_code=404,
                entity=None, not_found=True, source_error="", memories=[],
            )
        try:
            reader = sources.require_qdrant()
            entity = await reader.get_entity(point_id)
        except SourceUnavailable as exc:
            return self.render(
                request, "entity.html", nav="entities", entity=None,
                not_found=False, source_error=str(exc), memories=[],
            )
        if entity is None:
            return self.render(
                request, "entity.html", nav="entities", status_code=404,
                entity=None, not_found=True, source_error="", memories=[],
            )

        memories = await reader.get_memories(entity["linked_memory_ids"][:MAX_ENTITY_LINKS])
        found = {row["id"] for row in memories}
        missing = [i for i in entity["linked_memory_ids"] if i not in found]
        return self.render(
            request,
            "entity.html",
            nav="entities",
            entity=entity,
            memories=memories,
            missing=missing,
            raw_json=json.dumps(
                model.entity_raw(entity["payload"]), ensure_ascii=False, indent=2, default=str
            ),
            not_found=False,
            source_error="",
        )

    async def _queue_state(self, sources: Sources) -> dict[str, Any]:
        """Resumo + jobs da fila, com cache curto (o parcial faz polling)."""
        if not sources.queue_db:
            raise SourceUnavailable(
                "banco da fila não encontrado (~/.mem0/ingest_queue.db)"
            )
        from mem0_mcp_selfhosted.vault.memories.local_ro import QueueReader

        reader = QueueReader(sources.queue_db)

        async def _build() -> dict[str, Any]:
            summary = await reader.summary()
            active = await reader.jobs()
            dead = await reader.jobs(statuses=("dead",), limit=20)
            return {"summary": summary, "active": active, "dead": dead}

        return await sources.cache.get_or_build("queue", QUEUE_TTL_S, _build)

    @login_required
    async def queue_page(self, request: Request) -> Response:
        sources = _sources(request)
        try:
            state = await self._queue_state(sources)
            error = ""
        except SourceUnavailable as exc:
            state, error = {"summary": {}, "active": [], "dead": []}, str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vault memories: fila indisponível (%s)", exc)
            state, error = {"summary": {}, "active": [], "dead": []}, str(exc)
        return self.render(
            request, "queue.html", nav="queue", source_error=error, **state
        )

    @login_required
    async def queue_summary(self, request: Request) -> Response:
        """Fragmento recarregado por polling — sempre 200, como o de busca."""
        sources = _sources(request)
        try:
            state = await self._queue_state(sources)
            error = ""
        except Exception as exc:  # noqa: BLE001
            state, error = {"summary": {}, "active": [], "dead": []}, str(exc)
        return self.render(
            request, "partials/queue_summary.html", source_error=error, **state
        )

    @login_required
    async def memory_detail(self, request: Request) -> Response:
        sources = _sources(request)
        memory_id = request.path_params["memory_id"]
        if not model.is_point_id(memory_id):
            return self.render(
                request, "memory.html", nav="memories", status_code=404,
                memory=None, not_found=True, source_error="",
            )

        try:
            reader = sources.require_qdrant()
            payload = await reader.get_memory(memory_id)
        except SourceUnavailable as exc:
            return self.render(
                request, "memory.html", nav="memories", memory=None,
                not_found=False, source_error=str(exc),
            )

        if payload is None:
            # Registro de outro escopo e registro inexistente respondem igual:
            # distinguir os dois daria um oráculo de existência de graça.
            return self.render(
                request, "memory.html", nav="memories", status_code=404,
                memory=None, not_found=True, source_error="",
            )

        # As quatro leituras seguintes são independentes: em série somariam as
        # latências; num task group pagam a maior. Cada uma escreve no seu slot.
        extra: dict[str, Any] = {"chain": {"newer": [], "older": []}, "entities": [],
                                 "history": [], "delete_intent": None, "job": None}

        async def _chain() -> None:
            extra["chain"] = await reader.version_chain(payload)

        async def _entities() -> None:
            extra["entities"] = await reader.entities_for_memory(memory_id)

        async def _history() -> None:
            if sources.history_db:
                from mem0_mcp_selfhosted.vault.memories.local_ro import HistoryReader

                hist = HistoryReader(sources.history_db)
                extra["history"] = await hist.for_memory(memory_id)
                extra["delete_intent"] = await hist.delete_intent(memory_id)

        async def _job() -> None:
            task_id = payload.get("task_id")
            if task_id and sources.queue_db:
                from mem0_mcp_selfhosted.vault.memories.local_ro import QueueReader

                extra["job"] = await QueueReader(sources.queue_db).job(str(task_id))

        errors: list[str] = []

        async def _guarded(fn) -> None:
            """Um bloco que falha não pode levar a tela inteira junto."""
            try:
                await fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("vault memories: bloco do detalhe falhou (%s)", exc)
                errors.append(f"{type(exc).__name__}: {exc}")

        async with anyio.create_task_group() as tg:
            for block in (_chain, _entities, _history, _job):
                tg.start_soon(_guarded, block)

        return self.render(
            request,
            "memory.html",
            nav="memories",
            memory=model.memory_row(memory_id, payload),
            payload=payload,
            full_text=payload.get("data") or "",
            actr=model.actr_view(payload),
            provenance=model.provenance_view(payload),
            links=model.chain_ids(payload),
            # Serializado aqui e não no template: `tojson` do Jinja escaparia
            # para contexto de script, que não é onde isto vai (é <pre>).
            raw_json=json.dumps(
                model.raw_payload(payload), ensure_ascii=False, indent=2, default=str
            ),
            chain=extra["chain"],
            entities=extra["entities"],
            history=extra["history"],
            delete_intent=extra["delete_intent"],
            job=extra["job"],
            block_errors=errors,
            not_found=False,
            source_error="",
        )
