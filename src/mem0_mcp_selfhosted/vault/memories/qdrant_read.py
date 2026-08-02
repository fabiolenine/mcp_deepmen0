"""Leitor do Qdrant para a UI — SÓ LEITURA.

Esta classe não tem, e não deve ganhar, nenhum verbo de escrita: sem ``upsert``,
``delete``, ``set_payload``, ``create_*``. Há um teste que lê o fonte deste
arquivo e reprova se um deles aparecer. A UI é um console de leitura sobre o
corpus de produção; escrita passa pelo MCP, onde vive o pipeline (extração,
classificação, dedup, reforço, vínculo de entidade) que dá sentido ao dado.

O ``QdrantClient`` é síncrono e o cofre roda com ``workers=1``: toda chamada vai
para o threadpool (``anyio.to_thread``). Uma chamada síncrona no handler
bloquearia o processo inteiro — inclusive a tela de login.
"""

from __future__ import annotations

import logging
from typing import Any

import anyio

from mem0_mcp_selfhosted.vault.memories.model import (
    FACET_PROJECTION,
    LIST_PROJECTION,
    MAX_CHAIN_HOPS,
    Cursor,
    chain_ids,
    entity_row,
    facet_counts,
    memory_row,
)

logger = logging.getLogger(__name__)


class QdrantReader:
    """Acesso de leitura às collections de memória e de entidades."""

    def __init__(self, client: Any, collection: str, entity_collection: str, scope: dict[str, str]):
        self._c = client
        self.collection = collection
        self.entity_collection = entity_collection
        #: Escopo exibido pela UI (user_id/agent_id/run_id). Aplicado em toda
        #: consulta: a UI mostra UM escopo, não a soma de todos.
        self.scope = {k: v for k, v in (scope or {}).items() if v}

    # -------------------------------------------------------------- filtros

    def _models(self):
        from qdrant_client import models

        return models

    def _scope_conditions(self) -> list[Any]:
        m = self._models()
        return [
            m.FieldCondition(key=key, match=m.MatchValue(value=value))
            for key, value in self.scope.items()
        ]

    def _build_filter(self, filters: dict[str, Any] | None) -> Any | None:
        """Traduz os filtros da UI para um ``Filter`` do Qdrant.

        Campos sem índice (``domain``, ``memory_type``, ``project``) filtram
        assim mesmo — o Qdrant faz varredura e, no tamanho deste corpus, custa
        de 5 a 40 ms (medido). Não vale mudar o schema da collection por isso.
        """
        m = self._models()
        filters = filters or {}
        must: list[Any] = self._scope_conditions()
        must_not: list[Any] = []

        for key in ("domain", "memory_type", "project", "attributed_to"):
            if filters.get(key):
                must.append(
                    m.FieldCondition(key=key, match=m.MatchValue(value=filters[key]))
                )
        if filters.get("tag"):
            # `tags` é lista no payload; MatchValue casa "contém" em lista.
            must.append(m.FieldCondition(key="tags", match=m.MatchValue(value=filters["tag"])))
        if filters.get("only_superseded"):
            must_not.append(
                m.IsEmptyCondition(is_empty=m.PayloadField(key="superseded_at"))
            )
        if filters.get("has_event_date"):
            must_not.append(m.IsEmptyCondition(is_empty=m.PayloadField(key="event_date")))
        if filters.get("only_documents"):
            must_not.append(m.IsEmptyCondition(is_empty=m.PayloadField(key="source_doc")))

        if not must and not must_not:
            return None
        return m.Filter(must=must or None, must_not=must_not or None)

    # ---------------------------------------------------------------- lista

    async def list_memories(
        self, *, filters: dict[str, Any] | None, cursor: Cursor, page_size: int
    ) -> tuple[list[dict[str, Any]], Cursor | None]:
        """Uma página ordenada por ``created_at`` desc, mais o cursor seguinte.

        O over-fetch existe por causa do empate: ``start_from`` é inclusivo, então
        a chamada seguinte traz de volta todo o grupo do instante do boundary.
        pedimos página + (ids já entregues nesse instante) para que, depois de
        descartá-los, ainda sobre uma página cheia.
        """
        m = self._models()
        qfilter = self._build_filter(filters)
        seen = set(cursor.seen_ids)
        order = m.OrderBy(
            key="created_at",
            direction=m.Direction.DESC,
            **({"start_from": cursor.boundary_ts} if cursor.boundary_ts else {}),
        )

        def _scroll() -> list[Any]:
            points, _ = self._c.scroll(
                self.collection,
                limit=page_size + len(seen) + 1,
                scroll_filter=qfilter,
                order_by=order,
                with_payload=LIST_PROJECTION,
                with_vectors=False,
            )
            return points

        points = await anyio.to_thread.run_sync(_scroll)
        rows = [
            memory_row(str(p.id), p.payload or {})
            for p in points
            if str(p.id) not in seen
        ][:page_size]
        return rows, cursor.advance(rows)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        qfilter = self._build_filter(filters)

        def _count() -> int:
            return self._c.count(self.collection, count_filter=qfilter, exact=True).count

        return await anyio.to_thread.run_sync(_count)

    async def facets(self) -> dict[str, list[dict[str, Any]]]:
        """Contagem por valor das chaves da allowlist, no escopo da UI.

        Um scroll projetado do escopo inteiro (não a ``facet`` API — ver
        ``model.facet_counts`` para o porquê medido). Chamado sempre através do
        cache com TTL.
        """
        qfilter = self._build_filter(None)

        def _scan() -> list[dict[str, Any]]:
            payloads: list[dict[str, Any]] = []
            offset = None
            while True:
                points, offset = self._c.scroll(
                    self.collection,
                    limit=1000,
                    offset=offset,
                    scroll_filter=qfilter,
                    with_payload=FACET_PROJECTION,
                    with_vectors=False,
                )
                payloads.extend(p.payload or {} for p in points)
                if offset is None:
                    return payloads

        payloads = await anyio.to_thread.run_sync(_scan)
        return facet_counts(payloads)

    # --------------------------------------------------------------- ponto

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Payload completo de uma memória, ou None.

        Traz o payload inteiro de propósito: é a única tela onde os campos que a
        whitelist do MCP poda (ACT-R, proveniência, ontologia) ficam visíveis.
        """

        def _retrieve() -> list[Any]:
            return self._c.retrieve(
                self.collection, ids=[memory_id], with_payload=True, with_vectors=False
            )

        points = await anyio.to_thread.run_sync(_retrieve)
        if not points:
            return None
        point = points[0]
        payload = dict(point.payload or {})
        if not self._in_scope(payload):
            return None
        payload["_id"] = str(point.id)
        return payload

    async def get_memories(self, ids: list[str]) -> list[dict[str, Any]]:
        """Linhas de lista para um conjunto de ids (vínculos, cadeia de versões)."""
        if not ids:
            return []

        def _retrieve() -> list[Any]:
            return self._c.retrieve(
                self.collection, ids=ids, with_payload=LIST_PROJECTION, with_vectors=False
            )

        points = await anyio.to_thread.run_sync(_retrieve)
        return [memory_row(str(p.id), p.payload or {}) for p in points]

    async def entities_for_memory(self, memory_id: str, limit: int = 60) -> list[dict[str, Any]]:
        """Entidades vinculadas a uma memória.

        Dois caminhos de vínculo porque o store guarda os dois: a lista canônica
        ``linked_memory_ids`` e uma chave própria ``lnk_<id>`` por vínculo. A
        chave própria existe porque ``set_payload`` faz merge de CHAVES mas
        SUBSTITUI valor de lista — sem ela, dois processos ligando entidades
        diferentes à mesma linha apagariam o vínculo um do outro. Procurar pelos
        dois é o que torna a tela imune a linha cuja lista ficou dessincronizada.
        """
        m = self._models()
        qfilter = m.Filter(
            should=[
                m.FieldCondition(
                    key="linked_memory_ids", match=m.MatchValue(value=memory_id)
                ),
                m.Filter(
                    must_not=[m.IsEmptyCondition(is_empty=m.PayloadField(key=f"lnk_{memory_id}"))]
                ),
            ]
        )

        def _scroll() -> list[Any]:
            points, _ = self._c.scroll(
                self.entity_collection,
                limit=limit,
                scroll_filter=qfilter,
                with_payload=["data", "data_normalized", "entity_type", "linked_memory_ids"],
                with_vectors=False,
            )
            return points

        points = await anyio.to_thread.run_sync(_scroll)
        return [entity_row(str(p.id), p.payload or {}) for p in points]

    async def entity_index(self) -> list[dict[str, Any]]:
        """Todas as entidades do escopo, com projeção mínima.

        Varredura completa em vez de busca no servidor porque ``data_normalized``
        tem índice de palavra-chave (casamento EXATO), não de texto: filtrar por
        pedaço de palavra exigiria criar um índice full-text na collection de
        produção. Com 6267 linhas e projeção de 4 campos isso é barato, e o
        resultado fica em cache; o filtro por substring roda em Python sobre ele.
        Se o store crescer uma ordem de grandeza, trocar por índice ``MatchText``.
        """
        qfilter = self._build_filter(None)

        def _scan() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            offset = None
            while True:
                points, offset = self._c.scroll(
                    self.entity_collection,
                    limit=1000,
                    offset=offset,
                    scroll_filter=qfilter,
                    with_payload=[
                        "data", "data_normalized", "entity_type", "linked_memory_ids"
                    ],
                    with_vectors=False,
                )
                rows.extend(entity_row(str(p.id), p.payload or {}) for p in points)
                if offset is None:
                    return rows

        return await anyio.to_thread.run_sync(_scan)

    async def get_entity(self, point_id: str) -> dict[str, Any] | None:
        def _retrieve() -> list[Any]:
            return self._c.retrieve(
                self.entity_collection, ids=[point_id], with_payload=True, with_vectors=False
            )

        points = await anyio.to_thread.run_sync(_retrieve)
        if not points:
            return None
        payload = dict(points[0].payload or {})
        if not self._in_scope(payload):
            return None
        row = entity_row(str(points[0].id), payload)
        row["payload"] = payload
        return row

    async def version_chain(self, payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Vizinhança de versões: o que esta memória supersede e o que a supersede.

        Caminha para frente (``superseded_by``) e para trás (``supersedes``) com
        teto de saltos e conjunto de visitados — cadeia com ciclo ou id órfão é
        possível no corpus e não pode travar a requisição.
        """
        links = chain_ids(payload)
        seen: set[str] = {str(payload.get("_id") or "")}

        async def walk(start: str | None, key: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            current = start
            for _ in range(MAX_CHAIN_HOPS):
                if not current or current in seen:
                    break
                seen.add(current)
                found = await self.get_memories([current])
                if not found:
                    out.append({"id": current, "missing": True})
                    break
                row = found[0]
                out.append(row)
                full = await self.get_memory(current)
                current = str(chain_ids(full or {}).get(key) or "") or None
            return out

        newer = await walk(links["superseded_by"], "superseded_by")
        older: list[dict[str, Any]] = []
        for old_id in links["supersedes"][:MAX_CHAIN_HOPS]:
            if old_id in seen:
                continue
            seen.add(old_id)
            found = await self.get_memories([old_id])
            older.append(found[0] if found else {"id": old_id, "missing": True})
        return {"newer": newer, "older": older}

    def _in_scope(self, payload: dict[str, Any]) -> bool:
        """O ponto pertence ao escopo que esta UI mostra?

        ``retrieve`` por id não aceita filtro, então o escopo é conferido depois.
        Sem isso, um id de outro escopo colado na URL seria exibido — o filtro da
        listagem valeria só para quem navega, não para quem digita.
        """
        return all(payload.get(key) == value for key, value in self.scope.items())
