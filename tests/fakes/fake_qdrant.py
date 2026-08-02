"""Um Qdrant de mentira que reproduz o comportamento MEDIDO do servidor real.

Os detalhes abaixo não foram inventados: vieram de um spike de leitura contra o
Qdrant de produção (client 1.18). Eles importam porque são exatamente onde uma
paginação ingênua quebra:

- ``scroll`` com ``order_by`` devolve ``next_page_offset = None`` (a paginação
  passa a ser por VALOR, não por offset);
- ``start_from`` é INCLUSIVO: reabrir no último ``created_at`` traz de volta
  todos os pontos daquele mesmo instante;
- ``created_at`` NÃO é único (um instante do corpus real tem 67 pontos).

Um fake que devolvesse offset, ou que tratasse ``start_from`` como exclusivo,
aprovaria um cursor que perde registros em produção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _sortable(value: Any) -> datetime:
    """Normaliza timestamp para comparação.

    Necessário porque ``models.OrderBy`` é um modelo Pydantic: o ``start_from``
    entra como string ISO e SAI como ``datetime``. Comparar o que voltou de lá
    com a string crua do payload levantaria ``TypeError`` — no fake, e só no
    fake, porque no servidor real a comparação acontece do outro lado do fio.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class FakePoint:
    id: str
    payload: dict[str, Any] = field(default_factory=dict)


def point(pid: str, created_at: str, *, user_id: str = "ana", **payload: Any) -> FakePoint:
    return FakePoint(pid, {"created_at": created_at, "user_id": user_id, **payload})


@dataclass
class _Count:
    count: int


class FakeQdrant:
    """Implementa o subconjunto de leitura que o ``QdrantReader`` usa."""

    def __init__(self, collections: dict[str, list[FakePoint]]):
        self.collections = collections
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- filtro ---------------------------------------------------------

    def _matches(self, payload: dict[str, Any], qfilter: Any) -> bool:
        if qfilter is None:
            return True
        for cond in getattr(qfilter, "must", None) or []:
            if not self._match_condition(payload, cond):
                return False
        for cond in getattr(qfilter, "must_not", None) or []:
            if self._match_condition(payload, cond):
                return False
        return True

    def _match_condition(self, payload: dict[str, Any], cond: Any) -> bool:
        key = getattr(cond, "key", None)
        if key is None:  # IsEmptyCondition
            field_ = getattr(cond, "is_empty", None)
            key = getattr(field_, "key", None)
            value = payload.get(key)
            return value is None or value == "" or value == []
        match = getattr(cond, "match", None)
        if match is None:
            return True
        wanted = getattr(match, "value", None)
        actual = payload.get(key)
        if isinstance(actual, (list, tuple)):
            return wanted in actual
        return actual == wanted

    # -- leitura --------------------------------------------------------

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int = 10,
        offset: Any = None,
        scroll_filter: Any = None,
        order_by: Any = None,
        with_payload: Any = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> tuple[list[FakePoint], Any]:
        self.calls.append(("scroll", {"limit": limit, "order_by": order_by}))
        points = [
            p for p in self.collections.get(collection_name, [])
            if self._matches(p.payload, scroll_filter)
        ]

        if order_by is not None:
            key = getattr(order_by, "key", "created_at")
            descending = str(getattr(order_by, "direction", "desc")).lower().endswith("desc")
            # Ordem primária pelo valor; o id só desempata, e desempata sempre na
            # mesma direção — inverter o desempate junto com o valor faria o fake
            # embaralhar grupos empatados a cada mudança de direção.
            points.sort(key=lambda p: p.id)
            points.sort(key=lambda p: _sortable(p.payload.get(key)), reverse=descending)
            start_from = getattr(order_by, "start_from", None)
            if start_from is not None:
                # INCLUSIVO, como o servidor real: o próprio boundary volta.
                bound = _sortable(start_from)
                points = [
                    p
                    for p in points
                    if (
                        _sortable(p.payload.get(key)) <= bound
                        if descending
                        else _sortable(p.payload.get(key)) >= bound
                    )
                ]
            # Com order_by o servidor NÃO devolve cursor de página.
            return self._project(points[:limit], with_payload), None

        points.sort(key=lambda p: p.id)
        if offset is not None:
            points = [p for p in points if p.id >= str(offset)]
        page = points[:limit]
        rest = points[limit:]
        return self._project(page, with_payload), (rest[0].id if rest else None)

    def retrieve(
        self,
        collection_name: str,
        *,
        ids: list[str],
        with_payload: Any = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> list[FakePoint]:
        wanted = {str(i) for i in ids}
        found = [p for p in self.collections.get(collection_name, []) if p.id in wanted]
        return self._project(found, with_payload)

    def count(
        self, collection_name: str, *, count_filter: Any = None, exact: bool = True, **_: Any
    ) -> _Count:
        return _Count(
            sum(
                1
                for p in self.collections.get(collection_name, [])
                if self._matches(p.payload, count_filter)
            )
        )

    def _project(self, points: list[FakePoint], with_payload: Any) -> list[FakePoint]:
        if with_payload is True or with_payload is None:
            return [FakePoint(p.id, dict(p.payload)) for p in points]
        keys = set(with_payload)
        return [
            FakePoint(p.id, {k: v for k, v in p.payload.items() if k in keys}) for p in points
        ]
