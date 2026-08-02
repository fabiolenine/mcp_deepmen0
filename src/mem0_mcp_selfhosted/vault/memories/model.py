"""Presenters puros do read model: cursor, projeções, facetas, view-models.

Sem I/O — tudo aqui é testável com dicionários. O que fala com o Qdrant vive em
``qdrant_read.py``; o que decide o que MOSTRAR vive aqui.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import zlib
from typing import Any, Iterable

# --------------------------------------------------------------- projeções

#: Payload trazido na LISTA. Projetar é o que mantém a página barata: o payload
#: cheio traz `data` inteiro + `text_lemmatized` (uma segunda cópia do texto),
#: e nenhum dos dois cabe numa linha de tabela.
LIST_PROJECTION = [
    "data",
    "created_at",
    "importance",
    "domain",
    "memory_type",
    "project",
    "tags",
    "attributed_to",
    "actor_id",
    "event_date",
    "superseded_at",
    "superseded_by",
    "source_doc",
    "page_start",
    "page_end",
    "content_type",
    "memory_scope",
]

#: Campos usados para montar as facetas. Um scroll com esta projeção custou
#: ~36 ms / ~107 KiB no corpus de 1236 pontos (medido).
FACET_PROJECTION = ["domain", "memory_type", "project", "attributed_to"]

#: Chaves que viram filtro/faceta na UI. O corpus tem ~25 chaves ad-hoc de
#: ingestões manuais (doc, tipo, board, camada...) que, expostas, viram ruído
#: com uma opção cada. A allowlist é o que separa ontologia de resíduo.
FACET_ALLOWLIST = ("domain", "memory_type", "project", "attributed_to")

#: Filtros booleanos derivados de presença de campo (não de valor).
FLAG_FILTERS = ("only_superseded", "has_event_date", "only_documents")

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 25

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def is_point_id(value: str) -> bool:
    """Aceita o formato de id de ponto usado nas duas collections.

    MEDIDO: os 1236 pontos da collection principal e os 6267 da de entidades são
    TODOS UUID em string, sem exceção — as memórias com uuid4, as entidades com
    uuid5 determinístico de (escopo, chave). O Qdrant também aceitaria id
    inteiro, e por isso a verificação foi feita antes de fixar esta regra:
    validar como UUID não fecha porta nenhuma neste corpus, e fecha a porta de
    mandar texto arbitrário do path adiante.
    """
    return bool(_UUID_RE.match(value or ""))


# ----------------------------------------------------------------- cursor


class Cursor:
    """Posição na listagem ordenada por ``created_at`` desc.

    Por que não é um offset: ``scroll`` com ``order_by`` NÃO devolve
    ``next_page_offset`` (medido — vem ``None``), então a paginação é por VALOR
    (``start_from``). E ``start_from`` é INCLUSIVO: reabrir no último
    ``created_at`` traz de novo todos os pontos daquele instante.

    Por que os ids importam: ``created_at`` não é único. Os chunks de um mesmo
    documento nascem todos com ``created_at == submitted_at`` — no corpus real
    há um instante com 67 pontos, e outros dois acima de 25 (o tamanho da
    página). Guardar só o timestamp faria a página seguinte repetir o começo do
    grupo para sempre; guardar quantos já foram vistos (um skip numérico)
    dependeria de a ordem dentro do empate ser estável, o que a API não promete.
    Guardar os ids ENTREGUES naquele instante é exato sob qualquer ordem.
    """

    __slots__ = ("boundary_ts", "seen_ids")

    def __init__(self, boundary_ts: str | None = None, seen_ids: Iterable[str] = ()) -> None:
        self.boundary_ts = boundary_ts
        self.seen_ids = tuple(seen_ids)

    @property
    def is_first_page(self) -> bool:
        return not self.boundary_ts

    def encode(self) -> str:
        payload = json.dumps(
            {"ts": self.boundary_ts, "ids": list(self.seen_ids)}, separators=(",", ":")
        ).encode()
        return base64.urlsafe_b64encode(zlib.compress(payload, 6)).decode().rstrip("=")

    @classmethod
    def decode(cls, raw: str | None) -> "Cursor":
        """Cursor a partir da query string. Entrada inválida = primeira página.

        Um cursor corrompido (link velho, colado pela metade, adulterado) não é
        erro do operador: a resposta útil é a primeira página, não um 400.
        """
        if not raw:
            return cls()
        try:
            padded = raw + "=" * (-len(raw) % 4)
            data = json.loads(zlib.decompress(base64.urlsafe_b64decode(padded)))
            ts = data.get("ts")
            ids = data.get("ids") or []
            if not isinstance(ts, str) or not isinstance(ids, list):
                return cls()
            return cls(ts, [str(i) for i in ids if isinstance(i, (str, int))])
        except (ValueError, binascii.Error, zlib.error, TypeError):
            return cls()

    def advance(self, rows: list[dict[str, Any]]) -> "Cursor | None":
        """Cursor da PRÓXIMA página, ou None quando a listagem acabou.

        ``rows`` são as linhas efetivamente entregues nesta página, em ordem.
        """
        if not rows:
            return None
        last_ts = rows[-1].get("created_at")
        if not last_ts:
            return None
        # Ids já entregues NESTE instante — incluindo os das páginas anteriores,
        # se a página terminou no meio do mesmo grupo de empate.
        seen = [r["id"] for r in rows if r.get("created_at") == last_ts]
        if last_ts == self.boundary_ts:
            seen = list(self.seen_ids) + [i for i in seen if i not in set(self.seen_ids)]
        return Cursor(last_ts, seen)


# ------------------------------------------------------------- view-models


def _short(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def memory_row(point_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Uma linha da listagem."""
    return {
        "id": point_id,
        "excerpt": _short(payload.get("data") or ""),
        "created_at": payload.get("created_at"),
        "importance": payload.get("importance"),
        "domain": payload.get("domain"),
        "memory_type": payload.get("memory_type"),
        "project": payload.get("project"),
        "tags": payload.get("tags") or [],
        "attributed_to": payload.get("attributed_to"),
        "actor_id": payload.get("actor_id"),
        "event_date": payload.get("event_date"),
        "superseded": bool(payload.get("superseded_at") or payload.get("superseded_by")),
        "source_doc": payload.get("source_doc"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "memory_scope": payload.get("memory_scope"),
    }


#: Chaves que a tela de detalhe já mostra em campo próprio — o bloco de payload
#: bruto esconde estas para não repetir, e mostra TODO o resto (é o valor da
#: tela: os campos que a whitelist do MCP poda só aparecem aqui).
DETAIL_RENDERED_KEYS = {
    "data", "created_at", "updated_at", "hash", "text_lemmatized", "_id",
    "importance", "domain", "memory_type", "tags", "project",
    "attributed_to", "actor_id", "event_date",
    "superseded_at", "superseded_by", "supersedes",
    "source_doc", "page_start", "page_end", "chunk_index", "chunks_total",
    "reinforced_at", "access_count", "last_accessed", "reinforce_counts",
    "reinforced_by", "last_search_reinforced_at", "first_seen_at",
    "user_id", "agent_id", "run_id",
}

ACTR_FIELDS = (
    "reinforced_at", "access_count", "last_accessed", "reinforce_counts",
    "reinforced_by", "last_search_reinforced_at", "first_seen_at",
)

#: Teto de saltos ao caminhar a cadeia de versões. Uma cadeia real tem poucos
#: elos; o cap existe porque payload corrompido ou ciclo (v1→v2→v1) não pode
#: virar laço infinito numa requisição.
MAX_CHAIN_HOPS = 20


def actr_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Estado de ativação ACT-R de uma memória.

    A ativação NÃO é persistida — é derivada da timeline a cada leitura (é assim
    que o decaimento acontece sem job). Usa-se aqui a MESMA função do fork que
    o ranking usa em produção; recalcular por conta própria daria uma segunda
    verdade que divergiria em silêncio.
    """
    view: dict[str, Any] = {
        "has_history": bool(payload.get("reinforced_at")),
        "reinforced_at": payload.get("reinforced_at") or [],
        "access_count": payload.get("access_count"),
        "last_accessed": payload.get("last_accessed"),
        "reinforce_counts": payload.get("reinforce_counts") or {},
        "first_seen_at": payload.get("first_seen_at"),
        "activation": None,
        "boost": 0.0,
        "error": "",
    }
    if not view["has_history"]:
        return view
    try:
        from mem0.utils import dynamics

        view["activation"] = dynamics.base_level_activation(
            payload.get("reinforced_at"),
            payload.get("access_count"),
            first_seen=payload.get("first_seen_at") or payload.get("created_at"),
        )
        view["boost"] = dynamics.boost_from_payload(payload)
    except Exception as exc:  # noqa: BLE001 — sem o fork, a tela mostra o resto
        view["error"] = f"{type(exc).__name__}: {exc}"
    return view


def provenance_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Proveniência de documento (v0.5a/b), quando a memória veio de um."""
    return {
        "is_document": bool(payload.get("source_doc")),
        "source_doc": payload.get("source_doc"),
        "doc_sha256": payload.get("doc_sha256"),
        "content_type": payload.get("content_type"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "chunk_index": payload.get("chunk_index"),
        "chunks_total": payload.get("chunks_total"),
        "task_id": payload.get("task_id"),
    }


def raw_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """O que sobra do payload depois do que a tela já mostra em campo próprio."""
    return {k: v for k, v in sorted(payload.items()) if k not in DETAIL_RENDERED_KEYS}


def chain_ids(payload: dict[str, Any]) -> dict[str, Any]:
    """Ids vizinhos na cadeia de supersedência/versionamento.

    ``supersedes`` pode ser id único ou lista (o fork grava os dois formatos ao
    longo das versões); normalizar aqui evita que a tela tenha que saber disso.
    """
    supersedes = payload.get("supersedes")
    if isinstance(supersedes, str):
        supersedes = [supersedes]
    elif not isinstance(supersedes, (list, tuple)):
        supersedes = []
    return {
        "superseded_by": payload.get("superseded_by"),
        "superseded_at": payload.get("superseded_at"),
        "supersedes": [str(x) for x in supersedes if x],
        "version_prev": payload.get("_mem0_version_prev"),
        "version_next": payload.get("_mem0_version_next"),
    }


#: Parâmetros de busca que a tela expõe, e como cada um é convertido.
SEARCH_TEXT_FIELDS = ("query", "domain", "memory_type", "attributed_to", "actor_id",
                      "as_of", "event_from", "event_to")
SEARCH_NUM_FIELDS = {"limit": int, "threshold": float, "min_importance": float}


def search_params(params: Any, scope: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Monta os argumentos de ``search_memories`` a partir do formulário.

    Devolve também os avisos de validação. Campo numérico inválido é IGNORADO
    com aviso, em vez de derrubar a busca: quem digitou "abc" em limite quer
    buscar, não ver um erro de formulário.
    """
    args: dict[str, Any] = {}
    warnings: list[str] = []

    for key in SEARCH_TEXT_FIELDS:
        value = (params.get(key) or "").strip()
        if value:
            args[key] = value

    for key, caster in SEARCH_NUM_FIELDS.items():
        raw = (params.get(key) or "").strip()
        if not raw:
            continue
        try:
            args[key] = caster(raw)
        except ValueError:
            warnings.append(key)

    # `attributed_to` não é parâmetro da tool; vai como filtro estruturado.
    attributed = args.pop("attributed_to", None)
    if attributed:
        args["filters"] = {"attributed_to": attributed}

    if (params.get("historical") or "").strip() in ("1", "true", "on"):
        # A tool exige âncora: `historical` sem `as_of` seria recusado pelo
        # servidor. Ignorar aqui dá um aviso claro em vez de um 400 opaco.
        if args.get("as_of"):
            args["historical"] = True
        else:
            warnings.append("historical")

    args.setdefault("user_id", scope.get("user_id"))
    if scope.get("agent_id"):
        args.setdefault("agent_id", scope["agent_id"])
    if scope.get("run_id"):
        args.setdefault("run_id", scope["run_id"])
    return args, warnings


def search_result_view(item: dict[str, Any]) -> dict[str, Any]:
    """Um resultado de busca, com os sinais de ranking que a stack acrescenta."""
    metadata = item.get("metadata") or {}
    return {
        "id": item.get("id"),
        "text": item.get("memory") or item.get("data") or "",
        "score": item.get("score"),
        "rerank_score": item.get("rerank_score"),
        "superseded_penalty": item.get("superseded_penalty"),
        "event_proximity": item.get("event_proximity"),
        "has_newer_version": item.get("has_newer_version"),
        "created_at": item.get("created_at"),
        "actor_id": item.get("actor_id"),
        "attributed_to": item.get("attributed_to"),
        "memory_scope": item.get("memory_scope") or metadata.get("memory_scope"),
        "superseded_by": metadata.get("superseded_by") or item.get("superseded_by"),
        "event_date": metadata.get("event_date"),
        "domain": metadata.get("domain"),
        "memory_type": metadata.get("memory_type"),
        "importance": metadata.get("importance"),
        "tags": metadata.get("tags") or [],
        "source_doc": metadata.get("source_doc"),
    }


def search_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Envelope da busca: resultados mais os sinais que valem para a consulta toda."""
    raw = envelope.get("results")
    if not isinstance(raw, list):
        raw = []
    return {
        "results": [search_result_view(item) for item in raw if isinstance(item, dict)],
        "pending_ingest": envelope.get("pending_ingest"),
        "event_anchor": envelope.get("event_anchor"),
        "event_filter": envelope.get("event_filter"),
        "historical_recall": envelope.get("historical_recall"),
        "relations": envelope.get("relations"),
    }


def entity_row(point_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Uma linha do entity store.

    Os ids vinculados passam SEMPRE pelo normalizador do fork. O motivo é
    concreto: houve linhas gravadas com ``set(str)`` — que itera a string
    caractere a caractere — e ler ``linked_memory_ids`` cru delas devolveria uma
    lista de letras, não de ids (há um script de reparo no repo por causa disso).
    """
    links = payload.get("linked_memory_ids")
    try:
        from mem0.memory.utils import normalize_linked_memory_ids

        ids = list(normalize_linked_memory_ids(links))
    except Exception:  # noqa: BLE001 — sem o fork, degrada para o caminho conservador
        ids = [str(x) for x in links if isinstance(x, str) and len(str(x)) > 8] if isinstance(
            links, (list, tuple)
        ) else []
    return {
        "id": point_id,
        "data": payload.get("data"),
        "data_normalized": payload.get("data_normalized"),
        "entity_type": payload.get("entity_type"),
        "linked_memory_ids": ids,
        "link_count": len(ids),
    }


ENTITY_RENDERED_KEYS = {"data", "data_normalized", "entity_type", "linked_memory_ids",
                        "user_id", "agent_id", "run_id"}


def entity_raw(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload da entidade sem o que a tela já mostra e sem as chaves de vínculo.

    As chaves ``lnk_<uuid>`` são omitidas porque há uma por vínculo: numa
    entidade com 116 memórias elas sozinhas encobririam todo o resto do payload,
    sem acrescentar nada além do que a lista de vínculos já diz.
    """
    return {
        k: v
        for k, v in sorted(payload.items())
        if k not in ENTITY_RENDERED_KEYS and not k.startswith("lnk_")
    }


def filter_entities(
    rows: Iterable[dict[str, Any]], *, query: str = "", entity_type: str = ""
) -> list[dict[str, Any]]:
    """Filtra o índice de entidades em memória, por substring e por tipo.

    A comparação usa ``data_normalized`` (NFKC + casefold, a mesma identidade que
    o store usa para deduplicar), com fallback em ``data``: procurar por "fase"
    tem de achar tanto ``FASE`` quanto ``Fase``, que são a MESMA linha justamente
    porque a normalização existe.
    """
    needle = " ".join((query or "").split()).casefold()
    kind = (entity_type or "").strip()
    out = []
    for row in rows:
        if kind and row.get("entity_type") != kind:
            continue
        if needle:
            haystack = (row.get("data_normalized") or row.get("data") or "").casefold()
            if needle not in haystack:
                continue
        out.append(row)
    out.sort(key=lambda r: (-r.get("link_count", 0), (r.get("data") or "").casefold()))
    return out


def entity_kinds(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = row.get("entity_type")
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return [
        {"value": k, "count": v}
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def facet_counts(payloads: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Contagem por valor para cada chave da allowlist.

    Feito em Python, e não pela ``facet`` API do Qdrant, por uma razão MEDIDA:
    facetar exige índice de payload, e a collection só tem índice em
    ``actor_id, agent_id, attributed_to, created_at, data_normalized,
    event_date, memory_scope, run_id, superseded_at, user_id`` — ``domain`` e
    ``memory_type`` devolvem 400 ("No appropriate index for faceting"). Criar
    índice seria mudar o schema de uma collection de produção a partir de uma
    UI de leitura; contar em Python sobre um scroll projetado custa ~36 ms.
    """
    counters: dict[str, dict[str, int]] = {k: {} for k in FACET_ALLOWLIST}
    for payload in payloads:
        for key in FACET_ALLOWLIST:
            value = payload.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                continue
            counters[key][str(value)] = counters[key].get(str(value), 0) + 1
    return {
        key: [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        for key, counts in counters.items()
    }


def clean_filters(params: Any) -> dict[str, Any]:
    """Filtros vindos da query string, reduzidos ao que é permitido.

    Chave fora da allowlist é DESCARTADA em silêncio (não é erro do operador
    digitar um link velho), mas nunca vira filtro — é isso que impede que a URL
    escolha por qual campo do payload se filtra.
    """
    out: dict[str, Any] = {}
    for key in FACET_ALLOWLIST:
        value = (params.get(key) or "").strip()
        if value:
            out[key] = value
    tag = (params.get("tag") or "").strip()
    if tag:
        out["tag"] = tag
    for flag in FLAG_FILTERS:
        if (params.get(flag) or "").strip() in ("1", "true", "on"):
            out[flag] = True
    return out


def filters_query(filters: dict[str, Any], **overrides: Any) -> str:
    """Serializa os filtros de volta para uma query string (links de faceta)."""
    merged = dict(filters)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    parts = []
    for key, value in sorted(merged.items()):
        if value is True:
            parts.append(f"{key}=1")
        elif value not in (None, "", False):
            from urllib.parse import quote

            parts.append(f"{key}={quote(str(value))}")
    return "&".join(parts)
