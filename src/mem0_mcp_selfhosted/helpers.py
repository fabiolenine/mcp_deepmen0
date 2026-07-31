"""Shared utilities for mem0-mcp-selfhosted.

- patch_graph_sanitizer(): Monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- _mem0_call(): Error wrapper for all mem0ai calls
- call_with_graph(): Concurrency-safe enable_graph toggle
- safe_bulk_delete(): Iterate + individual delete (never memory.delete_all())
- get_default_user_id(): Default user_id injection
- list_entities_facet(): Qdrant Facet API entity listing with scroll fallback
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, NamedTuple

from mem0.memory.utils import normalize_scope_id
from mem0_mcp_selfhosted.env import env

#: Chaves de `filters` que são ESCOPO e por isso passam pela normalização.
#: As demais (metadata livre, operadores) seguem intocadas.
_SCOPE_KEYS = ("user_id", "agent_id", "run_id", "actor_id")

logger = logging.getLogger(__name__)

# Valid Neo4j relationship type: must start with a letter or underscore,
# followed by letters, digits, or underscores.
_NEO4J_VALID_TYPE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _make_enhanced_sanitizer(original_fn: Callable[[str], str]) -> Callable[[str], str]:
    """Wrap mem0ai's sanitize_relationship_for_cypher with Neo4j compliance fixes.

    Fixes two gaps in the upstream sanitizer:
    1. Hyphens and other ASCII characters not in the char_map
    2. Leading digits (Neo4j types must start with a letter or underscore)

    The wrapper calls the original first (preserving its 26+ special character
    mappings), then applies additional fixes.
    """

    def enhanced(relationship: str) -> str:
        # Run the original sanitizer first
        sanitized = original_fn(relationship)

        # Fix: replace hyphens (not in upstream char_map) with underscores
        sanitized = sanitized.replace("-", "_")

        # Fix: strip any remaining non-alphanumeric/underscore characters
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)

        # Collapse consecutive underscores and strip edges
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")

        # Fix: leading digit → prepend 'rel_' prefix
        if sanitized and sanitized[0].isdigit():
            sanitized = "rel_" + sanitized

        # Fallback for empty result
        if not sanitized:
            sanitized = "related_to"

        return sanitized

    return enhanced


def patch_graph_sanitizer() -> None:
    """Monkey-patch mem0ai's relationship sanitizer for full Neo4j compliance.

    Must be called AFTER mem0 modules are imported but BEFORE Memory.from_config().
    Patches both the utils module and the already-imported references in
    graph_memory/memgraph_memory.
    """
    import mem0.memory.utils as utils_module

    original = utils_module.sanitize_relationship_for_cypher
    enhanced = _make_enhanced_sanitizer(original)

    # Patch the source module
    utils_module.sanitize_relationship_for_cypher = enhanced

    # Patch already-imported references (from ... import creates local bindings)
    try:
        import mem0.memory.graph_memory as graph_module

        graph_module.sanitize_relationship_for_cypher = enhanced
    except (ImportError, AttributeError):
        pass

    try:
        import mem0.memory.memgraph_memory as memgraph_module

        memgraph_module.sanitize_relationship_for_cypher = enhanced
    except (ImportError, AttributeError):
        pass

    logger.info("Patched mem0ai relationship sanitizer for Neo4j compliance")


def patch_gemini_parse_response() -> None:
    """Monkey-patch mem0ai's GeminiLLM to guard against null content responses.

    The upstream ``GeminiLLM._parse_response`` accesses
    ``response.candidates[0].content.parts`` without checking that ``.content``
    is not ``None``.  When the Gemini API returns a candidate with null content
    (safety block, empty response, transient error), this raises
    ``AttributeError: 'NoneType' object has no attribute 'parts'``.

    Must be called AFTER mem0 modules are imported but BEFORE Memory.from_config().
    """
    try:
        from mem0.llms.gemini import GeminiLLM
    except ImportError:
        logger.debug("mem0.llms.gemini not available — skipping Gemini null guard patch")
        return

    original = getattr(GeminiLLM, "_parse_response", None)
    if original is None:
        logger.debug("GeminiLLM._parse_response not found — skipping patch")
        return

    def _safe_parse_response(self, response, *args, **kwargs):  # noqa: ANN001
        """Guarded _parse_response that handles null content gracefully."""
        if (
            response.candidates
            and response.candidates[0].content is not None
            and response.candidates[0].content.parts
        ):
            return original(self, response, *args, **kwargs)
        logger.warning("[mem0] Gemini returned null content — returning empty string")
        return ""

    GeminiLLM._parse_response = _safe_parse_response
    logger.info("Patched GeminiLLM._parse_response for null content guard")


# Serializes enable_graph mutation + full Memory method execution.
# Lock hold time is 2-20 seconds (see PRD §2.4).
_graph_lock = threading.Lock()


def get_default_user_id() -> str:
    """Get the default user_id from MEM0_USER_ID env var."""
    return env("MEM0_USER_ID", "user")


def _mem0_call(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Wrap a mem0ai call with structured error handling.

    Returns a JSON string in all cases (success or error).
    """
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        # Check if it's a MemoryError (imported lazily to avoid import issues)
        exc_type = type(exc).__name__
        is_memory_error = any(
            cls.__name__ == "MemoryError" for cls in type(exc).__mro__
        )
        if is_memory_error:
            logger.error("Mem0 call failed: %s", exc)
            return json.dumps(
                {
                    "error": str(exc),
                    "error_code": getattr(exc, "error_code", None),
                    "details": getattr(exc, "details", None),
                    "suggestion": getattr(exc, "suggestion", None),
                },
                ensure_ascii=False,
            )
        else:
            logger.error("Unexpected error: %s", exc)
            return json.dumps(
                {
                    "error": exc_type,
                    "detail": str(exc),
                },
                ensure_ascii=False,
            )
    return json.dumps(result, ensure_ascii=False)


def call_with_graph(
    memory: Any,
    enable_graph: bool | None,
    default_graph: bool,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a Memory method with per-request enable_graph context.

    Each tool call resolves its own effective enable_graph value and passes
    it here. The lock ensures no concurrent request can observe a stale flag.

    IMPORTANT: The lock is held for the full duration of func() (2-20s),
    because Memory.add() blocks on concurrent.futures.wait() internally.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Infrastructure may be unavailable.")
    effective = enable_graph if enable_graph is not None else default_graph
    with _graph_lock:
        memory.enable_graph = effective and memory.graph is not None
        return func(*args, **kwargs)


def _point_id(item: Any) -> str:
    """Memory id out of a Qdrant point / dict / whatever the store returned."""
    if hasattr(item, "id"):
        return item.id
    if isinstance(item, dict):
        return item.get("id")
    return str(item)


class BulkDeleteResult(NamedTuple):
    """Outcome of a bulk delete, stated honestly.

    ``vector_scope_drained`` is VERIFIED by a final scan, not inferred from the
    loop ending. Its absence is what made a half-finished delete look
    successful: the old helper returned a bare count, and a caller had no way to
    tell "deleted everything" from "deleted the first page".

    The name is deliberately narrow. It says the VECTOR scope is empty — it does
    NOT promise that entity links were cleaned or that graph data went with it.
    Calling that "complete" invited exactly the over-reading this whole exercise
    is about; ``graph_cleaned`` is reported separately (None = not attempted).

    ``remaining_ids`` is capped at one page, so treat it as a sample; use
    ``remaining_is_partial`` before quoting the number as a total.
    """

    targeted: int
    deleted: int          # ids REQUESTED that succeeded; lower bound on points
    failed_ids: list[str]
    remaining_ids: list[str]
    vector_scope_drained: bool
    remaining_is_partial: bool = False
    graph_cleaned: bool | None = None


def safe_bulk_delete(
    memory: Any,
    filters: dict[str, Any],
    *,
    graph_enabled: bool = False,
    page_size: int = 1000,
    max_pages: int = 1000,
) -> BulkDeleteResult:
    """Delete every memory matching ``filters``.

    NEVER calls memory.delete_all() (which triggers vector_store.reset()).

    Two defects this fixes:

    * ``vector_store.list()`` was called with NO ``top_k``, and Qdrant defaults
      to 100 — so one call deleted at most 100 memories and reported that count
      as the whole job. That is the real reason a bulk delete once "needed two
      passes".
    * ``memory.delete()`` deletes a whole UPDATE-VERSION CHAIN and, on partial
      failure, RETURNS a dict instead of raising, so a non-raising call was
      being scored as a success.

      KNOWN LIMITATION, stated rather than papered over: a SUCCESSFUL chain
      delete returns only ``{"message": "Memory deleted successfully!"}`` — no
      list of the points it removed. So ``deleted`` counts REQUESTED ids, and a
      chain of three versions still counts as one. The number is a lower bound
      on points removed; ``vector_scope_drained`` is the field that actually
      answers "is the scope empty". Fixing the count properly needs the fork to
      report the removed ids on success.

    Termination does not rest on "the page came back empty" — a row that fails
    to delete would be re-listed forever. Each pass must claim at least one id
    not already attempted, or the loop ends.

    Args:
        graph_enabled: Explicit graph state from caller (avoids reading
            mutable ``memory.enable_graph`` which races with ``call_with_graph``).
    """
    # Defesa em profundidade. A decisão de escopo pertence à fronteira da tool,
    # onde ainda dá para devolver erro acionável ao cliente — mas este helper é
    # genérico e pode ganhar outro chamador, e um escopo não normalizado aqui
    # não produz erro nenhum: produz um delete que não casa nada e responde
    # `deleted: 0` como sucesso. Reusa a regra do core em vez de reimplementá-la;
    # duas definições de "escopo válido" foi exatamente o que deixou
    # `importance='high'` entrar no corpus.
    filters = {
        k: (normalize_scope_id(v, k) if k in _SCOPE_KEYS else v)
        for k, v in (filters or {}).items()
    }

    attempted: set[str] = set()
    failed: list[str] = []
    deleted_ids: set[str] = set()

    for _ in range(max_pages):
        result = memory.vector_store.list(filters=filters, top_k=page_size)
        items = result[0] if isinstance(result, tuple) else result
        fresh = [i for i in items or [] if _point_id(i) not in attempted]
        if not fresh:
            break
        for item in fresh:
            memory_id = _point_id(item)
            attempted.add(memory_id)
            try:
                res = memory.delete(memory_id)
            except Exception as exc:
                logger.warning("Failed to delete memory %s: %s", memory_id, exc)
                failed.append(memory_id)
                continue
            # A PARTIAL version-chain delete reports itself; trust that over the
            # call merely not raising. (A successful chain reports nothing, so
            # `deleted` stays a lower bound — see the docstring.)
            if isinstance(res, dict) and res.get("remaining"):
                logger.warning("Partial version-chain delete for %s: remaining=%s",
                               memory_id, res["remaining"])
                deleted_ids.update(res.get("deleted") or [])
                failed.append(memory_id)
                continue
            if isinstance(res, dict) and res.get("deleted"):
                deleted_ids.update(res["deleted"])
            else:
                deleted_ids.add(memory_id)
    else:
        logger.warning("safe_bulk_delete: page cap (%d) reached for %s", max_pages, filters)

    # Verify rather than assume.
    verify = memory.vector_store.list(filters=filters, top_k=page_size)
    verify_items = verify[0] if isinstance(verify, tuple) else verify
    remaining_ids = [_point_id(i) for i in verify_items or []]
    drained = not remaining_ids
    # One page only: if it came back full, the true remainder may be larger.
    remaining_is_partial = len(remaining_ids) >= page_size

    # Mandatory graph cleanup — memory.delete() does NOT clean Neo4j (GitHub #3245).
    # Only when the scope really drained: running it on a partial delete would
    # strip graph data belonging to memories that are still alive.
    graph_cleaned = None
    if graph_enabled and hasattr(memory, "graph") and memory.graph is not None:
        if drained:
            try:
                memory.graph.delete_all(filters)
                graph_cleaned = True
            except Exception as exc:
                graph_cleaned = False
                logger.warning("Graph cleanup failed for filters %s: %s", filters, exc)
        else:
            graph_cleaned = False
            logger.warning(
                "Skipping graph cleanup for %s: %d memories still present — "
                "cleaning now would delete graph data for live memories.",
                filters, len(remaining_ids))

    return BulkDeleteResult(
        targeted=len(attempted),
        deleted=len(deleted_ids),
        failed_ids=failed,
        remaining_ids=remaining_ids,
        vector_scope_drained=drained,
        remaining_is_partial=remaining_is_partial,
        graph_cleaned=graph_cleaned,
    )


def list_entities_facet(memory: Any) -> dict[str, list[dict]]:
    """List entities using Qdrant Facet API with scroll fallback.

    Primary: Facet API (Qdrant v1.12+) — server-side distinct value aggregation.
    Fallback: scroll+dedupe for older Qdrant versions.

    Returns: {"users": [{"value": ..., "count": ...}], "agents": [...], "runs": [...]}
    """
    client = memory.vector_store.client
    collection = memory.vector_store.collection_name

    result: dict[str, list[dict]] = {"users": [], "agents": [], "runs": []}
    entity_keys = {"users": "user_id", "agents": "agent_id", "runs": "run_id"}

    try:
        for result_key, payload_key in entity_keys.items():
            facet_response = client.facet(
                collection_name=collection,
                key=payload_key,
            )
            result[result_key] = [
                {"value": hit.value, "count": hit.count}
                for hit in facet_response.hits
            ]
        return result
    except Exception as exc:
        # Facet API unavailable — fall back to scroll+dedupe
        logger.warning(
            "Qdrant Facet API unavailable (%s). Falling back to scroll+dedupe. "
            "Upgrade to Qdrant v1.12+ for better performance.",
            exc,
        )
        return _list_entities_scroll_fallback(memory)


def _list_entities_scroll_fallback(memory: Any) -> dict[str, list[dict]]:
    """Fallback entity listing via scroll+dedupe."""
    entities: dict[str, dict[str, int]] = {
        "user_id": {},
        "agent_id": {},
        "run_id": {},
    }

    # Real cursor scroll. Two bugs lived here: the kwarg was `limit=500`, but the
    # signature is `list(filters=None, top_k=100)` — so this raised TypeError the
    # moment the Facet API failed, i.e. exactly when the fallback was needed. And
    # renaming it to top_k alone would only trade a crash for silently counting
    # the first 500 memories and presenting that as the totals.
    client = getattr(memory.vector_store, "client", None)
    collection = getattr(memory.vector_store, "collection_name", None)
    all_memories: list = []
    if client is not None and collection is not None:
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection, with_payload=True,
                with_vectors=False, limit=500, offset=offset,
            )
            all_memories.extend(batch)
            if offset is None:
                break
    else:
        # No native cursor: take one big page and SAY that it may be truncated,
        # rather than reporting a partial count as if it were the whole corpus.
        top_k = 10000
        result = memory.vector_store.list(filters=None, top_k=top_k)
        all_memories = result[0] if isinstance(result, tuple) else result
        if len(all_memories) >= top_k:
            logger.warning(
                "Entity listing fallback hit the %d-row cap — counts are TRUNCATED.", top_k)

    for item in all_memories:
        payload = item.payload if hasattr(item, "payload") else item
        if isinstance(payload, dict):
            for key in entities:
                val = payload.get(key)
                if val:
                    entities[key][val] = entities[key].get(val, 0) + 1

    return {
        "users": [{"value": v, "count": c} for v, c in entities["user_id"].items()],
        "agents": [{"value": v, "count": c} for v, c in entities["agent_id"].items()],
        "runs": [{"value": v, "count": c} for v, c in entities["run_id"].items()],
    }
