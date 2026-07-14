"""FastMCP server for mem0-mcp-selfhosted.

Orchestrates: tool registration → transport → lazy Memory init on first call.
Memory initialization is deferred to the first tool invocation via _ensure_memory(),
allowing the server to respond to MCP initialize/tools/list without live infrastructure.
All 15 MCP tools + memory_assistant prompt.

Async ingest (v0.4): add_memory with infer=true enqueues into a durable SQLite
queue and acks immediately with a task_id envelope; a serial background worker
(ingest_worker.py) runs the 26-37s extraction pipeline off the client's clock.
Kill switch: MEM0_ASYNC_INGEST=false restores the synchronous path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mem0_mcp_selfhosted.config import ProviderInfo, build_config
from mem0_mcp_selfhosted.document_source import resolve_and_spool
from mem0_mcp_selfhosted.env import bool_env, env
from mem0_mcp_selfhosted.graph_tools import get_entity, search_graph
from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    call_with_graph,
    get_default_user_id,
    list_entities_facet,
    patch_gemini_parse_response,
    patch_graph_sanitizer,
    safe_bulk_delete,
)
from mem0_mcp_selfhosted.image_extract import vision_enabled
from mem0_mcp_selfhosted.ingest_queue import IngestQueue, idempotency_key
from mem0_mcp_selfhosted.ingest_worker import IngestWorker
from mem0_mcp_selfhosted.pdf_extract import EncryptedPdf, pdf_info

logger = logging.getLogger(__name__)

# --- Globals set during startup ---
memory = None
mcp: FastMCP | None = None
_enable_graph_default = False

# --- Lazy init state ---
_memory_init_lock = threading.Lock()
_last_init_failure: float = 0.0
_INIT_RETRY_COOLDOWN = 30.0  # seconds before retrying after a failed init

# --- Async ingest state ---
_ingest_queue: IngestQueue | None = None
_ingest_worker: IngestWorker | None = None
_ingest_lock = threading.Lock()


def _async_ingest_enabled() -> bool:
    return bool_env("MEM0_ASYNC_INGEST", "true")


def _queue_db_path() -> str:
    explicit = env("MEM0_QUEUE_DB_PATH")
    if explicit:
        return explicit
    history = env("MEM0_HISTORY_DB_PATH")
    if history:
        return str(Path(history).parent / "ingest_queue.db")
    return str(Path.home() / ".mem0" / "ingest_queue.db")


def _get_ingest() -> tuple[IngestQueue, IngestWorker]:
    """Lazy-create the queue + worker pair (thread-safe, idempotent).

    The worker thread autostarts unless MEM0_QUEUE_WORKER=false (tests disable
    it to exercise the queue without a live consumer racing them).
    """
    global _ingest_queue, _ingest_worker
    with _ingest_lock:
        if _ingest_queue is None:
            _ingest_queue = IngestQueue(_queue_db_path())
            _ingest_worker = IngestWorker(
                _ingest_queue,
                _ensure_memory,
                call_with_graph=call_with_graph,
            )
            if bool_env("MEM0_QUEUE_WORKER", "true"):
                _ingest_worker.start()
        return _ingest_queue, _ingest_worker


def _estimate_wait_s(queue: IngestQueue) -> int:
    """Kind-aware drain estimate: conversations cost EST_ADD_S each, documents
    cost EST_CHUNK_S per remaining chunk — queue_depth × 40s lies by an order
    of magnitude once a document is in line."""
    est_add = int(env("MEM0_QUEUE_EST_ADD_S", "40"))
    est_chunk = int(env("MEM0_DOC_EST_CHUNK_S", "35"))
    est_update = int(env("MEM0_QUEUE_EST_UPDATE_S", "15"))
    try:
        by_kind = queue.queue_status().get("depth_by_kind", {})
        conversations = by_kind.get("conversation", 0)
        updates = by_kind.get("update", 0)
        doc_chunks = queue.pending_document_chunks()
        return conversations * est_add + updates * est_update + doc_chunks * est_chunk
    except Exception:
        return queue.depth() * est_add


def register_providers(providers_info: list[ProviderInfo]) -> None:
    """Register custom LLM providers with mem0ai's LlmFactory.

    Maps provider names to their config classes and registers each.
    Config classes are lazy-imported to avoid pulling in unnecessary
    dependencies (e.g. ``anthropic`` package in Ollama-only mode).
    Safe to call multiple times (LlmFactory.register_provider is idempotent).
    """
    if not providers_info:
        return

    from mem0.utils.factory import LlmFactory

    for pi in providers_info:
        config_class = _resolve_config_class(pi["name"])
        if config_class is None:
            logger.warning("No config class for provider %r, skipping", pi["name"])
            continue
        LlmFactory.register_provider(
            name=pi["name"],
            class_path=pi["class_path"],
            config_class=config_class,
        )


def _resolve_config_class(provider_name: str) -> type | None:
    """Lazy-resolve the config class for a provider name.

    Imports are deferred so that unnecessary packages (e.g. ``anthropic``)
    are never loaded in a pure-Ollama setup.
    """
    if provider_name == "ollama":
        from mem0.configs.llms.ollama import OllamaConfig

        return OllamaConfig
    if provider_name in ("anthropic", "anthropic_oat"):
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        return AnthropicOATConfig
    return None


def _init_memory() -> Any:
    """Initialize mem0ai Memory with config and registered providers."""
    global memory, _enable_graph_default

    config_dict, providers_info, split_config = build_config()

    register_providers(providers_info)

    # Patch mem0ai's relationship sanitizer before Memory init
    patch_graph_sanitizer()
    patch_gemini_parse_response()

    # Initialize Memory
    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig

        router_config = SplitModelGraphLLMConfig(**split_config)
        memory.graph.llm = SplitModelGraphLLM(router_config)

    _enable_graph_default = bool_env("MEM0_ENABLE_GRAPH")
    return memory


def _ensure_memory() -> Any:
    """Lazy-initialize Memory on first tool call. Thread-safe with retry-after-delay.

    Returns the Memory instance, or None if initialization failed.
    After a failure, waits ``_INIT_RETRY_COOLDOWN`` seconds before retrying.
    Matches the lazy-init pattern used by ``graph_tools._get_driver()``.
    """
    global memory, _last_init_failure

    if memory is not None:
        return memory

    now = time.monotonic()
    if _last_init_failure and (now - _last_init_failure < _INIT_RETRY_COOLDOWN):
        return None  # Too soon to retry

    with _memory_init_lock:
        # Double-check after acquiring lock
        if memory is not None:
            return memory

        try:
            _init_memory()
            logger.info("mem0ai Memory initialized successfully (lazy)")
        except Exception as exc:
            _last_init_failure = time.monotonic()
            logger.error("Lazy Memory init failed: %s", exc)
            return None

    return memory


def _create_server() -> FastMCP:
    """Create and configure the FastMCP server with all tools and prompts."""
    global mcp

    host = env("MEM0_HOST", "0.0.0.0")
    port = int(env("MEM0_PORT", "8081"))

    mcp = FastMCP(
        "DeepMem0",
        host=host,
        port=port,
        instructions=(
            "DeepMem0 — memory tools for persistent cross-session memory. "
            "Use search_memories to find relevant context before starting work. "
            "Use add_memory to store important facts, preferences, and decisions; "
            "with infer=true (default) it is asynchronous — it acks immediately with a "
            "task_id while extraction runs in background (use memory_task_status to get "
            "the resulting memory_ids, memory_queue_status for queue health; a search "
            "response's pending_ingest field warns when queued facts are not searchable yet). "
            "Use add_document to ingest a PDF or image (PNG/JPEG) from a local path — "
            "facts are extracted per chunk with document/page provenance (scanned pages "
            "and images are read by a local vision model when MEM0_ENABLE_VISION is on); "
            "documents take minutes, poll memory_task_status for chunks_done progress. "
            "Use get_memories to browse stored memories with filters. "
            "Use search_graph to find relationships between entities. "
            "Use get_memory to retrieve a specific memory by ID. "
            "Use update_memory to modify existing memories. "
            "Use list_entities to see who/what has stored memories."
        ),
    )

    _register_tools(mcp)
    _register_prompts(mcp)

    return mcp


# ============================================================
# Memory Tools (7 tools)
# ============================================================


def _register_tools(mcp: FastMCP) -> None:
    """Register all 15 MCP tools on the server."""

    @mcp.tool()
    def add_memory(
        text: Annotated[str, Field(description="Text to store as a memory. Converted to messages format internally.")],
        messages: Annotated[list[dict] | None, Field(description="Structured conversation history (role/content dicts). When provided, takes precedence over text.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Arbitrary metadata JSON to store alongside the memory.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), LLM extracts key facts asynchronously: the call returns a queued envelope with a task_id immediately (use memory_task_status to fetch the resulting memory_ids). If false, stores raw text synchronously.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle for this call.")] = None,
    ) -> str:
        """Store a new memory. Requires at least one of user_id, agent_id, or run_id.

        Response contract (never a bare list):
        - {"status": "queued", "task_id", "submitted_at", "queue_depth", "estimated_wait_s"}
          — infer=true path; extraction runs in background, poll memory_task_status.
        - {"status": "stored", "memory_ids": [...], "results": [...]}
          — synchronous path; empty memory_ids carries "reason": "no_new_facts".
        - {"error": ...} — failure.
        """
        uid = user_id or get_default_user_id()

        # Build messages for mem0ai
        if messages:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": text}]

        eff_infer = True if infer is None else infer

        if eff_infer and _async_ingest_enabled():
            try:
                queue, worker = _get_ingest()
                params: dict[str, Any] = {}
                if metadata:
                    params["metadata"] = metadata
                # resolve the effective graph toggle NOW (the worker doesn't
                # know this server's default) and let it ride with the job
                params["enable_graph"] = enable_graph if enable_graph is not None else _enable_graph_default
                res = queue.enqueue(
                    user_id=uid, agent_id=agent_id or None, run_id=run_id or None,
                    messages=msgs, params=params,
                )
                worker.notify()
                envelope: dict[str, Any] = {
                    "status": "queued",
                    "task_id": res["task_id"],
                    "submitted_at": res["submitted_at"],
                    "queue_depth": res["queue_depth"],
                    "estimated_wait_s": _estimate_wait_s(queue),
                }
                if res["duplicate"]:
                    envelope["duplicate"] = True
                    envelope["note"] = "identical payload already queued; returning the existing task"
                return json.dumps(envelope, ensure_ascii=False)
            except Exception as exc:
                # A broken queue must not lose the fact — fall through to the
                # synchronous path and say so.
                logger.error("Async ingest enqueue failed, falling back to sync add: %s", exc)

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if metadata:
            kwargs["metadata"] = metadata
        if infer is not None:
            kwargs["infer"] = infer

        mem = _ensure_memory()

        def _do_add():
            return mem.add(msgs, **kwargs)

        def _do_add_enveloped():
            raw = call_with_graph(mem, enable_graph, _enable_graph_default, _do_add)
            results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
            memory_ids = [
                r["id"] for r in results
                if isinstance(r, dict) and r.get("event") in ("ADD", "UPDATE") and r.get("id")
            ]
            envelope: dict[str, Any] = {
                "status": "stored",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "memory_ids": memory_ids,
                "results": results,
            }
            if isinstance(raw, dict) and raw.get("relations") is not None:
                envelope["relations"] = raw["relations"]
            if not memory_ids:
                envelope["reason"] = "no_new_facts"
            return envelope

        return _mem0_call(_do_add_enveloped)

    @mcp.tool()
    def add_document(
        file_path: Annotated[str, Field(description="Absolute path on the server host (must live under MEM0_DOC_PATH_ALLOWLIST, default $HOME) of a PDF or an image (PNG/JPEG). Scanned PDFs and images need vision on (MEM0_ENABLE_VISION).")],
        filename: Annotated[str | None, Field(description="Display name stored as source_doc provenance. Defaults to the file's basename.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Extra metadata stored on every memory extracted from this document.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), the LLM extracts facts from each chunk; if false, raw chunks are stored as-is.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Graph extraction per chunk. Defaults to FALSE for documents (expensive and noisy).")] = None,
        force: Annotated[bool | None, Field(description="Re-ingest even if this exact document (same bytes + scope) was already ingested.")] = None,
    ) -> str:
        """Ingest a PDF or image asynchronously: extract text per page (digital
        PDF via poppler; scanned pages and images via a local vision model),
        chunk, and extract memorable facts with document/page provenance.

        Returns immediately with {"status": "queued", task_id, pages,
        chunks_estimate, estimated_wait_s} — a large document takes many
        minutes; poll memory_task_status(task_id) for chunks_done progress and
        the final memory_ids. Re-submitting the same file returns
        {"status": "already_ingested"} unless force=true. There is NO
        synchronous fallback: if the queue is unavailable the call errors.
        """
        if not bool_env("MEM0_DOC_ENABLED", "true"):
            return json.dumps({"error": "document ingestion is disabled (MEM0_DOC_ENABLED=false)"}, ensure_ascii=False)
        uid = user_id or get_default_user_id()

        def _do_submit():
            queue, worker = _get_ingest()
            info = resolve_and_spool(file_path)  # typed errors -> {"error": ...}
            is_image = info["content_type"].startswith("image/")
            if is_image:
                if not vision_enabled():
                    raise ValueError(
                        "image ingestion needs vision (set MEM0_ENABLE_VISION=true and MEM0_VLM_MODEL)"
                    )
                pages, chunks_estimate = 1, 1
            else:
                doc_meta = pdf_info(info["spool_path"])
                if doc_meta["encrypted"]:
                    raise EncryptedPdf("encrypted PDF — decrypt it before ingesting")
                max_pages = int(env("MEM0_DOC_MAX_PAGES", "50"))
                if doc_meta["pages"] > max_pages:
                    raise ValueError(
                        f"document has {doc_meta['pages']} pages; cap is {max_pages} "
                        f"(MEM0_DOC_MAX_PAGES) — split the file"
                    )
                pages = doc_meta["pages"]
                chunks_estimate = max(1, pages * 2)  # ~2 chunks/page at 1800 chars
            display_name = os.path.basename((filename or info["filename"]).strip()) or info["filename"]
            msgs = [{"role": "user", "content": f"[document sha256={info['doc_sha256']}]"}]

            if not force:
                done = queue.latest_done(idempotency_key(uid, agent_id or None, run_id or None, msgs))
                if done is not None:
                    return {
                        "status": "already_ingested",
                        "task_id": done["task_id"],
                        "finished_at": done.get("finished_at"),
                        "result": done.get("result"),
                        "note": "same bytes + scope already processed; resubmit with force=true to re-ingest",
                    }

            params: dict[str, Any] = {
                "spool_path": info["spool_path"],
                "doc_sha256": info["doc_sha256"],
                "content_type": info["content_type"],
                "filename": display_name,
                "pages": pages,
                "chunks_estimate": chunks_estimate,  # display only
                "enable_graph": enable_graph if enable_graph is not None else False,
            }
            if metadata:
                params["metadata"] = metadata
            if infer is not None:
                params["infer"] = infer

            res = queue.enqueue(
                user_id=uid, agent_id=agent_id or None, run_id=run_id or None,
                messages=msgs, params=params, kind="document",
            )
            worker.notify()
            envelope: dict[str, Any] = {
                "status": "queued",
                "task_id": res["task_id"],
                "submitted_at": res["submitted_at"],
                "source_doc": display_name,
                "content_type": info["content_type"],
                "pages": pages,
                "chunks_estimate": chunks_estimate,
                "queue_depth": res["queue_depth"],
                "estimated_wait_s": _estimate_wait_s(queue),
            }
            if res["duplicate"]:
                envelope["duplicate"] = True
                envelope["note"] = "this document is already queued; returning the existing task"
            return envelope

        return _mem0_call(_do_submit)

    @mcp.tool()
    def search_memories(
        query: Annotated[str, Field(description="Natural language description of what to find.")],
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        filters: Annotated[dict | None, Field(description="Additional structured filter clauses.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of results (default 10).")] = None,
        threshold: Annotated[float | None, Field(description="Minimum relevance score (0.0-1.0).")] = None,
        rerank: Annotated[bool | None, Field(description="Whether to apply reranking. Defaults to the server's MEM0_ENABLE_RERANK.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle.")] = None,
        min_importance: Annotated[float | None, Field(description="Keep only memories whose classified importance is >= this value (0.0-1.0).")] = None,
        domain: Annotated[str | None, Field(description="Keep only memories whose classified domain matches (e.g. career, ai, data, software_engineering, finance, trading, health, education, personal, legal, business, infrastructure).")] = None,
        memory_type: Annotated[str | None, Field(description="Keep only memories of this classified type: semantic, episodic, or procedural.")] = None,
        sort_by_importance: Annotated[bool | None, Field(description="Sort results by classified importance descending.")] = None,
        as_of: Annotated[str | None, Field(description="Temporal anchor (ISO date or datetime): return what was known/current on that date — memories created later are excluded and facts superseded only after the anchor carry no demotion.")] = None,
    ) -> str:
        """Semantic search across existing memories."""
        uid = user_id or get_default_user_id()

        kwargs: dict[str, Any] = {"user_id": uid, "query": query}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if filters:
            kwargs["filters"] = filters
        # mem0ai 2.0.7: Memory.search recebe top_k (limit cairia no **kwargs e seria
        # ignorado) e rerank default False — aqui o default vem de MEM0_ENABLE_RERANK
        # (o reranker residente é a razão de ser do :8081).
        eff_limit = limit if limit is not None else 10
        do_rerank = rerank if rerank is not None else bool_env("MEM0_ENABLE_RERANK")
        kwargs["rerank"] = do_rerank
        # DeepMem0 já over-fetcha e corta no core (rerank_pool); duplicar aqui
        # dobraria o pool do cross-encoder (20 -> 40 = ~2x a latência à toa).
        import mem0 as _m0
        _core_overfetches = bool(getattr(_m0, "__deepmem0__", False))
        if do_rerank and not _core_overfetches:
            # Over-fetch manual só no runtime mem0ai upstream: o rerank de lá só
            # reordena o top_k fundido; um pool maior resgata alvos que a fusão
            # aditiva enterra (golden set 06/07/2026: hit@1 0.857→0.886 com pool
            # 20; pool 30 = igual, 3x o custo).
            kwargs["top_k"] = max(2 * eff_limit, int(env("MEM0_RERANK_POOL", "20")))
        else:
            kwargs["top_k"] = eff_limit
        if threshold is not None:
            kwargs["threshold"] = threshold
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if domain:
            kwargs["domain"] = domain
        if memory_type:
            kwargs["memory_type"] = memory_type
        if sort_by_importance is not None:
            kwargs["sort_by_importance"] = sort_by_importance
        if as_of:
            # DeepMem0 v0.3: âncora temporal. No runtime mem0ai upstream o
            # parâmetro não existe — erro claro em vez de TypeError críptico.
            if not _core_overfetches:
                return json.dumps(
                    {"error": "as_of requer o runtime DeepMem0 >= 0.3 (mem0ai upstream não suporta)"},
                    ensure_ascii=False,
                )
            kwargs["as_of"] = as_of

        mem = _ensure_memory()

        # Classification keys clients actually filter/sort on; everything else in
        # metadata (text_lemmatized, entities, ...) only inflates client context.
        # v0.3: supersession/event-time fields are part of the contract clients
        # reason about (which fact is current, since when) — keep them visible.
        # task_id: provenance — which async submission a memory came from.
        # source_doc/page/chunk: document provenance (v0.5a) — which file and
        # page a fact was extracted from.
        _metadata_whitelist = {
            "importance", "domain", "tags", "memory_type",
            "superseded_by", "superseded_at", "supersedes", "event_date",
            "task_id", "source_doc", "page_start", "page_end", "chunk_index",
            "content_type",
        }

        def _do_search():
            res = mem.search(**kwargs)
            items = res.get("results") if isinstance(res, dict) else res
            if isinstance(items, list):
                # corta o over-fetch de volta ao limit pedido (pós-rerank/patch 4)
                if len(items) > eff_limit:
                    del items[eff_limit:]
                for r in items:
                    if isinstance(r, dict) and isinstance(r.get("metadata"), dict):
                        r["metadata"] = {
                            k: v for k, v in r["metadata"].items() if k in _metadata_whitelist
                        }
            # read-your-writes signal: facts still in the ingest queue are not
            # searchable yet — tell the caller the picture may be incomplete.
            if _async_ingest_enabled() and isinstance(res, dict):
                try:
                    queue, _ = _get_ingest()
                    res["pending_ingest"] = queue.pending_for_scope(uid)
                except Exception:
                    pass
            return res

        return _mem0_call(call_with_graph, mem, enable_graph, _enable_graph_default, _do_search)

    @mcp.tool()
    def get_memories(
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of memories to return.")] = None,
    ) -> str:
        """Page through memories using filters instead of search."""
        uid = user_id or get_default_user_id()

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit is not None:
            kwargs["limit"] = limit

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.get_all, **kwargs)

    @mcp.tool()
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to fetch.")],
    ) -> str:
        """Fetch a single memory by its ID."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.get, memory_id)

    @mcp.tool()
    def memory_history(
        memory_id: Annotated[str, Field(description="Exact memory UUID whose change history to fetch.")],
    ) -> str:
        """Full change timeline of a memory: ADD, UPDATE (old vs new text), SUPERSEDED (which fact replaced it) and DELETE events, oldest first."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.history, memory_id)

    @mcp.tool()
    def memory_task_status(
        task_id: Annotated[str, Field(description="Task id returned by a queued add_memory call (tsk_...).")],
    ) -> str:
        """Status of an asynchronous add_memory task.

        States: pending | processing | done | failed_retryable | dead.
        When done, ``result.memory_ids`` lists the memories created/updated
        (fetch them with get_memory) and ``result.events`` includes any
        SUPERSEDED markings. ``last_error`` explains failed/dead tasks.
        """
        def _do_status():
            queue, _ = _get_ingest()
            row = queue.task_status(task_id)
            if row is None:
                return {"error": f"unknown task_id: {task_id}"}
            return row

        return _mem0_call(_do_status)

    @mcp.tool()
    def memory_queue_status() -> str:
        """Health of the async ingest queue: depth (jobs waiting or running),
        per-status counts, age of the oldest pending job, estimated drain time,
        and whether the background worker is alive."""
        def _do_queue_status():
            queue, worker = _get_ingest()
            status = queue.queue_status()
            status["worker_alive"] = worker.is_alive()
            status["async_ingest_enabled"] = _async_ingest_enabled()
            status["estimated_drain_s"] = _estimate_wait_s(queue)
            return status

        return _mem0_call(_do_queue_status)

    @mcp.tool()
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
    ) -> str:
        """Update an existing memory's text.

        ASYNCHRONOUS by default (when MEM0_ASYNC_INGEST != false): validates the
        memory exists, then returns {"status": "queued", "task_id", ...} immediately
        while the re-embed + metadata re-classification (a slow llama3.1:8b call) run in
        the background worker — so the call never times out the client. Poll
        memory_task_status(task_id) for the result (memory_id / UPDATE event);
        memory_history(memory_id) shows the old-vs-new diff. An identical re-submit
        while the job is still active returns the same task_id (no double-apply). Set
        MEM0_ASYNC_INGEST=false for the synchronous path.
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        if _async_ingest_enabled():
            try:
                # Validate + resolve owner scope at submit: fail fast on a bad id
                # (never enqueue a doomed job) and scope the job to the memory's
                # owner so pending_ingest/purge stay correct.
                existing = mem.get(memory_id)
                if not existing:
                    return json.dumps({"error": "memory not found", "memory_id": memory_id}, ensure_ascii=False)
                uid = existing.get("user_id") or get_default_user_id()
                queue, worker = _get_ingest()
                # Sentinel messages encode memory_id+text so the idempotency key
                # (scope + messages) distinguishes distinct updates AND collapses an
                # identical retry onto the same task_id; the worker reads the real
                # memory_id/text from params.
                sentinel = [{"role": "user", "content": f"[update memory_id={memory_id}]\n{text}"}]
                res = queue.enqueue(
                    user_id=uid,
                    agent_id=existing.get("agent_id") or None,
                    run_id=existing.get("run_id") or None,
                    messages=sentinel,
                    params={"memory_id": memory_id, "text": text},
                    kind="update",
                )
                worker.notify()
                envelope: dict[str, Any] = {
                    "status": "queued",
                    "task_id": res["task_id"],
                    "submitted_at": res["submitted_at"],
                    "queue_depth": res["queue_depth"],
                    "estimated_wait_s": _estimate_wait_s(queue),
                }
                if res["duplicate"]:
                    envelope["duplicate"] = True
                    envelope["note"] = "identical update already queued; returning the existing task"
                return json.dumps(envelope, ensure_ascii=False)
            except Exception as exc:
                # A broken queue must not drop the update — fall through to sync.
                logger.error("Async update enqueue failed, falling back to sync update: %s", exc)

        def _do_update():
            mem.update(memory_id, data=text)
            return {"message": "Memory updated successfully!"}

        return _mem0_call(_do_update)

    @mcp.tool()
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to delete.")],
    ) -> str:
        """Delete a single memory."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_delete():
            mem.delete(memory_id)
            return {"message": "Memory deleted successfully!"}

        return _mem0_call(_do_delete)

    @mcp.tool()
    def delete_all_memories(
        user_id: Annotated[str | None, Field(description="User scope to delete.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope to delete.")] = None,
    ) -> str:
        """Bulk-delete all memories in the given scope. Requires at least one filter.

        NEVER calls memory.delete_all() — uses safe bulk-delete instead.
        """
        uid = user_id or get_default_user_id()
        if not any([uid, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if uid:
            filters["user_id"] = uid
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_bulk_delete():
            count = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return {"message": f"Deleted {count} memories.", "count": count}

        return _mem0_call(_do_bulk_delete)

    # ============================================================
    # Entity Tools (2 tools)
    # ============================================================

    @mcp.tool()
    def list_entities() -> str:
        """List which users/agents/runs currently hold memories.

        Uses Qdrant Facet API (v1.12+) for server-side aggregation,
        with scroll+dedupe fallback for older versions.
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_list():
            return list_entities_facet(mem)

        return _mem0_call(_do_list)

    @mcp.tool()
    def delete_entities(
        user_id: Annotated[str | None, Field(description="User entity to delete (cascades to all memories).")] = None,
        agent_id: Annotated[str | None, Field(description="Agent entity to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run entity to delete.")] = None,
    ) -> str:
        """Delete an entity and cascade-delete all its memories.

        Functionally equivalent to delete_all_memories in self-hosted mode.
        """
        if not any([user_id, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_delete_entity():
            count = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return {"message": f"Entity deleted. Removed {count} memories.", "count": count}

        return _mem0_call(_do_delete_entity)

    # ============================================================
    # Direct Neo4j Graph Tools
    # ============================================================

    @mcp.tool()
    def mcp_search_graph(
        query: Annotated[str, Field(description="Entity or topic to search for (e.g., 'Python', 'TypeScript').")],
    ) -> str:
        """Search entities by name/id substring matching in Neo4j knowledge graph."""
        return search_graph(query)

    @mcp.tool()
    def mcp_get_entity(
        name: Annotated[str, Field(description="Exact entity name to look up.")],
    ) -> str:
        """Get all relationships for a specific entity (bidirectional)."""
        return get_entity(name)


# ============================================================
# MCP Prompt
# ============================================================


def _register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts."""

    @mcp.prompt()
    def memory_assistant() -> str:
        """Quick-start guide for using the mem0 memory server."""
        return (
            "You are using the mem0 MCP server for long-term memory management.\n\n"
            "Quick Start:\n"
            "1. Store memories: Use add_memory to save facts, preferences, or conversations\n"
            "2. Search memories: Use search_memories for semantic queries\n"
            "3. Browse memories: Use get_memories for filtered listing\n"
            "4. Update/Delete: Use update_memory and delete_memory for modifications\n"
            "5. Graph exploration: Use search_graph and get_entity for entity relationships\n\n"
            "Tips:\n"
            "- user_id is automatically injected from MEM0_USER_ID default\n"
            "- Set enable_graph=true to include knowledge graph results\n"
            "- Use infer=false to store raw text without LLM extraction\n"
            "- Use threshold on search_memories to filter by relevance score\n"
            "- Use filters for structured queries: {\"key\": {\"eq\": \"value\"}}\n"
        )


# ============================================================
# Server Runner
# ============================================================


def run_server() -> None:
    """Entry point: create server and run.

    Memory initialization is deferred to the first tool call via
    ``_ensure_memory()``, allowing the server to respond to MCP
    ``initialize`` and ``tools/list`` without live infrastructure.
    """
    # Configure logging
    log_level = env("MEM0_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s | %(message)s",
    )

    # Load .env file
    load_dotenv()

    # Create and run server (Memory init deferred to first tool call)
    server = _create_server()

    # Start the ingest worker at boot so jobs left over from a previous run
    # drain without waiting for the first tool call.
    if _async_ingest_enabled():
        try:
            _get_ingest()
        except Exception as exc:
            logger.error("Async ingest init failed (adds fall back to sync): %s", exc)

    transport = env("MEM0_TRANSPORT", "stdio").lower()

    if transport == "sse":
        server.run(transport="sse")
    elif transport == "streamable-http":
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")
