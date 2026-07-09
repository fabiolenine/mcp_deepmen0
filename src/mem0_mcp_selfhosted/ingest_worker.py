"""Serial background worker that drains the ingest queue (ingest_queue.py).

One daemon thread, FIFO by ``submitted_at`` — two queued facts about the same
subject always process in submission order, so supersession resolves in the
right direction naturally; the born-superseded rule in the DeepMem0 core
covers the remaining race (a direct infer=false write overtaking the queue).

Job kinds:
- ``conversation`` (v0.4): one mem.add per job.
- ``document`` (v0.5a): spooled PDF -> poppler extraction -> page-aware chunks
  -> one mem.add per chunk with document provenance metadata and a
  document-specific extraction prompt. Progress heartbeats per chunk; between
  chunks the worker drains up to MEM0_DOC_INTERLEAVE_MAX pending conversation
  jobs so a 20-minute document never starves interactive adds.

Idempotency has two halves: the queue dedups client resubmissions
(idempotency_key), and this worker purges Qdrant points tagged with the job's
``task_id`` before each attempt — a crash mid-pipeline never becomes
duplicated ghost memories on retry. The purge also filters on
``created_at == submitted_at``: an UPDATE stamps the updating job's task_id
onto a PRE-EXISTING memory (fork's _update_memory merges metadata) but keeps
its original created_at, so the extra filter spares updated older memories
and deletes only points this job created.

Error classification: infrastructure errors (Ollama down/cold, Qdrant
unreachable, timeouts) are retryable with exponential backoff; payload errors
(validation, malformed JSON, unextractable PDFs) are poison and go straight
to ``dead``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from mem0_mcp_selfhosted.chunking import chunk_pages
from mem0_mcp_selfhosted.document_source import spool_gc
from mem0_mcp_selfhosted.env import bool_env, env
from mem0_mcp_selfhosted.image_extract import (
    VisionUnavailable,
    prepare_vision,
    release_vision,
    transcribe_image,
    vision_enabled,
)
from mem0_mcp_selfhosted.ingest_queue import IngestQueue
from mem0_mcp_selfhosted.pdf_extract import (
    FullyScannedPdf,
    extract_pages,
    pdf_info,
    rasterize_pages,
)

logger = logging.getLogger(__name__)

# Exception names that indicate a bad payload (poison) rather than sick infra.
_POISON_EXC_NAMES = {
    "Mem0ValidationError", "ValidationError", "ValueError", "TypeError", "KeyError",
    "JSONDecodeError", "AttributeError",
}

# Replaces the conversational custom instructions during document-chunk
# extraction: chunks are document excerpts, not dialogue — without this the
# extractor tends to attribute facts to the user ("User prefers...").
DOC_EXTRACTION_INSTRUCTIONS = (
    "These messages are excerpts from a DOCUMENT the user chose to memorize — "
    "not a conversation.\n"
    "- Extract durable, standalone factual statements: definitions, decisions, "
    "numbers, dates, procedures, conclusions.\n"
    "- Attribute facts to the document or its subject matter, never to the user "
    "(do not write \"User prefers/said...\").\n"
    "- Keep each fact self-contained and specific (include names and values); "
    "skip boilerplate, headers, page furniture and bibliographic references.\n"
    "- Write the facts in the same language as the document."
)


def _is_retryable(exc: BaseException) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__ in _POISON_EXC_NAMES:
            return False
    return True


def _observe(event: dict) -> None:
    """Best-effort metric push to OpenObserve (same stream as sitecustomize Patch 6)."""
    url = env("MEM0_OBSERVE_URL")
    if not url:
        return
    try:
        import requests

        event.setdefault("service", "mem0")
        event.setdefault("stage", "ingest_queue")
        event.setdefault("_timestamp", int(time.time() * 1_000_000))
        user, pw = env("MEM0_OBSERVE_USER"), env("MEM0_OBSERVE_PASS")
        requests.post(url, json=[event], auth=(user, pw) if user else None, timeout=3)
    except BaseException:
        pass


def _purge_task_points(mem: Any, task_id: str, created_at: str | None = None) -> None:
    """Delete Qdrant points left by a previous attempt of this job.

    ``created_at`` narrows the purge to points the job itself created
    (created_at == submitted_at is canonical for them); pre-existing memories
    that an UPDATE stamped with this task_id keep their original created_at
    and are spared.
    """
    try:
        vs = getattr(mem, "vector_store", None)
        client = getattr(vs, "client", None)
        if client is None:
            return
        from qdrant_client import models as qm

        must = [qm.FieldCondition(key="task_id", match=qm.MatchValue(value=task_id))]
        if created_at:
            must.append(qm.FieldCondition(
                key="created_at", range=qm.DatetimeRange(gte=created_at, lte=created_at),
            ))
        client.delete(
            collection_name=vs.collection_name,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
        )
    except Exception as e:
        logger.warning("Purge of prior attempt for %s failed (continuing): %s", task_id, e)


def _warmup_ollama() -> None:
    """Preload the extraction model after idleness so the first job's LLM call
    doesn't pay the cold-load inside its own timeout. Best-effort."""
    model = env("MEM0_LLM_MODEL")
    if not model or env("MEM0_LLM_PROVIDER", env("MEM0_PROVIDER", "anthropic")) != "ollama":
        return
    base = env("MEM0_LLM_URL") or env("MEM0_OLLAMA_URL") or "http://localhost:11434"
    try:
        import requests

        requests.post(
            f"{base.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": env("MEM0_OLLAMA_KEEP_ALIVE", "30m")},
            timeout=120,
        )
    except BaseException:
        pass


def _collect_results(raw: Any) -> tuple[list[str], list[dict]]:
    """Normalize a mem.add return into (memory_ids, events)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = {"results": []}
    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    memory_ids = [
        r["id"] for r in results
        if isinstance(r, dict) and r.get("event") in ("ADD", "UPDATE") and r.get("id")
    ]
    events = [
        {k: r.get(k) for k in ("id", "event", "memory", "superseded_by") if r.get(k) is not None}
        for r in results if isinstance(r, dict)
    ]
    return memory_ids, events


class IngestWorker:
    """Single serial consumer. Wake it with notify(); it also polls as fallback."""

    def __init__(
        self,
        queue: IngestQueue,
        memory_provider: Callable[[], Any],
        *,
        call_with_graph: Callable | None = None,
        enable_graph_default: bool = False,
        max_attempts: int | None = None,
        backoff_base_s: float | None = None,
        poll_interval_s: float | None = None,
        idle_warmup_s: float | None = None,
    ):
        self.queue = queue
        self.memory_provider = memory_provider
        self.call_with_graph = call_with_graph
        self.enable_graph_default = enable_graph_default
        self.max_attempts = max_attempts if max_attempts is not None else int(env("MEM0_QUEUE_MAX_ATTEMPTS", "4"))
        self.backoff_base_s = backoff_base_s if backoff_base_s is not None else float(env("MEM0_QUEUE_BACKOFF_BASE", "30"))
        self.poll_interval_s = poll_interval_s if poll_interval_s is not None else float(env("MEM0_QUEUE_POLL_INTERVAL", "2"))
        self.idle_warmup_s = idle_warmup_s if idle_warmup_s is not None else 300.0
        self.gc_interval_s = float(env("MEM0_QUEUE_GC_INTERVAL", "3600"))
        self.done_retention_s = float(env("MEM0_QUEUE_DONE_RETENTION", str(7 * 24 * 3600)))
        self.dead_retention_s = float(env("MEM0_QUEUE_DEAD_RETENTION", str(30 * 24 * 3600)))
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_job_finished = 0.0
        self._last_gc = 0.0

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mem0-ingest-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def notify(self) -> None:
        self._wake.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- main loop ------------------------------------------------------

    def _run(self) -> None:
        try:
            orphans = self.queue.recover_orphans()
            if orphans:
                logger.info("Ingest worker recovered %d orphaned job(s) from previous run", orphans)
        except Exception as e:
            logger.error("Ingest worker boot recovery failed: %s", e)
        logger.info("Ingest worker started (max_attempts=%d, backoff_base=%.0fs)", self.max_attempts, self.backoff_base_s)

        while not self._stop.is_set():
            try:
                job = self.queue.claim_next()
            except Exception as e:
                logger.error("Ingest queue claim failed: %s", e)
                job = None
            if job is None:
                self._maybe_gc()
                # nothing dispatchable: sleep until notify() or the earliest
                # scheduled retry (capped by the poll interval as a fallback)
                try:
                    retry_in = self.queue.next_wakeup_in_s()
                except Exception:
                    retry_in = None
                wait = self.poll_interval_s if retry_in is None else min(max(retry_in, 0.1), self.poll_interval_s * 5)
                self._wake.wait(timeout=wait)
                self._wake.clear()
                continue
            self._process(job)
            self._last_job_finished = time.monotonic()

    def _maybe_gc(self) -> None:
        """Opportunistic terminal-row pruning + spool cleanup on idle."""
        if time.monotonic() - self._last_gc < self.gc_interval_s:
            return
        self._last_gc = time.monotonic()
        try:
            purged = self.queue.gc(
                done_retention_s=self.done_retention_s,
                dead_retention_s=self.dead_retention_s,
            )
            if purged["done_purged"] or purged["dead_purged"]:
                logger.info(
                    "Ingest queue gc: pruned %d done, %d dead job(s)",
                    purged["done_purged"], purged["dead_purged"],
                )
        except Exception as e:
            logger.warning("Ingest queue gc failed: %s", e)
        try:
            removed = spool_gc(self.queue.referenced_doc_hashes())
            if removed:
                logger.info("Spool gc: removed %d unreferenced file(s)", removed)
        except Exception as e:
            logger.warning("Spool gc failed: %s", e)

    # -- job processing --------------------------------------------------

    def _process(self, job: dict[str, Any]) -> None:
        task_id = job["task_id"]
        t0 = time.monotonic()
        mem = None
        try:
            mem = self.memory_provider()
        except Exception as e:
            logger.error("Memory provider raised for %s: %s", task_id, e)
        if mem is None:
            status = self.queue.mark_failed(
                task_id, "Memory not initialized (infrastructure unavailable)",
                retryable=True, max_attempts=self.max_attempts, backoff_base_s=self.backoff_base_s,
            )
            _observe({"event": "ingest_failed", "task_id": task_id, "status": status, "error": "memory_init"})
            return

        idle_for = time.monotonic() - self._last_job_finished
        if self._last_job_finished == 0.0 or idle_for > self.idle_warmup_s:
            if bool_env("MEM0_QUEUE_WARMUP", "true"):
                _warmup_ollama()

        # Unconditional: a crash on the FIRST attempt leaves orphaned points but
        # recover_orphans() resets the job without bumping attempts — gating on
        # attempts>0 would reprocess without cleanup (mass duplication for
        # multi-chunk document jobs). With no prior points this is a cheap no-op.
        _purge_task_points(mem, task_id, created_at=job.get("submitted_at"))

        if job.get("kind", "conversation") == "document":
            self._process_document(mem, job, t0)
        else:
            self._process_conversation(mem, job, t0)

    def _fail(self, job: dict[str, Any], exc: BaseException, t0: float, extra: dict | None = None) -> None:
        task_id = job["task_id"]
        retryable = _is_retryable(exc)
        status = self.queue.mark_failed(
            task_id, f"{type(exc).__name__}: {exc}",
            retryable=retryable, max_attempts=self.max_attempts, backoff_base_s=self.backoff_base_s,
        )
        log = logger.error if status == "dead" else logger.warning
        log("Ingest job %s failed (%s, attempt -> %s): %s", task_id, type(exc).__name__, status, exc)
        event = {
            "event": "ingest_failed", "task_id": task_id, "status": status,
            "kind": job.get("kind", "conversation"), "retryable": retryable,
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
        if extra:
            event.update(extra)
        _observe(event)

    # -- conversation jobs (v0.4) -----------------------------------------

    def _process_conversation(self, mem: Any, job: dict[str, Any], t0: float) -> None:
        task_id = job["task_id"]
        params = job.get("params") or {}
        metadata = dict(params.get("metadata") or {})
        # submitted_at is the canonical fact time: it becomes created_at so the
        # record-time model (as_of, supersession direction) ignores queue delay.
        metadata["created_at"] = job["submitted_at"]
        metadata["task_id"] = task_id

        kwargs: dict[str, Any] = {"metadata": metadata, "infer": True}
        for scope_key in ("user_id", "agent_id", "run_id"):
            if job.get(scope_key):
                kwargs[scope_key] = job[scope_key]

        def _do_add():
            return mem.add(job["messages"], **kwargs)

        try:
            if self.call_with_graph is not None:
                raw = self.call_with_graph(
                    mem, params.get("enable_graph"), self.enable_graph_default, _do_add
                )
            else:
                raw = _do_add()
        except Exception as e:
            self._fail(job, e, t0)
            return

        memory_ids, events = _collect_results(raw)
        result = {
            "memory_ids": memory_ids,
            "events": events,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        if not memory_ids and not events:
            result["reason"] = "no_new_facts"
        self.queue.mark_done(task_id, result)
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info("Ingest job %s done in %dms (%d memory_ids)", task_id, duration_ms, len(memory_ids))
        _observe({
            "event": "ingest_done", "task_id": task_id, "duration_ms": duration_ms,
            "memory_count": len(memory_ids), "queue_depth": _safe_depth(self.queue),
        })

    # -- document jobs (v0.5a) ---------------------------------------------

    def _extract_document_text(self, params: dict) -> tuple[list[tuple[int, str]], list[int], int, list[int]]:
        """Phase A: produce (text_pages, skipped_pages, total_pages, ocr_pages).

        Digital PDF pages come from poppler; scanned pages and standalone
        images go through the VLM (v0.5b) when vision is on. All VLM work is
        confined here — strictly before the llama3.1:8b add-loop — so the GPU
        holds one model at a time (prepare/release force the two swaps).
        """
        spool_path = params.get("spool_path") or ""
        content_type = params.get("content_type") or "application/pdf"
        extract_timeout = float(env("MEM0_DOC_EXTRACT_TIMEOUT", "120"))
        vlm_timeout = float(env("MEM0_VLM_TIMEOUT", "300"))
        if not spool_path or not os.path.isfile(spool_path):
            raise ValueError(f"spool file missing: {spool_path} (was it gc-pruned?)")

        # standalone image: one "page", transcribed by the VLM
        if content_type.startswith("image/"):
            if not vision_enabled():
                raise VisionUnavailable(
                    "image ingestion needs vision (set MEM0_ENABLE_VISION=true and MEM0_VLM_MODEL)"
                )
            prepare_vision()
            try:
                text = transcribe_image(spool_path, timeout_s=vlm_timeout)
            finally:
                release_vision()
            return [(1, text)], [], 1, [1]

        # PDF: digital text from poppler; scanned pages via VLM if enabled
        try:
            pages = extract_pages(spool_path, timeout_s=extract_timeout)
            text_pages = [(p.number, p.text) for p in pages if p.has_text]
            scanned = [p.number for p in pages if not p.has_text]
            total_pages = len(pages)
        except FullyScannedPdf:
            if not vision_enabled():
                raise
            info = pdf_info(spool_path, timeout_s=min(extract_timeout, 30))
            total_pages = info["pages"]
            text_pages = []
            scanned = list(range(1, total_pages + 1))

        ocr_pages: list[int] = []
        if scanned and vision_enabled():
            images = rasterize_pages(
                spool_path, scanned, dpi=int(env("MEM0_VLM_DPI", "150")), timeout_s=extract_timeout,
            )
            prepare_vision()
            try:
                for num in scanned:
                    png = images.get(num)
                    if png is None:
                        continue
                    try:
                        text = transcribe_image(png, timeout_s=vlm_timeout)
                    except Exception as e:
                        logger.warning("OCR of page %d failed (skipping): %s", num, e)
                        continue
                    text_pages.append((num, text))
                    ocr_pages.append(num)
            finally:
                release_vision()
        skipped_pages = sorted(set(scanned) - set(ocr_pages))
        text_pages.sort(key=lambda t: t[0])
        return text_pages, skipped_pages, total_pages, ocr_pages

    def _process_document(self, mem: Any, job: dict[str, Any], t0: float) -> None:
        task_id = job["task_id"]
        params = job.get("params") or {}
        filename = params.get("filename") or "document.pdf"

        try:
            text_pages, skipped_pages, total_pages, ocr_pages = self._extract_document_text(params)
            chunks = chunk_pages(
                text_pages,
                chunk_chars=int(env("MEM0_DOC_CHUNK_CHARS", "1800")),
                overlap=int(env("MEM0_DOC_CHUNK_OVERLAP", "200")),
            )
            if not chunks:
                raise ValueError("document produced no ingestible text")
        except Exception as e:
            self._fail(job, e, t0)
            return

        chunks_total = len(chunks)
        self.queue.update_progress(task_id, {
            "kind": "document", "source_doc": filename,
            "chunks_done": 0, "chunks_total": chunks_total,
        })

        memory_ids: list[str] = []
        events: list[dict] = []
        for chunk in chunks:
            page_ref = (
                f"página {chunk.page_start}" if chunk.page_start == chunk.page_end
                else f"páginas {chunk.page_start}-{chunk.page_end}"
            )
            content = f"Trecho do documento '{filename}' ({page_ref} de {total_pages}):\n\n{chunk.text}"

            metadata = dict(params.get("metadata") or {})
            metadata.update({
                "created_at": job["submitted_at"],
                "task_id": task_id,
                "source_doc": filename,
                "doc_sha256": params.get("doc_sha256"),
                "content_type": params.get("content_type"),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_index": chunk.index,
                "chunks_total": chunks_total,
            })
            kwargs: dict[str, Any] = {
                "metadata": metadata,
                "infer": params.get("infer", True),
                "prompt": DOC_EXTRACTION_INSTRUCTIONS,
            }
            for scope_key in ("user_id", "agent_id", "run_id"):
                if job.get(scope_key):
                    kwargs[scope_key] = job[scope_key]

            def _do_add(content=content, kwargs=kwargs):
                return mem.add([{"role": "user", "content": content}], **kwargs)

            try:
                if self.call_with_graph is not None:
                    # documents default the graph OFF (entity extraction per chunk
                    # would inflate Neo4j and slow every chunk 2-3x)
                    raw = self.call_with_graph(
                        mem, params.get("enable_graph", False), False, _do_add
                    )
                else:
                    raw = _do_add()
            except Exception as e:
                self._fail(job, e, t0, extra={
                    "chunks_done": chunk.index, "chunks_total": chunks_total,
                })
                return

            ids, evts = _collect_results(raw)
            memory_ids.extend(ids)
            events.extend({k: e[k] for k in ("id", "event", "superseded_by") if k in e} for e in evts)
            self.queue.update_progress(task_id, {
                "chunks_done": chunk.index + 1, "chunks_total": chunks_total,
                "memory_ids": memory_ids,
            })
            _observe({
                "event": "ingest_progress", "task_id": task_id, "kind": "document",
                "source_doc": filename, "chunks_done": chunk.index + 1,
                "chunks_total": chunks_total, "memory_count": len(memory_ids),
            })

            self._drain_conversations(mem)

        result = {
            "memory_ids": memory_ids,
            "events": events,
            "source_doc": filename,
            "doc_sha256": params.get("doc_sha256"),
            "pages": total_pages,
            "skipped_pages": skipped_pages,
            "ocr_pages": ocr_pages,
            "chunks_total": chunks_total,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        if not memory_ids:
            result["reason"] = "no_new_facts"
        self.queue.mark_done(task_id, result)
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Document job %s done in %dms (%s: %d chunks, %d memory_ids, %d skipped pages)",
            task_id, duration_ms, filename, chunks_total, len(memory_ids), len(skipped_pages),
        )
        _observe({
            "event": "ingest_done", "task_id": task_id, "kind": "document",
            "duration_ms": duration_ms, "memory_count": len(memory_ids),
            "chunks_total": chunks_total, "queue_depth": _safe_depth(self.queue),
        })

    def _drain_conversations(self, mem: Any) -> None:
        """Between document chunks, let interactive adds jump the line.

        Bounded (MEM0_DOC_INTERLEAVE_MAX per chunk) so the document never
        starves; semantically safe because created_at=submitted_at plus
        born-superseded already make processing order irrelevant.
        """
        if not bool_env("MEM0_DOC_INTERLEAVE", "true"):
            return
        for _ in range(max(0, int(env("MEM0_DOC_INTERLEAVE_MAX", "2")))):
            if self._stop.is_set():
                return
            try:
                other = self.queue.claim_next(only_kind="conversation")
            except Exception:
                return
            if other is None:
                return
            _purge_task_points(mem, other["task_id"], created_at=other.get("submitted_at"))
            self._process_conversation(mem, other, time.monotonic())


def _safe_depth(queue: IngestQueue) -> int:
    try:
        return queue.depth()
    except Exception:
        return -1
