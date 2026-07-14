# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## MCP Servers

- **mem0**: Persistent memory across sessions. At the start of each session, `search_memories` for relevant context before asking the user to re-explain anything. Use `add_memory` whenever you discover project architecture, coding conventions, debugging insights, key decisions, or user preferences. Use `update_memory` when prior context changes. Save information like: "This project uses PostgreSQL with Prisma", "Tests run with pytest -v", "Auth uses JWT validated in middleware". When in doubt, save it — future sessions benefit from over-remembering.

## Build & Test Commands

```bash
pip install -e ".[dev]"              # Install with dev dependencies
python3 -m pytest tests/unit/ -v     # Unit tests (mocked, no infra needed)
python3 -m pytest tests/contract/ -v # Contract tests (validates mem0ai internals)
python3 -m pytest tests/integration/ -v  # Integration tests (requires live Qdrant + Neo4j + Ollama)
python3 -m pytest tests/ -v          # All tests
python3 -m pytest tests/ -m "not integration" -v  # Skip integration
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken -v  # Single test class
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken::test_oat_token_detected -v  # Single test
```

## Architecture

Self-hosted MCP server using `mem0ai` as a library. 15 tools (13 memory + 2 graph), FastMCP orchestrator.

**Module roles:**
- `server.py` — FastMCP orchestrator, registers all tools + `memory_assistant` prompt; owns the async ingest wiring (`_get_ingest()`, envelope contract in `add_memory`, submit-time validation in `add_document`)
- `ingest_queue.py` — Durable SQLite (WAL) queue for async `add_memory`/`add_document`: job kinds, idempotent enqueue, atomic FIFO claim (optionally kind-filtered), partial progress + heartbeat, exponential backoff → dead-letter, orphan recovery, retention gc (done 7d / dead 30d)
- `ingest_worker.py` — Single serial daemon thread draining the queue: injects `submitted_at` as `created_at` (canonical fact time) + `task_id` provenance, purge-on-retry (scoped by task_id AND created_at so UPDATEd pre-existing memories survive), poison-vs-retryable classification, Ollama warm-up, OpenObserve emit; document branch chunks PDFs and interleaves conversation adds between chunks; update branch (`kind="update"`) re-embeds + re-classifies an existing memory without stamping a new `created_at`
- `document_source.py` — file_path validation (allowlist/realpath/magic/caps) + content-addressed spool (`<sha256>.pdf`) with reference-counting gc
- `pdf_extract.py` — poppler wrapper: pdfinfo metadata, single-pass pdftotext + per-page fallback, NFC + dehyphenation, per-page text/scanned classification; `rasterize_pages()` renders pages to PNG (temp file — this poppler won't write PNG to stdout) for OCR
- `image_extract.py` — local Ollama VLM transcription (v0.5b): `transcribe_image()` for scanned pages and standalone images; `prepare_vision()`/`release_vision()` force the two model swaps (VLM and llama3.1:8b don't co-fit in 8GB); gated by `MEM0_ENABLE_VISION`+`MEM0_VLM_MODEL`
- `chunking.py` — pure page-aware chunker (no project imports; promotable to the fork core)
- `config.py` — Env vars → mem0ai `MemoryConfig` dict, handles all 5 graph LLM provider configs
- `auth.py` — 3-tier token fallback: `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY`
- `llm_anthropic.py` — Custom Anthropic provider registered with mem0ai's `LlmFactory`; handles OAT headers, structured outputs (JSON schema via `output_config`), and tool-call parsing
- `llm_router.py` — `SplitModelGraphLLM` routes by tool name: extraction tools → Gemini, contradiction tools → Claude
- `helpers.py` — `_mem0_call()` error wrapper, `call_with_graph()` threading lock for per-call graph toggle, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `patch_graph_sanitizer()` monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- `graph_tools.py` — Direct Neo4j Cypher queries with lazy driver init
- `__init__.py` — Suppresses mem0ai telemetry before any imports

**Critical implementation details:**
- `add_memory` with `infer=true` (default) is ASYNCHRONOUS when `MEM0_ASYNC_INGEST` != false: it returns `{"status": "queued", "task_id", ...}` immediately; results come via `memory_task_status`. Tests that want the sync path set `MEM0_ASYNC_INGEST=false` (the shared conftest does this globally and isolates the queue DB + doc spool in tmp)
- `update_memory` is ALSO ASYNCHRONOUS by default (same `MEM0_ASYNC_INGEST` switch, `kind="update"`): it validates the memory exists (`mem.get`, fail-fast on a bad id) and resolves the owner scope at submit, then enqueues and returns the queued envelope. The slow part is the metadata classifier (Patch 3 — an inline llama3.1:8b call, `MEM0_CLASSIFY_TIMEOUT`=300s) that used to time out synchronous MCP clients while the update actually succeeded server-side; running it in the worker removes that ambiguity. Idempotency key = scope + a sentinel message encoding `memory_id`+`text`, so an identical re-submit while the job is active collapses to the same `task_id` (no double-apply); the worker reads the real `memory_id`/`text` from `params`. The update preserves the memory's original `created_at`, so purge-on-retry (matches `created_at == submitted_at`) spares it. Sync fallback returns `{"message": "Memory updated successfully!"}` (the conftest sync path)
- `add_document` has NO synchronous fallback (a 20-minute document would hang any MCP client); a broken queue errors. Re-submitting the same bytes+scope returns `already_ingested` until the done row is gc-pruned (`force=true` escapes). Test PDFs are built in-code by `tests/pdf_builder.py` — never commit binary fixtures
- Vision (v0.5b) is off by default; `MEM0_ENABLE_VISION=true` + `MEM0_VLM_MODEL` (an `-instruct` VLM — the bare `qwen3-vl` thinking model returns empty transcriptions) turns on OCR for scanned pages and image ingestion. The VLM and the llama3.1:8b extractor don't co-fit in 8GB VRAM, so transcription (phase A) and fact extraction (phase B) are strictly separated with a model swap between; interleave runs only in phase B. Vision worker tests mock `rasterize_pages`/`transcribe_image` (poppler/VLM isolation)
- `memory.delete()` does NOT clean Neo4j nodes (mem0ai bug #3245) — `safe_bulk_delete()` explicitly calls `memory.graph.delete_all(filters)` after
- `memory.enable_graph` is mutable instance state — `call_with_graph()` holds a `threading.Lock` for the full duration of each Memory call (2-20s)
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- `Memory.update()` uses `data=` parameter, not `text=`
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- mem0ai's `sanitize_relationship_for_cypher()` has gaps (no hyphen handling, no leading-digit check) — `patch_graph_sanitizer()` wraps it at startup to ensure all relationship types match `^[a-zA-Z_][a-zA-Z0-9_]*$`
