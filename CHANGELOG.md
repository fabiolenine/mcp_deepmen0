# CHANGELOG


## v0.2.0 (2026-07-19)

### Documentation

- Restore fork note in README (lost in snapshot sync)
  ([`4143a4d`](https://github.com/fabiolenine/mcp_deepmen0/commit/4143a4d649e7cdbf47add759cd6bbe64da1830bf))

### Features

- **scope**: Passive memory_scope passthrough (ontology v1, step 2)
  ([`0bba795`](https://github.com/fabiolenine/mcp_deepmen0/commit/0bba795bdce85d2dc214ef5ef96422674b316d72))

Validates the 4-value scope enum (or null = absence) and leveled evidence on
  add_memory/add_document, stamps default provenance (version=1, source=manual), and exposes
  memory_scope + provenance fields through the metadata whitelist. No routing, no search behavior
  change — scope-aware retrieval is a later step gated on its own eval. Mirrors deepmen0 0.6.0
  (promoted key + keyword index).


## v0.1.0 (2026-07-14)

### Features

- **update**: Make update_memory asynchronous via the durable queue
  ([`cd51fb6`](https://github.com/fabiolenine/mcp_deepmen0/commit/cd51fb60443dcc0d6a68b24fe1c24dc9af2b1ae9))

Mirrors add_memory's async contract with a new kind="update": the tool validates the memory exists +
  resolves owner scope at submit, enqueues, and returns {status:"queued", task_id} immediately; the
  worker re-embeds + re-classifies the metadata in the background. An identical re-submit while the
  job is active returns the same task_id (sentinel idempotency key) — no double-apply. Fixes the
  ambiguous client-timeout on the previously synchronous update path (the update succeeded
  server-side while the client saw a timeout).


## v0.0.0 (2026-07-09)
