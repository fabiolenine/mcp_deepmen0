# CHANGELOG


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
