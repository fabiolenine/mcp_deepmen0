# CHANGELOG


## v0.3.2 (2026-03-13)

### Bug Fixes

- Cache-bust Glama badge URL to force fresh camo proxy fetch
  ([`205ecf9`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/205ecf9a6d8d95f23fa0d8fa27826e3348ab0728))


## v0.3.1 (2026-03-12)

### Bug Fixes

- Add .python-version for Glama uv sync compatibility
  ([`e4d1f09`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/e4d1f09008652a84ed1340db9372f621b8ffa785))

Pin Python 3.12 so uv sync resolves the correct interpreter in Glama's Docker build environment
  instead of picking up Debian's externally-managed Python 3.11.

### Chores

- Remove Dockerfile (Glama generates its own)
  ([`33f2f1d`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/33f2f1d25bdb1e4c85617e90b21a72c48fc9c2a2))

Glama's admin page generates a Dockerfile from configuration fields rather than using the repo's
  Dockerfile. No other Docker deployment workflow exists, so the file is unused.


## v0.3.0 (2026-03-12)

### Features

- Lazy Memory init + Glama submission packaging
  ([`c6f2b76`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/c6f2b76aa7fc1f243c86fbcd941825ef7861b539))

Defer Memory.from_config() to the first tool call via _ensure_memory(), allowing the MCP server to
  respond to initialize/tools/list without live Qdrant/Neo4j/Ollama. This unblocks Glama's
  Docker-based inspection pipeline which builds and runs the container in an ephemeral sandbox.

Add LICENSE (MIT), glama.json, Dockerfile, and Glama badge in README.


## v0.2.1 (2026-02-28)

### Bug Fixes

- Update hooks to nested format for Claude Code schema compatibility
  ([`2f86dee`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/2f86dee99c3fa73220270b721c1621881beea655))

Migrate hook installer from the deprecated flat format to the current nested schema (matcher group
  -> hooks array -> handler objects). Add legacy format detection and auto-migration so existing
  users upgrading do not end up with duplicate or broken entries.

### Documentation

- Clarify hooks and CLAUDE.md as complementary layers
  ([`94f29dc`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/94f29dca52582ee18ce9ae256fc06d8cf1adab30))

Update README to explain that hooks (automated memory at session boundaries) and CLAUDE.md
  (behavioral instructions for mid-session engagement) work best together rather than as
  alternatives.


## v0.2.0 (2026-02-28)

### Features

- Add Claude Code session hooks for cross-session memory
  ([`113df26`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/113df2678b05091dd0acffa2776c755d4c380644))

Add SessionStart and Stop hooks that give Claude Code automatic cross-session memory without
  requiring CLAUDE.md rules or manual tool calls.

- SessionStart hook (mem0-hook-context): searches mem0 with multi-query strategy, deduplicates by
  ID, injects formatted memories as additionalContext on startup and compact events - Stop hook
  (mem0-hook-stop): reads last ~3 exchanges from JSONL transcript via bounded deque, saves session
  summary to mem0 with infer=True for atomic fact extraction - CLI installer (mem0-install-hooks):
  patches .claude/settings.json with idempotent hook entries, supports --global and --project-dir -
  Graph force-disabled in hooks to stay within 15s/30s timeout budgets - Atomic settings.json write
  via tempfile + os.replace - 43 unit tests covering protocol, edge cases, and error handling - 6
  integration tests against live Qdrant + Ollama infrastructure - README updated with hooks
  documentation, architecture diagram, and test structure


## v0.1.1 (2026-02-27)

### Bug Fixes

- Use NEO4J_DATABASE env var instead of config dict for non-default database
  ([`74e1188`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/74e1188d38154846ec8b12602fde1d757197873b))

mem0ai's graph_memory.py passes config as positional args to Neo4jGraph() where pos 3 is `token`,
  not `database`. Setting database in the config dict causes it to land in the token parameter,
  resulting in AuthenticationError. Use NEO4J_DATABASE env var which langchain_neo4j reads via
  get_from_dict_or_env().

Upstream: mem0ai #3906, #3981, #4085 (none merged)

Resolves: PAR-57


## v0.1.0 (2026-02-27)

### Bug Fixes

- **ci**: Use angular parser compatible with PSR v9.15.2
  ([`b5bc6ab`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/b5bc6ab45edff26f07fc73774c7e0c57d22cb40d))

The v9 GitHub Action does not recognize "conventional" parser name (v10+ only). Reverts to "angular"
  and changelog.changelog_file format.

### Continuous Integration

- Add python-semantic-release configuration and GitHub Actions workflow
  ([`2473ee4`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/2473ee4ec9c0db90b2bb412d3714caae7dc41498))

Automated versioning via Conventional Commits analysis, changelog generation, git tagging
  (v{version}), and GitHub Release creation on push to main.
