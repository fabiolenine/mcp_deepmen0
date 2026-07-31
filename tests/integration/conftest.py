"""Integration test fixtures — skip gracefully when infrastructure is unavailable.

Fixture hierarchy:
    qdrant_url (session)  +  ollama_url (session)
                          ↓
                  memory_instance (session)  ←  uses mem0_integration_test collection
                          ↓                      drops collection in teardown
                  test_user_id (function)    ←  unique per test: "inttest-<test_name>"

    neo4j_available (session)  ←  independent, only for graph tests
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request

import pytest


def _load_dotenv_once() -> None:
    """Load .env for integration tests only.

    Unit tests use monkeypatch/patch.dict and must not be affected by a
    developer's local .env file.  Shell env vars take precedence
    (python-dotenv's default ``override=False``).
    """
    from dotenv import load_dotenv

    load_dotenv()


_load_dotenv_once()

# Mark all integration tests
pytestmark = pytest.mark.integration

TEST_COLLECTION = "mem0_integration_test"


def _resolve_qdrant_api_key() -> str | None:
    """Env, else the repo-root ``.qdrant.env`` — same resolution the offline
    scripts use (``scripts/qdrant_auth.py``).

    This Qdrant REQUIRES auth (since 2026-07-20). The integration suite was
    missed when the offline scripts were fixed, so it 401'd at fixture setup
    for anyone who had not exported the key by hand — which is why nobody
    noticed that ``safe_bulk_delete``'s return type had changed under it.
    ``/healthz`` is unauthenticated, so the health check passed and the failure
    only surfaced deep inside the first real operation.
    """
    key = os.environ.get("MEM0_QDRANT_API_KEY")
    if key:
        return key
    # tests/integration/conftest.py -> tests/ -> mem0-mcp-selfhosted/ -> repo root
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    try:
        with open(os.path.join(root, ".qdrant.env"), encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("MEM0_QDRANT_API_KEY="):
                    val = s.split("=", 1)[1]
                    if val:
                        os.environ["MEM0_QDRANT_API_KEY"] = val
                        return val
    except OSError:
        pass
    return None


@pytest.fixture(scope="session")
def qdrant_api_key():
    return _resolve_qdrant_api_key()


@pytest.fixture(scope="session")
def qdrant_url(qdrant_api_key):
    """Health-check Qdrant REST API; skip entire suite if unreachable.

    Probes an AUTHENTICATED endpoint: ``/healthz`` answers without a key, so it
    would happily green-light a suite that cannot do a single real operation.
    """
    url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")
    req = urllib.request.Request(
        f"{url}/collections",
        headers={"api-key": qdrant_api_key} if qdrant_api_key else {},
    )
    try:
        urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            pytest.skip(
                f"Qdrant at {url} requires an api-key and none was found "
                "(env MEM0_QDRANT_API_KEY or <repo>/.qdrant.env)"
            )
        pytest.skip(f"Qdrant not usable at {url}: {exc}")
    except Exception:
        pytest.skip(f"Qdrant not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def ollama_url():
    """Health-check Ollama API; skip entire suite if unreachable."""
    url = os.environ.get("MEM0_EMBED_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=3)
    except Exception:
        pytest.skip(f"Ollama not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def neo4j_available():
    """TCP-check Neo4j bolt port; skip graph tests if unreachable."""
    import mem0_mcp_selfhosted.graph_tools as gt

    url = os.environ.get("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
    # Parse host:port from bolt://host:port
    stripped = url.replace("bolt://", "").replace("neo4j://", "")
    if ":" in stripped:
        host, port_str = stripped.rsplit(":", 1)
        port = int(port_str)
    else:
        host = stripped
        port = 7687

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        sock.connect((host, port))
        sock.close()
    except (socket.timeout, ConnectionRefusedError, OSError):
        pytest.skip(f"Neo4j not reachable at {host}:{port}")

    # Reset lazy driver so it re-initializes from test env vars
    gt._driver = None
    yield
    gt._driver = None


def _is_neo4j_reachable() -> bool:
    """Check Neo4j reachability without skipping — for conditional graph init."""
    url = os.environ.get("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
    stripped = url.replace("bolt://", "").replace("neo4j://", "")
    if ":" in stripped:
        host, port_str = stripped.rsplit(":", 1)
        port = int(port_str)
    else:
        host = stripped
        port = 7687

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


@pytest.fixture(scope="session")
def memory_instance(qdrant_url, ollama_url):
    """Create a real Memory instance against live infrastructure.

    Uses a dedicated test collection, dropped entirely in teardown.
    Conditionally enables graph support when Neo4j is reachable.
    Anthropic token is only required when the resolved provider is anthropic.

    ⚠️ O provedor é resolvido com a MESMA precedência de `build_config()`
    (`MEM0_LLM_PROVIDER` -> `MEM0_PROVIDER` -> "anthropic"). Este fixture
    ignorava `MEM0_PROVIDER` e fixava "anthropic" na mão, o que é uma SEGUNDA
    definição de "qual LLM usamos" — e ela divergiu: o deployment roda
    `MEM0_PROVIDER=ollama`, sem token Anthropic nenhum, então a suíte
    exercitava um provedor que produção não usa e caía em
    `429 rate_limit_error` na fase de setup. MEDIDO: 8 falhas com o default
    antigo; 15 passed / 4 skipped com o provedor real.
    """
    llm_provider = (os.environ.get("MEM0_LLM_PROVIDER")
                    or os.environ.get("MEM0_PROVIDER")
                    or "anthropic")
    if llm_provider == "anthropic":
        from mem0_mcp_selfhosted.auth import resolve_token

        token = resolve_token()
        if not token:
            pytest.skip("No Anthropic token available (resolved provider is anthropic)")
    # Override collection name for test isolation
    original_collection = os.environ.get("MEM0_COLLECTION")
    original_graph = os.environ.get("MEM0_ENABLE_GRAPH")
    os.environ["MEM0_COLLECTION"] = TEST_COLLECTION

    # Enable graph if Neo4j is reachable (non-graph tests still work either way)
    neo4j_reachable = _is_neo4j_reachable()
    if neo4j_reachable:
        os.environ["MEM0_ENABLE_GRAPH"] = "true"

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info, split_config = build_config()
    register_providers(providers_info)

    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig
        memory.graph.llm = SplitModelGraphLLM(SplitModelGraphLLMConfig(**split_config))

    yield memory

    # Teardown: drop the entire test collection
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, api_key=_resolve_qdrant_api_key())
        client.delete_collection(TEST_COLLECTION)
        # The entity store is a SECOND collection; leaving it behind was how
        # `*_entities` leftovers accumulated from test runs.
        try:
            client.delete_collection(f"{TEST_COLLECTION}_entities")
        except Exception:
            pass
    except Exception:
        pass  # Collection may not exist

    # Restore original env
    if original_collection is not None:
        os.environ["MEM0_COLLECTION"] = original_collection
    else:
        os.environ.pop("MEM0_COLLECTION", None)
    if original_graph is not None:
        os.environ["MEM0_ENABLE_GRAPH"] = original_graph
    else:
        os.environ.pop("MEM0_ENABLE_GRAPH", None)


@pytest.fixture
def test_user_id(request):
    """Generate a unique user_id per test function."""
    return f"inttest-{request.node.name}"
