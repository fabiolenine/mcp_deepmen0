"""Integration tests: safe_bulk_delete and list_entities_facet against real Qdrant.

These validate custom code paths that bypass mem0ai's memory.delete_all().
"""

from __future__ import annotations

import pytest

from mem0_mcp_selfhosted.helpers import list_entities_facet, safe_bulk_delete

pytestmark = pytest.mark.integration


class TestBulkOperations:
    def test_safe_bulk_delete(self, memory_instance, test_user_id):
        """Add 3 memories, bulk-delete them, verify all removed."""
        personal_facts = [
            "I prefer Python for backend development and use FastAPI for APIs",
            "My favorite database is PostgreSQL for relational data",
            "I am learning Rust for systems programming",
        ]
        for content in personal_facts:
            memory_instance.add(
                [{"role": "user", "content": content}],
                user_id=test_user_id,
            )

        result = safe_bulk_delete(memory_instance, {"user_id": test_user_id})

        assert result.deleted >= 1  # LLM may merge similar facts; at least 1 must exist
        assert result.failed_ids == []
        # `complete` is the field that was structurally missing: a count alone
        # could not distinguish "drained the scope" from "drained one page".
        assert result.vector_scope_drained is True
        assert result.remaining_ids == []

        remaining = memory_instance.get_all(user_id=test_user_id)
        assert len(remaining.get("results", [])) == 0

    def test_safe_bulk_delete_crosses_the_page_boundary(self, memory_instance, test_user_id):
        """The production defect: `list()` defaults to top_k=100, so a scope
        larger than one page was silently half-deleted and reported as done.

        Seeds raw points (no LLM) — this exercises the paging logic, not extraction.
        """
        import uuid

        client = memory_instance.vector_store.client
        collection = memory_instance.vector_store.collection_name
        dims = memory_instance.vector_store.embedding_model_dims
        scope = f"{test_user_id}-page"
        points = [
            {"id": str(uuid.uuid4()), "vector": [0.0] * dims,
             "payload": {"data": f"page probe {i}", "user_id": scope}}
            for i in range(150)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)

        result = safe_bulk_delete(memory_instance, {"user_id": scope}, page_size=100)

        assert result.deleted == 150, "one page (100) would be the old behaviour"
        assert result.vector_scope_drained is True
        assert memory_instance.get_all(user_id=scope).get("results", []) == []

    def test_list_entities_facet(self, memory_instance):
        """Add memories with distinct user_ids, verify Facet API returns them."""
        user_a = "inttest-facet-user-a"
        user_b = "inttest-facet-user-b"

        # Add 2 memories for user_a
        memory_instance.add(
            [{"role": "user", "content": "I enjoy hiking in the Rocky Mountains every summer"}],
            user_id=user_a,
        )
        memory_instance.add(
            [{"role": "user", "content": "My favorite programming language is Go for microservices"}],
            user_id=user_a,
        )
        # Add 1 memory for user_b
        memory_instance.add(
            [{"role": "user", "content": "I work as a data engineer at a startup in Austin"}],
            user_id=user_b,
        )

        result = list_entities_facet(memory_instance)

        assert "users" in result
        user_map = {u["value"]: u["count"] for u in result["users"]}
        assert user_map.get(user_a, 0) >= 1  # At least 1 extracted fact per user
        assert user_map.get(user_b, 0) >= 1
