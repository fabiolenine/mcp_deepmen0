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

        count = safe_bulk_delete(memory_instance, {"user_id": test_user_id})

        assert count >= 1  # LLM may merge similar facts; at least 1 must exist

        remaining = memory_instance.get_all(user_id=test_user_id)
        assert len(remaining.get("results", [])) == 0

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
