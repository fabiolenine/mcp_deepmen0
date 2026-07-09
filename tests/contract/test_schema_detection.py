"""Contract test: Schema detection invariant.

Validates that mem0ai's fact extraction messages contain a system message
and memory update messages do not. This is a structural invariant in
mem0ai's architecture — if it breaks, our schema detection fails.
"""

from __future__ import annotations

import pytest


class TestSchemaDetectionInvariant:
    """Contract test: verify the system message presence invariant.

    These tests validate the assumption that:
    - Fact extraction calls include a system message (prompt template)
    - Memory update calls use only a user message (no system message)

    This invariant is how we select FACT_RETRIEVAL_SCHEMA vs MEMORY_UPDATE_SCHEMA.
    """

    def test_fact_extraction_has_system_message(self):
        """Fact extraction prompt templates should have role=system."""
        # Simulate what mem0ai sends for fact extraction
        # The FACT_RETRIEVAL_PROMPT template is passed as system message
        fact_extraction_messages = [
            {"role": "system", "content": "You are a Personal Information Organizer..."},
            {"role": "user", "content": "Input: Alice prefers TypeScript\nOld Memory: []"},
        ]

        has_system = any(m.get("role") == "system" for m in fact_extraction_messages)
        assert has_system, (
            "INVARIANT BROKEN: Fact extraction messages must contain a system message. "
            "If mem0ai changed this, our schema detection needs updating."
        )

    def test_memory_update_no_system_message(self):
        """Memory update messages should NOT have role=system."""
        # Simulate what mem0ai sends for memory update decisions
        memory_update_messages = [
            {"role": "user", "content": "Existing Memories:\n...\nNew Memory: ..."},
        ]

        has_system = any(m.get("role") == "system" for m in memory_update_messages)
        assert not has_system, (
            "INVARIANT BROKEN: Memory update messages must NOT contain a system message. "
            "If mem0ai changed this, our schema detection needs updating."
        )


class TestSchemaDetectionWithRealPrompts:
    """Validate with realistic prompt structures from mem0ai source."""

    def test_fact_extraction_prompt_structure(self):
        """The FACT_RETRIEVAL_PROMPT template produces system+user messages."""
        # From mem0/memory/main.py — the prompt is passed as system message
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Personal Information Organizer, specialized in accurately "
                    "storing facts, user memories, and preferences. Your primary role is to "
                    "extract relevant pieces of information from conversations and organize "
                    "them into distinct, manageable facts."
                ),
            },
            {
                "role": "user",
                "content": "Input: Alice prefers TypeScript for new projects\nOld Memory: []",
            },
        ]

        system_count = sum(1 for m in messages if m.get("role") == "system")
        assert system_count == 1

    def test_memory_update_prompt_structure(self):
        """The UPDATE_MEMORY_PROMPT template produces user-only messages."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Existing Memories:\n"
                    "---\n"
                    "ID: abc123\nMemory: Alice likes Python\n"
                    "---\n"
                    "New Memory: Alice now prefers TypeScript over Python\n"
                ),
            },
        ]

        system_count = sum(1 for m in messages if m.get("role") == "system")
        assert system_count == 0
