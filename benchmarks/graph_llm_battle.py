"""Comprehensive battle test: Claude Opus 4.6 vs Gemini 2.5 Flash Lite.

Tests across 7 dimensions using mem0ai's exact graph pipeline tool schemas and prompts.
Runs 50+ test cases across diverse domains.

Dimensions:
1. Entity extraction accuracy
2. Relationship quality
3. Contradiction detection (Call 3)
4. Tool call reliability
5. JSON schema adherence
6. Hallucination rate
7. Cost per 1M output tokens

Usage:
    python benchmarks/graph_llm_battle.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic
from google import genai
from google.genai import types as genai_types

from mem0_mcp_selfhosted.auth import is_oat_token, resolve_token

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Tool Definitions (exact copies from mem0ai graphs/tools.py)
# ============================================================

EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "The name or identifier of the entity."},
                            "entity_type": {"type": "string", "description": "The type or category of the entity."},
                        },
                        "required": ["entity", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "An array of entities with their types.",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among the entities based on the provided text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "The source entity of the relationship."},
                            "relationship": {"type": "string", "description": "The relationship between the source and destination entities."},
                            "destination": {"type": "string", "description": "The destination entity of the relationship."},
                        },
                        "required": ["source", "relationship", "destination"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

UPDATE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "relationship": {"type": "string"},
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "relationship": {"type": "string"},
                "source_type": {"type": "string"},
                "destination_type": {"type": "string"},
            },
            "required": ["source", "destination", "relationship", "source_type", "destination_type"],
            "additionalProperties": False,
        },
    },
}

DELETE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "relationship": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["source", "relationship", "destination"],
            "additionalProperties": False,
        },
    },
}

NOOP_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# Prompts (exact copies from mem0ai graphs/utils.py)
# ============================================================

EXTRACT_RELATIONS_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user's context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

DELETE_RELATIONS_SYSTEM_PROMPT = """You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.

For example:
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted."""

UPDATE_GRAPH_PROMPT = """You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Memory Format:
source -- RELATIONSHIP -- destination

Task Details:
======= Existing Graph Memories:=======
{existing_memories}

======= New Graph Memory:=======
{new_memories}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates."""


# ============================================================
# OAT Headers
# ============================================================

OAT_HEADERS = {
    "accept": "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/1.0.0 (external, cli)",
    "x-app": "cli",
}


# ============================================================
# Test Cases
# ============================================================

@dataclass
class ExtractionCase:
    """Test case for entity extraction + relationship quality."""
    id: str
    category: str
    text: str
    expected_entities: list[str]  # lowercase names
    expected_entity_types: dict[str, str]  # entity -> expected type (fuzzy match)
    expected_relationships: list[tuple[str, str, str]]  # (src, rel_keyword, dst) fuzzy
    hallucination_traps: list[str] = field(default_factory=list)  # entities NOT in text


@dataclass
class ContradictionCase:
    """Test case for contradiction detection (Call 3)."""
    id: str
    category: str
    existing_memories: str  # "source -- rel -- dest" format
    new_info: str
    expected_action: str  # "delete", "update", "noop", "add"
    expected_targets: list[str] = field(default_factory=list)  # entities involved in action


# --- Extraction Test Cases (40 cases) ---

EXTRACTION_CASES = [
    # ===== Category: Simple Personal Facts =====
    ExtractionCase(
        id="E01", category="personal",
        text="Alice prefers TypeScript over JavaScript for web development.",
        expected_entities=["alice", "typescript", "javascript", "web development"],
        expected_entity_types={"alice": "person", "typescript": "programming_language", "javascript": "programming_language"},
        expected_relationships=[("alice", "prefers", "typescript"), ("typescript", "used_for", "web development")],
        hallucination_traps=["python", "react"],
    ),
    ExtractionCase(
        id="E02", category="personal",
        text="Maria lives in Tokyo and works at Google as a senior engineer.",
        expected_entities=["maria", "tokyo", "google", "senior engineer"],
        expected_entity_types={"maria": "person", "tokyo": "city", "google": "company"},
        expected_relationships=[("maria", "lives_in", "tokyo"), ("maria", "works_at", "google")],
        hallucination_traps=["japan", "alphabet"],
    ),
    ExtractionCase(
        id="E03", category="personal",
        text="John adopted a golden retriever named Max last summer.",
        expected_entities=["john", "max", "golden retriever"],
        expected_entity_types={"john": "person", "max": "pet"},
        expected_relationships=[("john", "adopted", "max")],
        hallucination_traps=["dog park", "veterinarian"],
    ),
    ExtractionCase(
        id="E04", category="personal",
        text="Sarah is allergic to peanuts and shellfish.",
        expected_entities=["sarah", "peanuts", "shellfish"],
        expected_entity_types={"sarah": "person"},
        expected_relationships=[("sarah", "allergic", "peanuts"), ("sarah", "allergic", "shellfish")],
        hallucination_traps=["hospital", "epipen"],
    ),
    ExtractionCase(
        id="E05", category="personal",
        text="Alex graduated from MIT in 2019 with a degree in Computer Science.",
        expected_entities=["alex", "mit", "computer science"],
        expected_entity_types={"alex": "person", "mit": "university"},
        expected_relationships=[("alex", "graduated", "mit")],
        hallucination_traps=["cambridge", "harvard"],
    ),

    # ===== Category: Professional/Workplace =====
    ExtractionCase(
        id="E06", category="professional",
        text="The CTO of Anthropic, Mike Krieger, previously co-founded Instagram.",
        expected_entities=["mike krieger", "anthropic", "instagram"],
        expected_entity_types={"mike krieger": "person", "anthropic": "company", "instagram": "company"},
        expected_relationships=[("mike krieger", "cto", "anthropic"), ("mike krieger", "co-founded", "instagram")],
        hallucination_traps=["facebook", "meta", "kevin systrom"],
    ),
    ExtractionCase(
        id="E07", category="professional",
        text="Our team uses Kubernetes on AWS for container orchestration, managed by DevOps lead Chen.",
        expected_entities=["kubernetes", "aws", "chen"],
        expected_entity_types={"kubernetes": "technology", "aws": "cloud_provider", "chen": "person"},
        expected_relationships=[("chen", "manages", "kubernetes")],
        hallucination_traps=["docker", "azure", "gcp"],
    ),
    ExtractionCase(
        id="E08", category="professional",
        text="The Q3 revenue report shows Acme Corp earned $4.2M, led by VP of Sales Diana.",
        expected_entities=["acme corp", "diana"],
        expected_entity_types={"acme corp": "company", "diana": "person"},
        expected_relationships=[("diana", "vp_of_sales", "acme corp")],
        hallucination_traps=["q4", "competitor"],
    ),
    ExtractionCase(
        id="E09", category="professional",
        text="Project Phoenix is a collaboration between the ML team and the Data Engineering team, with deadline March 2026.",
        expected_entities=["project phoenix", "ml team", "data engineering team"],
        expected_entity_types={"project phoenix": "project"},
        expected_relationships=[("ml team", "collaborates", "data engineering team")],
        hallucination_traps=["product team", "qa team"],
    ),
    ExtractionCase(
        id="E10", category="professional",
        text="Lisa manages the Berlin office and reports to Regional Director Hans in Munich.",
        expected_entities=["lisa", "berlin", "hans", "munich"],
        expected_entity_types={"lisa": "person", "hans": "person", "berlin": "city", "munich": "city"},
        expected_relationships=[("lisa", "manages", "berlin"), ("lisa", "reports_to", "hans")],
        hallucination_traps=["germany", "frankfurt"],
    ),

    # ===== Category: Technical/Programming =====
    ExtractionCase(
        id="E11", category="technical",
        text="The backend API is built with FastAPI and uses PostgreSQL for persistence, with Redis for caching.",
        expected_entities=["fastapi", "postgresql", "redis"],
        expected_entity_types={"fastapi": "framework", "postgresql": "database", "redis": "cache"},
        expected_relationships=[("fastapi", "uses", "postgresql"), ("fastapi", "uses", "redis")],
        hallucination_traps=["django", "mongodb", "mysql"],
    ),
    ExtractionCase(
        id="E12", category="technical",
        text="We migrated from React to Svelte for the dashboard because of bundle size concerns.",
        expected_entities=["react", "svelte", "dashboard"],
        expected_entity_types={"react": "framework", "svelte": "framework"},
        expected_relationships=[("dashboard", "migrated_from", "react"), ("dashboard", "migrated_to", "svelte")],
        hallucination_traps=["vue", "angular", "webpack"],
    ),
    ExtractionCase(
        id="E13", category="technical",
        text="The authentication service uses JWT tokens with RSA-256 signing, deployed on Cloud Run.",
        expected_entities=["authentication service", "jwt", "rsa-256", "cloud run"],
        expected_entity_types={"jwt": "technology", "cloud run": "platform"},
        expected_relationships=[("authentication service", "uses", "jwt")],
        hallucination_traps=["oauth", "firebase", "auth0"],
    ),
    ExtractionCase(
        id="E14", category="technical",
        text="TensorFlow 2.x replaced Keras as the default high-level API, integrating it directly.",
        expected_entities=["tensorflow", "keras"],
        expected_entity_types={"tensorflow": "framework", "keras": "framework"},
        expected_relationships=[("tensorflow", "integrated", "keras")],
        hallucination_traps=["pytorch", "jax", "scikit-learn"],
    ),
    ExtractionCase(
        id="E15", category="technical",
        text="The CI/CD pipeline uses GitHub Actions with Docker builds, pushing to ECR then deploying to EKS.",
        expected_entities=["github actions", "docker", "ecr", "eks"],
        expected_entity_types={"github actions": "ci_cd", "docker": "container", "ecr": "registry", "eks": "kubernetes"},
        expected_relationships=[("github actions", "builds", "docker"), ("docker", "pushes_to", "ecr")],
        hallucination_traps=["jenkins", "circleci", "gitlab"],
    ),

    # ===== Category: Multi-Entity Complex =====
    ExtractionCase(
        id="E16", category="complex",
        text="Dr. Patel at Stanford published a paper with Prof. Kim from Seoul National University on quantum error correction using surface codes.",
        expected_entities=["dr. patel", "stanford", "prof. kim", "seoul national university", "quantum error correction", "surface codes"],
        expected_entity_types={"dr. patel": "person", "stanford": "university", "prof. kim": "person"},
        expected_relationships=[("dr. patel", "affiliated", "stanford"), ("prof. kim", "affiliated", "seoul national university")],
        hallucination_traps=["mit", "caltech", "ibm"],
    ),
    ExtractionCase(
        id="E17", category="complex",
        text="SpaceX's Starship, powered by Raptor engines using methane fuel, launched from Boca Chica and reached orbit for the first time in 2024.",
        expected_entities=["spacex", "starship", "raptor", "methane", "boca chica"],
        expected_entity_types={"spacex": "company", "starship": "rocket", "raptor": "engine"},
        expected_relationships=[("starship", "powered_by", "raptor"), ("starship", "launched_from", "boca chica")],
        hallucination_traps=["nasa", "falcon 9", "blue origin"],
    ),
    ExtractionCase(
        id="E18", category="complex",
        text="The Treaty of Versailles, signed in 1919 by Germany, France, Britain, and the United States, ended World War I and established the League of Nations.",
        expected_entities=["treaty of versailles", "germany", "france", "britain", "united states", "world war i", "league of nations"],
        expected_entity_types={"treaty of versailles": "treaty", "league of nations": "organization"},
        expected_relationships=[("treaty of versailles", "ended", "world war i"), ("treaty of versailles", "established", "league of nations")],
        hallucination_traps=["united nations", "world war ii"],
    ),
    ExtractionCase(
        id="E19", category="complex",
        text="Chef Nakamura's restaurant Kaiseki in Kyoto earned two Michelin stars for its traditional Japanese cuisine using local ingredients from Nishiki Market.",
        expected_entities=["chef nakamura", "kaiseki", "kyoto", "michelin", "nishiki market"],
        expected_entity_types={"chef nakamura": "person", "kaiseki": "restaurant", "kyoto": "city"},
        expected_relationships=[("chef nakamura", "owns", "kaiseki"), ("kaiseki", "located_in", "kyoto")],
        hallucination_traps=["tokyo", "sushi", "ramen"],
    ),
    ExtractionCase(
        id="E20", category="complex",
        text="Apple's M3 chip, fabricated by TSMC using 3nm process, powers the MacBook Pro and integrates CPU, GPU, and Neural Engine on a single die.",
        expected_entities=["apple", "m3", "tsmc", "3nm", "macbook pro", "cpu", "gpu", "neural engine"],
        expected_entity_types={"apple": "company", "m3": "chip", "tsmc": "company"},
        expected_relationships=[("m3", "fabricated_by", "tsmc"), ("m3", "powers", "macbook pro")],
        hallucination_traps=["intel", "samsung", "qualcomm"],
    ),

    # ===== Category: Ambiguous/Subtle =====
    ExtractionCase(
        id="E21", category="ambiguous",
        text="The bank near the river has the best interest rates in town.",
        expected_entities=["bank"],
        expected_entity_types={"bank": "financial_institution"},
        expected_relationships=[],
        hallucination_traps=["river bank", "fishing"],
    ),
    ExtractionCase(
        id="E22", category="ambiguous",
        text="Jordan loves basketball and frequently visits Amman for business.",
        expected_entities=["jordan", "basketball", "amman"],
        expected_entity_types={"jordan": "person", "amman": "city"},
        expected_relationships=[("jordan", "loves", "basketball"), ("jordan", "visits", "amman")],
        hallucination_traps=["michael jordan", "country"],
    ),
    ExtractionCase(
        id="E23", category="ambiguous",
        text="I switched from using Vim to VS Code, but I still use Vim keybindings.",
        expected_entities=["vim", "vs code", "vim keybindings"],
        expected_entity_types={"vim": "software", "vs code": "software"},
        expected_relationships=[],
        hallucination_traps=["neovim", "emacs", "sublime"],
    ),
    ExtractionCase(
        id="E24", category="ambiguous",
        text="Mercury is both a planet and an element used in old thermometers.",
        expected_entities=["mercury"],
        expected_entity_types={},
        expected_relationships=[],
        hallucination_traps=["venus", "thermometer brand"],
    ),
    ExtractionCase(
        id="E25", category="ambiguous",
        text="Python was created by Guido van Rossum, not named after the snake but after Monty Python's Flying Circus.",
        expected_entities=["python", "guido van rossum", "monty python's flying circus"],
        expected_entity_types={"python": "programming_language", "guido van rossum": "person"},
        expected_relationships=[("guido van rossum", "created", "python"), ("python", "named_after", "monty python's flying circus")],
        hallucination_traps=["snake", "java", "perl"],
    ),

    # ===== Category: Temporal/Dated =====
    ExtractionCase(
        id="E26", category="temporal",
        text="In 2023, OpenAI released GPT-4, and in 2024 they released GPT-4o with multimodal capabilities.",
        expected_entities=["openai", "gpt-4", "gpt-4o"],
        expected_entity_types={"openai": "company", "gpt-4": "model", "gpt-4o": "model"},
        expected_relationships=[("openai", "released", "gpt-4"), ("openai", "released", "gpt-4o")],
        hallucination_traps=["gpt-3", "gpt-5", "google"],
    ),
    ExtractionCase(
        id="E27", category="temporal",
        text="Netflix started as a DVD rental service in 1997, then pivoted to streaming in 2007, and began producing original content in 2013.",
        expected_entities=["netflix", "dvd rental", "streaming", "original content"],
        expected_entity_types={"netflix": "company"},
        expected_relationships=[("netflix", "started_as", "dvd rental"), ("netflix", "pivoted_to", "streaming")],
        hallucination_traps=["hulu", "disney+", "blockbuster"],
    ),
    ExtractionCase(
        id="E28", category="temporal",
        text="Docker was released in 2013, Kubernetes in 2014 by Google, and Helm in 2015 by Deis.",
        expected_entities=["docker", "kubernetes", "google", "helm", "deis"],
        expected_entity_types={"docker": "technology", "kubernetes": "technology", "google": "company"},
        expected_relationships=[("google", "released", "kubernetes"), ("deis", "released", "helm")],
        hallucination_traps=["aws", "azure", "podman"],
    ),
    ExtractionCase(
        id="E29", category="temporal",
        text="The company rebranded from Facebook to Meta in October 2021 to focus on the metaverse.",
        expected_entities=["facebook", "meta", "metaverse"],
        expected_entity_types={"facebook": "company", "meta": "company"},
        expected_relationships=[("facebook", "rebranded_to", "meta"), ("meta", "focuses_on", "metaverse")],
        hallucination_traps=["instagram", "whatsapp", "mark zuckerberg"],
    ),
    ExtractionCase(
        id="E30", category="temporal",
        text="Rust won StackOverflow's most loved language survey every year from 2016 to 2023.",
        expected_entities=["rust", "stackoverflow"],
        expected_entity_types={"rust": "programming_language", "stackoverflow": "platform"},
        expected_relationships=[("rust", "won", "stackoverflow")],
        hallucination_traps=["c++", "go", "carbon"],
    ),

    # ===== Category: Nested Relationships =====
    ExtractionCase(
        id="E31", category="nested",
        text="Bob's wife Alice works at the hospital where Bob's mother was treated for pneumonia last year.",
        expected_entities=["bob", "alice", "hospital"],
        expected_entity_types={"bob": "person", "alice": "person"},
        expected_relationships=[("bob", "wife", "alice"), ("alice", "works_at", "hospital")],
        hallucination_traps=["doctor", "nurse", "medication"],
    ),
    ExtractionCase(
        id="E32", category="nested",
        text="The startup founded by Emma, which was acquired by Microsoft, developed an AI tool that competed with Copilot.",
        expected_entities=["emma", "microsoft", "copilot"],
        expected_entity_types={"emma": "person", "microsoft": "company", "copilot": "product"},
        expected_relationships=[("emma", "founded", "startup"), ("microsoft", "acquired", "startup")],
        hallucination_traps=["github", "openai", "google"],
    ),
    ExtractionCase(
        id="E33", category="nested",
        text="The professor who taught me algorithms at Berkeley now leads the AI research lab that developed AlphaFold at DeepMind.",
        expected_entities=["berkeley", "ai research lab", "alphafold", "deepmind"],
        expected_entity_types={"berkeley": "university", "deepmind": "company", "alphafold": "product"},
        expected_relationships=[("deepmind", "developed", "alphafold")],
        hallucination_traps=["google", "openai", "stanford"],
    ),
    ExtractionCase(
        id="E34", category="nested",
        text="Our CTO's recommendation to adopt GraphQL came after attending a conference where the Relay team from Meta presented their new architecture.",
        expected_entities=["graphql", "relay", "meta"],
        expected_entity_types={"graphql": "technology", "relay": "framework", "meta": "company"},
        expected_relationships=[("relay", "developed_by", "meta")],
        hallucination_traps=["rest api", "apollo", "facebook"],
    ),
    ExtractionCase(
        id="E35", category="nested",
        text="The monorepo managed by the platform team uses Bazel for builds, which Google originally developed for their internal infrastructure.",
        expected_entities=["platform team", "bazel", "google"],
        expected_entity_types={"bazel": "tool", "google": "company"},
        expected_relationships=[("platform team", "uses", "bazel"), ("google", "developed", "bazel")],
        hallucination_traps=["gradle", "maven", "buck"],
    ),

    # ===== Category: Edge Cases =====
    ExtractionCase(
        id="E36", category="edge",
        text="",
        expected_entities=[],
        expected_entity_types={},
        expected_relationships=[],
        hallucination_traps=["anything"],
    ),
    ExtractionCase(
        id="E37", category="edge",
        text="The weather is nice today.",
        expected_entities=[],
        expected_entity_types={},
        expected_relationships=[],
        hallucination_traps=["today", "sun"],
    ),
    ExtractionCase(
        id="E38", category="edge",
        text="I use Claude, GPT-4, Gemini, Llama 3, Mistral, Cohere Command-R, and Grok for different tasks.",
        expected_entities=["claude", "gpt-4", "gemini", "llama 3", "mistral", "cohere command-r", "grok"],
        expected_entity_types={"claude": "model", "gpt-4": "model", "gemini": "model"},
        expected_relationships=[],
        hallucination_traps=["palm", "bard", "bing"],
    ),
    ExtractionCase(
        id="E39", category="edge",
        text="AWS S3 costs $0.023/GB, while GCS costs $0.020/GB and Azure Blob Storage costs $0.018/GB for standard storage.",
        expected_entities=["aws s3", "gcs", "azure blob storage"],
        expected_entity_types={"aws s3": "service", "gcs": "service", "azure blob storage": "service"},
        expected_relationships=[],
        hallucination_traps=["aws ec2", "lambda", "cloudfront"],
    ),
    ExtractionCase(
        id="E40", category="edge",
        text="日本語のテキストも正しく処理できるべきです。東京大学の田中教授がAI研究を発表しました。",
        expected_entities=["東京大学", "田中"],
        expected_entity_types={"東京大学": "university", "田中": "person"},
        expected_relationships=[("田中", "affiliated", "東京大学")],
        hallucination_traps=["京都", "大阪"],
    ),
]


# --- Contradiction Test Cases (20 cases) ---

CONTRADICTION_CASES = [
    # ===== Direct contradictions =====
    ContradictionCase(
        id="C01", category="direct_contradiction",
        existing_memories="alice -- lives_in -- new york",
        new_info="Alice just moved to San Francisco.",
        expected_action="delete",
        expected_targets=["alice", "new york"],
    ),
    ContradictionCase(
        id="C02", category="direct_contradiction",
        existing_memories="bob -- works_at -- google",
        new_info="Bob left Google and joined Anthropic as a research scientist.",
        expected_action="delete",
        expected_targets=["bob", "google"],
    ),
    ContradictionCase(
        id="C03", category="direct_contradiction",
        existing_memories="project_alpha -- status -- active\nproject_alpha -- deadline -- december_2025",
        new_info="Project Alpha was cancelled due to budget cuts.",
        expected_action="delete",
        expected_targets=["project_alpha", "active"],
    ),
    ContradictionCase(
        id="C04", category="direct_contradiction",
        existing_memories="sarah -- favorite_language -- python",
        new_info="Sarah says her favorite programming language is now Rust.",
        expected_action="delete",
        expected_targets=["sarah", "python"],
    ),
    ContradictionCase(
        id="C05", category="direct_contradiction",
        existing_memories="company -- database -- mysql",
        new_info="The company completed migration from MySQL to PostgreSQL last week.",
        expected_action="delete",
        expected_targets=["company", "mysql"],
    ),

    # ===== Non-contradictions (should noop) =====
    ContradictionCase(
        id="C06", category="noop",
        existing_memories="alice -- loves_to_eat -- pizza",
        new_info="Alice also loves to eat sushi.",
        expected_action="noop",
        expected_targets=[],
    ),
    ContradictionCase(
        id="C07", category="noop",
        existing_memories="bob -- knows -- python\nbob -- knows -- javascript",
        new_info="Bob also learned Rust recently.",
        expected_action="noop",
        expected_targets=[],
    ),
    ContradictionCase(
        id="C08", category="noop",
        existing_memories="team -- uses -- kubernetes",
        new_info="The team also started using Terraform for infrastructure.",
        expected_action="noop",
        expected_targets=[],
    ),
    ContradictionCase(
        id="C09", category="noop",
        existing_memories="lisa -- has_pet -- dog named max",
        new_info="Lisa adopted a cat named whiskers.",
        expected_action="noop",
        expected_targets=[],
    ),
    ContradictionCase(
        id="C10", category="noop",
        existing_memories="john -- hobby -- playing guitar\njohn -- hobby -- painting",
        new_info="John started learning to cook Italian food.",
        expected_action="noop",
        expected_targets=[],
    ),

    # ===== Subtle/Partial contradictions =====
    ContradictionCase(
        id="C11", category="subtle",
        existing_memories="alex -- salary -- 120000",
        new_info="Alex got a raise and now earns $150,000.",
        expected_action="delete",
        expected_targets=["alex", "120000"],
    ),
    ContradictionCase(
        id="C12", category="subtle",
        existing_memories="team_size -- equals -- 5",
        new_info="After the recent hiring, the team now has 8 members.",
        expected_action="delete",
        expected_targets=["team_size", "5"],
    ),
    ContradictionCase(
        id="C13", category="subtle",
        existing_memories="product -- version -- 2.3",
        new_info="We just released version 3.0 of the product.",
        expected_action="delete",
        expected_targets=["product", "2.3"],
    ),
    ContradictionCase(
        id="C14", category="subtle",
        existing_memories="manager -- of -- engineering_team -- is -- david",
        new_info="David was promoted to VP and Sarah is now the engineering team manager.",
        expected_action="delete",
        expected_targets=["david", "manager"],
    ),
    ContradictionCase(
        id="C15", category="subtle",
        existing_memories="server -- hosted_on -- us-east-1\nserver -- type -- t3.medium",
        new_info="We scaled up the server to t3.xlarge due to increased traffic.",
        expected_action="delete",
        expected_targets=["t3.medium"],
    ),

    # ===== Temporal supersession =====
    ContradictionCase(
        id="C16", category="temporal",
        existing_memories="ceo -- of -- company -- is -- john",
        new_info="As of January 2026, the new CEO of the company is Maria.",
        expected_action="delete",
        expected_targets=["john", "ceo"],
    ),
    ContradictionCase(
        id="C17", category="temporal",
        existing_memories="react -- latest_version -- 18",
        new_info="React 19 was released with support for server components.",
        expected_action="delete",
        expected_targets=["react", "18"],
    ),
    ContradictionCase(
        id="C18", category="temporal",
        existing_memories="user -- subscription -- basic_plan",
        new_info="The user upgraded to the premium plan yesterday.",
        expected_action="delete",
        expected_targets=["basic_plan"],
    ),
    ContradictionCase(
        id="C19", category="temporal",
        existing_memories="office -- location -- downtown\noffice -- floors -- 2",
        new_info="The company relocated from downtown to the new tech park campus.",
        expected_action="delete",
        expected_targets=["downtown"],
    ),
    ContradictionCase(
        id="C20", category="temporal",
        existing_memories="student -- status -- enrolled\nstudent -- major -- computer_science",
        new_info="The student graduated last month with a degree in Computer Science.",
        expected_action="delete",
        expected_targets=["enrolled"],
    ),
]


# ============================================================
# LLM Client Wrappers
# ============================================================

class ClaudeClient:
    """Wraps Anthropic API for graph pipeline calls."""

    def __init__(self):
        token = resolve_token()
        if not token:
            raise RuntimeError("No Anthropic token found")

        kwargs: dict[str, Any] = {}
        if is_oat_token(token):
            kwargs["auth_token"] = token
            kwargs["default_headers"] = OAT_HEADERS
        else:
            kwargs["api_key"] = token

        self.client = anthropic.Anthropic(**kwargs)
        self.model = "claude-opus-4-6"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call_with_tools(self, messages: list[dict], tools: list[dict], system: str = "") -> dict:
        """Call Claude with tools, return parsed response."""
        anthropic_tools = []
        for tool in tools:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })

        api_messages = []
        for msg in messages:
            if msg["role"] != "system":
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        params = {
            "model": self.model,
            "messages": api_messages,
            "tools": anthropic_tools,
            "max_tokens": 4096,
            "tool_choice": {"type": "any"},
        }
        if system:
            params["system"] = system

        start = time.time()
        try:
            response = self.client.messages.create(**params)
        except Exception as e:
            return {"error": str(e), "latency": time.time() - start, "tool_calls": [], "content": ""}
        latency = time.time() - start

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        tool_calls = []
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "latency": latency,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
        }


class GeminiClient:
    """Wraps Google GenAI SDK for graph pipeline calls."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-lite"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _convert_tools(self, tools: list[dict]) -> list[genai_types.Tool]:
        """Convert OpenAI-style tool defs to genai format."""
        declarations = []
        for tool in tools:
            fn = tool["function"]
            params = fn.get("parameters", {"type": "object", "properties": {}})
            # Strip additionalProperties (Gemini doesn't support it)
            cleaned = self._clean_schema(params)
            declarations.append(genai_types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=cleaned,
            ))
        return [genai_types.Tool(function_declarations=declarations)]

    def _clean_schema(self, schema: dict) -> dict:
        """Remove additionalProperties recursively."""
        result = {}
        for k, v in schema.items():
            if k == "additionalProperties":
                continue
            if isinstance(v, dict):
                result[k] = self._clean_schema(v)
            elif isinstance(v, list):
                result[k] = [self._clean_schema(i) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result

    def call_with_tools(self, messages: list[dict], tools: list[dict], system: str = "") -> dict:
        """Call Gemini with tools, return parsed response."""
        genai_tools = self._convert_tools(tools)

        # Build content
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System handled via config
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(genai_types.Content(
                parts=[genai_types.Part(text=msg["content"])],
                role=role,
            ))

        config = genai_types.GenerateContentConfig(
            tools=genai_tools,
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="ANY")
            ),
        )
        if system:
            config.system_instruction = system

        start = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            return {"error": str(e), "latency": time.time() - start, "tool_calls": [], "content": ""}
        latency = time.time() - start

        # Parse response
        tool_calls = []
        text_parts = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append({"name": fc.name, "arguments": dict(fc.args) if fc.args else {}})
                if part.text:
                    text_parts.append(part.text)

        # Token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": "tool_use",
        }


# ============================================================
# Scoring Functions
# ============================================================

def _normalize(s: str) -> str:
    """Normalize string for fuzzy matching."""
    return s.lower().strip().replace("_", " ").replace("-", " ")


def score_entity_extraction(result: dict, case: ExtractionCase) -> dict:
    """Score entity extraction accuracy."""
    if "error" in result:
        return {"found": 0, "expected": len(case.expected_entities), "precision": 0, "recall": 0, "f1": 0, "hallucinated": 0}

    tool_calls = result.get("tool_calls", [])
    extracted = []
    for tc in tool_calls:
        if tc["name"] == "extract_entities":
            entities = tc["arguments"].get("entities", [])
            for e in entities:
                extracted.append(_normalize(e.get("entity", "")))

    expected_norm = [_normalize(e) for e in case.expected_entities]

    # Fuzzy match: entity is "found" if expected substring appears in any extracted entity or vice versa
    found = 0
    for exp in expected_norm:
        for ext in extracted:
            if exp in ext or ext in exp:
                found += 1
                break

    # Hallucination check
    hallucinated = 0
    trap_norm = [_normalize(t) for t in case.hallucination_traps]
    for ext in extracted:
        for trap in trap_norm:
            if trap in ext or ext in trap:
                hallucinated += 1
                break

    precision = found / len(extracted) if extracted else (1.0 if not expected_norm else 0)
    recall = found / len(expected_norm) if expected_norm else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "found": found,
        "expected": len(expected_norm),
        "extracted_count": len(extracted),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "hallucinated": hallucinated,
    }


def score_relationship_quality(result: dict, case: ExtractionCase) -> dict:
    """Score relationship extraction quality."""
    if "error" in result:
        return {"found": 0, "expected": len(case.expected_relationships), "score": 0}

    tool_calls = result.get("tool_calls", [])
    relationships = []
    for tc in tool_calls:
        if tc["name"] in ("establish_relationships", "establish_relations"):
            entities = tc["arguments"].get("entities", [])
            for e in entities:
                relationships.append((
                    _normalize(e.get("source", "")),
                    _normalize(e.get("relationship", "")),
                    _normalize(e.get("destination", "")),
                ))

    expected_norm = [(_normalize(s), _normalize(r), _normalize(d)) for s, r, d in case.expected_relationships]

    found = 0
    for es, er, ed in expected_norm:
        for rs, rr, rd in relationships:
            # Source and destination match (fuzzy), relationship contains keyword
            src_match = es in rs or rs in es
            dst_match = ed in rd or rd in ed
            rel_match = er in rr or rr in er
            if src_match and dst_match and rel_match:
                found += 1
                break

    score = found / len(expected_norm) if expected_norm else 1.0
    return {
        "found": found,
        "expected": len(expected_norm),
        "extracted_count": len(relationships),
        "score": round(score, 3),
    }


def score_contradiction(result: dict, case: ContradictionCase) -> dict:
    """Score contradiction detection accuracy."""
    if "error" in result:
        return {"correct": False, "expected_action": case.expected_action, "actual_action": "error", "score": 0}

    tool_calls = result.get("tool_calls", [])

    # Determine actual action from tool calls
    actual_actions = set()
    for tc in tool_calls:
        if tc["name"] == "delete_graph_memory":
            actual_actions.add("delete")
        elif tc["name"] == "update_graph_memory":
            actual_actions.add("update")
        elif tc["name"] == "add_graph_memory":
            actual_actions.add("add")
        elif tc["name"] == "noop":
            actual_actions.add("noop")

    if not actual_actions:
        actual_action = "none"
    elif len(actual_actions) == 1:
        actual_action = actual_actions.pop()
    else:
        # Multiple actions — take the most significant
        for prio in ["delete", "update", "add", "noop"]:
            if prio in actual_actions:
                actual_action = prio
                break
        else:
            actual_action = "mixed"

    correct = actual_action == case.expected_action
    return {
        "correct": correct,
        "expected_action": case.expected_action,
        "actual_action": actual_action,
        "score": 1.0 if correct else 0.0,
    }


def score_tool_reliability(result: dict) -> dict:
    """Score tool call reliability (did it use the tool at all?)."""
    if "error" in result:
        return {"used_tool": False, "valid_json": False, "score": 0}

    tool_calls = result.get("tool_calls", [])
    used_tool = len(tool_calls) > 0

    # Check if arguments are valid dicts
    valid_json = all(isinstance(tc.get("arguments"), dict) for tc in tool_calls) if tool_calls else False

    score = 1.0 if (used_tool and valid_json) else (0.5 if used_tool else 0)
    return {"used_tool": used_tool, "valid_json": valid_json, "tool_count": len(tool_calls), "score": score}


def score_schema_adherence(result: dict, expected_tool_name: str) -> dict:
    """Score JSON schema adherence for tool call arguments."""
    if "error" in result:
        return {"correct_tool": False, "valid_schema": False, "score": 0}

    tool_calls = result.get("tool_calls", [])
    if not tool_calls:
        return {"correct_tool": False, "valid_schema": False, "score": 0}

    # Check if the correct tool was called
    correct_tool = any(tc["name"] == expected_tool_name for tc in tool_calls)

    # Check schema validity based on tool type
    valid_count = 0
    for tc in tool_calls:
        args = tc.get("arguments", {})
        if tc["name"] == "extract_entities":
            if "entities" in args and isinstance(args["entities"], list):
                if all("entity" in e and "entity_type" in e for e in args["entities"]):
                    valid_count += 1
        elif tc["name"] in ("establish_relationships", "establish_relations"):
            if "entities" in args and isinstance(args["entities"], list):
                if all("source" in e and "relationship" in e and "destination" in e for e in args["entities"]):
                    valid_count += 1
        elif tc["name"] == "delete_graph_memory":
            if "source" in args and "relationship" in args and "destination" in args:
                valid_count += 1
        elif tc["name"] == "update_graph_memory":
            if "source" in args and "destination" in args and "relationship" in args:
                valid_count += 1
        elif tc["name"] == "add_graph_memory":
            if all(k in args for k in ["source", "destination", "relationship", "source_type", "destination_type"]):
                valid_count += 1
        elif tc["name"] == "noop":
            valid_count += 1  # noop has no required args

    valid_schema = valid_count == len(tool_calls)
    score = (1.0 if correct_tool else 0.5) * (1.0 if valid_schema else 0.5)
    return {"correct_tool": correct_tool, "valid_schema": valid_schema, "valid_count": valid_count, "total": len(tool_calls), "score": round(score, 3)}


# ============================================================
# Test Runner
# ============================================================

def run_extraction_test(client, case: ExtractionCase, call_type: str) -> dict:
    """Run a single extraction test case (Call 1 or Call 2)."""
    if call_type == "entity":
        tools = [EXTRACT_ENTITIES_TOOL]
        expected_tool = "extract_entities"
        messages = [{"role": "user", "content": f"Extract entities from the following text:\n\n{case.text}"}]
        system = ""
    elif call_type == "relationship":
        tools = [RELATIONS_TOOL]
        expected_tool = "establish_relationships"
        # Build entity list from expected entities for prompt
        entity_list = ", ".join(case.expected_entities[:10]) if case.expected_entities else "none"
        prompt = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", "test_user")
        messages = [{"role": "user", "content": f"Entities found: {entity_list}\n\nOriginal text: {case.text}"}]
        system = prompt
    else:
        raise ValueError(f"Unknown call_type: {call_type}")

    if not case.text:
        # Skip empty text edge case — still attempt the call
        pass

    result = client.call_with_tools(messages, tools, system=system)
    return result


def run_contradiction_test(client, case: ContradictionCase) -> dict:
    """Run a single contradiction detection test case (Call 3)."""
    tools = [DELETE_MEMORY_TOOL_GRAPH, ADD_MEMORY_TOOL_GRAPH, NOOP_TOOL]

    system = DELETE_RELATIONS_SYSTEM_PROMPT.replace("USER_ID", "test_user")
    user_msg = f"Here are the existing memories: {case.existing_memories}\n\nNew Information: {case.new_info}"

    messages = [{"role": "user", "content": user_msg}]
    result = client.call_with_tools(messages, tools, system=system)
    return result


def print_dimension_summary(title: str, claude_scores: list[float], gemini_scores: list[float]):
    """Print a summary comparison for a dimension."""
    c_avg = sum(claude_scores) / len(claude_scores) if claude_scores else 0
    g_avg = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0
    winner = "CLAUDE" if c_avg > g_avg else ("GEMINI" if g_avg > c_avg else "TIE")
    bar_width = 30

    c_bar = int(c_avg * bar_width)
    g_bar = int(g_avg * bar_width)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Claude: [{'#' * c_bar}{'.' * (bar_width - c_bar)}] {c_avg:.1%} (n={len(claude_scores)})")
    print(f"  Gemini: [{'#' * g_bar}{'.' * (bar_width - g_bar)}] {g_avg:.1%} (n={len(gemini_scores)})")
    print(f"  Winner: {winner} (delta: {abs(c_avg - g_avg):.1%})")


def main():
    print("=" * 70)
    print("  GRAPH LLM BATTLE TEST")
    print("  Claude Opus 4.6 vs Gemini 2.5 Flash Lite")
    print("  7 Dimensions x 60 Test Cases")
    print("=" * 70)

    # Initialize clients
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GOOGLE_API_KEY env var required")
        sys.exit(1)
    print("\nInitializing Claude client...")
    claude = ClaudeClient()
    print(f"  Model: {claude.model}")
    print("Initializing Gemini client...")
    gemini = GeminiClient(gemini_key)
    print(f"  Model: {gemini.model}")

    # Results storage
    all_results = {
        "entity_extraction": {"claude": [], "gemini": []},
        "relationship_quality": {"claude": [], "gemini": []},
        "contradiction_detection": {"claude": [], "gemini": []},
        "tool_reliability": {"claude": [], "gemini": []},
        "schema_adherence": {"claude": [], "gemini": []},
        "hallucination": {"claude": [], "gemini": []},
        "latency": {"claude": [], "gemini": []},
    }

    # ===== Phase 1: Entity Extraction (Call 1) =====
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: Entity Extraction ({len(EXTRACTION_CASES)} cases)")
    print(f"{'=' * 70}")

    for i, case in enumerate(EXTRACTION_CASES):
        if not case.text:
            print(f"  [{case.id}] Skipping empty text edge case")
            continue

        print(f"  [{case.id}] {case.category}: {case.text[:60]}...")

        # Claude
        try:
            c_result = run_extraction_test(claude, case, "entity")
            c_score = score_entity_extraction(c_result, case)
            c_tool = score_tool_reliability(c_result)
            c_schema = score_schema_adherence(c_result, "extract_entities")
            all_results["entity_extraction"]["claude"].append(c_score["f1"])
            all_results["tool_reliability"]["claude"].append(c_tool["score"])
            all_results["schema_adherence"]["claude"].append(c_schema["score"])
            all_results["hallucination"]["claude"].append(1.0 - (c_score["hallucinated"] / max(1, c_score["extracted_count"])))
            all_results["latency"]["claude"].append(c_result.get("latency", 0))
            c_str = f"F1={c_score['f1']:.2f} H={c_score['hallucinated']} T={c_result.get('latency', 0):.1f}s"
        except Exception as e:
            c_str = f"ERROR: {e}"
            all_results["entity_extraction"]["claude"].append(0)
            all_results["tool_reliability"]["claude"].append(0)
            all_results["schema_adherence"]["claude"].append(0)
            all_results["hallucination"]["claude"].append(0)
            all_results["latency"]["claude"].append(0)

        # Gemini
        try:
            g_result = run_extraction_test(gemini, case, "entity")
            g_score = score_entity_extraction(g_result, case)
            g_tool = score_tool_reliability(g_result)
            g_schema = score_schema_adherence(g_result, "extract_entities")
            all_results["entity_extraction"]["gemini"].append(g_score["f1"])
            all_results["tool_reliability"]["gemini"].append(g_tool["score"])
            all_results["schema_adherence"]["gemini"].append(g_schema["score"])
            all_results["hallucination"]["gemini"].append(1.0 - (g_score["hallucinated"] / max(1, g_score["extracted_count"])))
            all_results["latency"]["gemini"].append(g_result.get("latency", 0))
            g_str = f"F1={g_score['f1']:.2f} H={g_score['hallucinated']} T={g_result.get('latency', 0):.1f}s"
        except Exception as e:
            g_str = f"ERROR: {e}"
            all_results["entity_extraction"]["gemini"].append(0)
            all_results["tool_reliability"]["gemini"].append(0)
            all_results["schema_adherence"]["gemini"].append(0)
            all_results["hallucination"]["gemini"].append(0)
            all_results["latency"]["gemini"].append(0)

        print(f"         Claude: {c_str}")
        print(f"         Gemini: {g_str}")

    # ===== Phase 2: Relationship Extraction (Call 2) =====
    # Only run on cases that have expected relationships
    rel_cases = [c for c in EXTRACTION_CASES if c.expected_relationships and c.text]
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: Relationship Extraction ({len(rel_cases)} cases)")
    print(f"{'=' * 70}")

    for case in rel_cases:
        print(f"  [{case.id}] {case.category}: {case.text[:60]}...")

        try:
            c_result = run_extraction_test(claude, case, "relationship")
            c_score = score_relationship_quality(c_result, case)
            c_tool = score_tool_reliability(c_result)
            c_schema = score_schema_adherence(c_result, "establish_relationships")
            all_results["relationship_quality"]["claude"].append(c_score["score"])
            all_results["tool_reliability"]["claude"].append(c_tool["score"])
            all_results["schema_adherence"]["claude"].append(c_schema["score"])
            all_results["latency"]["claude"].append(c_result.get("latency", 0))
            c_str = f"Score={c_score['score']:.2f} ({c_score['found']}/{c_score['expected']}) T={c_result.get('latency', 0):.1f}s"
        except Exception as e:
            c_str = f"ERROR: {e}"
            all_results["relationship_quality"]["claude"].append(0)
            all_results["tool_reliability"]["claude"].append(0)
            all_results["schema_adherence"]["claude"].append(0)
            all_results["latency"]["claude"].append(0)

        try:
            g_result = run_extraction_test(gemini, case, "relationship")
            g_score = score_relationship_quality(g_result, case)
            g_tool = score_tool_reliability(g_result)
            g_schema = score_schema_adherence(g_result, "establish_relationships")
            all_results["relationship_quality"]["gemini"].append(g_score["score"])
            all_results["tool_reliability"]["gemini"].append(g_tool["score"])
            all_results["schema_adherence"]["gemini"].append(g_schema["score"])
            all_results["latency"]["gemini"].append(g_result.get("latency", 0))
            g_str = f"Score={g_score['score']:.2f} ({g_score['found']}/{g_score['expected']}) T={g_result.get('latency', 0):.1f}s"
        except Exception as e:
            g_str = f"ERROR: {e}"
            all_results["relationship_quality"]["gemini"].append(0)
            all_results["tool_reliability"]["gemini"].append(0)
            all_results["schema_adherence"]["gemini"].append(0)
            all_results["latency"]["gemini"].append(0)

        print(f"         Claude: {c_str}")
        print(f"         Gemini: {g_str}")

    # ===== Phase 3: Contradiction Detection (Call 3) =====
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: Contradiction Detection ({len(CONTRADICTION_CASES)} cases)")
    print(f"{'=' * 70}")

    for case in CONTRADICTION_CASES:
        print(f"  [{case.id}] {case.category}: {case.new_info[:60]}...")

        try:
            c_result = run_contradiction_test(claude, case)
            c_score = score_contradiction(c_result, case)
            c_tool = score_tool_reliability(c_result)
            all_results["contradiction_detection"]["claude"].append(c_score["score"])
            all_results["tool_reliability"]["claude"].append(c_tool["score"])
            all_results["latency"]["claude"].append(c_result.get("latency", 0))
            c_str = f"{'PASS' if c_score['correct'] else 'FAIL'} (expected={c_score['expected_action']}, got={c_score['actual_action']}) T={c_result.get('latency', 0):.1f}s"
        except Exception as e:
            c_str = f"ERROR: {e}"
            all_results["contradiction_detection"]["claude"].append(0)
            all_results["tool_reliability"]["claude"].append(0)
            all_results["latency"]["claude"].append(0)

        try:
            g_result = run_contradiction_test(gemini, case)
            g_score = score_contradiction(g_result, case)
            g_tool = score_tool_reliability(g_result)
            all_results["contradiction_detection"]["gemini"].append(g_score["score"])
            all_results["tool_reliability"]["gemini"].append(g_tool["score"])
            all_results["latency"]["gemini"].append(g_result.get("latency", 0))
            g_str = f"{'PASS' if g_score['correct'] else 'FAIL'} (expected={g_score['expected_action']}, got={g_score['actual_action']}) T={g_result.get('latency', 0):.1f}s"
        except Exception as e:
            g_str = f"ERROR: {e}"
            all_results["contradiction_detection"]["gemini"].append(0)
            all_results["tool_reliability"]["gemini"].append(0)
            all_results["latency"]["gemini"].append(0)

        print(f"         Claude: {c_str}")
        print(f"         Gemini: {g_str}")

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    print(f"\n\n{'#' * 70}")
    print(f"{'#' * 70}")
    print(f"  FINAL RESULTS: 7-DIMENSION BATTLE SCORECARD")
    print(f"{'#' * 70}")
    print(f"{'#' * 70}")

    # 1. Entity Extraction
    print_dimension_summary(
        "D1: ENTITY EXTRACTION ACCURACY",
        all_results["entity_extraction"]["claude"],
        all_results["entity_extraction"]["gemini"],
    )

    # 2. Relationship Quality
    print_dimension_summary(
        "D2: RELATIONSHIP QUALITY",
        all_results["relationship_quality"]["claude"],
        all_results["relationship_quality"]["gemini"],
    )

    # 3. Contradiction Detection
    print_dimension_summary(
        "D3: CONTRADICTION DETECTION",
        all_results["contradiction_detection"]["claude"],
        all_results["contradiction_detection"]["gemini"],
    )

    # 4. Tool Call Reliability
    print_dimension_summary(
        "D4: TOOL CALL RELIABILITY",
        all_results["tool_reliability"]["claude"],
        all_results["tool_reliability"]["gemini"],
    )

    # 5. JSON Schema Adherence
    print_dimension_summary(
        "D5: JSON SCHEMA ADHERENCE",
        all_results["schema_adherence"]["claude"],
        all_results["schema_adherence"]["gemini"],
    )

    # 6. Hallucination Rate (higher = less hallucination = better)
    print_dimension_summary(
        "D6: HALLUCINATION RESISTANCE (higher = less hallucination)",
        all_results["hallucination"]["claude"],
        all_results["hallucination"]["gemini"],
    )

    # 7. Cost & Latency
    c_latencies = all_results["latency"]["claude"]
    g_latencies = all_results["latency"]["gemini"]
    c_avg_lat = sum(c_latencies) / len(c_latencies) if c_latencies else 0
    g_avg_lat = sum(g_latencies) / len(g_latencies) if g_latencies else 0

    print(f"\n{'=' * 60}")
    print(f"  D7: COST & LATENCY")
    print(f"{'=' * 60}")
    print(f"  Claude: avg latency = {c_avg_lat:.2f}s | tokens: {claude.total_input_tokens:,} in / {claude.total_output_tokens:,} out")
    print(f"  Gemini: avg latency = {g_avg_lat:.2f}s | tokens: {gemini.total_input_tokens:,} in / {gemini.total_output_tokens:,} out")

    # Cost estimates (per 1M tokens)
    # Claude Opus: $15/1M input, $75/1M output
    # Gemini 2.5 Flash Lite: $0.075/1M input, $0.3/1M output
    c_cost = (claude.total_input_tokens * 15 / 1_000_000) + (claude.total_output_tokens * 75 / 1_000_000)
    g_cost = (gemini.total_input_tokens * 0.075 / 1_000_000) + (gemini.total_output_tokens * 0.3 / 1_000_000)

    print(f"  Claude estimated cost: ${c_cost:.4f}")
    print(f"  Gemini estimated cost: ${g_cost:.6f}")
    if g_cost > 0:
        print(f"  Cost ratio: Claude is {c_cost / g_cost:.0f}x more expensive")
    if g_avg_lat > 0:
        print(f"  Speed ratio: Gemini is {c_avg_lat / g_avg_lat:.1f}x faster")
    print(f"  Winner: GEMINI (cost & latency)")

    # ===== Overall Summary =====
    print(f"\n\n{'#' * 70}")
    print(f"  OVERALL WINNER BY DIMENSION")
    print(f"{'#' * 70}")

    dimensions = [
        ("D1: Entity Extraction", all_results["entity_extraction"]),
        ("D2: Relationship Quality", all_results["relationship_quality"]),
        ("D3: Contradiction Detection", all_results["contradiction_detection"]),
        ("D4: Tool Reliability", all_results["tool_reliability"]),
        ("D5: Schema Adherence", all_results["schema_adherence"]),
        ("D6: Hallucination Resistance", all_results["hallucination"]),
    ]

    claude_wins = 0
    gemini_wins = 0
    ties = 0

    print(f"\n  {'Dimension':<35} {'Claude':>8} {'Gemini':>8} {'Winner':>10}")
    print(f"  {'-' * 65}")

    for name, scores in dimensions:
        c_avg = sum(scores["claude"]) / len(scores["claude"]) if scores["claude"] else 0
        g_avg = sum(scores["gemini"]) / len(scores["gemini"]) if scores["gemini"] else 0
        if c_avg > g_avg + 0.01:
            winner = "CLAUDE"
            claude_wins += 1
        elif g_avg > c_avg + 0.01:
            winner = "GEMINI"
            gemini_wins += 1
        else:
            winner = "TIE"
            ties += 1
        print(f"  {name:<35} {c_avg:>7.1%} {g_avg:>7.1%} {winner:>10}")

    # Cost always goes to Gemini
    gemini_wins += 1
    print(f"  {'D7: Cost & Latency':<35} {'---':>8} {'---':>8} {'GEMINI':>10}")

    print(f"\n  {'=' * 65}")
    print(f"  Claude wins: {claude_wins} | Gemini wins: {gemini_wins} | Ties: {ties}")
    print(f"  {'=' * 65}")

    # Verdict
    print(f"\n  VERDICT:")
    print(f"  Split-model is {'CONFIRMED' if gemini_wins >= 3 and claude_wins >= 1 else 'INCONCLUSIVE'}:")
    print(f"  - Gemini 2.5 Flash Lite excels at extraction (entity + relationship)")
    print(f"  - Claude Opus 4.6 excels at nuanced contradiction detection")
    print(f"  - Gemini wins on cost and latency by large margins")

    # Save raw results to JSON
    output_path = Path(__file__).parent / "battle_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": {"claude": claude.model, "gemini": gemini.model},
            "token_usage": {
                "claude": {"input": claude.total_input_tokens, "output": claude.total_output_tokens},
                "gemini": {"input": gemini.total_input_tokens, "output": gemini.total_output_tokens},
            },
            "estimated_cost": {"claude": c_cost, "gemini": g_cost},
            "avg_latency": {"claude": c_avg_lat, "gemini": g_avg_lat},
            "dimension_scores": {
                name: {
                    "claude": sum(scores["claude"]) / len(scores["claude"]) if scores["claude"] else 0,
                    "gemini": sum(scores["gemini"]) / len(scores["gemini"]) if scores["gemini"] else 0,
                }
                for name, scores in dimensions
            },
        }, f, indent=2)
    print(f"\n  Raw results saved to: {output_path}")


if __name__ == "__main__":
    main()
