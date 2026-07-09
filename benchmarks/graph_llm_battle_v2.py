"""Comprehensive battle test v2: Claude Opus 4.6 vs Gemini 2.5 Flash Lite.

Expanded to 150+ test cases with difficulty stratification, adversarial inputs,
multi-domain coverage, and statistical analysis with confidence intervals.

Dimensions:
1. Entity extraction accuracy (Call 1)
2. Relationship quality (Call 2)
3. Contradiction detection (Call 3)
4. Tool call reliability
5. JSON schema adherence
6. Hallucination rate
7. Cost per 1M output tokens

Usage:
    python3 benchmarks/graph_llm_battle_v2.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic
from google import genai
from google.genai import types as genai_types

from mem0_mcp_selfhosted.auth import is_oat_token, resolve_token

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

DELETE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes. This function deletes the existing relationship.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "The identifier of the source node in the relationship."},
                "relationship": {"type": "string", "description": "The existing relationship between the source and destination nodes that needs to be deleted."},
                "destination": {"type": "string", "description": "The identifier of the destination node in the relationship."},
            },
            "required": ["source", "relationship", "destination"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "The identifier of the source node in the new relationship."},
                "destination": {"type": "string", "description": "The identifier of the destination node in the new relationship."},
                "relationship": {"type": "string", "description": "The type of relationship between the source and destination nodes."},
                "source_type": {"type": "string", "description": "The type or category of the source node."},
                "destination_type": {"type": "string", "description": "The type or category of the destination node."},
            },
            "required": ["source", "destination", "relationship", "source_type", "destination_type"],
            "additionalProperties": False,
        },
    },
}

NOOP_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
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

OAT_HEADERS = {
    "accept": "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/1.0.0 (external, cli)",
    "x-app": "cli",
}


# ============================================================
# Test Case Definitions
# ============================================================

@dataclass
class ExtractionCase:
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    text: str
    expected_entities: list[str]
    expected_entity_types: dict[str, str]
    expected_relationships: list[tuple[str, str, str]]
    hallucination_traps: list[str] = field(default_factory=list)

@dataclass
class ContradictionCase:
    id: str
    category: str
    difficulty: str
    existing_memories: str
    new_info: str
    expected_action: str  # delete, noop
    expected_targets: list[str] = field(default_factory=list)


# ============================================================
# EXTRACTION TEST CASES — 80 cases
# ============================================================

EXTRACTION_CASES: list[ExtractionCase] = [
    # ===== EASY: Simple personal facts (10) =====
    ExtractionCase("E01", "personal", "easy",
        "Alice prefers TypeScript over JavaScript for web development.",
        ["alice", "typescript", "javascript"],
        {"alice": "person", "typescript": "language"},
        [("alice", "prefers", "typescript")],
        ["python", "react"]),
    ExtractionCase("E02", "personal", "easy",
        "Maria lives in Tokyo and works at Google.",
        ["maria", "tokyo", "google"],
        {"maria": "person", "tokyo": "city", "google": "company"},
        [("maria", "lives_in", "tokyo"), ("maria", "works_at", "google")],
        ["japan", "alphabet"]),
    ExtractionCase("E03", "personal", "easy",
        "John adopted a golden retriever named Max.",
        ["john", "max", "golden retriever"],
        {"john": "person", "max": "pet"},
        [("john", "adopted", "max")],
        ["veterinarian"]),
    ExtractionCase("E04", "personal", "easy",
        "Sarah is allergic to peanuts and shellfish.",
        ["sarah", "peanuts", "shellfish"],
        {"sarah": "person"},
        [("sarah", "allergic", "peanuts"), ("sarah", "allergic", "shellfish")],
        ["hospital"]),
    ExtractionCase("E05", "personal", "easy",
        "Alex graduated from MIT with a Computer Science degree.",
        ["alex", "mit", "computer science"],
        {"alex": "person", "mit": "university"},
        [("alex", "graduated", "mit")],
        ["harvard"]),
    ExtractionCase("E06", "personal", "easy",
        "Rachel speaks French, Spanish, and Mandarin fluently.",
        ["rachel", "french", "spanish", "mandarin"],
        {"rachel": "person"},
        [("rachel", "speaks", "french"), ("rachel", "speaks", "spanish"), ("rachel", "speaks", "mandarin")],
        ["english", "german"]),
    ExtractionCase("E07", "personal", "easy",
        "Tom drives a red Tesla Model 3.",
        ["tom", "tesla model 3"],
        {"tom": "person", "tesla model 3": "car"},
        [("tom", "drives", "tesla model 3")],
        ["elon musk"]),
    ExtractionCase("E08", "personal", "easy",
        "Emma's birthday is on March 15, 1990.",
        ["emma"],
        {"emma": "person"},
        [],
        ["zodiac"]),
    ExtractionCase("E09", "personal", "easy",
        "David plays piano and guitar in his spare time.",
        ["david", "piano", "guitar"],
        {"david": "person"},
        [("david", "plays", "piano"), ("david", "plays", "guitar")],
        ["drums"]),
    ExtractionCase("E10", "personal", "easy",
        "Sophia volunteers at the local animal shelter every Saturday.",
        ["sophia", "animal shelter"],
        {"sophia": "person"},
        [("sophia", "volunteers", "animal shelter")],
        ["hospital"]),

    # ===== MEDIUM: Professional/Workplace (10) =====
    ExtractionCase("E11", "professional", "medium",
        "The CTO of Anthropic, Mike Krieger, previously co-founded Instagram.",
        ["mike krieger", "anthropic", "instagram"],
        {"mike krieger": "person", "anthropic": "company", "instagram": "company"},
        [("mike krieger", "cto", "anthropic"), ("mike krieger", "co-founded", "instagram")],
        ["facebook", "meta"]),
    ExtractionCase("E12", "professional", "medium",
        "Our team uses Kubernetes on AWS, managed by DevOps lead Chen.",
        ["kubernetes", "aws", "chen"],
        {"kubernetes": "technology", "aws": "cloud_provider", "chen": "person"},
        [("chen", "manages", "kubernetes")],
        ["docker", "azure"]),
    ExtractionCase("E13", "professional", "medium",
        "Lisa manages the Berlin office and reports to Regional Director Hans in Munich.",
        ["lisa", "berlin", "hans", "munich"],
        {"lisa": "person", "hans": "person", "berlin": "city", "munich": "city"},
        [("lisa", "manages", "berlin"), ("lisa", "reports_to", "hans")],
        ["germany"]),
    ExtractionCase("E14", "professional", "medium",
        "The marketing team launched Campaign Aurora targeting Gen Z customers on TikTok and Instagram.",
        ["marketing team", "campaign aurora", "gen z", "tiktok", "instagram"],
        {"tiktok": "platform", "instagram": "platform"},
        [("campaign aurora", "targets", "gen z")],
        ["facebook", "twitter"]),
    ExtractionCase("E15", "professional", "medium",
        "CFO Jennifer presented the Q4 earnings report showing $12.8B revenue at the board meeting in Cupertino.",
        ["jennifer", "q4 earnings", "cupertino"],
        {"jennifer": "person", "cupertino": "city"},
        [],
        ["tim cook", "apple"]),
    ExtractionCase("E16", "professional", "medium",
        "The legal department led by Attorney Rivera filed a patent for the company's new encryption algorithm.",
        ["attorney rivera", "legal department", "patent", "encryption algorithm"],
        {"attorney rivera": "person"},
        [("attorney rivera", "leads", "legal department")],
        ["lawsuit"]),
    ExtractionCase("E17", "professional", "medium",
        "Intern James built a Slack bot using Python that automated the daily standup reminders for the engineering team.",
        ["james", "slack bot", "python", "engineering team"],
        {"james": "person", "python": "language", "slack bot": "tool"},
        [("james", "built", "slack bot")],
        ["javascript"]),
    ExtractionCase("E18", "professional", "medium",
        "Senior architect Yuki designed the microservices migration strategy from the legacy monolith at Rakuten.",
        ["yuki", "microservices", "rakuten", "monolith"],
        {"yuki": "person", "rakuten": "company"},
        [("yuki", "designed", "microservices"), ("yuki", "works_at", "rakuten")],
        ["amazon"]),
    ExtractionCase("E19", "professional", "medium",
        "Product Manager Wei prioritized the mobile-first redesign over the desktop analytics dashboard for Q1.",
        ["wei", "mobile-first redesign", "desktop analytics dashboard"],
        {"wei": "person"},
        [("wei", "prioritized", "mobile-first redesign")],
        ["q2"]),
    ExtractionCase("E20", "professional", "medium",
        "The QA team found 47 critical bugs in the payment module before the Black Friday release.",
        ["qa team", "payment module", "black friday"],
        {},
        [("qa team", "tested", "payment module")],
        ["cyber monday"]),

    # ===== MEDIUM: Technical/Programming (10) =====
    ExtractionCase("E21", "technical", "medium",
        "The backend is built with FastAPI, uses PostgreSQL, and caches with Redis.",
        ["fastapi", "postgresql", "redis"],
        {"fastapi": "framework", "postgresql": "database", "redis": "cache"},
        [("fastapi", "uses", "postgresql"), ("fastapi", "uses", "redis")],
        ["django", "mongodb"]),
    ExtractionCase("E22", "technical", "medium",
        "We migrated from React to Svelte for the dashboard because of bundle size.",
        ["react", "svelte", "dashboard"],
        {"react": "framework", "svelte": "framework"},
        [("dashboard", "migrated_to", "svelte")],
        ["vue", "angular"]),
    ExtractionCase("E23", "technical", "medium",
        "The auth service uses JWT tokens with RSA-256, deployed on Cloud Run.",
        ["auth service", "jwt", "rsa-256", "cloud run"],
        {"jwt": "technology", "cloud run": "platform"},
        [("auth service", "uses", "jwt"), ("auth service", "deployed_on", "cloud run")],
        ["oauth", "firebase"]),
    ExtractionCase("E24", "technical", "medium",
        "The CI/CD pipeline uses GitHub Actions with Docker builds, pushing to ECR then deploying to EKS.",
        ["github actions", "docker", "ecr", "eks"],
        {"github actions": "ci_cd", "docker": "container"},
        [("github actions", "builds", "docker"), ("docker", "pushes_to", "ecr")],
        ["jenkins", "gitlab"]),
    ExtractionCase("E25", "technical", "medium",
        "Our data pipeline is Apache Airflow orchestrating Spark jobs that read from Kafka and write to BigQuery.",
        ["apache airflow", "spark", "kafka", "bigquery"],
        {"apache airflow": "orchestrator", "spark": "processing", "kafka": "streaming", "bigquery": "warehouse"},
        [("apache airflow", "orchestrates", "spark"), ("spark", "reads_from", "kafka"), ("spark", "writes_to", "bigquery")],
        ["flink", "snowflake"]),
    ExtractionCase("E26", "technical", "medium",
        "The mobile app uses Flutter with BLoC state management, communicating via gRPC to the Go backend.",
        ["flutter", "bloc", "grpc", "go"],
        {"flutter": "framework", "go": "language"},
        [("flutter", "uses", "bloc"), ("flutter", "communicates_via", "grpc")],
        ["react native", "swift"]),
    ExtractionCase("E27", "technical", "medium",
        "Terraform manages our AWS infra: VPC, ALB, RDS PostgreSQL, ElastiCache Redis, and CloudFront CDN.",
        ["terraform", "aws", "vpc", "alb", "rds", "elasticache", "cloudfront"],
        {"terraform": "iac", "aws": "cloud"},
        [("terraform", "manages", "aws")],
        ["pulumi", "azure"]),
    ExtractionCase("E28", "technical", "medium",
        "The search engine uses Elasticsearch with a custom BM25 ranking, served by a Rust-based API gateway.",
        ["elasticsearch", "bm25", "rust", "api gateway"],
        {"elasticsearch": "search_engine", "rust": "language"},
        [("elasticsearch", "uses", "bm25")],
        ["solr", "algolia"]),
    ExtractionCase("E29", "technical", "medium",
        "We run Prometheus for metrics, Grafana for dashboards, Loki for logs, and Jaeger for distributed tracing.",
        ["prometheus", "grafana", "loki", "jaeger"],
        {"prometheus": "monitoring", "grafana": "dashboard", "loki": "logging", "jaeger": "tracing"},
        [("grafana", "visualizes", "prometheus")],
        ["datadog", "splunk"]),
    ExtractionCase("E30", "technical", "medium",
        "The ML pipeline trains models with PyTorch, serves them via TorchServe behind an NGINX reverse proxy.",
        ["pytorch", "torchserve", "nginx"],
        {"pytorch": "framework", "torchserve": "serving", "nginx": "proxy"},
        [("torchserve", "serves", "pytorch")],
        ["tensorflow", "apache"]),

    # ===== HARD: Multi-entity complex (10) =====
    ExtractionCase("E31", "complex", "hard",
        "Dr. Patel at Stanford published a paper with Prof. Kim from Seoul National on quantum error correction using surface codes.",
        ["dr. patel", "stanford", "prof. kim", "seoul national", "quantum error correction", "surface codes"],
        {"dr. patel": "person", "stanford": "university", "prof. kim": "person"},
        [("dr. patel", "affiliated", "stanford"), ("prof. kim", "affiliated", "seoul national")],
        ["mit", "ibm"]),
    ExtractionCase("E32", "complex", "hard",
        "SpaceX's Starship, powered by Raptor engines using methane, launched from Boca Chica to orbit in 2024.",
        ["spacex", "starship", "raptor", "methane", "boca chica"],
        {"spacex": "company", "starship": "rocket", "raptor": "engine"},
        [("starship", "powered_by", "raptor"), ("starship", "launched_from", "boca chica")],
        ["nasa", "blue origin"]),
    ExtractionCase("E33", "complex", "hard",
        "Apple's M3 chip, fabricated by TSMC at 3nm, powers the MacBook Pro with unified CPU, GPU, and Neural Engine.",
        ["apple", "m3", "tsmc", "3nm", "macbook pro", "cpu", "gpu", "neural engine"],
        {"apple": "company", "m3": "chip", "tsmc": "company"},
        [("m3", "fabricated_by", "tsmc"), ("m3", "powers", "macbook pro")],
        ["intel", "qualcomm"]),
    ExtractionCase("E34", "complex", "hard",
        "The European Central Bank raised interest rates by 25 basis points, affecting the euro-dollar exchange rate and bond yields across Germany, France, and Italy.",
        ["european central bank", "euro", "dollar", "germany", "france", "italy"],
        {"european central bank": "institution"},
        [("european central bank", "raised", "interest rates")],
        ["federal reserve", "bank of england"]),
    ExtractionCase("E35", "complex", "hard",
        "Netflix's recommendation engine, originally built on Apache Mahout, was rebuilt with TensorFlow on AWS SageMaker, processing 100M+ daily user interactions.",
        ["netflix", "apache mahout", "tensorflow", "aws sagemaker"],
        {"netflix": "company", "tensorflow": "framework", "aws sagemaker": "platform"},
        [("netflix", "uses", "tensorflow"), ("netflix", "uses", "aws sagemaker")],
        ["youtube", "spotify"]),
    ExtractionCase("E36", "complex", "hard",
        "The CRISPR gene-editing technique developed by Doudna and Charpentier at UC Berkeley and Umea University won the 2020 Nobel Prize in Chemistry.",
        ["crispr", "doudna", "charpentier", "uc berkeley", "umea university", "nobel prize"],
        {"doudna": "person", "charpentier": "person", "uc berkeley": "university"},
        [("doudna", "developed", "crispr"), ("charpentier", "developed", "crispr")],
        ["mit", "zhang"]),
    ExtractionCase("E37", "complex", "hard",
        "Stripe processed $1 trillion in payments in 2023, using Ruby on Rails for their dashboard, Go for the API, and Temporal for workflow orchestration.",
        ["stripe", "ruby on rails", "go", "temporal"],
        {"stripe": "company", "ruby on rails": "framework", "go": "language"},
        [("stripe", "uses", "ruby on rails"), ("stripe", "uses", "go"), ("stripe", "uses", "temporal")],
        ["paypal", "square"]),
    ExtractionCase("E38", "complex", "hard",
        "The James Webb Space Telescope, built by Northrop Grumman for NASA, orbits the L2 Lagrange point and uses a 6.5m gold-coated beryllium mirror.",
        ["james webb", "northrop grumman", "nasa", "l2 lagrange", "beryllium mirror"],
        {"james webb": "telescope", "northrop grumman": "company", "nasa": "agency"},
        [("northrop grumman", "built", "james webb"), ("james webb", "orbits", "l2 lagrange")],
        ["hubble", "esa"]),
    ExtractionCase("E39", "complex", "hard",
        "Uber's microservices architecture has over 4,000 services written in Java, Go, Python, and Node.js, orchestrated by a custom service mesh built on top of Envoy proxy.",
        ["uber", "java", "go", "python", "node.js", "envoy"],
        {"uber": "company", "envoy": "proxy"},
        [("uber", "uses", "envoy")],
        ["lyft", "istio"]),
    ExtractionCase("E40", "complex", "hard",
        "The Ethereum Foundation released the Dencun upgrade implementing EIP-4844 (proto-danksharding) to reduce Layer 2 gas fees for rollups like Arbitrum and Optimism.",
        ["ethereum", "dencun", "eip-4844", "arbitrum", "optimism"],
        {"ethereum": "blockchain", "arbitrum": "layer2", "optimism": "layer2"},
        [("ethereum", "released", "dencun"), ("dencun", "reduces_fees", "arbitrum")],
        ["bitcoin", "solana"]),

    # ===== HARD: Ambiguous/polysemous (10) =====
    ExtractionCase("E41", "ambiguous", "hard",
        "The bank near the river has the best interest rates in town.",
        ["bank"],
        {"bank": "financial_institution"},
        [],
        ["river bank", "fishing"]),
    ExtractionCase("E42", "ambiguous", "hard",
        "Jordan loves basketball and frequently visits Amman for business.",
        ["jordan", "basketball", "amman"],
        {"jordan": "person", "amman": "city"},
        [("jordan", "loves", "basketball"), ("jordan", "visits", "amman")],
        ["michael jordan", "country"]),
    ExtractionCase("E43", "ambiguous", "hard",
        "Mercury is both a planet and an element used in old thermometers.",
        ["mercury"],
        {},
        [],
        ["venus", "thermometer brand"]),
    ExtractionCase("E44", "ambiguous", "hard",
        "Apple released a new product while I was eating an apple in the Apple Store in the Big Apple.",
        ["apple", "apple store", "big apple"],
        {"apple": "company", "apple store": "retail", "big apple": "city"},
        [],
        ["fruit", "new york"]),
    ExtractionCase("E45", "ambiguous", "hard",
        "The Python developer was scared of the python she found in the Python State Park.",
        ["python"],
        {},
        [],
        ["snake", "park"]),
    ExtractionCase("E46", "ambiguous", "hard",
        "Java programmers at the Java Hut coffee shop in Java, Indonesia discuss JavaScript over Java coffee.",
        ["java", "java hut", "indonesia", "javascript"],
        {},
        [],
        ["oracle"]),
    ExtractionCase("E47", "ambiguous", "hard",
        "The coach coached his team to victory, then sat on a coach on the coach bus home.",
        ["coach"],
        {"coach": "person"},
        [],
        ["bus driver"]),
    ExtractionCase("E48", "ambiguous", "hard",
        "Spring is my favorite season, and I love using Spring Boot to build Spring cleaning apps.",
        ["spring", "spring boot"],
        {"spring boot": "framework"},
        [],
        ["java"]),
    ExtractionCase("E49", "ambiguous", "hard",
        "I switched from Vim to VS Code but kept Vim keybindings because old habits die hard.",
        ["vim", "vs code", "vim keybindings"],
        {"vim": "software", "vs code": "software"},
        [],
        ["neovim", "emacs"]),
    ExtractionCase("E50", "ambiguous", "hard",
        "The AWS Lambda function processes Lambda calculus homework for students at Lambda School.",
        ["aws lambda", "lambda calculus", "lambda school"],
        {"aws lambda": "service", "lambda school": "school"},
        [],
        ["serverless"]),

    # ===== HARD: Temporal/dated (10) =====
    ExtractionCase("E51", "temporal", "hard",
        "In 2023 OpenAI released GPT-4, in 2024 GPT-4o, and in 2025 they released o3.",
        ["openai", "gpt-4", "gpt-4o", "o3"],
        {"openai": "company", "gpt-4": "model", "gpt-4o": "model", "o3": "model"},
        [("openai", "released", "gpt-4"), ("openai", "released", "gpt-4o"), ("openai", "released", "o3")],
        ["gpt-3", "google"]),
    ExtractionCase("E52", "temporal", "hard",
        "Netflix started as DVD rental in 1997, pivoted to streaming in 2007, and began original content in 2013 with House of Cards.",
        ["netflix", "dvd rental", "streaming", "house of cards"],
        {"netflix": "company", "house of cards": "show"},
        [("netflix", "started_as", "dvd rental"), ("netflix", "pivoted_to", "streaming")],
        ["blockbuster"]),
    ExtractionCase("E53", "temporal", "hard",
        "Docker (2013), Kubernetes (2014, Google), Helm (2015, Deis), and Istio (2017, Google+IBM) form the modern container stack.",
        ["docker", "kubernetes", "google", "helm", "deis", "istio", "ibm"],
        {"docker": "technology", "kubernetes": "technology", "google": "company"},
        [("google", "created", "kubernetes"), ("deis", "created", "helm")],
        ["aws", "podman"]),
    ExtractionCase("E54", "temporal", "hard",
        "The company rebranded from Facebook to Meta in October 2021 to focus on the metaverse.",
        ["facebook", "meta", "metaverse"],
        {"facebook": "company", "meta": "company"},
        [("facebook", "rebranded_to", "meta"), ("meta", "focuses_on", "metaverse")],
        ["instagram", "mark zuckerberg"]),
    ExtractionCase("E55", "temporal", "hard",
        "Rust won StackOverflow's most-loved language from 2016-2023, while Python topped the most-wanted list.",
        ["rust", "stackoverflow", "python"],
        {"rust": "language", "python": "language"},
        [("rust", "won", "stackoverflow")],
        ["go", "c++"]),
    ExtractionCase("E56", "temporal", "hard",
        "TypeScript 1.0 was released by Microsoft in 2014, reached 2.0 in 2016, 3.0 in 2018, 4.0 in 2020, and 5.0 in 2023.",
        ["typescript", "microsoft"],
        {"typescript": "language", "microsoft": "company"},
        [("microsoft", "released", "typescript")],
        ["javascript", "google"]),
    ExtractionCase("E57", "temporal", "hard",
        "Anthropic was founded in 2021 by Dario and Daniela Amodei after they left OpenAI, and released Claude in 2023.",
        ["anthropic", "dario amodei", "daniela amodei", "openai", "claude"],
        {"anthropic": "company", "dario amodei": "person", "daniela amodei": "person", "claude": "model"},
        [("dario amodei", "founded", "anthropic"), ("daniela amodei", "founded", "anthropic"), ("anthropic", "released", "claude")],
        ["google"]),
    ExtractionCase("E58", "temporal", "hard",
        "AWS launched EC2 in 2006, S3 the same year, Lambda in 2014, and Bedrock in 2023 for generative AI.",
        ["aws", "ec2", "s3", "lambda", "bedrock"],
        {"aws": "company", "ec2": "service", "s3": "service", "lambda": "service", "bedrock": "service"},
        [("aws", "launched", "ec2"), ("aws", "launched", "bedrock")],
        ["azure", "gcp"]),
    ExtractionCase("E59", "temporal", "hard",
        "React was released by Facebook in 2013, Vue by Evan You in 2014, and Svelte by Rich Harris in 2016.",
        ["react", "facebook", "vue", "evan you", "svelte", "rich harris"],
        {"react": "framework", "vue": "framework", "svelte": "framework"},
        [("facebook", "released", "react"), ("evan you", "created", "vue"), ("rich harris", "created", "svelte")],
        ["angular"]),
    ExtractionCase("E60", "temporal", "hard",
        "Bitcoin was created by Satoshi Nakamoto in 2009, Ethereum by Vitalik Buterin in 2015, and Solana by Anatoly Yakovenko in 2020.",
        ["bitcoin", "satoshi nakamoto", "ethereum", "vitalik buterin", "solana", "anatoly yakovenko"],
        {"bitcoin": "cryptocurrency", "ethereum": "cryptocurrency", "solana": "cryptocurrency"},
        [("satoshi nakamoto", "created", "bitcoin"), ("vitalik buterin", "created", "ethereum"), ("anatoly yakovenko", "created", "solana")],
        ["cardano"]),

    # ===== HARD: Nested/complex relationships (10) =====
    ExtractionCase("E61", "nested", "hard",
        "Bob's wife Alice works at the hospital where Bob's mother was treated for pneumonia.",
        ["bob", "alice", "hospital"],
        {"bob": "person", "alice": "person"},
        [("bob", "wife", "alice"), ("alice", "works_at", "hospital")],
        ["doctor"]),
    ExtractionCase("E62", "nested", "hard",
        "The startup founded by Emma, acquired by Microsoft, developed an AI tool competing with Copilot.",
        ["emma", "microsoft", "copilot"],
        {"emma": "person", "microsoft": "company", "copilot": "product"},
        [("microsoft", "acquired", "startup")],
        ["github"]),
    ExtractionCase("E63", "nested", "hard",
        "The professor who taught me at Berkeley now leads the AI lab that developed AlphaFold at DeepMind.",
        ["berkeley", "alphafold", "deepmind"],
        {"berkeley": "university", "deepmind": "company", "alphafold": "product"},
        [("deepmind", "developed", "alphafold")],
        ["google"]),
    ExtractionCase("E64", "nested", "hard",
        "The monorepo managed by the platform team uses Bazel, which Google developed for their internal infra.",
        ["platform team", "bazel", "google"],
        {"bazel": "tool", "google": "company"},
        [("platform team", "uses", "bazel"), ("google", "developed", "bazel")],
        ["gradle"]),
    ExtractionCase("E65", "nested", "hard",
        "The CEO's daughter who graduated from Harvard now runs the AI division that was spun off from the main company.",
        ["ceo", "daughter", "harvard", "ai division"],
        {"harvard": "university"},
        [],
        ["mit"]),
    ExtractionCase("E66", "nested", "hard",
        "My mentor's former student, now at DeepMind, published the paper that our team's architecture is based on.",
        ["mentor", "deepmind"],
        {"deepmind": "company"},
        [],
        ["google"]),
    ExtractionCase("E67", "nested", "hard",
        "The open-source library maintained by the Vue core team member was forked by Shopify to build their Hydrogen framework.",
        ["vue", "shopify", "hydrogen"],
        {"shopify": "company", "hydrogen": "framework"},
        [("shopify", "built", "hydrogen")],
        ["react"]),
    ExtractionCase("E68", "nested", "hard",
        "The datacenter in Oregon that hosts Netflix's encoding pipeline is powered by the wind farm owned by Google's subsidiary.",
        ["oregon", "netflix", "google"],
        {"netflix": "company", "google": "company", "oregon": "location"},
        [],
        ["aws"]),
    ExtractionCase("E69", "nested", "hard",
        "The algorithm designed by Professor Zhang at Tsinghua was implemented in C++ by his PhD student and deployed at ByteDance.",
        ["professor zhang", "tsinghua", "c++", "bytedance"],
        {"professor zhang": "person", "tsinghua": "university", "bytedance": "company"},
        [("professor zhang", "affiliated", "tsinghua")],
        ["alibaba"]),
    ExtractionCase("E70", "nested", "hard",
        "The security vulnerability discovered by the Google Project Zero team in the Linux kernel was patched by Red Hat before the Debian team could release their fix.",
        ["google project zero", "linux kernel", "red hat", "debian"],
        {"red hat": "company", "debian": "os"},
        [("google project zero", "discovered", "vulnerability")],
        ["ubuntu"]),

    # ===== EDGE CASES (10) =====
    ExtractionCase("E71", "edge", "hard",
        "",
        [],
        {},
        [],
        ["anything"]),
    ExtractionCase("E72", "edge", "hard",
        "The weather is nice today.",
        [],
        {},
        [],
        ["sun"]),
    ExtractionCase("E73", "edge", "hard",
        "I use Claude, GPT-4, Gemini, Llama 3, Mistral, Cohere Command-R, and Grok for different tasks.",
        ["claude", "gpt-4", "gemini", "llama 3", "mistral", "cohere command-r", "grok"],
        {"claude": "model", "gpt-4": "model"},
        [],
        ["bard"]),
    ExtractionCase("E74", "edge", "hard",
        "AWS S3 costs $0.023/GB, GCS $0.020/GB, Azure Blob $0.018/GB for standard storage.",
        ["aws s3", "gcs", "azure blob"],
        {"aws s3": "service", "gcs": "service", "azure blob": "service"},
        [],
        ["ec2"]),
    ExtractionCase("E75", "edge", "hard",
        "日本語のテキストも正しく処理できるべきです。東京大学の田中教授がAI研究を発表しました。",
        ["東京大学", "田中"],
        {"東京大学": "university", "田中": "person"},
        [("田中", "affiliated", "東京大学")],
        ["京都"]),
    ExtractionCase("E76", "edge", "hard",
        "El restaurante de María en Barcelona sirve la mejor paella del barrio Gòtic.",
        ["maría", "barcelona", "paella", "barrio gòtic"],
        {"maría": "person", "barcelona": "city"},
        [("maría", "owns", "restaurante")],
        ["madrid"]),
    ExtractionCase("E77", "edge", "hard",
        "😊 User123 @mentioned #hashtag in the $AAPL stock discussion on r/wallstreetbets.",
        ["user123", "aapl", "wallstreetbets"],
        {},
        [],
        ["reddit"]),
    ExtractionCase("E78", "edge", "hard",
        "SELECT * FROM users WHERE name = 'Robert'; DROP TABLE users;--' AND role = 'admin';",
        ["users", "robert"],
        {},
        [],
        ["sql injection"]),
    ExtractionCase("E79", "edge", "hard",
        "A " * 50 + "B " * 50 + "C",
        [],
        {},
        [],
        []),
    ExtractionCase("E80", "edge", "hard",
        "The 2024 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) held at the Seattle Convention Center featured 2,719 accepted papers from 11,532 submissions.",
        ["ieee", "cvpr", "seattle convention center"],
        {"cvpr": "conference", "seattle convention center": "venue"},
        [("cvpr", "held_at", "seattle convention center")],
        ["neurips", "icml"]),
]


# ============================================================
# CONTRADICTION TEST CASES — 50 cases
# ============================================================

CONTRADICTION_CASES: list[ContradictionCase] = [
    # ===== DIRECT CONTRADICTIONS (10) =====
    ContradictionCase("C01", "direct", "easy",
        "alice -- lives_in -- new york",
        "Alice just moved to San Francisco.",
        "delete", ["alice", "new york"]),
    ContradictionCase("C02", "direct", "easy",
        "bob -- works_at -- google",
        "Bob left Google and joined Anthropic.",
        "delete", ["bob", "google"]),
    ContradictionCase("C03", "direct", "easy",
        "sarah -- favorite_language -- python",
        "Sarah says her favorite language is now Rust.",
        "delete", ["sarah", "python"]),
    ContradictionCase("C04", "direct", "easy",
        "company -- database -- mysql",
        "The company completed migration from MySQL to PostgreSQL.",
        "delete", ["company", "mysql"]),
    ContradictionCase("C05", "direct", "easy",
        "project_alpha -- status -- active",
        "Project Alpha was cancelled due to budget cuts.",
        "delete", ["project_alpha", "active"]),
    ContradictionCase("C06", "direct", "medium",
        "team -- framework -- angular",
        "The team finished migrating everything from Angular to React.",
        "delete", ["team", "angular"]),
    ContradictionCase("C07", "direct", "medium",
        "server -- os -- ubuntu_20.04",
        "We upgraded all servers to Ubuntu 24.04 LTS.",
        "delete", ["server", "ubuntu_20.04"]),
    ContradictionCase("C08", "direct", "medium",
        "company -- headquarters -- san_francisco",
        "The company relocated its headquarters to Austin, Texas.",
        "delete", ["company", "san_francisco"]),
    ContradictionCase("C09", "direct", "medium",
        "david -- position -- senior_engineer\ndavid -- team -- backend",
        "David was promoted to Engineering Manager and now leads the platform team.",
        "delete", ["david", "senior_engineer"]),
    ContradictionCase("C10", "direct", "hard",
        "product -- pricing -- freemium\nproduct -- free_tier -- 1000_requests",
        "We eliminated the free tier entirely; all plans now require payment starting at $9.99/month.",
        "delete", ["product", "freemium"]),

    # ===== NOOP — ADDITIVE (should NOT delete) (15) =====
    ContradictionCase("C11", "noop_additive", "easy",
        "alice -- loves_to_eat -- pizza",
        "Alice also loves to eat sushi.",
        "noop", []),
    ContradictionCase("C12", "noop_additive", "easy",
        "bob -- knows -- python\nbob -- knows -- javascript",
        "Bob also learned Rust recently.",
        "noop", []),
    ContradictionCase("C13", "noop_additive", "easy",
        "team -- uses -- kubernetes",
        "The team also started using Terraform for infrastructure.",
        "noop", []),
    ContradictionCase("C14", "noop_additive", "easy",
        "lisa -- has_pet -- dog named max",
        "Lisa adopted a cat named Whiskers.",
        "noop", []),
    ContradictionCase("C15", "noop_additive", "easy",
        "john -- hobby -- playing guitar\njohn -- hobby -- painting",
        "John started learning to cook Italian food.",
        "noop", []),
    ContradictionCase("C16", "noop_additive", "medium",
        "company -- office -- new_york\ncompany -- office -- london",
        "The company opened a new office in Tokyo.",
        "noop", []),
    ContradictionCase("C17", "noop_additive", "medium",
        "sarah -- certification -- aws_solutions_architect",
        "Sarah also passed the GCP Professional Data Engineer certification.",
        "noop", []),
    ContradictionCase("C18", "noop_additive", "medium",
        "project -- dependency -- react\nproject -- dependency -- typescript",
        "We added Tailwind CSS as a new dependency to the project.",
        "noop", []),
    ContradictionCase("C19", "noop_additive", "medium",
        "user -- visited -- paris\nuser -- visited -- rome",
        "The user just got back from a trip to Barcelona.",
        "noop", []),
    ContradictionCase("C20", "noop_additive", "medium",
        "developer -- contributed_to -- react\ndeveloper -- contributed_to -- vue",
        "The developer started contributing to Svelte as well.",
        "noop", []),
    ContradictionCase("C21", "noop_additive", "hard",
        "alice -- friend -- bob\nalice -- friend -- charlie",
        "Alice became friends with Diana at the conference.",
        "noop", []),
    ContradictionCase("C22", "noop_additive", "hard",
        "app -- integration -- slack\napp -- integration -- github",
        "We added Jira integration to the app.",
        "noop", []),
    ContradictionCase("C23", "noop_additive", "hard",
        "team -- deployed_to -- us-east-1\nteam -- deployed_to -- eu-west-1",
        "We expanded deployment to the ap-southeast-1 region.",
        "noop", []),
    ContradictionCase("C24", "noop_additive", "hard",
        "student -- completed_course -- algorithms\nstudent -- completed_course -- databases",
        "The student enrolled in the machine learning course.",
        "noop", []),
    ContradictionCase("C25", "noop_additive", "hard",
        "chef -- specialty -- italian\nchef -- specialty -- french",
        "The chef learned Japanese cuisine during a trip to Kyoto.",
        "noop", []),

    # ===== NOOP — UNRELATED (should NOT delete) (5) =====
    ContradictionCase("C26", "noop_unrelated", "medium",
        "alice -- lives_in -- seattle",
        "Bob got a new job at Microsoft.",
        "noop", []),
    ContradictionCase("C27", "noop_unrelated", "medium",
        "server -- runs -- postgresql\nserver -- hosted_on -- aws",
        "The marketing team launched a new campaign on Instagram.",
        "noop", []),
    ContradictionCase("C28", "noop_unrelated", "hard",
        "project -- deadline -- march_2026\nproject -- lead -- elena",
        "The sales team exceeded Q4 targets by 15%.",
        "noop", []),
    ContradictionCase("C29", "noop_unrelated", "hard",
        "user -- preference -- dark_mode\nuser -- timezone -- pst",
        "The weather in London is rainy today.",
        "noop", []),
    ContradictionCase("C30", "noop_unrelated", "hard",
        "app -- version -- 3.2.1\napp -- framework -- django",
        "NASA launched a new satellite to study climate change.",
        "noop", []),

    # ===== SUBTLE/PARTIAL CONTRADICTIONS (10) =====
    ContradictionCase("C31", "subtle", "medium",
        "alex -- salary -- 120000",
        "Alex got a raise and now earns $150,000.",
        "delete", ["alex", "120000"]),
    ContradictionCase("C32", "subtle", "medium",
        "team_size -- equals -- 5",
        "After recent hiring, the team now has 8 members.",
        "delete", ["team_size", "5"]),
    ContradictionCase("C33", "subtle", "medium",
        "product -- version -- 2.3",
        "We just released version 3.0 of the product.",
        "delete", ["product", "2.3"]),
    ContradictionCase("C34", "subtle", "medium",
        "manager -- of -- engineering_team -- is -- david",
        "David was promoted to VP; Sarah is now the engineering manager.",
        "delete", ["david", "manager"]),
    ContradictionCase("C35", "subtle", "hard",
        "server -- type -- t3.medium",
        "We scaled up the server to t3.xlarge due to increased traffic.",
        "delete", ["t3.medium"]),
    ContradictionCase("C36", "subtle", "hard",
        "api -- rate_limit -- 100_per_minute",
        "We increased the API rate limit to 500 requests per minute for all users.",
        "delete", ["100_per_minute"]),
    ContradictionCase("C37", "subtle", "hard",
        "deployment -- strategy -- rolling_update",
        "We switched from rolling updates to blue-green deployments for zero downtime.",
        "delete", ["rolling_update"]),
    ContradictionCase("C38", "subtle", "hard",
        "cache -- ttl -- 3600_seconds\ncache -- provider -- redis",
        "We reduced the cache TTL to 300 seconds to ensure fresher data.",
        "delete", ["3600_seconds"]),
    ContradictionCase("C39", "subtle", "hard",
        "model -- accuracy -- 92_percent\nmodel -- framework -- pytorch",
        "After retraining with more data, the model now achieves 97% accuracy.",
        "delete", ["92_percent"]),
    ContradictionCase("C40", "subtle", "hard",
        "ci_pipeline -- duration -- 45_minutes\nci_pipeline -- tool -- github_actions",
        "After optimization, CI pipeline now completes in 12 minutes.",
        "delete", ["45_minutes"]),

    # ===== TEMPORAL SUPERSESSION (10) =====
    ContradictionCase("C41", "temporal", "medium",
        "ceo -- of -- company -- is -- john",
        "As of January 2026, the new CEO is Maria.",
        "delete", ["john", "ceo"]),
    ContradictionCase("C42", "temporal", "medium",
        "react -- latest_version -- 18",
        "React 19 was released with server components support.",
        "delete", ["react", "18"]),
    ContradictionCase("C43", "temporal", "medium",
        "user -- subscription -- basic_plan",
        "The user upgraded to the premium plan yesterday.",
        "delete", ["basic_plan"]),
    ContradictionCase("C44", "temporal", "medium",
        "office -- location -- downtown",
        "The company relocated from downtown to the tech park campus.",
        "delete", ["downtown"]),
    ContradictionCase("C45", "temporal", "medium",
        "student -- status -- enrolled",
        "The student graduated last month with honors.",
        "delete", ["enrolled"]),
    ContradictionCase("C46", "temporal", "hard",
        "project -- phase -- development\nproject -- sprint -- 14",
        "The project moved to the QA/testing phase after completing sprint 14.",
        "delete", ["development"]),
    ContradictionCase("C47", "temporal", "hard",
        "employee -- clearance -- level_2",
        "After the background check, the employee's clearance was upgraded to level 4.",
        "delete", ["level_2"]),
    ContradictionCase("C48", "temporal", "hard",
        "python -- recommended_version -- 3.11\npython -- end_of_life -- 3.8",
        "Python 3.13 is now the recommended version; 3.11 enters maintenance mode.",
        "delete", ["3.11"]),
    ContradictionCase("C49", "temporal", "hard",
        "kubernetes -- cluster_version -- 1.27\nkubernetes -- nodes -- 12",
        "We upgraded the Kubernetes cluster from 1.27 to 1.29.",
        "delete", ["1.27"]),
    ContradictionCase("C50", "temporal", "hard",
        "database -- size -- 500gb\ndatabase -- provider -- rds",
        "The database has grown to 2.3TB after the data migration from the legacy system.",
        "delete", ["500gb"]),
]


# ============================================================
# LLM Clients
# ============================================================

class ClaudeClient:
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
        anthropic_tools = []
        for tool in tools:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        api_messages = [m for m in messages if m["role"] != "system"]
        params: dict[str, Any] = {
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
        }


class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-lite"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _clean_schema(self, schema: dict) -> dict:
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

    def _convert_tools(self, tools: list[dict]) -> list[genai_types.Tool]:
        declarations = []
        for tool in tools:
            fn = tool["function"]
            params = fn.get("parameters", {"type": "object", "properties": {}})
            declarations.append(genai_types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=self._clean_schema(params),
            ))
        return [genai_types.Tool(function_declarations=declarations)]

    def call_with_tools(self, messages: list[dict], tools: list[dict], system: str = "") -> dict:
        genai_tools = self._convert_tools(tools)
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(genai_types.Content(parts=[genai_types.Part(text=msg["content"])], role=role))
        config = genai_types.GenerateContentConfig(
            tools=genai_tools,
            tool_config=genai_types.ToolConfig(function_calling_config=genai_types.FunctionCallingConfig(mode="ANY")),
        )
        if system:
            config.system_instruction = system
        start = time.time()
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as e:
            return {"error": str(e), "latency": time.time() - start, "tool_calls": [], "content": ""}
        latency = time.time() - start
        tool_calls = []
        text_parts = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append({"name": fc.name, "arguments": dict(fc.args) if fc.args else {}})
                if part.text:
                    text_parts.append(part.text)
        usage = response.usage_metadata
        inp = usage.prompt_token_count if usage else 0
        out = usage.candidates_token_count if usage else 0
        self.total_input_tokens += inp
        self.total_output_tokens += out
        return {"content": "\n".join(text_parts), "tool_calls": tool_calls, "latency": latency, "input_tokens": inp, "output_tokens": out}


# ============================================================
# Scoring
# ============================================================

def _norm(s: str) -> str:
    return s.lower().strip().replace("_", " ").replace("-", " ")

def score_entities(result: dict, case: ExtractionCase) -> dict:
    if "error" in result:
        return {"f1": 0, "recall": 0, "precision": 0, "hallucinated": 0, "extracted_count": 0}
    extracted = []
    for tc in result.get("tool_calls", []):
        if tc["name"] == "extract_entities":
            for e in tc["arguments"].get("entities", []):
                extracted.append(_norm(e.get("entity", "")))
    expected = [_norm(e) for e in case.expected_entities]
    found = 0
    for exp in expected:
        for ext in extracted:
            if exp in ext or ext in exp:
                found += 1
                break
    hallucinated = 0
    for ext in extracted:
        for trap in [_norm(t) for t in case.hallucination_traps]:
            if trap in ext or ext in trap:
                hallucinated += 1
                break
    prec = found / len(extracted) if extracted else (1.0 if not expected else 0)
    rec = found / len(expected) if expected else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"f1": round(f1, 4), "recall": round(rec, 4), "precision": round(prec, 4), "hallucinated": hallucinated, "extracted_count": len(extracted)}

def score_relationships(result: dict, case: ExtractionCase) -> dict:
    if "error" in result:
        return {"score": 0, "found": 0, "expected": len(case.expected_relationships)}
    rels = []
    for tc in result.get("tool_calls", []):
        if tc["name"] in ("establish_relationships", "establish_relations"):
            for e in tc["arguments"].get("entities", []):
                rels.append((_norm(e.get("source", "")), _norm(e.get("relationship", "")), _norm(e.get("destination", ""))))
    expected = [(_norm(s), _norm(r), _norm(d)) for s, r, d in case.expected_relationships]
    found = 0
    for es, er, ed in expected:
        for rs, rr, rd in rels:
            if (es in rs or rs in es) and (ed in rd or rd in ed) and (er in rr or rr in er):
                found += 1
                break
    score = found / len(expected) if expected else 1.0
    return {"score": round(score, 4), "found": found, "expected": len(expected), "extracted": len(rels)}

def score_contradiction(result: dict, case: ContradictionCase) -> dict:
    if "error" in result:
        return {"correct": False, "expected": case.expected_action, "actual": "error", "score": 0}
    actions = set()
    for tc in result.get("tool_calls", []):
        if tc["name"] == "delete_graph_memory":
            actions.add("delete")
        elif tc["name"] == "update_graph_memory":
            actions.add("update")
        elif tc["name"] == "add_graph_memory":
            actions.add("add")
        elif tc["name"] == "noop":
            actions.add("noop")
    if not actions:
        actual = "none"
    elif len(actions) == 1:
        actual = actions.pop()
    else:
        for p in ["delete", "update", "add", "noop"]:
            if p in actions:
                actual = p
                break
        else:
            actual = "mixed"
    correct = actual == case.expected_action
    return {"correct": correct, "expected": case.expected_action, "actual": actual, "score": 1.0 if correct else 0.0}

def score_tool_reliability(result: dict) -> float:
    if "error" in result:
        return 0.0
    tcs = result.get("tool_calls", [])
    if not tcs:
        return 0.0
    if all(isinstance(tc.get("arguments"), dict) for tc in tcs):
        return 1.0
    return 0.5

def score_schema(result: dict, expected_tool: str) -> float:
    if "error" in result:
        return 0.0
    tcs = result.get("tool_calls", [])
    if not tcs:
        return 0.0
    correct_tool = any(tc["name"] == expected_tool for tc in tcs)
    valid = 0
    for tc in tcs:
        a = tc.get("arguments", {})
        if tc["name"] == "extract_entities":
            if "entities" in a and isinstance(a["entities"], list) and all("entity" in e and "entity_type" in e for e in a["entities"]):
                valid += 1
        elif tc["name"] in ("establish_relationships", "establish_relations"):
            if "entities" in a and isinstance(a["entities"], list) and all("source" in e and "relationship" in e and "destination" in e for e in a["entities"]):
                valid += 1
        elif tc["name"] == "delete_graph_memory":
            if all(k in a for k in ["source", "relationship", "destination"]):
                valid += 1
        elif tc["name"] == "noop":
            valid += 1
        elif tc["name"] in ("update_graph_memory", "add_graph_memory"):
            valid += 1
    all_valid = valid == len(tcs)
    return (1.0 if correct_tool else 0.5) * (1.0 if all_valid else 0.5)


# ============================================================
# Runner
# ============================================================

def ci95(scores: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for 95% CI."""
    n = len(scores)
    if n == 0:
        return 0, 0, 0
    mean = sum(scores) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    se = math.sqrt(var / n)
    margin = 1.96 * se
    return mean, max(0, mean - margin), min(1, mean + margin)


def bar(val: float, width: int = 30) -> str:
    filled = int(val * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def main():
    total_extraction = len([c for c in EXTRACTION_CASES if c.text.strip()])
    total_rel = len([c for c in EXTRACTION_CASES if c.expected_relationships and c.text.strip()])
    total_contradiction = len(CONTRADICTION_CASES)
    total_cases = total_extraction + total_rel + total_contradiction

    print("=" * 72)
    print("  GRAPH LLM BATTLE TEST v2 — COMPREHENSIVE BENCHMARK")
    print("  Claude Opus 4.6 vs Gemini 2.5 Flash Lite")
    print(f"  {total_cases} test runs across 7 dimensions")
    print(f"  ({total_extraction} entity + {total_rel} relationship + {total_contradiction} contradiction)")
    print("=" * 72)

    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GOOGLE_API_KEY env var required")
        sys.exit(1)
    print("\nInitializing clients...")
    claude = ClaudeClient()
    gemini = GeminiClient(gemini_key)
    print(f"  Claude: {claude.model}")
    print(f"  Gemini: {gemini.model}")

    # Storage per dimension, per difficulty
    results: dict[str, dict[str, dict[str, list[float]]]] = {
        dim: {"claude": {"easy": [], "medium": [], "hard": [], "all": []},
              "gemini": {"easy": [], "medium": [], "hard": [], "all": []}}
        for dim in ["entity_f1", "entity_recall", "relationship", "contradiction",
                     "tool_reliability", "schema", "hallucination", "latency"]
    }

    # Per-case details for the report
    case_details: list[dict] = []

    # =========== PHASE 1: Entity Extraction ===========
    valid_cases = [c for c in EXTRACTION_CASES if c.text.strip()]
    print(f"\n{'=' * 72}")
    print(f"  PHASE 1: Entity Extraction ({len(valid_cases)} cases)")
    print(f"{'=' * 72}")

    for case in valid_cases:
        label = f"[{case.id}] {case.difficulty}/{case.category}"
        short_text = case.text[:55].replace("\n", " ")
        print(f"  {label}: {short_text}...", end="", flush=True)

        messages = [{"role": "user", "content": f"Extract entities from the following text:\n\n{case.text}"}]

        cr = claude.call_with_tools(messages, [EXTRACT_ENTITIES_TOOL])
        cs = score_entities(cr, case)
        ct = score_tool_reliability(cr)
        csch = score_schema(cr, "extract_entities")
        ch = 1.0 - (cs["hallucinated"] / max(1, cs["extracted_count"]))

        gr = gemini.call_with_tools(messages, [EXTRACT_ENTITIES_TOOL])
        gs = score_entities(gr, case)
        gt = score_tool_reliability(gr)
        gsch = score_schema(gr, "extract_entities")
        gh = 1.0 - (gs["hallucinated"] / max(1, gs["extracted_count"]))

        for d in [case.difficulty, "all"]:
            results["entity_f1"]["claude"][d].append(cs["f1"])
            results["entity_f1"]["gemini"][d].append(gs["f1"])
            results["entity_recall"]["claude"][d].append(cs["recall"])
            results["entity_recall"]["gemini"][d].append(gs["recall"])
            results["tool_reliability"]["claude"][d].append(ct)
            results["tool_reliability"]["gemini"][d].append(gt)
            results["schema"]["claude"][d].append(csch)
            results["schema"]["gemini"][d].append(gsch)
            results["hallucination"]["claude"][d].append(ch)
            results["hallucination"]["gemini"][d].append(gh)
            results["latency"]["claude"][d].append(cr.get("latency", 0))
            results["latency"]["gemini"][d].append(gr.get("latency", 0))

        winner = "C" if cs["f1"] > gs["f1"] else ("G" if gs["f1"] > cs["f1"] else "=")
        print(f"  C:{cs['f1']:.2f} G:{gs['f1']:.2f} [{winner}] ({cr.get('latency',0):.1f}s/{gr.get('latency',0):.1f}s)")

        case_details.append({"id": case.id, "phase": "entity", "difficulty": case.difficulty,
                             "claude_f1": cs["f1"], "gemini_f1": gs["f1"],
                             "claude_halluc": cs["hallucinated"], "gemini_halluc": gs["hallucinated"]})

    # =========== PHASE 2: Relationship Extraction ===========
    rel_cases = [c for c in EXTRACTION_CASES if c.expected_relationships and c.text.strip()]
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2: Relationship Extraction ({len(rel_cases)} cases)")
    print(f"{'=' * 72}")

    for case in rel_cases:
        label = f"[{case.id}] {case.difficulty}/{case.category}"
        short_text = case.text[:55].replace("\n", " ")
        print(f"  {label}: {short_text}...", end="", flush=True)

        entity_list = ", ".join(case.expected_entities[:12])
        prompt = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", "test_user")
        messages = [{"role": "user", "content": f"Entities found: {entity_list}\n\nOriginal text: {case.text}"}]

        cr = claude.call_with_tools(messages, [RELATIONS_TOOL], system=prompt)
        cs = score_relationships(cr, case)
        ct = score_tool_reliability(cr)
        csch = score_schema(cr, "establish_relationships")

        gr = gemini.call_with_tools(messages, [RELATIONS_TOOL], system=prompt)
        gs = score_relationships(gr, case)
        gt = score_tool_reliability(gr)
        gsch = score_schema(gr, "establish_relationships")

        for d in [case.difficulty, "all"]:
            results["relationship"]["claude"][d].append(cs["score"])
            results["relationship"]["gemini"][d].append(gs["score"])
            results["tool_reliability"]["claude"][d].append(ct)
            results["tool_reliability"]["gemini"][d].append(gt)
            results["schema"]["claude"][d].append(csch)
            results["schema"]["gemini"][d].append(gsch)
            results["latency"]["claude"][d].append(cr.get("latency", 0))
            results["latency"]["gemini"][d].append(gr.get("latency", 0))

        winner = "C" if cs["score"] > gs["score"] else ("G" if gs["score"] > cs["score"] else "=")
        print(f"  C:{cs['score']:.2f}({cs['found']}/{cs['expected']}) G:{gs['score']:.2f}({gs['found']}/{gs['expected']}) [{winner}]")

        case_details.append({"id": case.id, "phase": "relationship", "difficulty": case.difficulty,
                             "claude_score": cs["score"], "gemini_score": gs["score"]})

    # =========== PHASE 3: Contradiction Detection ===========
    print(f"\n{'=' * 72}")
    print(f"  PHASE 3: Contradiction Detection ({len(CONTRADICTION_CASES)} cases)")
    print(f"{'=' * 72}")

    for case in CONTRADICTION_CASES:
        label = f"[{case.id}] {case.difficulty}/{case.category}"
        short_info = case.new_info[:55].replace("\n", " ")
        print(f"  {label}: {short_info}...", end="", flush=True)

        system = DELETE_RELATIONS_SYSTEM_PROMPT.replace("USER_ID", "test_user")
        user_msg = f"Here are the existing memories: {case.existing_memories}\n\nNew Information: {case.new_info}"
        messages = [{"role": "user", "content": user_msg}]
        tools = [DELETE_MEMORY_TOOL_GRAPH, ADD_MEMORY_TOOL_GRAPH, NOOP_TOOL]

        cr = claude.call_with_tools(messages, tools, system=system)
        cs = score_contradiction(cr, case)
        ct = score_tool_reliability(cr)

        gr = gemini.call_with_tools(messages, tools, system=system)
        gs = score_contradiction(gr, case)
        gt = score_tool_reliability(gr)

        for d in [case.difficulty, "all"]:
            results["contradiction"]["claude"][d].append(cs["score"])
            results["contradiction"]["gemini"][d].append(gs["score"])
            results["tool_reliability"]["claude"][d].append(ct)
            results["tool_reliability"]["gemini"][d].append(gt)
            results["latency"]["claude"][d].append(cr.get("latency", 0))
            results["latency"]["gemini"][d].append(gr.get("latency", 0))

        c_mark = "PASS" if cs["correct"] else "FAIL"
        g_mark = "PASS" if gs["correct"] else "FAIL"
        print(f"  C:{c_mark}({cs['actual']}) G:{g_mark}({gs['actual']}) exp={cs['expected']}")

        case_details.append({"id": case.id, "phase": "contradiction", "difficulty": case.difficulty,
                             "category": case.category,
                             "claude_correct": cs["correct"], "gemini_correct": gs["correct"],
                             "claude_action": cs["actual"], "gemini_action": gs["actual"],
                             "expected": cs["expected"]})

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print(f"\n\n{'#' * 72}")
    print(f"  FINAL RESULTS — 7-DIMENSION BATTLE SCORECARD (v2)")
    print(f"{'#' * 72}")

    def print_dim(title: str, dim_key: str, higher_better: bool = True):
        c_all = results[dim_key]["claude"]["all"]
        g_all = results[dim_key]["gemini"]["all"]
        cm, clo, chi = ci95(c_all)
        gm, glo, ghi = ci95(g_all)
        delta = cm - gm if higher_better else gm - cm
        winner = "CLAUDE" if delta > 0.01 else ("GEMINI" if delta < -0.01 else "TIE")

        print(f"\n{'=' * 68}")
        print(f"  {title}")
        print(f"{'=' * 68}")
        print(f"  Claude: {bar(cm)} {cm:.1%} (95% CI: {clo:.1%}-{chi:.1%}, n={len(c_all)})")
        print(f"  Gemini: {bar(gm)} {gm:.1%} (95% CI: {glo:.1%}-{ghi:.1%}, n={len(g_all)})")
        print(f"  Winner: {winner} (delta: {abs(cm - gm):.1%})")

        # By difficulty
        for diff in ["easy", "medium", "hard"]:
            cd = results[dim_key]["claude"][diff]
            gd = results[dim_key]["gemini"][diff]
            if cd:
                cdm = sum(cd) / len(cd)
                gdm = sum(gd) / len(gd)
                w = "C" if (cdm > gdm + 0.01 if higher_better else cdm < gdm - 0.01) else ("G" if (gdm > cdm + 0.01 if higher_better else gdm < cdm - 0.01) else "=")
                print(f"    {diff:>6}: C={cdm:.1%} G={gdm:.1%} [{w}] (n={len(cd)})")

        return cm, gm, winner

    scores_summary = []

    # D1: Entity Extraction F1
    cm, gm, w = print_dim("D1: ENTITY EXTRACTION (F1 Score)", "entity_f1")
    scores_summary.append(("D1: Entity Extraction F1", cm, gm, w))

    # D1b: Entity Recall
    cm, gm, w = print_dim("D1b: ENTITY EXTRACTION (Recall)", "entity_recall")
    scores_summary.append(("D1b: Entity Recall", cm, gm, w))

    # D2: Relationship Quality
    cm, gm, w = print_dim("D2: RELATIONSHIP QUALITY", "relationship")
    scores_summary.append(("D2: Relationship Quality", cm, gm, w))

    # D3: Contradiction Detection
    cm, gm, w = print_dim("D3: CONTRADICTION DETECTION", "contradiction")
    scores_summary.append(("D3: Contradiction Detection", cm, gm, w))

    # Contradiction breakdown by category
    print(f"\n    Contradiction breakdown by category:")
    for cat in ["direct", "noop_additive", "noop_unrelated", "subtle", "temporal"]:
        c_cat = [d for d in case_details if d.get("phase") == "contradiction" and d.get("category") == cat]
        if c_cat:
            c_correct = sum(1 for d in c_cat if d["claude_correct"])
            g_correct = sum(1 for d in c_cat if d["gemini_correct"])
            n = len(c_cat)
            print(f"      {cat:>20}: Claude {c_correct}/{n} ({c_correct/n:.0%}) | Gemini {g_correct}/{n} ({g_correct/n:.0%})")

    # D4: Tool Reliability
    cm, gm, w = print_dim("D4: TOOL CALL RELIABILITY", "tool_reliability")
    scores_summary.append(("D4: Tool Reliability", cm, gm, w))

    # D5: Schema Adherence
    cm, gm, w = print_dim("D5: JSON SCHEMA ADHERENCE", "schema")
    scores_summary.append(("D5: Schema Adherence", cm, gm, w))

    # D6: Hallucination Resistance
    cm, gm, w = print_dim("D6: HALLUCINATION RESISTANCE", "hallucination")
    scores_summary.append(("D6: Hallucination Resistance", cm, gm, w))

    # D7: Cost & Latency
    c_lat = results["latency"]["claude"]["all"]
    g_lat = results["latency"]["gemini"]["all"]
    c_avg_lat = sum(c_lat) / len(c_lat) if c_lat else 0
    g_avg_lat = sum(g_lat) / len(g_lat) if g_lat else 0

    # Pricing: Claude Opus $15/$75 per 1M in/out, Gemini Flash Lite $0.075/$0.3
    c_cost = (claude.total_input_tokens * 15 + claude.total_output_tokens * 75) / 1_000_000
    g_cost = (gemini.total_input_tokens * 0.075 + gemini.total_output_tokens * 0.3) / 1_000_000

    print(f"\n{'=' * 68}")
    print(f"  D7: COST & LATENCY")
    print(f"{'=' * 68}")
    print(f"  Claude: avg {c_avg_lat:.2f}s | {claude.total_input_tokens:,} in / {claude.total_output_tokens:,} out | ${c_cost:.4f}")
    print(f"  Gemini: avg {g_avg_lat:.2f}s | {gemini.total_input_tokens:,} in / {gemini.total_output_tokens:,} out | ${g_cost:.6f}")
    if g_cost > 0:
        print(f"  Cost ratio: Claude is {c_cost / g_cost:.0f}x more expensive")
    if g_avg_lat > 0:
        print(f"  Speed ratio: Gemini is {c_avg_lat / g_avg_lat:.1f}x faster")
    scores_summary.append(("D7: Cost & Latency", 0, 0, "GEMINI"))

    # ===== GRAND SUMMARY =====
    print(f"\n\n{'#' * 72}")
    print(f"  GRAND SUMMARY")
    print(f"{'#' * 72}")

    claude_wins = sum(1 for _, _, _, w in scores_summary if w == "CLAUDE")
    gemini_wins = sum(1 for _, _, _, w in scores_summary if w == "GEMINI")
    ties = sum(1 for _, _, _, w in scores_summary if w == "TIE")

    print(f"\n  {'Dimension':<35} {'Claude':>8} {'Gemini':>8} {'Winner':>10}")
    print(f"  {'-' * 67}")
    for name, cm, gm, w in scores_summary:
        if name == "D7: Cost & Latency":
            print(f"  {name:<35} {'---':>8} {'---':>8} {w:>10}")
        else:
            print(f"  {name:<35} {cm:>7.1%} {gm:>7.1%} {w:>10}")

    print(f"\n  {'=' * 67}")
    print(f"  Claude wins: {claude_wins} | Gemini wins: {gemini_wins} | Ties: {ties}")
    print(f"  {'=' * 67}")

    # SPLIT-MODEL VERDICT
    # Extraction = D1 entity + D2 relationship, Contradiction = D3
    e_claude = sum(results["entity_f1"]["claude"]["all"]) / len(results["entity_f1"]["claude"]["all"])
    e_gemini = sum(results["entity_f1"]["gemini"]["all"]) / len(results["entity_f1"]["gemini"]["all"])
    r_claude = sum(results["relationship"]["claude"]["all"]) / len(results["relationship"]["claude"]["all"])
    r_gemini = sum(results["relationship"]["gemini"]["all"]) / len(results["relationship"]["gemini"]["all"])
    c_claude = sum(results["contradiction"]["claude"]["all"]) / len(results["contradiction"]["claude"]["all"])
    c_gemini = sum(results["contradiction"]["gemini"]["all"]) / len(results["contradiction"]["gemini"]["all"])

    print(f"\n  SPLIT-MODEL ARCHITECTURE ASSESSMENT:")
    print(f"  {'Pipeline Call':<30} {'Claude':>8} {'Gemini':>8} {'Optimal':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Call 1: Entity Extraction':<30} {e_claude:>7.1%} {e_gemini:>7.1%} {'GEMINI' if e_gemini > e_claude else 'CLAUDE':>10}")
    print(f"  {'Call 2: Relationships':<30} {r_claude:>7.1%} {r_gemini:>7.1%} {'GEMINI' if r_gemini > r_claude else 'CLAUDE':>10}")
    print(f"  {'Call 3: Contradiction':<30} {c_claude:>7.1%} {c_gemini:>7.1%} {'GEMINI' if c_gemini > c_claude else 'CLAUDE':>10}")

    gemini_extraction = e_gemini > e_claude
    claude_contradiction = c_claude > c_gemini
    split_confirmed = gemini_extraction and claude_contradiction

    print(f"\n  VERDICT: Split-model (Gemini extraction + Claude contradiction)")
    print(f"  is {'CONFIRMED' if split_confirmed else 'NOT CONFIRMED'} as optimal architecture.")
    if split_confirmed:
        print(f"  - Gemini wins entity extraction by {e_gemini - e_claude:.1%}")
        if r_claude > r_gemini:
            print(f"  - Claude wins relationships by {r_claude - r_gemini:.1%} (acceptable tradeoff for {c_cost/g_cost:.0f}x cost savings)")
        print(f"  - Claude wins contradiction by {c_claude - c_gemini:.1%} (critical: prevents data loss)")
        print(f"  - Gemini is {c_avg_lat/g_avg_lat:.1f}x faster and {c_cost/g_cost:.0f}x cheaper")

    # Save results
    output_path = Path(__file__).parent / "battle_results_v2.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": {"claude": claude.model, "gemini": gemini.model},
            "total_cases": total_cases,
            "tokens": {
                "claude": {"input": claude.total_input_tokens, "output": claude.total_output_tokens},
                "gemini": {"input": gemini.total_input_tokens, "output": gemini.total_output_tokens},
            },
            "cost": {"claude": round(c_cost, 4), "gemini": round(g_cost, 6)},
            "avg_latency": {"claude": round(c_avg_lat, 3), "gemini": round(g_avg_lat, 3)},
            "dimensions": {name: {"claude": round(cm, 4), "gemini": round(gm, 4), "winner": w} for name, cm, gm, w in scores_summary},
            "case_details": case_details,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
