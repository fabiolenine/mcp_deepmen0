"""End-to-End MCP Configuration Battle Test.

Tests the complete MCP operational surface through the actual Memory object:
fact extraction, semantic search, updates, graph knowledge building,
contradiction handling, and robustness.

Compares two configurations:
  Config A — "Ollama Only" (fully local): qwen3:14b + bge-m3 + ollama graph
  Config B — "Hybrid/Optimal" (cloud + local): claude-opus-4-6 + bge-m3 + gemini_split graph

Both share the same embedder (bge-m3), Qdrant, and Neo4j — isolating LLM quality.

Dimensions:
  D1. Fact Extraction F1
  D2. Search Relevance
  D3. Update Fidelity
  D4. Graph Entity Accuracy
  D5. Graph Relationship Accuracy
  D6. Contradiction Resolution
  D7. Robustness
  D8. End-to-End Latency (P50, P95, mean)
  D9. Cost Efficiency

Usage:
    python3 benchmarks/mcp_e2e_battle.py
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Suppress mem0 telemetry before any mem0 import
os.environ["MEM0_TELEMETRY"] = "false"


# ============================================================
# Test Case Dataclasses
# ============================================================

@dataclass
class FactExtractionCase:
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    messages: list[dict[str, str]]
    expected_facts: list[str]
    hallucination_traps: list[str] = field(default_factory=list)


@dataclass
class SearchCase:
    id: str
    category: str
    difficulty: str
    seed_messages: list[list[dict[str, str]]]  # multiple add() calls
    query: str
    expected_relevant: list[str]  # substrings expected in results
    expected_absent: list[str] = field(default_factory=list)  # should NOT appear


@dataclass
class UpdateCase:
    id: str
    category: str
    difficulty: str
    initial_message: list[dict[str, str]]
    update_text: str
    verify_present: list[str]
    verify_absent: list[str]


@dataclass
class GraphEntityCase:
    id: str
    category: str
    difficulty: str
    messages: list[dict[str, str]]
    expected_entities: list[str]
    expected_relationships: list[tuple[str, str, str]]  # (src, rel, dst)


@dataclass
class ContradictionCase:
    id: str
    category: str
    difficulty: str
    initial_message: list[dict[str, str]]
    contradicting_message: list[dict[str, str]]
    expect_old_removed: bool  # True=old should be gone, False=old should remain
    old_relationship_key: str  # substring to check in graph
    new_relationship_key: str


@dataclass
class ComplexScenarioCase:
    id: str
    category: str
    difficulty: str
    messages_sequence: list[list[dict[str, str]]]
    search_queries: list[str]
    expected_search_hits: list[list[str]]  # per-query expected substrings
    expected_entities: list[str]
    expected_relationships: list[tuple[str, str, str]]


@dataclass
class RobustnessCase:
    id: str
    category: str
    difficulty: str
    messages: list[dict[str, str]]
    expect_success: bool
    min_facts: int  # minimum number of facts that should be extracted


# ============================================================
# Phase 1: Fact Extraction Cases (20)
# ============================================================

FACT_EXTRACTION_CASES: list[FactExtractionCase] = [
    # --- Personal (5) ---
    FactExtractionCase("FE01", "personal", "easy",
        [{"role": "user", "content": "My name is Alice and I live in San Francisco."}],
        ["alice", "san francisco"],
        ["new york", "bob"]),
    FactExtractionCase("FE02", "personal", "easy",
        [{"role": "user", "content": "I'm a vegetarian and I'm allergic to peanuts."}],
        ["vegetarian", "allergic", "peanuts"],
        ["vegan", "shellfish"]),
    FactExtractionCase("FE03", "personal", "easy",
        [{"role": "user", "content": "My birthday is March 15th and I'm 32 years old."}],
        ["march 15", "32"],
        ["april", "33"]),
    FactExtractionCase("FE04", "personal", "medium",
        [{"role": "user", "content": "I prefer dark mode, use Vim keybindings, and my terminal theme is Dracula."}],
        ["dark mode", "vim", "dracula"],
        ["light mode", "emacs"]),
    FactExtractionCase("FE05", "personal", "medium",
        [{"role": "user", "content": "I have two cats named Luna and Mochi, both are 3 years old."}],
        ["luna", "mochi", "cats"],
        ["dog", "fish"]),

    # --- Multi-fact (5) ---
    FactExtractionCase("FE06", "multi-fact", "medium",
        [{"role": "user", "content": "I work at Anthropic as a senior engineer. I use Python and TypeScript daily. My team uses FastAPI for the backend."}],
        ["anthropic", "senior engineer", "python", "typescript", "fastapi"],
        ["google", "java"]),
    FactExtractionCase("FE07", "multi-fact", "medium",
        [{"role": "user", "content": "I graduated from MIT in 2018 with a CS degree, then worked at Google for 3 years before joining a startup."}],
        ["mit", "2018", "cs", "google", "startup"],
        ["stanford", "2020"]),
    FactExtractionCase("FE08", "multi-fact", "hard",
        [{"role": "user", "content": "Our tech stack is React 19 frontend, FastAPI backend, PostgreSQL database, Redis cache, and we deploy to AWS EKS with Terraform."}],
        ["react", "fastapi", "postgresql", "redis", "aws", "eks", "terraform"],
        ["vue", "django", "mongodb"]),
    FactExtractionCase("FE09", "multi-fact", "hard",
        [{"role": "user", "content": "I speak English natively, French fluently, and I'm learning Japanese. I lived in Paris for 2 years and Tokyo for 6 months."}],
        ["english", "french", "japanese", "paris", "tokyo"],
        ["spanish", "london"]),
    FactExtractionCase("FE10", "multi-fact", "hard",
        [{"role": "user", "content": "The CI pipeline runs pytest for unit tests, mypy for type checking, ruff for linting, and takes about 12 minutes on GitHub Actions."}],
        ["pytest", "mypy", "ruff", "12 minutes", "github actions"],
        ["jenkins", "eslint"]),

    # --- Technical (3) ---
    FactExtractionCase("FE11", "technical", "medium",
        [{"role": "user", "content": "The API rate limit is 1000 requests per minute with a burst of 50."}],
        ["1000", "rate limit", "burst", "50"],
        ["500"]),
    FactExtractionCase("FE12", "technical", "hard",
        [{"role": "user", "content": "We use OIDC with Auth0 for SSO, JWT tokens with RS256, and refresh tokens expire after 30 days."}],
        ["oidc", "auth0", "sso", "jwt", "rs256", "30 days"],
        ["oauth1", "hs256"]),
    FactExtractionCase("FE13", "technical", "hard",
        [{"role": "user", "content": "The model serves at P99 latency of 180ms, throughput of 500 QPS on 4x A100 GPUs."}],
        ["180ms", "500 qps", "a100"],
        ["h100"]),

    # --- Temporal (3) ---
    FactExtractionCase("FE14", "temporal", "medium",
        [{"role": "user", "content": "I started learning Rust in January 2024 and passed my AWS Solutions Architect exam in March."}],
        ["rust", "january 2024", "aws solutions architect", "march"],
        ["go", "gcp"]),
    FactExtractionCase("FE15", "temporal", "hard",
        [{"role": "user", "content": "The project launched in Q1 2024, hit 10k users by Q2, and we're targeting 100k by EOY 2025."}],
        ["q1 2024", "10k", "q2", "100k", "2025"],
        ["2023"]),
    FactExtractionCase("FE16", "temporal", "hard",
        [{"role": "user", "content": "Version 1.0 shipped in 2022, v2.0 in 2023 with breaking changes, and v3.0 is planned for summer 2025."}],
        ["1.0", "2022", "2.0", "2023", "3.0", "2025"],
        ["4.0"]),

    # --- Ambiguous (2) ---
    FactExtractionCase("FE17", "ambiguous", "hard",
        [{"role": "user", "content": "I love Python but find the GIL frustrating. I use Java for Android development."}],
        ["python", "gil", "java", "android"],
        ["kotlin", "swift"]),
    FactExtractionCase("FE18", "ambiguous", "hard",
        [{"role": "user", "content": "Spring is my favorite framework but fall is my favorite season."}],
        ["spring", "framework", "fall", "season"],
        ["winter"]),

    # --- Jargon (2) ---
    FactExtractionCase("FE19", "jargon", "hard",
        [{"role": "user", "content": "We run a k8s cluster with Istio service mesh, Kiali dashboard, and Jaeger for distributed tracing."}],
        ["k8s", "istio", "kiali", "jaeger"],
        ["envoy"]),
    FactExtractionCase("FE20", "jargon", "hard",
        [{"role": "user", "content": "The LLM pipeline uses RAG with pgvector for hybrid search, BM25 + dense retrieval, and HyDE for query expansion."}],
        ["rag", "pgvector", "bm25", "hyde"],
        ["elasticsearch"]),
]


# ============================================================
# Phase 2: Semantic Search Cases (15)
# ============================================================

SEARCH_CASES: list[SearchCase] = [
    # --- Exact match (4) ---
    SearchCase("SR01", "exact", "easy",
        [[{"role": "user", "content": "Alice works at Anthropic as a software engineer."}]],
        "Where does Alice work?",
        ["anthropic", "software engineer"]),
    SearchCase("SR02", "exact", "easy",
        [[{"role": "user", "content": "Bob's favorite programming language is Rust."}]],
        "What is Bob's favorite language?",
        ["rust"]),
    SearchCase("SR03", "exact", "easy",
        [[{"role": "user", "content": "The database server runs PostgreSQL 16 on Ubuntu 24.04."}]],
        "What database does the server use?",
        ["postgresql"]),
    SearchCase("SR04", "exact", "medium",
        [[{"role": "user", "content": "The project deadline is March 31, 2026."}]],
        "When is the project deadline?",
        ["march", "2026"]),

    # --- Paraphrase (4) ---
    SearchCase("SR05", "paraphrase", "medium",
        [[{"role": "user", "content": "Charlie is allergic to shellfish and cannot eat seafood."}]],
        "What food restrictions does Charlie have?",
        ["shellfish", "seafood"]),
    SearchCase("SR06", "paraphrase", "medium",
        [[{"role": "user", "content": "The team uses Kubernetes deployed on AWS EKS."}]],
        "How does the team handle container orchestration?",
        ["kubernetes"]),
    SearchCase("SR07", "paraphrase", "hard",
        [[{"role": "user", "content": "Our authentication uses JWT tokens with RS256 signing."}]],
        "How do we verify user identity?",
        ["jwt"]),
    SearchCase("SR08", "paraphrase", "hard",
        [[{"role": "user", "content": "We migrated from monolith to microservices architecture last year."}]],
        "What was the recent architectural change?",
        ["microservices"]),

    # --- Related (3) ---
    SearchCase("SR09", "related", "medium",
        [[{"role": "user", "content": "Our backend is built with FastAPI and uses SQLAlchemy ORM."}],
         [{"role": "user", "content": "The frontend uses React 19 with TypeScript."}]],
        "What technologies does the project use?",
        ["fastapi"]),
    SearchCase("SR10", "related", "hard",
        [[{"role": "user", "content": "Emily manages the Berlin engineering office."}],
         [{"role": "user", "content": "Emily reports to the VP of Engineering, David."}]],
        "Who works in the Berlin office?",
        ["emily"]),
    SearchCase("SR11", "related", "hard",
        [[{"role": "user", "content": "The ML pipeline trains models on A100 GPUs."}],
         [{"role": "user", "content": "Model inference runs on T4 GPUs for cost efficiency."}]],
        "What hardware does the ML system use?",
        ["a100"]),

    # --- Unrelated/Negative (4) ---
    SearchCase("SR12", "unrelated", "medium",
        [[{"role": "user", "content": "The team uses PostgreSQL for the main database."}]],
        "What is the company's vacation policy?",
        [],
        ["postgresql"]),
    SearchCase("SR13", "unrelated", "medium",
        [[{"role": "user", "content": "Alice prefers VS Code with the Dracula theme."}]],
        "What is the weather forecast for tomorrow?",
        [],
        ["vs code"]),
    SearchCase("SR14", "negative", "hard",
        [[{"role": "user", "content": "We use Python 3.12 for backend development."}]],
        "Do we use Java for any projects?",
        [],
        ["java"]),
    SearchCase("SR15", "negative", "hard",
        [[{"role": "user", "content": "The company headquarters is in San Francisco."}]],
        "Does the company have an office in Tokyo?",
        [],
        ["tokyo"]),
]


# ============================================================
# Phase 3: Memory Update Cases (8)
# ============================================================

UPDATE_CASES: list[UpdateCase] = [
    UpdateCase("UP01", "simple", "easy",
        [{"role": "user", "content": "Bob works at Google."}],
        "Bob now works at Anthropic.",
        ["anthropic"], ["google"]),
    UpdateCase("UP02", "simple", "easy",
        [{"role": "user", "content": "The project uses Python 3.11."}],
        "The project has been upgraded to Python 3.13.",
        ["3.13"], ["3.11"]),
    UpdateCase("UP03", "simple", "medium",
        [{"role": "user", "content": "Alice's favorite color is blue."}],
        "Alice's favorite color changed to green.",
        ["green"], ["blue"]),
    UpdateCase("UP04", "role-change", "medium",
        [{"role": "user", "content": "David is a senior engineer on the backend team."}],
        "David was promoted to engineering manager and now leads the platform team.",
        ["manager", "platform"], ["senior engineer"]),
    UpdateCase("UP05", "technical", "medium",
        [{"role": "user", "content": "The API rate limit is 500 requests per minute."}],
        "The API rate limit was increased to 2000 requests per minute.",
        ["2000"], ["500"]),
    UpdateCase("UP06", "technical", "hard",
        [{"role": "user", "content": "We deploy using rolling updates with 25% max unavailable."}],
        "We switched to blue-green deployments for zero-downtime releases.",
        ["blue-green", "zero-downtime"], ["rolling"]),
    UpdateCase("UP07", "location", "medium",
        [{"role": "user", "content": "The company headquarters is in Seattle."}],
        "The company relocated headquarters to Austin, Texas.",
        ["austin"], ["seattle"]),
    UpdateCase("UP08", "complex", "hard",
        [{"role": "user", "content": "The database is MySQL 5.7 with 500GB storage on a single node."}],
        "We migrated to PostgreSQL 16 with 2TB storage in a 3-node cluster.",
        ["postgresql", "2tb", "cluster"], ["mysql", "500gb", "single"]),
]


# ============================================================
# Phase 4: Graph Entity/Relationship Extraction Cases (15)
# ============================================================

GRAPH_ENTITY_CASES: list[GraphEntityCase] = [
    # --- Single entity (5) ---
    GraphEntityCase("GE01", "single-entity", "easy",
        [{"role": "user", "content": "Alice works at Anthropic."}],
        ["alice", "anthropic"],
        [("alice", "works_at", "anthropic")]),
    GraphEntityCase("GE02", "single-entity", "easy",
        [{"role": "user", "content": "Bob lives in Tokyo, Japan."}],
        ["bob", "tokyo"],
        [("bob", "lives_in", "tokyo")]),
    GraphEntityCase("GE03", "single-entity", "easy",
        [{"role": "user", "content": "Charlie speaks French fluently."}],
        ["charlie", "french"],
        [("charlie", "speaks", "french")]),
    GraphEntityCase("GE04", "single-entity", "medium",
        [{"role": "user", "content": "Diana graduated from MIT with a PhD in Computer Science."}],
        ["diana", "mit"],
        [("diana", "graduated_from", "mit")]),
    GraphEntityCase("GE05", "single-entity", "medium",
        [{"role": "user", "content": "Eve drives a Tesla Model 3."}],
        ["eve", "tesla"],
        [("eve", "drives", "tesla")]),

    # --- Multi-entity (5) ---
    GraphEntityCase("GE06", "multi-entity", "medium",
        [{"role": "user", "content": "Frank works at Google and his wife Grace works at Meta."}],
        ["frank", "google", "grace", "meta"],
        [("frank", "works_at", "google"), ("grace", "works_at", "meta")]),
    GraphEntityCase("GE07", "multi-entity", "medium",
        [{"role": "user", "content": "The startup was co-founded by Hank and Iris in San Francisco."}],
        ["hank", "iris", "san francisco"],
        [("hank", "co_founded", "startup"), ("iris", "co_founded", "startup")]),
    GraphEntityCase("GE08", "multi-entity", "hard",
        [{"role": "user", "content": "Jack manages the Berlin office and reports to Regional Director Karen in Munich."}],
        ["jack", "berlin", "karen", "munich"],
        [("jack", "manages", "berlin"), ("jack", "reports_to", "karen")]),
    GraphEntityCase("GE09", "multi-entity", "hard",
        [{"role": "user", "content": "Leo, a Python developer at Stripe, mentors Mia who is a junior engineer."}],
        ["leo", "stripe", "mia"],
        [("leo", "works_at", "stripe"), ("leo", "mentors", "mia")]),
    GraphEntityCase("GE10", "multi-entity", "hard",
        [{"role": "user", "content": "Nina from the ML team built a recommendation engine using TensorFlow and deployed it on AWS SageMaker."}],
        ["nina", "tensorflow", "sagemaker"],
        [("nina", "built", "recommendation engine")]),

    # --- Complex (5) ---
    GraphEntityCase("GE11", "complex", "hard",
        [{"role": "user", "content": "Dr. Patel at Stanford collaborated with Prof. Kim from Seoul National University on a quantum computing paper."}],
        ["patel", "stanford", "kim", "seoul national"],
        [("patel", "affiliated", "stanford"), ("kim", "affiliated", "seoul national")]),
    GraphEntityCase("GE12", "complex", "hard",
        [{"role": "user", "content": "The backend uses FastAPI with PostgreSQL, and the frontend uses React with TypeScript, all deployed on AWS EKS."}],
        ["fastapi", "postgresql", "react", "typescript", "aws"],
        [("fastapi", "uses", "postgresql")]),
    GraphEntityCase("GE13", "complex", "hard",
        [{"role": "user", "content": "Anthropic, founded by Dario and Daniela Amodei, released Claude which competes with OpenAI's GPT-4."}],
        ["anthropic", "dario", "daniela", "claude", "openai", "gpt-4"],
        [("dario", "founded", "anthropic"), ("anthropic", "released", "claude")]),
    GraphEntityCase("GE14", "complex", "hard",
        [{"role": "user", "content": "The CI/CD pipeline uses GitHub Actions to build Docker images, push to ECR, and deploy to EKS using Helm charts."}],
        ["github actions", "docker", "ecr", "eks", "helm"],
        [("github actions", "builds", "docker")]),
    GraphEntityCase("GE15", "complex", "hard",
        [{"role": "user", "content": "The data pipeline orchestrated by Apache Airflow reads from Kafka, processes with Spark, and writes to BigQuery."}],
        ["airflow", "kafka", "spark", "bigquery"],
        [("airflow", "orchestrates", "spark"), ("spark", "reads_from", "kafka")]),
]


# ============================================================
# Phase 5: Contradiction Handling Cases (10)
# ============================================================

CONTRADICTION_CASES: list[ContradictionCase] = [
    # --- Direct contradiction (4) ---
    ContradictionCase("CT01", "direct", "easy",
        [{"role": "user", "content": "Alice lives in New York."}],
        [{"role": "user", "content": "Alice just moved to San Francisco."}],
        True, "new york", "san francisco"),
    ContradictionCase("CT02", "direct", "easy",
        [{"role": "user", "content": "Bob works at Google."}],
        [{"role": "user", "content": "Bob left Google and joined Anthropic."}],
        True, "google", "anthropic"),
    ContradictionCase("CT03", "direct", "medium",
        [{"role": "user", "content": "The company uses MySQL as its primary database."}],
        [{"role": "user", "content": "The company completed migration from MySQL to PostgreSQL."}],
        True, "mysql", "postgresql"),
    ContradictionCase("CT04", "direct", "hard",
        [{"role": "user", "content": "David is a senior engineer on the backend team."}],
        [{"role": "user", "content": "David was promoted to Engineering Manager and now leads the platform team."}],
        True, "senior engineer", "manager"),

    # --- Noop/additive (3) ---
    ContradictionCase("CT05", "noop-additive", "easy",
        [{"role": "user", "content": "Alice loves eating pizza."}],
        [{"role": "user", "content": "Alice also loves eating sushi."}],
        False, "pizza", "sushi"),
    ContradictionCase("CT06", "noop-additive", "medium",
        [{"role": "user", "content": "Bob knows Python and JavaScript."}],
        [{"role": "user", "content": "Bob also learned Rust recently."}],
        False, "python", "rust"),
    ContradictionCase("CT07", "noop-additive", "hard",
        [{"role": "user", "content": "The company has offices in New York and London."}],
        [{"role": "user", "content": "The company opened a new office in Tokyo."}],
        False, "new york", "tokyo"),

    # --- Temporal (3) ---
    ContradictionCase("CT08", "temporal", "medium",
        [{"role": "user", "content": "The recommended Python version is 3.11."}],
        [{"role": "user", "content": "Python 3.13 is now the recommended version."}],
        True, "python", "python"),
    ContradictionCase("CT09", "temporal", "hard",
        [{"role": "user", "content": "The server instance type is t3.medium."}],
        [{"role": "user", "content": "We scaled up the server to t3.xlarge due to increased traffic."}],
        True, "server", "server"),
    ContradictionCase("CT10", "temporal", "hard",
        [{"role": "user", "content": "The team has 5 members."}],
        [{"role": "user", "content": "After recent hiring, the team now has 8 members."}],
        True, "team", "team"),
]


# ============================================================
# Phase 6: Complex Multi-Memory Scenarios (5)
# ============================================================

COMPLEX_CASES: list[ComplexScenarioCase] = [
    ComplexScenarioCase("CX01", "knowledge-synthesis", "hard",
        [
            [{"role": "user", "content": "Alice works at Anthropic as a senior ML engineer."}],
            [{"role": "user", "content": "Alice graduated from Stanford with a PhD in NLP."}],
            [{"role": "user", "content": "Alice's team is building a new LLM evaluation framework."}],
            [{"role": "user", "content": "Alice previously worked at Google Brain for 3 years."}],
            [{"role": "user", "content": "Alice lives in San Francisco and commutes to the Anthropic office."}],
        ],
        ["Tell me about Alice's career", "What is Alice's educational background?"],
        [["anthropic", "ml engineer"], ["stanford", "phd"]],
        ["alice", "anthropic", "stanford"],
        [("alice", "works_at", "anthropic")]),
    ComplexScenarioCase("CX02", "tech-stack", "hard",
        [
            [{"role": "user", "content": "Our backend API is built with FastAPI and Python 3.12."}],
            [{"role": "user", "content": "The frontend uses React 19 with TypeScript and Tailwind CSS."}],
            [{"role": "user", "content": "We use PostgreSQL 16 as our primary database with Redis for caching."}],
            [{"role": "user", "content": "CI/CD is handled by GitHub Actions, deploying to AWS EKS."}],
            [{"role": "user", "content": "Monitoring uses Prometheus, Grafana, and Loki for the observability stack."}],
        ],
        ["What is our tech stack?", "How do we handle monitoring?"],
        [["fastapi", "react"], ["prometheus", "grafana"]],
        ["fastapi", "react", "postgresql", "redis"],
        []),
    ComplexScenarioCase("CX03", "team-structure", "hard",
        [
            [{"role": "user", "content": "Maria is the CTO and oversees all engineering teams."}],
            [{"role": "user", "content": "Carlos leads the backend team of 6 engineers."}],
            [{"role": "user", "content": "Yuki leads the frontend team with 4 developers."}],
            [{"role": "user", "content": "Carlos and Yuki both report to Maria."}],
            [{"role": "user", "content": "The company has a total engineering team of 15 people."}],
        ],
        ["Who reports to Maria?", "Tell me about the engineering team structure"],
        [["carlos", "yuki"], ["maria", "cto"]],
        ["maria", "carlos", "yuki"],
        [("carlos", "reports_to", "maria"), ("yuki", "reports_to", "maria")]),
    ComplexScenarioCase("CX04", "project-history", "hard",
        [
            [{"role": "user", "content": "Project Phoenix started in January 2024."}],
            [{"role": "user", "content": "The alpha release was in March 2024 with 100 beta testers."}],
            [{"role": "user", "content": "Version 1.0 launched in September 2024 to 5,000 users."}],
            [{"role": "user", "content": "By January 2025, Project Phoenix had 50,000 active users."}],
            [{"role": "user", "content": "The team is now working on version 2.0 with AI features."}],
        ],
        ["What is the history of Project Phoenix?", "How many users does Phoenix have?"],
        [["january 2024", "phoenix"], ["50,000"]],
        ["phoenix"],
        []),
    ComplexScenarioCase("CX05", "cross-reference", "hard",
        [
            [{"role": "user", "content": "Our ML models are trained on A100 GPUs in the us-east-1 region."}],
            [{"role": "user", "content": "The training pipeline uses PyTorch 2.3 with DeepSpeed for distributed training."}],
            [{"role": "user", "content": "Model serving uses TorchServe behind an NGINX load balancer."}],
            [{"role": "user", "content": "The P99 latency for inference is 45ms at 1000 QPS."}],
            [{"role": "user", "content": "We retrain models weekly with new data from the data lake."}],
        ],
        ["What ML infrastructure do we use?", "What is our model serving setup?"],
        [["a100", "pytorch"], ["torchserve", "nginx"]],
        ["pytorch", "torchserve", "a100"],
        []),
]


# ============================================================
# Phase 7: Robustness Cases (8)
# ============================================================

ROBUSTNESS_CASES: list[RobustnessCase] = [
    RobustnessCase("RB01", "empty", "easy",
        [{"role": "user", "content": ""}],
        True, 0),
    RobustnessCase("RB02", "japanese", "medium",
        [{"role": "user", "content": "東京大学の田中教授がAI研究について発表しました。"}],
        True, 1),
    RobustnessCase("RB03", "spanish", "medium",
        [{"role": "user", "content": "El restaurante de Maria en Barcelona sirve la mejor paella."}],
        True, 1),
    RobustnessCase("RB04", "emoji", "medium",
        [{"role": "user", "content": "User123 loves Python and hates bugs."}],
        True, 1),
    RobustnessCase("RB05", "long-text", "hard",
        [{"role": "user", "content": "The system architecture involves " + ", ".join(f"component_{i}" for i in range(50)) + " all interconnected via a message bus."}],
        True, 1),
    RobustnessCase("RB06", "sql-injection", "hard",
        [{"role": "user", "content": "Robert'; DROP TABLE users;-- works at the security department."}],
        True, 1),
    RobustnessCase("RB07", "many-entities", "hard",
        [{"role": "user", "content": "The conference featured talks by Dr. Smith, Prof. Jones, Dr. Lee, Prof. Patel, Dr. Kim, Prof. Garcia, Dr. Chen, Prof. Mueller, Dr. Tanaka, and Prof. Rossi."}],
        True, 3),
    RobustnessCase("RB08", "repeated-chars", "hard",
        [{"role": "user", "content": "Aaaaaa Bbbbbb works at Cccccc Dddddd company in Eeeeee Ffffff city."}],
        True, 0),
]


# ============================================================
# Scoring Functions
# ============================================================

def _norm(s: str) -> str:
    """Normalize text for comparison."""
    return s.lower().strip().replace("_", " ").replace("-", " ")


def _fuzzy_match(needle: str, haystack: str) -> bool:
    """Check if normalized needle is a substring of normalized haystack."""
    return _norm(needle) in _norm(haystack)


def _fuzzy_match_any(needle: str, texts: list[str]) -> bool:
    """Check if normalized needle is found in any of the texts."""
    nn = _norm(needle)
    return any(nn in _norm(t) for t in texts)


def score_fact_extraction(memories: list[dict], case: FactExtractionCase) -> dict:
    """Score fact extraction: F1 of expected facts + hallucination penalty."""
    all_text = " ".join(m.get("memory", "") for m in memories).lower()
    memory_texts = [m.get("memory", "") for m in memories]

    found = sum(1 for f in case.expected_facts if _fuzzy_match(f, all_text))
    total_expected = len(case.expected_facts)

    hallucinated = sum(1 for t in case.hallucination_traps if _fuzzy_match(t, all_text))

    recall = found / total_expected if total_expected > 0 else 1.0
    # Precision penalized by hallucinations
    total_extracted = len(memories)
    precision = found / (found + hallucinated) if (found + hallucinated) > 0 else (1.0 if total_expected == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "found": found,
        "expected": total_expected,
        "hallucinated": hallucinated,
        "extracted_count": total_extracted,
    }


def score_search(results: list[dict], case: SearchCase) -> dict:
    """Score search quality: recall of expected results + false positive penalty."""
    all_text = " ".join(r.get("memory", "") for r in results).lower()

    found_relevant = sum(1 for e in case.expected_relevant if _fuzzy_match(e, all_text))
    total_relevant = len(case.expected_relevant)

    false_positives = sum(1 for a in case.expected_absent if _fuzzy_match(a, all_text))

    recall = found_relevant / total_relevant if total_relevant > 0 else 1.0
    fp_penalty = false_positives * 0.2  # Each FP costs 20%
    score = max(0.0, recall - fp_penalty)

    return {
        "score": round(score, 4),
        "recall": round(recall, 4),
        "found": found_relevant,
        "expected": total_relevant,
        "false_positives": false_positives,
    }


def score_update(search_results: list[dict], case: UpdateCase) -> dict:
    """Score update fidelity: new content present + old content absent."""
    all_text = " ".join(r.get("memory", "") for r in search_results).lower()

    present_found = sum(1 for p in case.verify_present if _fuzzy_match(p, all_text))
    absent_found = sum(1 for a in case.verify_absent if _fuzzy_match(a, all_text))

    presence_score = present_found / len(case.verify_present) if case.verify_present else 1.0
    absence_score = 1.0 - (absent_found / len(case.verify_absent)) if case.verify_absent else 1.0

    score = (presence_score + absence_score) / 2.0
    return {
        "score": round(score, 4),
        "presence": round(presence_score, 4),
        "absence": round(absence_score, 4),
        "present_found": present_found,
        "absent_found": absent_found,
    }


def score_graph_entities(found_entities: list[str], expected: list[str]) -> dict:
    """Score graph entity extraction: F1."""
    found = 0
    for exp in expected:
        if _fuzzy_match_any(exp, found_entities):
            found += 1
    total = len(expected)
    recall = found / total if total > 0 else 1.0
    precision = found / len(found_entities) if found_entities else (1.0 if not expected else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": round(f1, 4), "recall": round(recall, 4), "found": found, "expected": total}


def score_graph_relationships(found_rels: list[tuple[str, str, str]], expected: list[tuple[str, str, str]]) -> dict:
    """Score graph relationship extraction: fraction of expected triples found."""
    found = 0
    for es, er, ed in expected:
        for fs, fr, fd in found_rels:
            if (_fuzzy_match(es, fs) and _fuzzy_match(ed, fd)):
                found += 1
                break
    total = len(expected)
    score = found / total if total > 0 else 1.0
    return {"score": round(score, 4), "found": found, "expected": total}


def score_contradiction(graph_text: str, case: ContradictionCase) -> dict:
    """Score contradiction handling based on graph state after both adds."""
    old_present = _fuzzy_match(case.old_relationship_key, graph_text)
    new_present = _fuzzy_match(case.new_relationship_key, graph_text)

    if case.expect_old_removed:
        # Direct/temporal contradiction: old should be gone, new should be present
        correct = not old_present and new_present
        score = (0.5 if not old_present else 0.0) + (0.5 if new_present else 0.0)
    else:
        # Additive: both old and new should be present
        correct = old_present and new_present
        score = (0.5 if old_present else 0.0) + (0.5 if new_present else 0.0)

    return {
        "correct": correct,
        "score": round(score, 4),
        "old_present": old_present,
        "new_present": new_present,
        "expect_old_removed": case.expect_old_removed,
    }


def score_robustness(success: bool, memories: list[dict], case: RobustnessCase) -> dict:
    """Score robustness: success + minimum facts threshold."""
    num_facts = len(memories)
    meets_threshold = num_facts >= case.min_facts
    score = (0.5 if success else 0.0) + (0.5 if meets_threshold else 0.0)
    return {
        "score": round(score, 4),
        "success": success,
        "num_facts": num_facts,
        "min_required": case.min_facts,
        "meets_threshold": meets_threshold,
    }


def ci95(scores: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for 95% CI."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    se = math.sqrt(var / n)
    margin = 1.96 * se
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def bar(val: float, width: int = 30) -> str:
    """ASCII progress bar."""
    filled = int(val * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def percentile(values: list[float], pct: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ============================================================
# Infrastructure Health Checks
# ============================================================

def check_qdrant() -> str | None:
    url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")
    try:
        urllib.request.urlopen(f"{url}/healthz", timeout=5)
        return url
    except Exception as e:
        return None


def check_ollama() -> str | None:
    url = os.environ.get("MEM0_EMBED_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        return url
    except Exception as e:
        return None


def check_neo4j() -> bool:
    url = os.environ.get("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
    stripped = url.replace("bolt://", "").replace("neo4j://", "")
    if ":" in stripped:
        host, port_str = stripped.rsplit(":", 1)
        port = int(port_str)
    else:
        host = stripped
        port = 7687
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((host, port))
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
    finally:
        sock.close()


def check_cloud_creds() -> tuple[bool, bool]:
    """Check for cloud credentials. Returns (has_google, has_anthropic).

    For long benchmarks, prefers ANTHROPIC_API_KEY over OAT tokens to avoid
    token expiry during multi-hour runs. If ANTHROPIC_API_KEY is available,
    it is set as MEM0_ANTHROPIC_TOKEN so build_config() uses it.
    """
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))

    # Prefer stable API key over OAT for benchmarks
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        os.environ["MEM0_ANTHROPIC_TOKEN"] = api_key
        has_anthropic = True
        print("  INFO: Using ANTHROPIC_API_KEY (stable, no expiry risk)")
    else:
        try:
            from mem0_mcp_selfhosted.auth import resolve_token
            token = resolve_token()
            has_anthropic = bool(token)
            if has_anthropic:
                from mem0_mcp_selfhosted.auth import is_oat_token
                if is_oat_token(token):
                    print("  WARN: Using OAT token — may expire during long runs (>1hr)")
                    print("        Set ANTHROPIC_API_KEY for stable auth in benchmarks")
        except Exception:
            has_anthropic = False
    return has_google, has_anthropic


# ============================================================
# Memory Instance Factory
# ============================================================

def _make_memory(
    config_name: str,
    qdrant_url: str,
    collection: str,
    neo4j_base_label: str,
    llm_provider: str,
    llm_model: str,
    graph_llm_provider: str,
    enable_graph: bool = True,
) -> tuple[Any, Callable[[], None]]:
    """Create a Memory instance with specific configuration.

    Follows the pattern from tests/integration/test_graph.py:_make_gemini_memory().
    Returns (memory, cleanup_fn).
    """
    # Save original env vars
    originals = {}
    keys_to_override = [
        "MEM0_LLM_PROVIDER", "MEM0_LLM_MODEL",
        "MEM0_EMBED_PROVIDER", "MEM0_EMBED_MODEL",
        "MEM0_ENABLE_GRAPH", "MEM0_GRAPH_LLM_PROVIDER",
        "MEM0_COLLECTION", "MEM0_NEO4J_BASE_LABEL",
    ]
    for k in keys_to_override:
        originals[k] = os.environ.get(k)

    # Set env vars for this config
    os.environ["MEM0_LLM_PROVIDER"] = llm_provider
    os.environ["MEM0_LLM_MODEL"] = llm_model
    os.environ["MEM0_EMBED_PROVIDER"] = "ollama"
    os.environ["MEM0_EMBED_MODEL"] = "bge-m3"
    os.environ["MEM0_ENABLE_GRAPH"] = "true" if enable_graph else "false"
    os.environ["MEM0_GRAPH_LLM_PROVIDER"] = graph_llm_provider
    os.environ["MEM0_COLLECTION"] = collection
    os.environ["MEM0_NEO4J_BASE_LABEL"] = neo4j_base_label

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers
    from mem0_mcp_selfhosted.helpers import patch_graph_sanitizer, patch_gemini_parse_response

    config_dict, providers_info, split_config = build_config()
    register_providers(providers_info)
    patch_graph_sanitizer()
    patch_gemini_parse_response()

    from mem0 import Memory
    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig
        memory.graph.llm = SplitModelGraphLLM(SplitModelGraphLLMConfig(**split_config))

    def cleanup():
        # Drop Qdrant collection
        try:
            from qdrant_client import QdrantClient
            QdrantClient(url=qdrant_url).delete_collection(collection)
        except Exception:
            pass
        # Clean Neo4j nodes by base label
        try:
            from mem0_mcp_selfhosted.graph_tools import _run_query
            _run_query(f"MATCH (n:{neo4j_base_label}) DETACH DELETE n")
        except Exception:
            pass
        # Restore env
        for k, v in originals.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    return memory, cleanup


# ============================================================
# Graph Inspection Helpers
# ============================================================
#
# mem0ai stores graph entities with entity-type labels (person, company, etc.)
# NOT with the configured base_label. Queries must search by entity name
# across all labels. Since test cases use unique entity names per case,
# cross-contamination between configs is minimal.

def query_graph_entities_by_names(expected_names: list[str]) -> list[str]:
    """Search Neo4j for specific entity names (case-insensitive).

    Handles mem0ai's underscore normalization: "New York" → "new_york".
    Searches both the original name and underscore/space variants.
    """
    if not expected_names:
        return []
    try:
        from mem0_mcp_selfhosted.graph_tools import _run_query
        # Build expanded name list with underscore/space variants
        expanded = []
        for name in expected_names:
            expanded.append(name)
            if " " in name:
                expanded.append(name.replace(" ", "_"))
            if "_" in name:
                expanded.append(name.replace("_", " "))
        cypher = """
        MATCH (n)
        WHERE any(name IN $names WHERE toLower(n.name) CONTAINS toLower(name))
           OR any(name IN $names WHERE toLower(replace(n.name, '_', ' ')) CONTAINS toLower(name))
        RETURN DISTINCT n.name AS entity
        LIMIT 100
        """
        records = _run_query(cypher, {"names": expanded})
        return [r["entity"] for r in records if r.get("entity")]
    except Exception:
        return []


def query_graph_relationships_for_entities(entity_names: list[str]) -> list[tuple[str, str, str]]:
    """Get (src, rel_type, dst) triples involving any of the given entity names.

    Handles mem0ai's underscore normalization: "New York" → "new_york".
    """
    if not entity_names:
        return []
    try:
        from mem0_mcp_selfhosted.graph_tools import _run_query
        # Build expanded name list with underscore/space variants
        expanded = []
        for name in entity_names:
            expanded.append(name)
            if " " in name:
                expanded.append(name.replace(" ", "_"))
            if "_" in name:
                expanded.append(name.replace("_", " "))
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE any(name IN $names WHERE toLower(n.name) CONTAINS toLower(name))
           OR any(name IN $names WHERE toLower(m.name) CONTAINS toLower(name))
           OR any(name IN $names WHERE toLower(replace(n.name, '_', ' ')) CONTAINS toLower(name))
           OR any(name IN $names WHERE toLower(replace(m.name, '_', ' ')) CONTAINS toLower(name))
        RETURN DISTINCT n.name AS source, type(r) AS rel_type, m.name AS target
        LIMIT 200
        """
        records = _run_query(cypher, {"names": expanded})
        return [(r["source"], r["rel_type"], r["target"]) for r in records
                if r.get("source") and r.get("rel_type") and r.get("target")]
    except Exception:
        return []


def query_graph_text_for_entities(entity_names: list[str]) -> str:
    """Dump graph content involving specific entities as text for substring matching."""
    entities = query_graph_entities_by_names(entity_names)
    rels = query_graph_relationships_for_entities(entity_names)
    parts = []
    for e in entities:
        parts.append(e)
    for s, r, d in rels:
        parts.append(f"{s} {r} {d}")
    return " ".join(parts).lower()


# ============================================================
# Operation Timer
# ============================================================

class OperationTimer:
    """Track latencies by operation type."""

    def __init__(self):
        self.latencies: dict[str, list[float]] = {}

    def record(self, op_type: str, duration: float):
        self.latencies.setdefault(op_type, []).append(duration)

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for op, vals in self.latencies.items():
            result[op] = {
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 3) if vals else 0,
                "p50": round(percentile(vals, 50), 3),
                "p95": round(percentile(vals, 95), 3),
                "total": round(sum(vals), 3),
            }
        return result


# ============================================================
# Main Battle Runner
# ============================================================

def main():
    from dotenv import load_dotenv
    load_dotenv()

    # Suppress noisy Neo4j driver warnings about non-existent labels
    import logging as _logging
    _logging.getLogger("neo4j").setLevel(_logging.ERROR)

    total_cases = (
        len(FACT_EXTRACTION_CASES) + len(SEARCH_CASES) + len(UPDATE_CASES)
        + len(GRAPH_ENTITY_CASES) + len(CONTRADICTION_CASES)
        + len(COMPLEX_CASES) + len(ROBUSTNESS_CASES)
    )

    print("=" * 76)
    print("  MCP END-TO-END CONFIGURATION BATTLE TEST")
    print("  Ollama Only vs Hybrid/Optimal")
    print(f"  {total_cases} test cases across 7 phases, 9 scoring dimensions")
    print("=" * 76)

    # --- Pre-flight Checks ---
    print("\n  Pre-flight checks:")
    qdrant_url = check_qdrant()
    if not qdrant_url:
        print("  FAIL: Qdrant not reachable. Exiting.")
        sys.exit(1)
    print(f"  OK  Qdrant at {qdrant_url}")

    ollama_url = check_ollama()
    if not ollama_url:
        print("  FAIL: Ollama not reachable. Exiting.")
        sys.exit(1)
    print(f"  OK  Ollama at {ollama_url}")

    neo4j_ok = check_neo4j()
    if not neo4j_ok:
        print("  FAIL: Neo4j not reachable. Exiting.")
        sys.exit(1)
    print("  OK  Neo4j reachable")

    has_google, has_anthropic = check_cloud_creds()
    run_hybrid = has_google and has_anthropic

    if run_hybrid:
        print("  OK  GOOGLE_API_KEY set")
        print("  OK  Anthropic token resolved")
    else:
        print("  WARN: Cloud credentials incomplete — running Ollama-only config")
        if not has_google:
            print("        Missing: GOOGLE_API_KEY")
        if not has_anthropic:
            print("        Missing: Anthropic token")

    # --- Determine Configurations ---
    configs: dict[str, dict[str, str]] = {}
    configs["ollama"] = {
        "label": "Ollama Only",
        "llm_provider": "ollama",
        "llm_model": "qwen3:14b",
        "graph_llm_provider": "ollama",
        "collection": "bench_e2e_ollama",
        "neo4j_base_label": "BenchOllama",
    }
    if run_hybrid:
        configs["hybrid"] = {
            "label": "Hybrid/Optimal",
            "llm_provider": "anthropic",
            "llm_model": "claude-opus-4-6",
            "graph_llm_provider": "gemini_split",
            "collection": "bench_e2e_hybrid",
            "neo4j_base_label": "BenchHybrid",
        }

    config_names = list(configs.keys())
    print(f"\n  Configurations: {', '.join(c['label'] for c in configs.values())}")

    # --- Create Memory Instances ---
    print("\n  Initializing Memory instances...")
    memories: dict[str, Any] = {}
    cleanups: dict[str, Callable] = {}
    timers: dict[str, OperationTimer] = {}

    # Reset Neo4j driver before creating memories
    import mem0_mcp_selfhosted.graph_tools as gt
    gt._driver = None

    for cfg_name, cfg in configs.items():
        print(f"    Creating {cfg['label']}...", end="", flush=True)
        try:
            mem, cleanup = _make_memory(
                cfg_name, qdrant_url,
                cfg["collection"], cfg["neo4j_base_label"],
                cfg["llm_provider"], cfg["llm_model"],
                cfg["graph_llm_provider"],
            )
            memories[cfg_name] = mem
            cleanups[cfg_name] = cleanup
            timers[cfg_name] = OperationTimer()
            print(" OK")
        except Exception as e:
            print(f" FAIL: {e}")
            traceback.print_exc()
            # Clean up already-created instances
            for c in cleanups.values():
                try:
                    c()
                except Exception:
                    pass
            sys.exit(1)

    # --- Results Storage ---
    # Dimensions: per config -> list of scores
    results: dict[str, dict[str, list[float]]] = {
        dim: {cfg: [] for cfg in config_names}
        for dim in ["D1_fact_f1", "D2_search", "D3_update", "D4_graph_entity",
                     "D5_graph_rel", "D6_contradiction", "D7_robustness"]
    }
    # Per-case details for JSON export
    case_details: list[dict] = []

    from mem0_mcp_selfhosted.helpers import call_with_graph

    try:
        # ========================================================
        # PHASE 1: Fact Extraction Quality
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 1: Fact Extraction Quality ({len(FACT_EXTRACTION_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in FACT_EXTRACTION_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result: dict[str, Any] = {"id": case.id, "phase": "fact_extraction",
                                           "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                cfg = configs[cfg_name]
                user_id = f"bench-p1-{case.id}-{cfg_name}"
                tag = "O" if cfg_name == "ollama" else "H"

                start = time.time()
                try:
                    result = mem.add(case.messages, user_id=user_id)
                    duration = time.time() - start
                    timers[cfg_name].record("add", duration)

                    # Get stored memories
                    stored = mem.get_all(user_id=user_id)
                    if isinstance(stored, dict):
                        stored_list = stored.get("results", stored.get("memories", []))
                    elif isinstance(stored, list):
                        stored_list = stored
                    else:
                        stored_list = []

                    scores = score_fact_extraction(stored_list, case)
                    results["D1_fact_f1"][cfg_name].append(scores["f1"])
                    case_result[f"{cfg_name}_f1"] = scores["f1"]
                    case_result[f"{cfg_name}_recall"] = scores["recall"]
                    case_result[f"{cfg_name}_halluc"] = scores["hallucinated"]
                    case_result[f"{cfg_name}_latency"] = round(duration, 2)
                    print(f" {tag}:{scores['f1']:.2f}", end="", flush=True)
                except Exception as e:
                    duration = time.time() - start
                    timers[cfg_name].record("add", duration)
                    results["D1_fact_f1"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_f1"] = 0.0
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            # Determine winner
            scores_map = {cfg: case_result.get(f"{cfg}_f1", 0) for cfg in config_names}
            if len(config_names) == 2:
                o, h = scores_map.get("ollama", 0), scores_map.get("hybrid", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 2: Semantic Search Quality
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 2: Semantic Search Quality ({len(SEARCH_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in SEARCH_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "search",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                user_id = f"bench-p2-{case.id}-{cfg_name}"
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    # Seed memories
                    for seed_msgs in case.seed_messages:
                        start = time.time()
                        mem.add(seed_msgs, user_id=user_id)
                        timers[cfg_name].record("add", time.time() - start)

                    # Small delay for indexing
                    time.sleep(0.5)

                    # Search
                    start = time.time()
                    search_result = mem.search(query=case.query, user_id=user_id)
                    duration = time.time() - start
                    timers[cfg_name].record("search", duration)

                    if isinstance(search_result, dict):
                        search_list = search_result.get("results", search_result.get("memories", []))
                    elif isinstance(search_result, list):
                        search_list = search_result
                    else:
                        search_list = []

                    scores = score_search(search_list, case)
                    results["D2_search"][cfg_name].append(scores["score"])
                    case_result[f"{cfg_name}_score"] = scores["score"]
                    case_result[f"{cfg_name}_recall"] = scores["recall"]
                    case_result[f"{cfg_name}_fp"] = scores["false_positives"]
                    case_result[f"{cfg_name}_latency"] = round(duration, 2)
                    print(f" {tag}:{scores['score']:.2f}", end="", flush=True)
                except Exception as e:
                    results["D2_search"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_score"] = 0.0
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            if len(config_names) == 2:
                o, h = case_result.get("ollama_score", 0), case_result.get("hybrid_score", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 3: Memory Update Fidelity
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 3: Memory Update Fidelity ({len(UPDATE_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in UPDATE_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "update",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                user_id = f"bench-p3-{case.id}-{cfg_name}"
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    # Add initial memory
                    start = time.time()
                    add_result = mem.add(case.initial_message, user_id=user_id)
                    timers[cfg_name].record("add", time.time() - start)

                    # Get the memory ID
                    if isinstance(add_result, dict):
                        mem_results = add_result.get("results", [])
                    else:
                        mem_results = []

                    if mem_results and "id" in mem_results[0]:
                        memory_id = mem_results[0]["id"]

                        # Update the memory
                        start = time.time()
                        mem.update(memory_id, data=case.update_text)
                        duration = time.time() - start
                        timers[cfg_name].record("update", duration)

                        time.sleep(0.5)

                        # Verify via search
                        start = time.time()
                        search_result = mem.search(query=case.update_text, user_id=user_id)
                        timers[cfg_name].record("search", time.time() - start)

                        if isinstance(search_result, dict):
                            search_list = search_result.get("results", search_result.get("memories", []))
                        elif isinstance(search_result, list):
                            search_list = search_result
                        else:
                            search_list = []

                        scores = score_update(search_list, case)
                        results["D3_update"][cfg_name].append(scores["score"])
                        case_result[f"{cfg_name}_score"] = scores["score"]
                        case_result[f"{cfg_name}_presence"] = scores["presence"]
                        case_result[f"{cfg_name}_absence"] = scores["absence"]
                        print(f" {tag}:{scores['score']:.2f}", end="", flush=True)
                    else:
                        results["D3_update"][cfg_name].append(0.0)
                        case_result[f"{cfg_name}_score"] = 0.0
                        case_result[f"{cfg_name}_error"] = "no memory id returned"
                        print(f" {tag}:NO_ID", end="", flush=True)
                except Exception as e:
                    results["D3_update"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_score"] = 0.0
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            if len(config_names) == 2:
                o, h = case_result.get("ollama_score", 0), case_result.get("hybrid_score", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 4: Graph Entity/Relationship Extraction
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 4: Graph Entity/Relationship Extraction ({len(GRAPH_ENTITY_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in GRAPH_ENTITY_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "graph_entity",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                cfg = configs[cfg_name]
                user_id = f"bench-p4-{case.id}-{cfg_name}"
                base_label = cfg["neo4j_base_label"]
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    # Add with graph enabled
                    start = time.time()
                    call_with_graph(mem, True, True, mem.add, case.messages, user_id=user_id)
                    duration = time.time() - start
                    timers[cfg_name].record("add_graph", duration)

                    # Wait for graph processing
                    time.sleep(1.0)

                    # Query graph for entities by name
                    found_entities = query_graph_entities_by_names(case.expected_entities)
                    entity_scores = score_graph_entities(found_entities, case.expected_entities)
                    results["D4_graph_entity"][cfg_name].append(entity_scores["f1"])
                    case_result[f"{cfg_name}_entity_f1"] = entity_scores["f1"]

                    # Query graph for relationships
                    all_entity_names = list(set(case.expected_entities + [s for s, _, d in case.expected_relationships] + [d for _, _, d in case.expected_relationships]))
                    found_rels = query_graph_relationships_for_entities(all_entity_names)
                    rel_scores = score_graph_relationships(found_rels, case.expected_relationships)
                    results["D5_graph_rel"][cfg_name].append(rel_scores["score"])
                    case_result[f"{cfg_name}_rel_score"] = rel_scores["score"]
                    case_result[f"{cfg_name}_latency"] = round(duration, 2)

                    print(f" {tag}:E{entity_scores['f1']:.2f}/R{rel_scores['score']:.2f}", end="", flush=True)
                except Exception as e:
                    results["D4_graph_entity"][cfg_name].append(0.0)
                    results["D5_graph_rel"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_entity_f1"] = 0.0
                    case_result[f"{cfg_name}_rel_score"] = 0.0
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            if len(config_names) == 2:
                oe = case_result.get("ollama_entity_f1", 0)
                he = case_result.get("hybrid_entity_f1", 0)
                winner = "[O]" if oe > he else ("[H]" if he > oe else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 5: Contradiction Handling
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 5: Contradiction Handling ({len(CONTRADICTION_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in CONTRADICTION_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "contradiction",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                cfg = configs[cfg_name]
                user_id = f"bench-p5-{case.id}-{cfg_name}"
                base_label = cfg["neo4j_base_label"]
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    # Add initial memory with graph
                    start = time.time()
                    call_with_graph(mem, True, True, mem.add, case.initial_message, user_id=user_id)
                    timers[cfg_name].record("add_graph", time.time() - start)

                    time.sleep(1.0)

                    # Add contradicting memory with graph
                    start = time.time()
                    call_with_graph(mem, True, True, mem.add, case.contradicting_message, user_id=user_id)
                    duration = time.time() - start
                    timers[cfg_name].record("add_graph_contradiction", duration)

                    time.sleep(1.0)

                    # Inspect graph state using entity-relevant keywords
                    search_terms = [case.old_relationship_key, case.new_relationship_key]
                    graph_text = query_graph_text_for_entities(search_terms)
                    scores = score_contradiction(graph_text, case)

                    results["D6_contradiction"][cfg_name].append(scores["score"])
                    case_result[f"{cfg_name}_score"] = scores["score"]
                    case_result[f"{cfg_name}_correct"] = scores["correct"]
                    case_result[f"{cfg_name}_old_present"] = scores["old_present"]
                    case_result[f"{cfg_name}_new_present"] = scores["new_present"]
                    case_result[f"{cfg_name}_latency"] = round(duration, 2)

                    mark = "PASS" if scores["correct"] else "FAIL"
                    print(f" {tag}:{mark}", end="", flush=True)
                except Exception as e:
                    results["D6_contradiction"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_score"] = 0.0
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            if len(config_names) == 2:
                o = case_result.get("ollama_score", 0)
                h = case_result.get("hybrid_score", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 6: Complex Multi-Memory Scenarios
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 6: Complex Multi-Memory Scenarios ({len(COMPLEX_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in COMPLEX_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "complex",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                cfg = configs[cfg_name]
                user_id = f"bench-p6-{case.id}-{cfg_name}"
                base_label = cfg["neo4j_base_label"]
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    # Add all messages in sequence with graph
                    for msgs in case.messages_sequence:
                        start = time.time()
                        call_with_graph(mem, True, True, mem.add, msgs, user_id=user_id)
                        timers[cfg_name].record("add_graph", time.time() - start)
                        time.sleep(0.5)

                    time.sleep(1.0)

                    # Test search queries
                    search_scores = []
                    for i, query in enumerate(case.search_queries):
                        start = time.time()
                        search_result = mem.search(query=query, user_id=user_id)
                        timers[cfg_name].record("search", time.time() - start)

                        if isinstance(search_result, dict):
                            search_list = search_result.get("results", search_result.get("memories", []))
                        elif isinstance(search_result, list):
                            search_list = search_result
                        else:
                            search_list = []

                        all_text = " ".join(r.get("memory", "") for r in search_list).lower()
                        expected = case.expected_search_hits[i] if i < len(case.expected_search_hits) else []
                        found = sum(1 for e in expected if _fuzzy_match(e, all_text))
                        search_scores.append(found / len(expected) if expected else 1.0)

                    avg_search = sum(search_scores) / len(search_scores) if search_scores else 0.0
                    results["D2_search"][cfg_name].append(avg_search)

                    # Test graph entities
                    found_entities = query_graph_entities_by_names(case.expected_entities)
                    entity_scores = score_graph_entities(found_entities, case.expected_entities)
                    results["D4_graph_entity"][cfg_name].append(entity_scores["f1"])

                    # Test graph relationships
                    if case.expected_relationships:
                        all_entity_names = list(set(case.expected_entities + [s for s, _, d in case.expected_relationships] + [d for _, _, d in case.expected_relationships]))
                        found_rels = query_graph_relationships_for_entities(all_entity_names)
                        rel_scores = score_graph_relationships(found_rels, case.expected_relationships)
                        results["D5_graph_rel"][cfg_name].append(rel_scores["score"])
                        case_result[f"{cfg_name}_rel_score"] = rel_scores["score"]

                    case_result[f"{cfg_name}_search"] = round(avg_search, 4)
                    case_result[f"{cfg_name}_entity_f1"] = entity_scores["f1"]
                    print(f" {tag}:S{avg_search:.2f}/E{entity_scores['f1']:.2f}", end="", flush=True)
                except Exception as e:
                    results["D2_search"][cfg_name].append(0.0)
                    results["D4_graph_entity"][cfg_name].append(0.0)
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:ERR({e.__class__.__name__}: {e})", end="", flush=True)

            if len(config_names) == 2:
                o = case_result.get("ollama_search", 0)
                h = case_result.get("hybrid_search", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

        # ========================================================
        # PHASE 7: Robustness
        # ========================================================
        print(f"\n{'=' * 76}")
        print(f"  PHASE 7: Robustness ({len(ROBUSTNESS_CASES)} cases)")
        print(f"{'=' * 76}")

        for case in ROBUSTNESS_CASES:
            label = f"[{case.id}] {case.difficulty}/{case.category}"
            print(f"  {label}:", end="", flush=True)

            case_result = {"id": case.id, "phase": "robustness",
                          "difficulty": case.difficulty, "category": case.category}

            for cfg_name in config_names:
                mem = memories[cfg_name]
                user_id = f"bench-p7-{case.id}-{cfg_name}"
                tag = "O" if cfg_name == "ollama" else "H"

                try:
                    start = time.time()
                    mem.add(case.messages, user_id=user_id)
                    duration = time.time() - start
                    timers[cfg_name].record("add", duration)
                    success = True

                    stored = mem.get_all(user_id=user_id)
                    if isinstance(stored, dict):
                        stored_list = stored.get("results", stored.get("memories", []))
                    elif isinstance(stored, list):
                        stored_list = stored
                    else:
                        stored_list = []

                    scores = score_robustness(success, stored_list, case)
                    results["D7_robustness"][cfg_name].append(scores["score"])
                    case_result[f"{cfg_name}_score"] = scores["score"]
                    case_result[f"{cfg_name}_success"] = scores["success"]
                    case_result[f"{cfg_name}_num_facts"] = scores["num_facts"]
                    print(f" {tag}:{scores['score']:.2f}", end="", flush=True)
                except Exception as e:
                    # Even errors count as partial success for robustness
                    scores = score_robustness(False, [], case)
                    results["D7_robustness"][cfg_name].append(scores["score"])
                    case_result[f"{cfg_name}_score"] = scores["score"]
                    case_result[f"{cfg_name}_error"] = str(e)
                    print(f" {tag}:{scores['score']:.2f}(err)", end="", flush=True)

            if len(config_names) == 2:
                o = case_result.get("ollama_score", 0)
                h = case_result.get("hybrid_score", 0)
                winner = "[O]" if o > h else ("[H]" if h > o else "[=]")
            else:
                winner = ""
            print(f" {winner}")
            case_details.append(case_result)

    finally:
        # --- Cleanup ---
        print("\n  Cleaning up...")
        for cfg_name, cleanup in cleanups.items():
            try:
                cleanup()
                print(f"    {configs[cfg_name]['label']}: cleaned up")
            except Exception as e:
                print(f"    {configs[cfg_name]['label']}: cleanup error: {e}")

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print(f"\n\n{'#' * 76}")
    print(f"  FINAL RESULTS - MCP E2E CONFIGURATION BATTLE SCORECARD")
    print(f"{'#' * 76}")

    dimension_labels = {
        "D1_fact_f1": "D1: Fact Extraction F1",
        "D2_search": "D2: Search Relevance",
        "D3_update": "D3: Update Fidelity",
        "D4_graph_entity": "D4: Graph Entity Accuracy",
        "D5_graph_rel": "D5: Graph Relationship Accuracy",
        "D6_contradiction": "D6: Contradiction Resolution",
        "D7_robustness": "D7: Robustness",
    }

    scores_summary: list[tuple[str, dict[str, float], str]] = []

    for dim_key, dim_label in dimension_labels.items():
        print(f"\n{'=' * 72}")
        print(f"  {dim_label}")
        print(f"{'=' * 72}")

        dim_scores: dict[str, tuple[float, float, float]] = {}
        for cfg_name in config_names:
            vals = results[dim_key][cfg_name]
            m, lo, hi = ci95(vals)
            dim_scores[cfg_name] = (m, lo, hi)
            cfg_label = configs[cfg_name]["label"]
            print(f"  {cfg_label:>20}: {bar(m)} {m:.1%} (95% CI: {lo:.1%}-{hi:.1%}, n={len(vals)})")

        # Determine winner
        if len(config_names) == 2:
            om, _, _ = dim_scores["ollama"]
            hm, _, _ = dim_scores["hybrid"]
            delta = hm - om
            if delta > 0.01:
                winner = "HYBRID"
            elif delta < -0.01:
                winner = "OLLAMA"
            else:
                winner = "TIE"
            print(f"  Winner: {winner} (delta: {abs(delta):.1%})")
        else:
            om, _, _ = dim_scores["ollama"]
            winner = "OLLAMA"

        means = {cfg: dim_scores[cfg][0] for cfg in config_names}
        scores_summary.append((dim_label, means, winner))

    # --- D8: Latency Summary ---
    print(f"\n{'=' * 72}")
    print(f"  D8: End-to-End Latency")
    print(f"{'=' * 72}")

    for cfg_name in config_names:
        cfg_label = configs[cfg_name]["label"]
        timer_summary = timers[cfg_name].summary()
        print(f"\n  {cfg_label}:")
        print(f"    {'Operation':<25} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'Total':>8}")
        print(f"    {'-' * 63}")
        for op, stats in sorted(timer_summary.items()):
            print(f"    {op:<25} {stats['count']:>6} {stats['mean']:>7.2f}s {stats['p50']:>7.2f}s {stats['p95']:>7.2f}s {stats['total']:>7.1f}s")

    # --- D9: Cost Efficiency ---
    print(f"\n{'=' * 72}")
    print(f"  D9: Cost Efficiency")
    print(f"{'=' * 72}")

    for cfg_name in config_names:
        cfg = configs[cfg_name]
        cfg_label = cfg["label"]
        total_ops = sum(len(v) for v in timers[cfg_name].latencies.values())
        total_time = sum(sum(v) for v in timers[cfg_name].latencies.values())

        if cfg_name == "ollama":
            cost_est = "$0.00 (local compute only)"
        else:
            # Rough estimate: ~$0.015 per Claude call, ~$0.0003 per Gemini call
            graph_ops = sum(len(timers[cfg_name].latencies.get(op, [])) for op in ["add_graph", "add_graph_contradiction"])
            non_graph_ops = total_ops - graph_ops
            est_cost = non_graph_ops * 0.015 + graph_ops * 0.03  # graph = 2 Gemini + 1 Claude call
            cost_est = f"~${est_cost:.2f} estimated"

        print(f"  {cfg_label:>20}: {total_ops} operations, {total_time:.1f}s total, {cost_est}")

    # ===== GRAND SUMMARY =====
    print(f"\n\n{'#' * 76}")
    print(f"  GRAND SUMMARY")
    print(f"{'#' * 76}")

    if len(config_names) == 2:
        ollama_wins = sum(1 for _, _, w in scores_summary if w == "OLLAMA")
        hybrid_wins = sum(1 for _, _, w in scores_summary if w == "HYBRID")
        ties = sum(1 for _, _, w in scores_summary if w == "TIE")

        print(f"\n  {'Dimension':<40} {'Ollama':>8} {'Hybrid':>8} {'Winner':>10}")
        print(f"  {'-' * 70}")
        for name, means, winner in scores_summary:
            om = means.get("ollama", 0)
            hm = means.get("hybrid", 0)
            print(f"  {name:<40} {om:>7.1%} {hm:>7.1%} {winner:>10}")

        print(f"\n  {'=' * 70}")
        print(f"  Ollama wins: {ollama_wins} | Hybrid wins: {hybrid_wins} | Ties: {ties}")
        print(f"  {'=' * 70}")

        if hybrid_wins > ollama_wins:
            print(f"\n  VERDICT: Hybrid/Optimal configuration is superior across {hybrid_wins}/{len(scores_summary)} dimensions.")
            print(f"  The cloud LLMs provide measurably better quality at the cost of API fees.")
        elif ollama_wins > hybrid_wins:
            print(f"\n  VERDICT: Ollama-only configuration is competitive or superior across {ollama_wins}/{len(scores_summary)} dimensions.")
            print(f"  The fully-local setup provides good quality without any API costs.")
        else:
            print(f"\n  VERDICT: Both configurations are closely matched.")
    else:
        print(f"\n  {'Dimension':<40} {'Ollama':>8}")
        print(f"  {'-' * 50}")
        for name, means, winner in scores_summary:
            om = means.get("ollama", 0)
            print(f"  {name:<40} {om:>7.1%}")
        print(f"\n  (Hybrid config not tested — missing cloud credentials)")

    # --- Save Results ---
    output_path = Path(__file__).parent / "mcp_e2e_battle_results.json"

    json_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configurations": {name: {
            "label": cfg["label"],
            "llm_provider": cfg["llm_provider"],
            "llm_model": cfg["llm_model"],
            "graph_llm_provider": cfg["graph_llm_provider"],
        } for name, cfg in configs.items()},
        "total_cases": total_cases,
        "dimensions": {},
        "latency": {cfg: timers[cfg].summary() for cfg in config_names},
        "case_details": case_details,
    }

    for name, means, winner in scores_summary:
        json_results["dimensions"][name] = {
            **{cfg: round(means.get(cfg, 0), 4) for cfg in config_names},
            "winner": winner,
        }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
