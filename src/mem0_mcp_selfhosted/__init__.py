"""mem0-mcp-selfhosted: Self-hosted mem0 MCP server for Claude Code."""

import os

# CRITICAL: Suppress mem0ai telemetry BEFORE any mem0 import.
# The telemetry module reads this env var at module import time and sends
# events to PostHog when enabled. Must be set before `import mem0`.
os.environ["MEM0_TELEMETRY"] = "false"

__version__ = "0.1.0"


def main():
    """Entry point for the mem0-mcp-selfhosted server."""
    from mem0_mcp_selfhosted.server import run_server

    run_server()
