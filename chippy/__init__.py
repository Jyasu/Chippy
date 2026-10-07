"""
Chippy: single-file, zero-dependency sandboxed agent harness.

Features:
- Strict canonical path enforcement (pathlib.resolve) to prevent directory traversal
- Native Python file operations (list, read, write, edit) with zero shell execution
- Human-in-the-loop diff approval for all writes/modifications
- AGENTS.md auto-discovery and OpenCode-style initial context payload
"""

__version__ = "0.2.0"
