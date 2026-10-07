"""Defaults, limits and runtime settings shared across the harness."""

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
# Empty means the explore sub-agent uses the main model.
DEFAULT_EXPLORE_MODEL = os.getenv("LLM_EXPLORE_MODEL", "")
# The key is only read from the environment: CLI arguments leak into shell history and `ps`.
API_KEY_ENV = "OPENAI_API_KEY"

# Tool output limits keep a single call from flooding the model's context.
READ_MAX_LINES = 500
READ_MAX_CHARS = 256 * 1024
READ_MAX_LINE_CHARS = 2000
BINARY_SNIFF_BYTES = 8192
LIST_MAX_ENTRIES = 500
SEARCH_MAX_MATCHES = 100
SEARCH_MAX_FILES = 5000
SEARCH_MAX_FILE_BYTES = 1024 * 1024
SEARCH_MAX_LINE_CHARS = 300

# AGENTS.md up to this size goes into the prompt verbatim; longer files are sent as an outline.
AGENTS_MD_INLINE_CHARS = 8 * 1024

# Initial file map sent in the system prompt.
INVENTORY_MAX_FILES = 100
INVENTORY_MAX_DEPTH = 3
# Skipped by the inventory and by search_files.
WALK_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv"})

# Tool results at least this long are replaced by a stub once they are more than one request old.
ELIDE_MIN_CHARS = 1000

# Diff lines shown before the approval prompt; the rest is available on request.
APPROVAL_DIFF_PREVIEW_LINES = 200

DEFAULT_MAX_STEPS = 25
DEFAULT_EXPLORE_MAX_STEPS = 15
HTTP_TIMEOUT_SECONDS = 120
HTTP_MAX_RETRIES = 4
HTTP_MAX_BACKOFF_SECONDS = 60


@dataclass(frozen=True)
class Settings:
    model: str
    url: str
    api_key: str = ""
    temperature: Optional[float] = None
    max_steps: int = DEFAULT_MAX_STEPS
    explore_model: str = ""
    explore_max_steps: int = DEFAULT_EXPLORE_MAX_STEPS
