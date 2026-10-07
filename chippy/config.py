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
SEARCH_MAX_CONTEXT_LINES = 5

# AGENTS.md up to this size goes into the prompt verbatim; longer files are sent as an outline.
AGENTS_MD_INLINE_CHARS = 8 * 1024

# Initial file map sent in the system prompt.
INVENTORY_MAX_FILES = 100
INVENTORY_MAX_DEPTH = 3
# Skipped by the inventory and by search_files.
WALK_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv"})

# Elision: tool results at least ELIDE_MIN_CHARS long, and string arguments of tool calls
# (write_file content, edit_file strings) at least ELIDE_ARG_MIN_CHARS long, become short stubs.
ELIDE_MIN_CHARS = 1000
ELIDE_ARG_MIN_CHARS = 300

# Context window management. Sizes are API-reported prompt tokens when available,
# otherwise estimated at CHARS_PER_TOKEN characters per token.
DEFAULT_CONTEXT_LIMIT = 128_000
MIN_CONTEXT_LIMIT = 4_000
CHARS_PER_TOKEN = 4
COMPACT_TRIGGER_RATIO = 0.8     # auto-compact once the conversation reaches this share of the limit
COMPACT_TARGET_RATIO = 0.5      # summarize if eliding alone doesn't get it below this share
COMPACT_KEEP_STEPS = 2          # the latest model steps are always kept verbatim
COMPACT_SUMMARY_INPUT_RATIO = 0.5
COMPACT_TRANSCRIPT_ARG_CHARS = 300
COMPACT_TRANSCRIPT_RESULT_CHARS = 1500

# Explore sub-agent limits.
EXPLORE_MODES = ("auto", "always", "never")
EXPLORE_READ_MAX_LINES = 200
EXPLORE_CONTEXT_RATIO = 0.5     # the explorer must report once its own context reaches this share
EXPLORE_SNIPPET_MAX_LINES = 80

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
    explore_mode: str = "auto"
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    log_path: str = ""
