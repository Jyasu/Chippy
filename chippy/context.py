"""Builds the initial system prompt: environment, workspace inventory and AGENTS.md rules."""

import json
import os
import platform
from pathlib import Path

from chippy.config import (
    AGENTS_MD_MAX_CHARS,
    INVENTORY_MAX_DEPTH,
    INVENTORY_MAX_FILES,
    INVENTORY_SKIP_DIRS,
    READ_MAX_LINES,
)
from chippy.sandbox import check_readable
from chippy.tools import TOOL_HANDLERS


def load_agents_md(workspace: Path) -> tuple:
    """Returns (status for the banner, rules text for the prompt)."""
    # Goes through the sandbox: a symlinked AGENTS.md must not pull files from outside the workspace.
    try:
        path = check_readable("AGENTS.md", workspace)
    except PermissionError as e:
        return f"Ignored ({e})", "None provided."
    if not path.is_file():
        return "None", "None provided."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Unreadable ({e})", f"Error reading AGENTS.md: {e}"
    if len(text) > AGENTS_MD_MAX_CHARS:
        text = text[:AGENTS_MD_MAX_CHARS] + "\n... [AGENTS.md truncated]"
    return "Detected", text


def build_inventory(workspace: Path) -> list:
    """Sorted file map (depth-limited, capped) so the prompt is identical across runs."""
    base = workspace.resolve()
    files = []
    for cur_root, dirs, names in os.walk(base):
        rel = Path(cur_root).relative_to(base)
        if len(rel.parts) < INVENTORY_MAX_DEPTH:
            dirs[:] = sorted(d for d in dirs if d not in INVENTORY_SKIP_DIRS)
        else:
            dirs[:] = []
        for name in sorted(names):
            files.append((rel / name).as_posix())
            if len(files) > INVENTORY_MAX_FILES:
                return files[:INVENTORY_MAX_FILES] + ["... [truncated]"]
    return files


def build_system_context(workspace: Path) -> tuple:
    """Returns (system prompt, AGENTS.md status)."""
    agents_status, agents_rules = load_agents_md(workspace)

    payload = {
        "agent_environment": {
            "os": platform.system(),
            "working_directory": str(workspace.resolve()),
            "isolation_mode": "No shell access; file tools are confined to the workspace",
        },
        "workspace_inventory": {
            "root_folder": workspace.name,
            "visible_files": build_inventory(workspace),
        },
        "agents_md_rules": agents_rules,
        "operating_constraints": [
            "You CANNOT execute terminal or shell commands (no bash, no powershell).",
            f"You only have access to these tools: {', '.join(TOOL_HANDLERS)}.",
            "All paths must be relative to the scoped workspace.",
            "Any path referencing parent directories outside the workspace will trigger a hard security denial.",
            ".git/ and likely secrets files (.env, private keys) are off limits.",
            f"read_file returns at most {READ_MAX_LINES} lines per call; page through longer files with 'offset' before changing them.",
            "Prefer edit_file for changes to existing files; use write_file to create files or replace them entirely.",
            "Every write is shown to a human as a diff and may be rejected.",
            "Follow all conventions defined in 'agents_md_rules'.",
        ],
    }

    prompt = (
        "You are an engineering assistant working in a sandboxed directory environment.\n"
        "Here is your initialization context payload:\n\n"
        "```json\n"
        + json.dumps(payload, indent=2) +
        "\n```"
    )
    return prompt, agents_status
