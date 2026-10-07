"""Builds the system prompts: environment, workspace inventory and AGENTS.md rules."""

import json
import os
import platform
import re
from pathlib import Path

from chippy.config import (
    AGENTS_MD_INLINE_CHARS,
    INVENTORY_MAX_DEPTH,
    INVENTORY_MAX_FILES,
    READ_MAX_LINES,
    WALK_SKIP_DIRS,
)
from chippy.sandbox import check_readable
from chippy.tools import EXPLORE_TOOL_NAME, READ_ONLY_TOOL_NAMES, TOOL_HANDLERS

_HEADING = re.compile(r"^#{1,6}\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")


def outline_markdown(text: str) -> list:
    """Headings with the line ranges they cover, e.g. 'L20-30: ## Testing'. Ignores '#' lines inside code fences."""
    lines = text.splitlines()
    starts, in_fence = [], False
    for number, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and _HEADING.match(line):
            starts.append((number, line.strip()))
    if not starts:
        return []
    sections = []
    if starts[0][0] > 1 and any(line.strip() for line in lines[:starts[0][0] - 1]):
        sections.append(f"L1-{starts[0][0] - 1}: (text before the first heading)")
    for index, (start, heading) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        sections.append(f"L{start}-{end}: {heading}")
    return sections


def load_agents_md(workspace: Path) -> tuple:
    """
    Returns (status for the banner, rules for the prompt). Short files are sent verbatim;
    long ones as an outline the model reads from on demand, so they don't fill every request.
    """
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
    if len(text) <= AGENTS_MD_INLINE_CHARS:
        return "Detected", text

    sections = outline_markdown(text)
    if sections:
        return f"Detected ({len(text):,} chars, sent as an outline of {len(sections)} sections)", {
            "note": ("AGENTS.md is long, so only its outline is shown. Before acting, read the sections "
                     "relevant to your task with read_file('AGENTS.md', offset, limit)."),
            "sections": sections,
        }
    head = text[:AGENTS_MD_INLINE_CHARS]
    next_line = head.count("\n") + 1
    return (f"Detected ({len(text):,} chars, truncated)",
            head + f"\n... [AGENTS.md truncated; read the rest with read_file('AGENTS.md', offset={next_line})]")


def build_inventory(workspace: Path) -> list:
    """Sorted file map (depth-limited, capped) so the prompt is identical across runs."""
    base = workspace.resolve()
    files = []
    for cur_root, dirs, names in os.walk(base):
        rel = Path(cur_root).relative_to(base)
        if len(rel.parts) < INVENTORY_MAX_DEPTH:
            dirs[:] = sorted(d for d in dirs if d not in WALK_SKIP_DIRS)
        else:
            dirs[:] = []
        for name in sorted(names):
            files.append((rel / name).as_posix())
            if len(files) > INVENTORY_MAX_FILES:
                return files[:INVENTORY_MAX_FILES] + ["... [truncated]"]
    return files


def _render_prompt(intro: str, workspace: Path, agents_rules, tool_names, constraints: list) -> str:
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
            f"You only have access to these tools: {', '.join(tool_names)}.",
            "All paths must be relative to the scoped workspace.",
            "Any path referencing parent directories outside the workspace will trigger a hard security denial.",
            ".git/ and likely secrets files (.env, private keys) are off limits.",
            f"read_file returns at most {READ_MAX_LINES} lines per call; page through longer files with 'offset'.",
            "Locate code with search_files, then read only the lines you need with read_file offset/limit.",
            *constraints,
            "Follow all conventions defined in 'agents_md_rules'.",
        ],
    }
    return (
        intro + "\n"
        "Here is your initialization context payload:\n\n"
        "```json\n"
        + json.dumps(payload, indent=2) +
        "\n```"
    )


EXPLORE_GUIDANCE = {
    "auto": [
        "Before multi-file changes, or when you don't know where the relevant code is, call explore: "
        "it gathers context in a separate conversation and returns a compact brief. "
        "For small, local tasks use search_files and read_file directly.",
    ],
    "always": [
        "Every request starts with an explore call: describe the task and what you need to know in 'task'. "
        "Call explore again later only if the brief turns out to be missing something.",
    ],
    "never": [],
}
EXPLORE_SNIPPET_GUIDANCE = (
    "Snippets in an explore brief are copied from the files by the harness, so they are exact and can serve "
    "as edit_file old_string; if an edit fails, the file changed since, so re-read the lines."
)


def build_system_context(workspace: Path, explore_mode: str = "auto") -> tuple:
    """Returns (system prompt for the main agent, AGENTS.md status)."""
    agents_status, agents_rules = load_agents_md(workspace)
    explore_lines = EXPLORE_GUIDANCE[explore_mode] + ([EXPLORE_SNIPPET_GUIDANCE] if explore_mode != "never" else [])
    tool_names = [*TOOL_HANDLERS] + ([EXPLORE_TOOL_NAME] if explore_mode != "never" else [])
    prompt = _render_prompt(
        "You are an engineering assistant working in a sandboxed directory environment.",
        workspace, agents_rules, tool_names,
        [
            *explore_lines,
            "Read a whole file before rewriting it with write_file. "
            "Prefer edit_file for changes to existing files; use write_file to create files or replace them entirely.",
            "Every write is shown to a human as a diff and may be rejected.",
            "Old tool output may be elided, and the conversation may be replaced by a summary, to save context; "
            "call a tool again if you need its output.",
        ],
    )
    return prompt, agents_status


EXPLORE_BRIEF_FORMAT = """{
  "summary": "one or two sentences: what the task touches and how",
  "relevant": [{"file": "path", "lines": "40-62", "why": "..."}],
  "rules": ["AGENTS.md L20-30: the rule, paraphrased"],
  "assumptions": ["things you deduced but did not confirm"],
  "open_questions": ["only questions that block the task and cannot be answered from the code"]
}"""


def build_explorer_prompt(workspace: Path) -> str:
    """System prompt for the read-only explore sub-agent."""
    _, agents_rules = load_agents_md(workspace)
    return _render_prompt(
        "You are a read-only context scout. Another agent will carry out the task you are given; "
        "your job is to find exactly the context it needs, so it doesn't have to read whole files.",
        workspace, agents_rules, READ_ONLY_TOOL_NAMES,
        [
            "You cannot change files. Search first, then read narrow line ranges; avoid reading whole files.",
            "Collect the AGENTS.md rules that apply to the task (read the relevant sections if only an outline is shown).",
            "Resolve ambiguity by reading the code where you can. Ask open_questions only when the answer "
            "changes what should be done and the code cannot tell you; the user will be asked them.",
            "Don't copy code into the brief: give exact 'lines' ranges and the harness attaches the text of each. "
            "Make each range cover exactly what the task needs (e.g. the whole function to change).",
            "Keep the brief compact: include only what the task needs, not everything you read.",
            "When done, reply with ONLY a JSON object in this format (no prose, no code fence):\n"
            + EXPLORE_BRIEF_FORMAT,
        ],
    )
