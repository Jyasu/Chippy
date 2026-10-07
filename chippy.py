#!/usr/bin/env python3
"""
Single-file, zero-dependency Sandboxed Agent Harness.
Features:
- Strict canonical path enforcement (pathlib.resolve) to prevent directory traversal
- Native Python file operations (read, write, list) with zero shell execution
- Human-in-the-loop approval prompt for all writes/modifications
- AGENTS.md auto-discovery and OpenCode-style initial context payload
"""

import os
import sys
import json
import platform
import argparse
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


# ==========================================
# 1. Sandboxing & Path Boundary Verification
# ==========================================

def resolve_scoped_path(rel_or_abs_path: str, workspace: Path) -> Path:
    """
    Resolves target path to its canonical form and verifies it strictly resides
    within the target workspace. Raises PermissionError on any escape attempt.
    """
    base = workspace.resolve()
    target = (base / rel_or_abs_path).resolve()

    # The canonical target path must be base or a descendant of base
    if target != base and base not in target.parents:
        raise PermissionError(
            f"Security Violation: Path '{rel_or_abs_path}' resolves to '{target}', "
            f"which escapes the designated workspace '{base}'."
        )
    return target


# ==========================================
# 2. Workspace File Tools (Zero Shell Access)
# ==========================================

def tool_list_directory(directory_path: str, workspace: Path) -> dict:
    """Lists entries inside a specified directory relative to workspace root."""
    try:
        target = resolve_scoped_path(directory_path, workspace)
        if not target.exists():
            return {"status": "error", "message": f"Directory '{directory_path}' does not exist."}
        if not target.is_dir():
            return {"status": "error", "message": f"Path '{directory_path}' is not a directory."}

        entries = []
        for entry in sorted(target.iterdir()):
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size_bytes": entry.stat().st_size if entry.is_file() else None
            })
        return {"status": "success", "entries": entries}
    except PermissionError as pe:
        return {"status": "denied", "message": str(pe)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_read_file(file_path: str, workspace: Path, max_lines: int = 500) -> dict:
    """Safely reads contents of a file within the workspace."""
    try:
        target = resolve_scoped_path(file_path, workspace)
        if not target.exists():
            return {"status": "error", "message": f"File '{file_path}' does not exist."}
        if not target.is_file():
            return {"status": "error", "message": f"Path '{file_path}' is not a regular file."}

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(max_lines)]
            content = "".join(lines)
            truncated = bool(f.readline())

        return {
            "status": "success",
            "content": content,
            "truncated": truncated,
            "note": f"Truncated to {max_lines} lines" if truncated else "Full file content"
        }
    except PermissionError as pe:
        return {"status": "denied", "message": str(pe)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_write_file(file_path: str, content: str, workspace: Path) -> dict:
    """Prompts human approval before writing/overwriting a file in the workspace."""
    try:
        target = resolve_scoped_path(file_path, workspace)

        print("\n" + "=" * 50)
        print("[HUMAN APPROVAL REQUIRED: FILE WRITE]")
        print(f"Target Path: {target}")
        print(f"Operation  : {'Overwrite' if target.exists() else 'Create new file'}")
        print(f"Size       : {len(content.encode('utf-8'))} bytes")
        print("-" * 50)
        # Show preview
        preview_lines = content.splitlines()[:10]
        print("\n".join(preview_lines))
        if len(content.splitlines()) > 10:
            print("... [preview truncated]")
        print("=" * 50)

        decision = input("Allow write operation? [y/N]: ").strip().lower()
        if decision not in ("y", "yes"):
            return {"status": "denied", "message": "File write rejected by human supervisor."}

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"status": "success", "message": f"Successfully wrote {len(content)} characters to {file_path}."}
    except PermissionError as pe:
        return {"status": "denied", "message": str(pe)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==========================================
# 3. Context Payload & Tool Schema
# ==========================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories within a given relative folder path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Relative directory path. Use '.' or '' for root."
                    }
                },
                "required": ["directory_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a UTF-8 text file up to 500 lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of the file to inspect."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file within the scoped workspace. Requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of the target file to create or update."
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact full text content to write."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    }
]


def build_system_context(workspace: Path) -> str:
    agents_md_path = workspace / "AGENTS.md"
    agents_rules = "None provided."
    if agents_md_path.is_file():
        try:
            agents_rules = agents_md_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            agents_rules = f"Error reading AGENTS.md: {e}"

    # Initial file map (max 100 entries, depth 3)
    inventory = []
    for cur_root, dirs, files in os.walk(workspace):
        rel = Path(cur_root).relative_to(workspace)
        if len(rel.parts) > 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", ".venv"}]
        for f in files:
            p = rel / f if str(rel) != "." else Path(f)
            inventory.append(str(p).replace("\\", "/"))
            if len(inventory) >= 100:
                inventory.append("... [truncated]")
                break
        if len(inventory) >= 100:
            break

    payload = {
        "agent_environment": {
            "os": platform.system(),
            "working_directory": str(workspace.resolve()),
            "isolation_mode": "Strict Python Canonical Sandbox (No shell access)"
        },
        "workspace_inventory": {
            "root_folder": workspace.name,
            "visible_files": inventory
        },
        "agents_md_rules": agents_rules,
        "operating_constraints": [
            "You CANNOT execute terminal or shell commands (no bash, no powershell).",
            "You only have access to read_file, write_file, and list_directory.",
            "All paths must be relative to the scoped workspace.",
            "Any path referencing parent directories outside the workspace will trigger a hard security denial.",
            "Follow all conventions defined in 'agents_md_rules'."
        ]
    }

    return (
        "You are an engineering assistant working in a sandboxed directory environment.\n"
        "Here is your initialization context payload:\n\n"
        "```json\n"
        + json.dumps(payload, indent=2) +
        "\n```"
    )


# ==========================================
# 4. Zero-Dependency API & Loop
# ==========================================

def call_llm_api(messages, model, url, api_key):
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        print(f"\n[API Error {e.code}]: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n[Network Error]: {e.reason}", file=sys.stderr)
        sys.exit(1)


def run_agent(workspace: Path, model: str, url: str, api_key: str):
    system_prompt = build_system_context(workspace)
    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 60)
    print(f"Directory Sandbox : {workspace.resolve()}")
    print(f"Shell Disabled    : True (Pure Python file APIs only)")
    print(f"AGENTS.md         : {'Detected' if (workspace / 'AGENTS.md').exists() else 'None'}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            response_msg = call_llm_api(messages, model, url, api_key)
            messages.append(response_msg)

            tool_calls = response_msg.get("tool_calls")
            if not tool_calls:
                print(f"\nAgent > {response_msg.get('content', '')}\n")
                break

            for tool in tool_calls:
                func_name = tool["function"]["name"]
                args = json.loads(tool["function"]["arguments"])
                call_id = tool["id"]

                print(f"[Tool Call] {func_name}({args})")

                if func_name == "read_file":
                    result = tool_read_file(args.get("file_path", ""), workspace)
                elif func_name == "write_file":
                    result = tool_write_file(args.get("file_path", ""), args.get("content", ""), workspace)
                elif func_name == "list_directory":
                    result = tool_list_directory(args.get("directory_path", "."), workspace)
                else:
                    result = {"status": "error", "message": f"Unknown tool: {func_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps(result)
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-dependency Sandboxed Agent Runner")
    parser.add_argument("-d", "--dir", default=".", help="Target directory to lock the harness to.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="API Key.")

    args = parser.parse_args()
    target_dir = Path(args.dir).resolve()

    if not target_dir.is_dir():
        print(f"Error: Scoped folder '{target_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    run_agent(target_dir, args.model, args.url, args.key)