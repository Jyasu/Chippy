"""Workspace file tools exposed to the model (zero shell access), their schemas and the dispatcher."""

import fnmatch
import functools
import inspect
import json
import os
import re
import stat
import tempfile
from pathlib import Path

from chippy.approval import request_write_approval
from chippy.config import (
    BINARY_SNIFF_BYTES,
    LIST_MAX_ENTRIES,
    READ_MAX_CHARS,
    READ_MAX_LINE_CHARS,
    READ_MAX_LINES,
    SEARCH_MAX_FILE_BYTES,
    SEARCH_MAX_FILES,
    SEARCH_MAX_LINE_CHARS,
    SEARCH_MAX_MATCHES,
    WALK_SKIP_DIRS,
)
from chippy.sandbox import (
    check_listable,
    check_readable,
    check_writable,
    resolve_scoped_path,
    write_warnings,
)
from chippy.terminal import clip


def _tool_errors(func):
    """Turns exceptions into tool results so a failing tool never ends the session."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError as pe:
            return {"status": "denied", "message": str(pe)}
        except Exception as e:
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}
    return wrapper


def _as_int(value, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer, got {value!r}") from None


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\x00" in f.read(BINARY_SNIFF_BYTES)


def _read_text(path: Path, strict: bool) -> tuple:
    """
    Returns (text, newline). Text always uses '\\n' line endings; `newline` is the
    file's original ending so writes can restore it.
    """
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8", errors="strict" if strict else "replace")
    return text.replace("\r\n", "\n"), newline


def _default_file_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _atomic_write(target: Path, text: str, newline: str) -> None:
    """Writes via a temp file + rename so an interrupted write never leaves a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else _default_file_mode()
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _iter_lines(f, max_chars: int):
    """Yields (line, clipped), capping each line at max_chars without loading over-long lines into memory."""
    while True:
        line = f.readline(max_chars)
        if not line:
            return
        clipped = False
        if len(line) >= max_chars and not line.endswith(("\n", "\r")):
            ending = ""
            rest = f.readline(max_chars)
            while rest:
                if rest.endswith(("\n", "\r")):
                    ending = "\n"
                    break
                rest = f.readline(max_chars)
            clipped = True
            line = f"{line}... [line clipped at {max_chars} chars]{ending}"
        yield line.replace("\r\n", "\n"), clipped


def _walk_files(root: Path):
    """Yields files under root (or root itself), sorted, skipping WALK_SKIP_DIRS. Symlinked dirs are not followed."""
    if not root.is_dir():
        yield root
        return
    for cur_root, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in WALK_SKIP_DIRS)
        for name in sorted(names):
            yield Path(cur_root) / name


# ==========================================
# Tools
# ==========================================

@_tool_errors
def tool_list_directory(workspace: Path, directory_path: str = ".") -> dict:
    """Lists entries inside a specified directory relative to workspace root."""
    target = check_listable(directory_path or ".", workspace)
    if not target.exists():
        return {"status": "error", "message": f"Directory '{directory_path}' does not exist."}
    if not target.is_dir():
        return {"status": "error", "message": f"Path '{directory_path}' is not a directory."}

    children = sorted(target.iterdir())
    entries = []
    for entry in children[:LIST_MAX_ENTRIES]:
        item = {"name": entry.name}
        if entry.is_symlink():
            item["is_symlink"] = True
            try:
                resolve_scoped_path(str(entry), workspace)
            except PermissionError:
                # Don't stat it: that would leak metadata about files outside the workspace.
                item["outside_workspace"] = True
                entries.append(item)
                continue
        try:
            st = entry.stat()
        except OSError:
            item["error"] = "unreadable (broken symlink?)"
            entries.append(item)
            continue
        is_dir = stat.S_ISDIR(st.st_mode)
        item["is_dir"] = is_dir
        item["size_bytes"] = None if is_dir else st.st_size
        entries.append(item)

    result = {"status": "success", "entries": entries}
    if len(children) > LIST_MAX_ENTRIES:
        result["truncated"] = True
        result["note"] = f"Showing the first {LIST_MAX_ENTRIES} of {len(children)} entries."
    return result


@_tool_errors
def tool_read_file(workspace: Path, file_path: str, offset=1, limit=READ_MAX_LINES) -> dict:
    """Safely reads a window of lines from a text file within the workspace."""
    target = check_readable(file_path, workspace)
    if not target.exists():
        return {"status": "error", "message": f"File '{file_path}' does not exist."}
    if not target.is_file():
        return {"status": "error", "message": f"Path '{file_path}' is not a regular file."}
    offset = _as_int(offset, "offset")
    if offset < 1:
        return {"status": "error", "message": "'offset' must be 1 or greater."}
    limit = max(1, min(_as_int(limit, "limit"), READ_MAX_LINES))
    if _is_binary(target):
        return {"status": "error", "message": f"File '{file_path}' appears to be binary."}

    lines, chars, clipped_count, line_no, has_more = [], 0, 0, 0, False
    with open(target, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line_no, (line, clipped) in enumerate(_iter_lines(f, READ_MAX_LINE_CHARS), start=1):
            if line_no < offset:
                continue
            if len(lines) >= limit or chars + len(line) > READ_MAX_CHARS:
                has_more = True
                break
            lines.append(line)
            chars += len(line)
            clipped_count += clipped

    if not lines and offset > 1:
        return {"status": "error", "message": f"'offset' {offset} is past the end of the file ({line_no} lines)."}

    end_line = offset + len(lines) - 1
    result = {
        "status": "success",
        "content": "".join(lines),
        "start_line": offset,
        "end_line": end_line,
        "has_more": has_more,
    }
    notes = []
    if has_more:
        result["next_offset"] = end_line + 1
        notes.append(f"Showing lines {offset}-{end_line}. Call read_file with offset={end_line + 1} to continue.")
    if clipped_count:
        notes.append(f"{clipped_count} over-long line(s) were clipped to {READ_MAX_LINE_CHARS} characters.")
    if not notes:
        notes.append("Full file content" if offset == 1 else "End of file")
    result["note"] = " ".join(notes)
    return result


@_tool_errors
def tool_search_files(workspace: Path, pattern: str, path: str = ".", glob: str = "", ignore_case=False) -> dict:
    """Regex search over text files, returning 'path:line: text' hits so the model can read just those lines."""
    if not isinstance(pattern, str) or not pattern:
        return {"status": "error", "message": "'pattern' must be a non-empty string."}
    try:
        regex = re.compile(pattern, re.IGNORECASE if _as_bool(ignore_case) else 0)
    except re.error as e:
        return {"status": "error", "message": f"Invalid regular expression: {e}"}
    root = check_listable(path or ".", workspace)
    if not root.exists():
        return {"status": "error", "message": f"Path '{path}' does not exist."}

    base = workspace.resolve()
    matches, files_matched, files_scanned, stop_reason = [], 0, 0, ""
    for candidate in _walk_files(root):
        rel = candidate.relative_to(base).as_posix()
        if glob and not fnmatch.fnmatch(rel if "/" in glob else candidate.name, glob):
            continue
        try:
            # Resolves symlinks and refuses secrets, exactly like read_file.
            target = check_readable(rel, workspace)
        except PermissionError:
            continue
        if not target.is_file() or target.stat().st_size > SEARCH_MAX_FILE_BYTES or _is_binary(target):
            continue
        if files_scanned >= SEARCH_MAX_FILES:
            stop_reason = f"Stopped after scanning {SEARCH_MAX_FILES} files; narrow 'path' or 'glob'."
            break
        files_scanned += 1
        found = False
        with open(target, "r", encoding="utf-8", errors="replace", newline="") as f:
            for line_no, (line, _) in enumerate(_iter_lines(f, READ_MAX_LINE_CHARS), start=1):
                if not regex.search(line):
                    continue
                if len(matches) >= SEARCH_MAX_MATCHES:
                    stop_reason = f"Showing the first {SEARCH_MAX_MATCHES} matches; narrow the pattern, 'path' or 'glob'."
                    break
                found = True
                matches.append(f"{rel}:{line_no}: {clip(line.rstrip(), SEARCH_MAX_LINE_CHARS)}")
        files_matched += found
        if stop_reason:
            break

    result = {"status": "success", "matches": matches, "files_matched": files_matched}
    if stop_reason:
        result["truncated"] = True
        result["note"] = stop_reason
    elif not matches:
        result["note"] = f"No matches in {files_scanned} file(s) searched."
    return result


@_tool_errors
def tool_write_file(workspace: Path, file_path: str, content: str) -> dict:
    """Prompts human approval (with a diff) before creating or overwriting a file."""
    target = check_writable(file_path, workspace)
    if target.is_dir():
        return {"status": "error", "message": f"Path '{file_path}' is a directory."}
    if not isinstance(content, str):
        return {"status": "error", "message": "'content' must be a string."}

    exists = target.exists()
    old_text, newline, old_is_binary = "", "\n", False
    if exists:
        old_is_binary = _is_binary(target)
        if not old_is_binary:
            old_text, newline = _read_text(target, strict=False)
    new_text = content.replace("\r\n", "\n")
    if exists and not old_is_binary and new_text == old_text:
        return {"status": "success", "message": f"No changes: '{file_path}' already has this content."}

    approved = request_write_approval(
        target, workspace, "Overwrite" if exists else "Create new file",
        old_text, new_text, write_warnings(target, workspace), old_is_binary=old_is_binary,
    )
    if not approved:
        return {"status": "denied", "message": "File write rejected by human supervisor."}

    _atomic_write(target, new_text, newline)
    return {"status": "success", "message": f"Wrote {len(new_text.splitlines())} lines to {file_path}."}


@_tool_errors
def tool_edit_file(workspace: Path, file_path: str, old_string: str, new_string: str, replace_all=False) -> dict:
    """Replaces an exact, unique snippet in an existing file, after human approval."""
    # Readable check too: match/no-match errors would otherwise reveal the contents of secret files.
    target = check_readable(file_path, workspace)
    check_writable(file_path, workspace)
    if not target.is_file():
        return {"status": "error", "message": f"File '{file_path}' does not exist. Use write_file to create it."}
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return {"status": "error", "message": "'old_string' and 'new_string' must be strings."}
    if not old_string:
        return {"status": "error", "message": "'old_string' must not be empty. Use write_file to create or replace whole files."}
    if old_string == new_string:
        return {"status": "error", "message": "'old_string' and 'new_string' are identical."}
    if _is_binary(target):
        return {"status": "error", "message": f"File '{file_path}' appears to be binary."}
    try:
        text, newline = _read_text(target, strict=True)
    except UnicodeDecodeError:
        return {"status": "error", "message": f"File '{file_path}' is not valid UTF-8; editing it could corrupt it."}

    old = old_string.replace("\r\n", "\n")
    new = new_string.replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        return {"status": "error", "message": "'old_string' was not found. Read the file again and copy the text exactly, including whitespace."}
    replace_all = _as_bool(replace_all)
    if count > 1 and not replace_all:
        return {"status": "error", "message": f"'old_string' appears {count} times. Include more surrounding context to make it unique, or set replace_all=true."}

    replaced = count if replace_all else 1
    new_text = text.replace(old, new, -1 if replace_all else 1)
    approved = request_write_approval(
        target, workspace, f"Edit ({replaced} replacement{'s' if replaced > 1 else ''})",
        text, new_text, write_warnings(target, workspace),
    )
    if not approved:
        return {"status": "denied", "message": "File edit rejected by human supervisor."}

    _atomic_write(target, new_text, newline)
    return {"status": "success", "message": f"Replaced {replaced} occurrence(s) in {file_path}."}


# ==========================================
# Registry: the single source of truth for tool names, schemas and handlers
# ==========================================

_REGISTRY = (
    (tool_list_directory, {
        "name": "list_directory",
        "description": f"List files and subdirectories within a given relative folder path (up to {LIST_MAX_ENTRIES} entries).",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Relative directory path. Use '.' or '' for root.",
                },
            },
            "required": ["directory_path"],
        },
    }),
    (tool_read_file, {
        "name": "read_file",
        "description": (
            f"Read a UTF-8 text file, at most {READ_MAX_LINES} lines per call. "
            "Read only the lines you need (offset/limit, e.g. around a search_files hit) instead of whole files. "
            "If the result has has_more=true, call again with offset=next_offset to read the rest. "
            "Always read the whole file before rewriting it with write_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path of the file to inspect."},
                "offset": {"type": "integer", "description": "1-based line number to start from. Defaults to 1."},
                "limit": {"type": "integer", "description": f"Maximum lines to return (1-{READ_MAX_LINES}). Defaults to {READ_MAX_LINES}."},
            },
            "required": ["file_path"],
        },
    }),
    (tool_search_files, {
        "name": "search_files",
        "description": (
            f"Search text files for a regular expression (Python syntax). Returns up to {SEARCH_MAX_MATCHES} "
            "'path:line: text' matches. Use it to find where something is defined or used, then read_file "
            "just those lines with offset/limit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Relative file or directory to search. Defaults to the workspace root."},
                "glob": {"type": "string", "description": "Only search files matching this pattern, e.g. '*.py' or 'src/*.ts'."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
            },
            "required": ["pattern"],
        },
    }),
    (tool_write_file, {
        "name": "write_file",
        "description": (
            "Create a new file, or replace an existing file's entire content. Requires human approval. "
            "Prefer edit_file for changes to existing files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path of the target file to create or replace."},
                "content": {"type": "string", "description": "The exact full text content to write."},
            },
            "required": ["file_path", "content"],
        },
    }),
    (tool_edit_file, {
        "name": "edit_file",
        "description": (
            "Replace an exact snippet of an existing file with new text. Requires human approval. "
            "old_string must match the file exactly (including whitespace) and be unique unless replace_all is true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path of the file to edit."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    }),
)

TOOL_HANDLERS = {schema["name"]: handler for handler, schema in _REGISTRY}
TOOLS = [{"type": "function", "function": schema} for _, schema in _REGISTRY]

# The explore sub-agent gets these only: it gathers context and never changes files.
READ_ONLY_TOOL_NAMES = ("list_directory", "read_file", "search_files")
READ_ONLY_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] in READ_ONLY_TOOL_NAMES]

# Run by the agent loop rather than dispatch_tool, since it starts a sub-agent (see explore.py).
EXPLORE_TOOL_NAME = "explore"
EXPLORE_TOOL = {"type": "function", "function": {
    "name": EXPLORE_TOOL_NAME,
    "description": (
        "Hand a context-gathering task to a read-only sub-agent with a fresh context. It searches and reads "
        "the workspace and returns a compact brief: the relevant file:line ranges with verbatim snippets, "
        "applicable AGENTS.md rules, assumptions, and the user's answers to any blocking questions. "
        "Use it before multi-file changes or when you don't know where the relevant code is; "
        "skip it for small, local tasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What you are about to do and what you need to know to do it.",
            },
        },
        "required": ["task"],
    },
}}
AGENT_TOOLS = TOOLS + [EXPLORE_TOOL]


def parse_tool_arguments(raw_arguments) -> tuple:
    """Returns (arguments dict, None) or (None, error result)."""
    if isinstance(raw_arguments, dict):
        args = raw_arguments
    else:
        try:
            args = json.loads(raw_arguments) if raw_arguments else {}
        except (TypeError, ValueError) as e:
            return None, {"status": "error", "message": f"Arguments are not valid JSON ({e}). Retry with a JSON object."}
    if not isinstance(args, dict):
        return None, {"status": "error", "message": "Arguments must be a JSON object."}
    return args, None


def tool_result_message(call: dict, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.get("id"),
        "name": (call.get("function") or {}).get("name") or "",
        "content": json.dumps(result),
    }


def dispatch_tool(name: str, raw_arguments, workspace: Path, allowed=None) -> dict:
    """
    Runs one model tool call, limited to the `allowed` tool names if given.
    Bad names or arguments become error results the model can correct.
    """
    available = tuple(TOOL_HANDLERS) if allowed is None else tuple(allowed)
    handler = TOOL_HANDLERS.get(name) if name in available else None
    if handler is None:
        return {"status": "error", "message": f"Unknown tool: {name!r}. Available tools: {', '.join(available)}."}

    args, error = parse_tool_arguments(raw_arguments)
    if error:
        return error

    try:
        inspect.signature(handler).bind(workspace, **args)
    except TypeError as e:
        return {"status": "error", "message": f"Invalid arguments for {name}: {e}"}
    return handler(workspace, **args)
