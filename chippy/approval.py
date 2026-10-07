"""Human-in-the-loop approval for file modifications."""

import difflib
from pathlib import Path

from chippy.config import APPROVAL_DIFF_PREVIEW_LINES
from chippy.terminal import sanitize


def _diff_lines(old_text: str, new_text: str, label: str) -> list:
    return list(difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(),
        fromfile=f"a/{label}", tofile=f"b/{label}", lineterm="",
    ))


def _print_diff(diff: list, limit=None) -> bool:
    """Prints the diff, returning True if it was cut short."""
    shown = diff if limit is None else diff[:limit]
    for line in shown:
        print(sanitize(line))
    if len(shown) < len(diff):
        print(f"... [{len(diff) - len(shown)} more diff lines]")
        return True
    return False


def request_write_approval(target: Path, workspace: Path, operation: str, old_text: str,
                           new_text: str, warnings: list, old_is_binary: bool = False) -> bool:
    """Shows the full change as a diff and asks the human to approve it."""
    label = target.relative_to(workspace.resolve()).as_posix()
    diff = _diff_lines("" if old_is_binary else old_text, new_text, label)
    old_lines = "binary" if old_is_binary else str(len(old_text.splitlines()))

    print("\n" + "=" * 60)
    print("[HUMAN APPROVAL REQUIRED: FILE WRITE]")
    print(f"Target Path: {sanitize(target)}")
    print(f"Operation  : {sanitize(operation)}")
    print(f"Size       : {len(new_text.encode('utf-8'))} bytes")
    print(f"Lines      : {old_lines} -> {len(new_text.splitlines())}")
    for warning in warnings:
        print(f"!! WARNING : {warning}")
    if old_is_binary:
        print("!! WARNING : the existing file is binary and will be replaced; diff is against an empty file")
    print("-" * 60)
    truncated = _print_diff(diff, APPROVAL_DIFF_PREVIEW_LINES)
    print("=" * 60)

    prompt = "Allow write operation? [y/N" + ("/v = view full diff" if truncated else "") + "]: "
    while True:
        try:
            decision = input(prompt).strip().lower()
        except EOFError:
            return False
        if decision in ("y", "yes"):
            return True
        if decision == "v" and truncated:
            _print_diff(diff)
            continue
        return False
