"""Workspace boundary checks. Every tool path goes through here before touching disk."""

import fnmatch
from pathlib import Path

# Never readable or writable by the agent: hooks under .git/ run code on commit,
# and .git/config can hold credentials in remote URLs.
PROTECTED_DIR_NAMES = frozenset({".git"})

# Reads need no human approval and their contents go straight to the API, so likely
# secrets are refused. Patterns without "/" match the file name, others the relative path.
SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".netrc", ".npmrc", ".pypirc",
)
SECRET_FILE_EXCEPTIONS = (".env.example", ".env.sample", ".env.template")

# Files that cause code to run later. Writes are still allowed, but the approval prompt flags them.
RISKY_WRITE_PATTERNS = (
    (".envrc", "direnv executes this when you enter the directory"),
    ("makefile", "make targets run shell commands"),
    ("gnumakefile", "make targets run shell commands"),
    ("package.json", "npm scripts run shell commands, including install hooks"),
    ("setup.py", "runs when the package is built or installed"),
    ("pyproject.toml", "configures build backends and hooks"),
    ("conftest.py", "pytest imports and runs this automatically"),
    (".pre-commit-config.yaml", "pre-commit runs these hooks on every commit"),
    (".vscode/*", "editor tasks and settings can run commands"),
    (".idea/*", "IDE run configurations can run commands"),
    (".github/workflows/*", "CI runs these with repository secrets"),
    (".gitlab-ci.yml", "CI runs these with repository secrets"),
    ("*.sh", "shell script"),
)


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


def _relative(target: Path, workspace: Path) -> Path:
    return target.relative_to(workspace.resolve())


def _matches(rel: Path, pattern: str) -> bool:
    # Case-insensitive: macOS and Windows filesystems treat ".GIT" and ".git" as the same directory.
    pattern = pattern.lower()
    subject = rel.as_posix() if "/" in pattern else rel.name
    return fnmatch.fnmatchcase(subject.lower(), pattern)


def is_protected(target: Path, workspace: Path) -> bool:
    return any(part.lower() in PROTECTED_DIR_NAMES for part in _relative(target, workspace).parts)


def is_secret(target: Path, workspace: Path) -> bool:
    rel = _relative(target, workspace)
    if any(_matches(rel, pattern) for pattern in SECRET_FILE_EXCEPTIONS):
        return False
    return any(_matches(rel, pattern) for pattern in SECRET_FILE_PATTERNS)


def check_listable(path: str, workspace: Path) -> Path:
    target = resolve_scoped_path(path, workspace)
    if is_protected(target, workspace):
        raise PermissionError(f"Access to '{path}' is blocked: version-control internals are off limits.")
    return target


def check_readable(path: str, workspace: Path) -> Path:
    target = check_listable(path, workspace)
    if is_secret(target, workspace):
        raise PermissionError(f"Access to '{path}' is blocked: it looks like a secrets file.")
    return target


def check_writable(path: str, workspace: Path) -> Path:
    return check_listable(path, workspace)


def write_warnings(target: Path, workspace: Path) -> list:
    """Reasons a write to `target` deserves extra scrutiny from the human approver."""
    rel = _relative(target, workspace)
    warnings = [reason for pattern, reason in RISKY_WRITE_PATTERNS if _matches(rel, pattern)]
    if is_secret(target, workspace):
        warnings.append("looks like a secrets file")
    return warnings
