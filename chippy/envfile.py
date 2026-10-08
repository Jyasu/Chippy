"""Loads chippy.env: NAME=value settings for machines where environment variables are awkward to set."""

import os
import re
from pathlib import Path

ENV_FILE_NAME = "chippy.env"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class EnvFileError(Exception):
    """The env file is unreadable or has a line that is not NAME=value."""


def default_env_file() -> Path:
    """chippy.env next to chippy.py; in development, at the repo root beside the chippy/ package."""
    here = Path(__file__).resolve().parent
    return (here.parent if __package__ else here) / ENV_FILE_NAME


def parse_env_file(text: str, label: str = ENV_FILE_NAME) -> dict:
    """
    Parses NAME=value lines. Blank lines and lines starting with # are skipped, an `export `
    prefix is allowed, and a value may be wrapped in matching quotes. Unquoted values end
    at ` #`, so a trailing comment does not become part of a key.
    """
    values = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not _ENV_NAME.fullmatch(name):
            raise EnvFileError(f"{label}:{number}: expected NAME=value, got {raw.strip()[:40]!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0]
        values[name] = value
    return values


def load_env_file(path: Path, required: bool = False) -> tuple:
    """
    Copies the file's settings into os.environ and returns (names set, names skipped because
    the environment already has a non-empty value). A missing file is fine unless required.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")  # Notepad may save a BOM
    except FileNotFoundError:
        if required:
            raise EnvFileError(f"{path} does not exist.") from None
        return [], []
    except (OSError, UnicodeDecodeError) as e:
        raise EnvFileError(f"Could not read {path}: {e}") from e
    loaded, skipped = [], []
    for name, value in parse_env_file(text, str(path)).items():
        if os.environ.get(name):
            skipped.append(name)
        else:
            os.environ[name] = value
            loaded.append(name)
    return loaded, skipped
