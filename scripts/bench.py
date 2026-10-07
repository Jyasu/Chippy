#!/usr/bin/env python3
"""
Runs reference tasks against a real model to compare token usage and success, between
settings (e.g. --modes auto,never) or before and after a change to the harness.

Each run copies a small fixture project into a temporary directory, sends one task,
approves every write automatically, skips explore questions, then checks the result.
Checks import the edited fixture in a subprocess, so they run code the model wrote,
inside the temporary copy.

Needs OPENAI_API_KEY (LLM_BASE_URL and LLM_MODEL are optional, as for chippy itself).

Usage:
    python3 scripts/bench.py                              # every task, explore auto and never
    python3 scripts/bench.py --tasks rename --modes always --repeat 3
    python3 scripts/bench.py --json results.json          # also save the raw numbers
"""

import argparse
import builtins
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chippy.agent import run_turn  # noqa: E402
from chippy.compact import ContextMeter  # noqa: E402
from chippy.config import API_KEY_ENV, DEFAULT_CONTEXT_LIMIT, DEFAULT_MODEL, DEFAULT_URL, EXPLORE_MODES, Settings  # noqa: E402
from chippy.context import build_system_context  # noqa: E402
from chippy.llm import LLMError, Usage  # noqa: E402

FIXTURE = {
    "AGENTS.md": """# Shop project conventions

## Style
- Python 3.9+, 4-space indents, docstrings on public functions.

## Errors
- Raise ValueError for bad input, with a message of the form "invalid <thing>: <value>",
  for example "invalid price: -1".

## Logging
- Never use print in library code. Use a module-level `logger = logging.getLogger(__name__)`
  and log at info level.

## Testing
- Tests live in tests/ and use unittest.
""",
    "shop/__init__.py": '"""A tiny shop."""\n',
    "shop/pricing.py": '''"""Prices and totals."""


def parse_price(text):
    """Parses a price like '2.50' into a float."""
    return float(text)


def calc_total(prices, tax_rate=0.0):
    """Sum of prices plus tax, rounded to cents."""
    subtotal = sum(prices)
    return round(subtotal * (1 + tax_rate), 2)
''',
    "shop/cart.py": '''"""Shopping cart."""

from shop.pricing import calc_total, parse_price


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price_text):
        self.items.append((name, parse_price(price_text)))

    def total(self, tax_rate=0.0):
        return calc_total([price for _, price in self.items], tax_rate)
''',
    "shop/report.py": '''"""Sales report."""

from shop.pricing import calc_total


def print_report(carts):
    for index, cart in enumerate(carts, start=1):
        print(f"Cart {index}: {cart.total()}")
    print(f"All carts: {calc_total([cart.total() for cart in carts])}")
''',
}


def _python(workspace: Path, code: str) -> tuple:
    """Runs code against the edited fixture. Returns (passed, detail)."""
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=workspace, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "check timed out"
    return result.returncode == 0, result.stderr.strip().splitlines()[-1] if result.returncode else ""


def check_rename(workspace: Path) -> tuple:
    sources = {p.name: p.read_text(encoding="utf-8") for p in (workspace / "shop").glob("*.py")}
    if any("calc_total" in text for text in sources.values()):
        return False, "calc_total is still referenced"
    if "def compute_total" not in sources.get("pricing.py", ""):
        return False, "pricing.py does not define compute_total"
    return _python(workspace, "import shop.cart, shop.report\n"
                              "from shop.pricing import compute_total\n"
                              "assert compute_total([1, 2]) == 3")


def check_validate(workspace: Path) -> tuple:
    return _python(workspace, """
from shop.pricing import parse_price
assert parse_price('2.50') == 2.5
try:
    parse_price('-1')
except ValueError as e:
    assert str(e).startswith('invalid price'), f'message does not follow AGENTS.md: {e}'
else:
    raise AssertionError('parse_price accepted -1')
""")


def check_logging(workspace: Path) -> tuple:
    text = (workspace / "shop" / "report.py").read_text(encoding="utf-8")
    if "print(" in text:
        return False, "report.py still prints"
    if "getLogger(__name__)" not in text:
        return False, "report.py has no module-level logger"
    return _python(workspace, "import shop.report")


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    check: Callable


TASKS = (
    Task("rename", "Rename the function calc_total to compute_total everywhere it is defined or used.", check_rename),
    Task("validate", "Make parse_price reject negative prices, following the error conventions in AGENTS.md.",
         check_validate),
    Task("logging", "Make shop/report.py follow the logging rules in AGENTS.md.", check_logging),
)


def _auto_answer(prompt: str = "") -> str:
    """Approves every write; skips explore's questions so the agent decides on its own."""
    return "y" if prompt.startswith("Allow") else ""


def write_fixture(workspace: Path) -> None:
    for rel, text in FIXTURE.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_task(task: Task, settings: Settings, verbose: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "project"
        write_fixture(workspace)
        messages = [
            {"role": "system", "content": build_system_context(workspace, settings.explore_mode)[0]},
            {"role": "user", "content": task.prompt},
        ]
        session, transcript, error = Usage(), io.StringIO(), ""
        original_input = builtins.input
        builtins.input = _auto_answer
        try:
            with redirect_stdout(sys.stdout if verbose else transcript):
                run_turn(messages, workspace, settings, ContextMeter(), session)
        except LLMError as e:
            error = str(e)
        finally:
            builtins.input = original_input
        passed, detail = (False, error) if error else task.check(workspace)
    return {
        "task": task.name,
        "explore": settings.explore_mode,
        "passed": passed,
        "detail": detail,
        "calls": session.calls,
        "by_source": session.by_source,
        "prompt_tokens": session.prompt_tokens,
        "cached_tokens": session.cached_tokens,
        "completion_tokens": session.completion_tokens,
        "largest_prompt": session.peak_prompt_tokens,
        "estimated": session.estimated > 0,
    }


def print_table(results: list) -> None:
    header = ("task", "explore", "ok", "calls", "prompt", "cached", "completion", "largest", "detail")
    rows = [(r["task"], r["explore"], "yes" if r["passed"] else "NO", str(r["calls"]),
             f"{'~' if r['estimated'] else ''}{r['prompt_tokens']:,}", f"{r['cached_tokens']:,}",
             f"{r['completion_tokens']:,}", f"{r['largest_prompt']:,}", r["detail"][:60]) for r in results]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare chippy's token usage and success on reference tasks.")
    parser.add_argument("--tasks", default=",".join(t.name for t in TASKS), help="Comma-separated task names.")
    parser.add_argument("--modes", default="auto,never", help="Comma-separated explore modes to compare.")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per task and mode (models are not deterministic).")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--explore-model", default="")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    parser.add_argument("--json", metavar="PATH", help="Also write the results to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the agent's output.")
    args = parser.parse_args(argv)

    by_name = {task.name: task for task in TASKS}
    names, modes = args.tasks.split(","), args.modes.split(",")
    unknown = [n for n in names if n not in by_name] + [m for m in modes if m not in EXPLORE_MODES]
    if unknown:
        parser.error(f"unknown task or mode: {', '.join(unknown)}")

    results = []
    for mode in modes:
        settings = Settings(model=args.model, url=args.url, api_key=os.getenv(API_KEY_ENV, ""),
                            explore_model=args.explore_model, explore_mode=mode, context_limit=args.context_limit)
        for name in names:
            for run in range(args.repeat):
                print(f"Running {name} (explore {mode}, run {run + 1}/{args.repeat})...", file=sys.stderr)
                results.append(run_task(by_name[name], settings, args.verbose))

    print_table(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
