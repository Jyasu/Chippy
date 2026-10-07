"""Command-line entry point: `python -m chippy` in development, `python chippy.py` once bundled."""

import argparse
import os
import sys
from pathlib import Path

from chippy import __version__
from chippy.agent import run_agent
from chippy.config import (
    API_KEY_ENV,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_EXPLORE_MAX_STEPS,
    DEFAULT_EXPLORE_MODEL,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_URL,
    EXPLORE_MODES,
    MIN_CONTEXT_LIMIT,
    Settings,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="chippy",
        description="Zero-dependency Sandboxed Agent Runner",
        epilog=f"The API key is read from the {API_KEY_ENV} environment variable.",
    )
    parser.add_argument("-d", "--dir", default=".", help="Target directory to lock the harness to.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Chat completions endpoint URL.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Omitted from requests when not set.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum model calls per message before the agent pauses.")
    parser.add_argument("--explore-model", default=DEFAULT_EXPLORE_MODEL,
                        help="Model for the explore sub-agent, e.g. a cheaper one. Defaults to --model "
                             "(or the LLM_EXPLORE_MODEL environment variable).")
    parser.add_argument("--explore-max-steps", type=int, default=DEFAULT_EXPLORE_MAX_STEPS,
                        help="Maximum model calls for one explore run before it must report back.")
    parser.add_argument("--explore", choices=EXPLORE_MODES, default="auto",
                        help="auto: the agent decides when to explore; always: every request starts with "
                             "an explore; never: no explore tool.")
    parser.add_argument("--context-limit", type=int, default=os.getenv("LLM_CONTEXT_LIMIT", DEFAULT_CONTEXT_LIMIT),
                        help="The model's context window in tokens; the conversation is compacted automatically "
                             f"as it gets close (default {DEFAULT_CONTEXT_LIMIT:,}, or LLM_CONTEXT_LIMIT).")
    parser.add_argument("--log", default="", metavar="PATH",
                        help="Append one JSON line per model call (source, model, token usage) to this file.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.explore_max_steps < 1:
        parser.error("--explore-max-steps must be at least 1")
    if args.context_limit < MIN_CONTEXT_LIMIT:
        parser.error(f"--context-limit must be at least {MIN_CONTEXT_LIMIT:,}")

    target_dir = Path(args.dir).resolve()
    if not target_dir.is_dir():
        print(f"Error: Scoped folder '{target_dir}' does not exist.", file=sys.stderr)
        return 1

    settings = Settings(
        model=args.model,
        url=args.url,
        api_key=os.getenv(API_KEY_ENV, ""),
        temperature=args.temperature,
        max_steps=args.max_steps,
        explore_model=args.explore_model,
        explore_max_steps=args.explore_max_steps,
        explore_mode=args.explore,
        context_limit=args.context_limit,
        log_path=args.log,
    )
    run_agent(target_dir, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
