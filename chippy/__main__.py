"""Command-line entry point: `python -m chippy` in development, `python chippy.py` once bundled."""

import argparse
import os
import sys
from pathlib import Path

from chippy import __version__
from chippy.agent import run_agent
from chippy.config import (
    API_KEY_ENV,
    DEFAULT_EXPLORE_MAX_STEPS,
    DEFAULT_EXPLORE_MODEL,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_URL,
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.explore_max_steps < 1:
        parser.error("--explore-max-steps must be at least 1")

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
    )
    run_agent(target_dir, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
