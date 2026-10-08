"""Command-line entry point: `python -m chippy` in development, `python chippy.py` once bundled."""

import argparse
import os
import sys
from pathlib import Path

from chippy import __version__
from chippy.agent import run_agent
from chippy.config import (
    API_KEY_ENV,
    CONTEXT_LIMIT_ENV,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_EXPLORE_MAX_STEPS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_URL,
    EXPLORE_MODEL_ENV,
    EXPLORE_MODES,
    MIN_CONTEXT_LIMIT,
    MODEL_ENV,
    URL_ENV,
    Settings,
)
from chippy.envfile import ENV_FILE_NAME, EnvFileError, default_env_file, load_env_file


def _epilog() -> str:
    return f"""configuration:
  {API_KEY_ENV:<19} API key (never an option: it would leak into shell history)
  {URL_ENV:<19} endpoint, as --url
  {MODEL_ENV:<19} as --model
  {EXPLORE_MODEL_ENV:<19} as --explore-model
  {CONTEXT_LIMIT_ENV:<19} as --context-limit

Set these as environment variables, or as NAME=value lines in
  {default_env_file()}
or the file given with --env-file. Environment variables win over the file,
and command-line options win over both. Example {ENV_FILE_NAME}:
  {API_KEY_ENV}=sk-...
  {URL_ENV}=https://example.com/v1/chat/completions
  {MODEL_ENV}=gpt-4o"""


def _from_env(value, env_name: str, default):
    """A command-line value, else the environment (or env file), else the default."""
    return value if value is not None else os.getenv(env_name) or default


def parse_settings(argv=None) -> tuple:
    """Parses the command line and loads the env file. Returns (target dir, Settings); exits on bad input."""
    parser = argparse.ArgumentParser(
        prog="chippy",
        description="Zero-dependency Sandboxed Agent Runner",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--dir", default=".", help="Target directory to lock the harness to.")
    parser.add_argument("-m", "--model", help=f"Model name (default {DEFAULT_MODEL}).")
    parser.add_argument("--url", help="Chat completions endpoint URL (default OpenAI's).")
    parser.add_argument("--env-file", type=Path, metavar="PATH",
                        help=f"Read settings from this file instead of {ENV_FILE_NAME} next to chippy.py.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Omitted from requests when not set.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum model calls per message before the agent pauses.")
    parser.add_argument("--explore-model",
                        help="Model for the explore sub-agent, e.g. a cheaper one. Defaults to --model.")
    parser.add_argument("--explore-max-steps", type=int, default=DEFAULT_EXPLORE_MAX_STEPS,
                        help="Maximum model calls for one explore run before it must report back.")
    parser.add_argument("--explore", choices=EXPLORE_MODES, default="auto",
                        help="auto: the agent decides when to explore; always: every request starts with "
                             "an explore; never: no explore tool.")
    parser.add_argument("--context-limit", type=int,
                        help="The model's context window in tokens; the conversation is compacted automatically "
                             f"as it gets close (default {DEFAULT_CONTEXT_LIMIT:,}).")
    parser.add_argument("--log", default="", metavar="PATH",
                        help="Append one JSON line per model call (source, model, token usage) to this file.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)
    env_path = args.env_file or default_env_file()
    try:
        loaded, skipped = load_env_file(env_path, required=args.env_file is not None)
    except EnvFileError as e:
        parser.error(str(e))

    context_limit = args.context_limit
    if context_limit is None:
        raw = os.getenv(CONTEXT_LIMIT_ENV)
        try:
            context_limit = int(raw) if raw else DEFAULT_CONTEXT_LIMIT
        except ValueError:
            parser.error(f"{CONTEXT_LIMIT_ENV} must be a whole number of tokens, not {raw!r}")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.explore_max_steps < 1:
        parser.error("--explore-max-steps must be at least 1")
    if context_limit < MIN_CONTEXT_LIMIT:
        parser.error(f"--context-limit must be at least {MIN_CONTEXT_LIMIT:,}")

    api_key = os.getenv(API_KEY_ENV, "")
    if API_KEY_ENV in loaded:
        key_source = str(env_path)
    elif API_KEY_ENV in skipped:
        key_source = f"the {API_KEY_ENV} environment variable (overriding {env_path})"
    else:
        key_source = f"the {API_KEY_ENV} environment variable" if api_key else ""

    settings = Settings(
        model=_from_env(args.model, MODEL_ENV, DEFAULT_MODEL),
        url=_from_env(args.url, URL_ENV, DEFAULT_URL),
        api_key=api_key,
        api_key_source=key_source,
        temperature=args.temperature,
        max_steps=args.max_steps,
        explore_model=_from_env(args.explore_model, EXPLORE_MODEL_ENV, ""),
        explore_max_steps=args.explore_max_steps,
        explore_mode=args.explore,
        context_limit=context_limit,
        log_path=args.log,
    )
    return Path(args.dir).resolve(), settings


def main(argv=None) -> int:
    target_dir, settings = parse_settings(argv)
    if not target_dir.is_dir():
        print(f"Error: Scoped folder '{target_dir}' does not exist.", file=sys.stderr)
        return 1
    run_agent(target_dir, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
