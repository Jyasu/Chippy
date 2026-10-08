# Chippy

A zero-dependency, sandboxed agent harness that lives in one file.

## Use it anywhere

[`chippy.py`](chippy.py) is the whole harness. It needs only Python 3.9+, with no
clone, no install and no other files. Copy it (or open it raw and paste it into a file)
on any machine and run:

```sh
export OPENAI_API_KEY=...            # LLM_BASE_URL, LLM_MODEL, LLM_EXPLORE_MODEL and LLM_CONTEXT_LIMIT are optional
python3 chippy.py -d path/to/project
```

On Windows, `export` doesn't exist. In PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
python chippy.py -d path\to\project
```

In cmd, use `set OPENAI_API_KEY=...` (no quotes).

### Without environment variables

If you can't set environment variables, put the settings in a file named `chippy.env`
next to `chippy.py`:

```ini
# chippy.env: one NAME=value per line
OPENAI_API_KEY=...
LLM_BASE_URL=https://api.example.com/v1/chat/completions
LLM_MODEL=gpt-4o
```

You can also keep the file somewhere else and pass `--env-file PATH`. Environment variables
override the file, and command-line options override both. The file can set any variable,
for example `HTTPS_PROXY`. The agent can't read a file named `chippy.env`, even inside the
project. Keep the file out of version control. The API key is never accepted as a
command-line option, because it would be saved in shell history and visible in the
process list.

At startup, chippy prints the endpoint and where the API key came from. If no key is set,
it says so before you send anything.

`chippy.py` is generated from the `chippy/` package. Don't edit it by hand.

At the prompt, `/compact [focus]` summarizes the conversation to free context, `/usage`
shows token usage for the session, and `/help` lists commands.

## Keeping context small

- **Targeted reads.** `search_files` (with optional context lines) finds `path:line` hits so
  the agent reads only the lines it needs. Re-reading the same unchanged lines in one
  request returns a short note instead of the text again.
- **Explore.** `explore` hands context gathering to a read-only sub-agent with a fresh
  conversation (optionally a cheaper model, `--explore-model`). Only its brief reaches the
  main agent: file:line ranges whose exact text the harness copies from disk, applicable
  `AGENTS.md` rules, and your answers to any blocking questions. `--explore auto|always|never`
  controls when it runs.
- **Elision.** Bulky tool output, and large `write_file`/`edit_file` arguments, older than
  the previous request are replaced by stubs.
- **Compaction.** Near `--context-limit` (default 128,000 tokens), older tool output is
  elided, and if that isn't enough, earlier conversation is summarized. If the API still
  rejects a request as too long, everything is compacted and the request retried once.
- **AGENTS.md outline.** A long `AGENTS.md` is sent as an outline of its sections, which
  the agent reads on demand.
- **Measurement.** Token usage is printed after every request (estimated when the API
  doesn't report it), and `--log PATH` writes one JSON line per model call.
  `scripts/bench.py` runs reference tasks against a real model to compare settings or
  harness changes.

## Layout

| Path | Purpose |
|---|---|
| `chippy.py` | **Built, committed release file**: the only thing users need |
| `chippy/config.py` | Defaults, limits, `Settings` |
| `chippy/sandbox.py` | Workspace boundary and path policy (`.git/`, secrets, risky writes) |
| `chippy/terminal.py` | Escaping model output before it reaches the terminal |
| `chippy/approval.py` | Diff-based human approval prompt |
| `chippy/tools.py` | File tools, their schemas and the dispatcher |
| `chippy/context.py` | System prompts: inventory and `AGENTS.md` (verbatim or outline) |
| `chippy/llm.py` | Chat completions client with timeouts and retries, token usage |
| `chippy/compact.py` | Context window management: elision, summaries, size tracking |
| `chippy/explore.py` | Read-only explore sub-agent that returns a compact brief |
| `chippy/agent.py` | Interactive loop and `/` commands |
| `chippy/__main__.py` | CLI entry point |
| `scripts/build.py` | Bundler: `chippy/` → `chippy.py` |
| `scripts/bench.py` | Reference tasks for comparing token usage and success (needs an API key) |
| `tests/` | `unittest` suite, runnable against the package or the bundle |

## Development

```sh
python3 -m chippy -d path/to/project          # run from source (the package wins over chippy.py)
python3 -m unittest discover -s tests          # test the package
python3 scripts/build.py                       # rebuild chippy.py
python3 scripts/build.py --check               # rebuild, smoke-test, and run the tests against both
python3 scripts/build.py --verify              # exit 1 if chippy.py is out of date (CI / pre-commit)
```

**Any change under `chippy/` must be committed together with a rebuilt `chippy.py`.**
`--verify` catches commits that forget this.

## Bundling rules

`scripts/build.py` concatenates the modules in dependency order into one namespace,
removes internal imports and hoists standard-library imports. The build fails if the
code breaks any rule that would make the bundle behave differently from the package:

1. Import internal names at module top level: `from chippy.tools import dispatch_tool`.
   No `import chippy.x`, no module objects, no `as` aliases, no `*`.
2. Top-level names, including `_private` helpers, must be unique across modules.
3. Only `__main__.py` may have an `if __name__ == "__main__":` block.
4. No import cycles, no subpackages, standard library only.
5. The output must be pure ASCII, since non-ASCII characters are the ones clipboards and
   terminals tend to mangle. Use escapes such as `"✓"` in source.

Tests share the same constraint: patch shared modules (`builtins.input`,
`urllib.request.urlopen`, `time.sleep`), never names inside a `chippy` module, so the
same test works against both targets.

## Releasing

1. Bump `__version__` in `chippy/__init__.py`.
2. `python3 scripts/build.py --check`
3. Commit `chippy/` and `chippy.py` together, then tag the commit.
