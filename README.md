# Chippy

A zero-dependency, sandboxed agent harness that lives in one file.

## Use it anywhere

[`chippy.py`](chippy.py) is the whole harness. It needs only Python 3.9+, with no
clone, no install and no other files. Copy it (or open it raw and paste it into a file)
on any machine and run:

```sh
export OPENAI_API_KEY=...            # LLM_BASE_URL and LLM_MODEL are optional
python3 chippy.py -d path/to/project
```

`chippy.py` is generated from the `chippy/` package. Don't edit it by hand.

## Layout

| Path | Purpose |
|---|---|
| `chippy.py` | **Built, committed release file**: the only thing users need |
| `chippy/config.py` | Defaults, limits, `Settings` |
| `chippy/sandbox.py` | Workspace boundary and path policy (`.git/`, secrets, risky writes) |
| `chippy/terminal.py` | Escaping model output before it reaches the terminal |
| `chippy/approval.py` | Diff-based human approval prompt |
| `chippy/tools.py` | File tools, their schemas and the dispatcher |
| `chippy/context.py` | System prompt: inventory and `AGENTS.md` |
| `chippy/llm.py` | Chat completions client with timeouts and retries |
| `chippy/agent.py` | Interactive loop |
| `chippy/__main__.py` | CLI entry point |
| `scripts/build.py` | Bundler: `chippy/` → `chippy.py` |
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
