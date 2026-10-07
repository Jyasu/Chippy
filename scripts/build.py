#!/usr/bin/env python3
"""
Bundles the chippy/ package into one self-contained script, chippy.py at the repo root.

That file is committed: it is the release artifact people copy-paste into new
environments, so it must stay in sync with the sources (see --verify).

Modules are concatenated in dependency order into a single namespace: internal
imports are removed, stdlib imports are de-duplicated and hoisted to the top.
That only works if the package follows these rules, and the build fails with a
clear message when one is broken:

1. Import internal names at module top level with `from chippy.<module> import name`
   (or `from .<module> import name`). No `import chippy.x`, no importing modules
   as objects, no `as` aliases, no `*`.
2. Top-level names must be unique across modules (private `_helpers` included).
3. Only __main__.py may contain an `if __name__ == "__main__":` block.
4. No import cycles and no subpackages.
5. Standard library only; the bundle must run on a bare Python install.

Usage:
    python scripts/build.py            # writes chippy.py
    python scripts/build.py --verify   # exit 1 if chippy.py is stale (for CI / pre-commit)
    python scripts/build.py --check    # also runs the test suite against the package and the bundle
"""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_DIR = ROOT / "chippy"
DEFAULT_OUT = ROOT / "chippy.py"
ENTRY_MODULE = "__main__"
INIT_MODULE = "__init__"
MIN_PYTHON = (3, 9)


class BuildError(Exception):
    pass


class Module:
    def __init__(self, path: Path, package: str):
        self.name = path.stem
        self.label = f"{package}/{path.name}"
        self.source = path.read_text(encoding="utf-8")
        try:
            self.tree = ast.parse(self.source, filename=self.label, feature_version=MIN_PYTHON)
        except SyntaxError as e:
            raise BuildError(f"{self.label}:{e.lineno}: not valid Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} syntax: {e.msg}") from e
        self.deps = {}       # internal module name -> names imported from it
        self.future = set()  # __future__ features
        self.external = []   # (kind, module, name, asname) for hoisted stdlib imports
        self.drop = []       # (first_line, last_line) spans removed from the output
        self.defined = set()

    def where(self, node) -> str:
        return f"{self.label}:{node.lineno}"


# ==========================================
# Analysis
# ==========================================

def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_main_guard(node) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _internal_source(node, package: str):
    """The internal module a `from ... import` reads from, or None if it is external."""
    if isinstance(node, ast.Import):
        return None
    if node.level == 1:
        return node.module or INIT_MODULE
    if node.level > 1:
        return ".."
    if node.module == package:
        return INIT_MODULE
    if node.module and node.module.startswith(package + "."):
        return node.module[len(package) + 1:]
    return None


def _is_internal(node, package: str) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == package or a.name.startswith(package + ".") for a in node.names)
    return _internal_source(node, package) is not None


def _external_binding(kind: str, source, name: str, asname) -> tuple:
    """(bound name, what it refers to) for an external import, so equal imports in two modules don't collide."""
    if kind == "import":
        if asname:
            return asname, ("module", name)
        root = name.split(".")[0]
        return root, ("module", root)
    return asname or name, ("from", source, name)


def _bound_names(stmts, nested=False):
    """Top-level names a block of statements binds (descending into if/try/with/for blocks)."""
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        yield n.id
        elif nested and isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                yield a.asname or a.name.split(".")[0]
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)):
            for block in ("body", "orelse", "finalbody"):
                yield from _bound_names(getattr(node, block, []), nested=True)
            for handler in getattr(node, "handlers", []):
                yield from _bound_names(handler.body, nested=True)


def analyze(module: Module, package: str, module_names: set) -> None:
    body = module.tree.body
    if body and _is_docstring(body[0]):
        module.drop.append((body[0].lineno, body[0].end_lineno))

    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" and node.level == 0:
            module.future.update(a.name for a in node.names)
            module.drop.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.ImportFrom) and _is_internal(node, package):
            source = _internal_source(node, package)
            if source == ".." or "." in source:
                raise BuildError(f"{module.where(node)}: subpackages and multi-level relative imports are not supported.")
            if source not in module_names:
                raise BuildError(f"{module.where(node)}: there is no module {package}.{source}.")
            for a in node.names:
                if a.name == "*":
                    raise BuildError(f"{module.where(node)}: `import *` is not supported; import names explicitly.")
                if a.asname and a.asname != a.name:
                    raise BuildError(f"{module.where(node)}: internal imports cannot be aliased ({a.name} as {a.asname}); "
                                     "the bundle has one namespace, so the name must stay the same.")
                if source == INIT_MODULE and a.name in module_names:
                    raise BuildError(f"{module.where(node)}: import names from {package}.{a.name}, not the module itself.")
                module.deps.setdefault(source, set()).add(a.name)
            module.drop.append((node.lineno, node.end_lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if _is_internal(node, package):
                raise BuildError(f"{module.where(node)}: use `from {package}.<module> import name` instead of importing modules.")
            for a in node.names:
                if a.name == "*":
                    raise BuildError(f"{module.where(node)}: `import *` is not supported; import names explicitly.")
                if isinstance(node, ast.Import):
                    module.external.append(("import", None, a.name, a.asname))
                else:
                    module.external.append(("from", node.module, a.name, a.asname))
            module.drop.append((node.lineno, node.end_lineno))
        elif _is_main_guard(node) and module.name != ENTRY_MODULE:
            raise BuildError(f"{module.where(node)}: only {ENTRY_MODULE}.py may have an `if __name__ == ...` block; "
                             "in the bundle it would run on startup.")

    # Nested internal imports would survive bundling and fail at runtime.
    top_level = {id(node) for node in body}
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level and _is_internal(node, package):
            raise BuildError(f"{module.where(node)}: internal imports must be at module top level.")

    # Dropping lines is only safe when no kept statement shares a line with a dropped one.
    dropped = {line for first, last in module.drop for line in range(first, last + 1)}
    for node in body:
        span = (node.lineno, node.end_lineno)
        if span not in module.drop and dropped.intersection(range(span[0], span[1] + 1)):
            raise BuildError(f"{module.where(node)}: put imports on their own lines.")

    module.defined = set(_bound_names(body))


def check_names(modules: dict) -> None:
    owners = {}  # bound name -> (identity, label)

    def claim(name, identity, label):
        if name in owners and owners[name][0] != identity:
            raise BuildError(f"Top-level name '{name}' is bound in both {owners[name][1]} and {label}. "
                             "Names must be unique across modules; rename one.")
        owners.setdefault(name, (identity, label))

    for module in modules.values():
        for name in module.defined:
            claim(name, ("def", module.name), module.label)
        for entry in module.external:
            bound, identity = _external_binding(*entry)
            claim(bound, identity, f"{module.label} (import)")

    for module in modules.values():
        for dep, names in module.deps.items():
            dep_module = modules[dep]
            available = dep_module.defined | {_external_binding(*entry)[0] for entry in dep_module.external}
            missing = sorted(names - available)
            if missing:
                raise BuildError(f"{module.label} imports {', '.join(missing)} from {dep_module.label}, which does not define it.")


def order_modules(modules: dict) -> list:
    """Dependency order (depth-first, alphabetical tie-break) with __main__ last."""
    for module in modules.values():
        if ENTRY_MODULE in module.deps:
            raise BuildError(f"{module.label} imports from {ENTRY_MODULE}.py; move shared code into another module.")

    order, state = [], {}

    def visit(name, stack):
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise BuildError("Import cycle: " + " -> ".join(stack + [name]))
        state[name] = "visiting"
        for dep in sorted(modules[name].deps):
            visit(dep, stack + [name])
        state[name] = "done"
        order.append(name)

    for name in sorted(modules, key=lambda n: (n == ENTRY_MODULE, n)):
        visit(name, [])
    return order


# ==========================================
# Rendering
# ==========================================

def _tidy(lines: list) -> list:
    """Strips leading/trailing blank lines and collapses runs of blank lines to two."""
    out, blanks = [], 0
    for line in lines:
        if line.strip():
            if out and blanks:
                out.extend([""] * min(blanks, 2))
            out.append(line)
            blanks = 0
        else:
            blanks += 1
    return out


def render_module(module: Module) -> list:
    dropped = {line for first, last in module.drop for line in range(first, last + 1)}
    kept = [line for number, line in enumerate(module.source.splitlines(), start=1) if number not in dropped]
    return _tidy(kept)


def render_imports(modules: dict) -> list:
    plain, froms = set(), {}
    for module in modules.values():
        for kind, source, name, asname in module.external:
            if kind == "import":
                plain.add((name, asname or ""))
            else:
                froms.setdefault(source, set()).add((name, asname or ""))

    def fmt(name, asname):
        return f"{name} as {asname}" if asname else name

    lines = [f"import {fmt(name, asname)}" for name, asname in sorted(plain)]
    for source in sorted(froms):
        lines.append(f"from {source} import {', '.join(fmt(n, a) for n, a in sorted(froms[source]))}")
    return lines


def _version(init: Module) -> str:
    for node in init.tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
                and isinstance(node.value, ast.Constant)):
            return str(node.value.value)
    return "unknown"


def load_modules(package_dir: Path) -> dict:
    package = package_dir.name
    for required in (INIT_MODULE, ENTRY_MODULE):
        if not (package_dir / f"{required}.py").is_file():
            raise BuildError(f"{package}/{required}.py is required.")
    for child in package_dir.iterdir():
        if child.is_dir() and child.name != "__pycache__" and any(child.glob("*.py")):
            raise BuildError(f"{package}/{child.name}/: subpackages are not supported.")
    return {path.stem: Module(path, package) for path in sorted(package_dir.glob("*.py"))}


def render_bundle(package_dir: Path = DEFAULT_PACKAGE_DIR) -> str:
    """Returns the bundled source for the package, or raises BuildError."""
    package_dir = package_dir.resolve()
    package = package_dir.name
    modules = load_modules(package_dir)
    for module in modules.values():
        analyze(module, package, set(modules))
    check_names(modules)
    order = order_modules(modules)

    init = modules[INIT_MODULE]
    docstring = ""
    if init.tree.body and _is_docstring(init.tree.body[0]):
        docstring = ast.get_source_segment(init.source, init.tree.body[0])

    out = ["#!/usr/bin/env python3"]
    if docstring:
        out.append(docstring)
    out += [
        f"# {package} v{_version(init)} - self-contained, standard library only (Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+).",
        f"# Usage: python3 {package}.py -d <project-dir>   (API key from the OPENAI_API_KEY environment variable)",
        f"# Generated by scripts/build.py from {package}/. Do not edit here: change the modules and rebuild.",
    ]
    future = sorted(set().union(*(m.future for m in modules.values())))
    if future:
        out.append(f"from __future__ import {', '.join(future)}")
    out += ["", *render_imports(modules)]
    for name in order:
        body = render_module(modules[name])
        if body:
            out += ["", "", f"# {'=' * 20} {modules[name].label} {'=' * 20}", "", *body]
    text = "\n".join(out) + "\n"

    # The file is meant to be copy-pasted between machines; non-ASCII characters are
    # the ones clipboards, terminals and editors most often mangle.
    for number, line in enumerate(text.splitlines(), start=1):
        bad = next((ch for ch in line if ord(ch) > 127), None)
        if bad is not None:
            raise BuildError(f"Bundled output line {number} contains non-ASCII {bad!r}; "
                             f"use an escape such as \\u{ord(bad):04x} in the source instead.")

    try:
        compile(text, f"{package}.py", "exec")
    except SyntaxError as e:
        raise BuildError(f"Bundled output does not compile (line {e.lineno}): {e.msg}") from e
    return text


def build(package_dir: Path = DEFAULT_PACKAGE_DIR, out_path: Path = DEFAULT_OUT) -> Path:
    text = render_bundle(package_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out_path.chmod(0o755)
    return out_path


def is_up_to_date(package_dir: Path = DEFAULT_PACKAGE_DIR, out_path: Path = DEFAULT_OUT) -> bool:
    """True if the committed bundle matches what the current sources would build."""
    try:
        with open(out_path, encoding="utf-8", newline="") as f:
            return f.read() == render_bundle(package_dir)
    except FileNotFoundError:
        return False


def run_checks(out_path: Path) -> bool:
    smoke = subprocess.run([sys.executable, str(out_path), "--version"], capture_output=True, text=True)
    if smoke.returncode != 0:
        print(f"Smoke test failed:\n{smoke.stdout}{smoke.stderr}", file=sys.stderr)
        return False
    print(f"Smoke test: {smoke.stdout.strip()}")

    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    for label, extra_env in (("package", {}), ("bundle", {"CHIPPY_BUNDLE": str(out_path)})):
        env = {k: v for k, v in os.environ.items() if k != "CHIPPY_BUNDLE"}
        env.update(extra_env)
        print(f"\nRunning tests against the {label}...")
        if subprocess.run(test_cmd, cwd=ROOT, env=env).returncode != 0:
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bundle the chippy package into a single Python file.")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="Output path (default: chippy.py).")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="Run the test suite against the package and the bundle.")
    parser.add_argument("--verify", action="store_true",
                        help="Don't write anything; exit 1 if the bundle is out of date with the sources.")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            if is_up_to_date(args.package, args.out):
                print(f"{args.out} is up to date.")
                return 0
            print(f"{args.out} is out of date: run `python3 scripts/build.py` and commit the result.", file=sys.stderr)
            return 1
        out_path = build(args.package, args.out)
    except BuildError as e:
        print(f"Build failed: {e}", file=sys.stderr)
        return 1
    print(f"Built {out_path} ({out_path.stat().st_size:,} bytes)")
    if args.check and not run_checks(out_path):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
