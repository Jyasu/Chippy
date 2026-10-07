"""
Loads the code under test and shared test helpers.

Tests run against the chippy package by default. Set CHIPPY_BUNDLE=chippy.py to
run the same tests against the bundled single file instead. Either way, `chippy` below
is one flat namespace, so tests must patch shared modules (builtins.input,
urllib.request.urlopen, time.sleep) rather than names inside a chippy module.
"""

import contextlib
import importlib
import importlib.util
import io
import json
import os
import pkgutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    bundle = os.environ.get("CHIPPY_BUNDLE")
    if bundle:
        spec = importlib.util.spec_from_file_location("chippy_bundle", bundle)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    sys.path.insert(0, str(ROOT))
    import chippy as package
    namespace = types.SimpleNamespace(__version__=package.__version__)
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"chippy.{info.name}")
        vars(namespace).update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    return namespace


chippy = _load()


def quiet():
    """Swallows terminal output from approval prompts and the agent loop."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def api_response(content=None, tool_calls=None) -> bytes:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": message}]}).encode("utf-8")


def tool_call(call_id: str, name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class WorkspaceTestCase(unittest.TestCase):
    """Provides self.ws, an empty workspace inside self.root (so 'outside' paths exist too)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.ws = self.root / "ws"
        self.ws.mkdir()

    def write(self, rel: str, text: str = "", data: bytes = None) -> Path:
        path = self.ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is not None:
            path.write_bytes(data)
        else:
            path.write_text(text, encoding="utf-8")
        return path
