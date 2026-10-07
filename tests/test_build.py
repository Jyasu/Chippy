import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from support import ROOT, chippy as C


def _load_build():
    spec = importlib.util.spec_from_file_location("chippy_build", ROOT / "scripts" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def make_package(self, **modules):
        pkg = self.tmp / "chippy"
        pkg.mkdir()
        files = {"__init__": '"""Doc."""\n__version__ = "1"\n', "__main__": "print('main')\n"}
        files.update(modules)
        for name, source in files.items():
            (pkg / f"{name}.py").write_text(textwrap.dedent(source))
        return pkg

    def assert_build_fails(self, message, **modules):
        with self.assertRaises(build.BuildError) as ctx:
            build.build(self.make_package(**modules), self.tmp / "out.py")
        self.assertIn(message, str(ctx.exception))

    def test_real_package_bundles_and_runs(self):
        out = build.build(build.DEFAULT_PACKAGE_DIR, self.tmp / "chippy.py")
        self.assertIsNone(re.search(r"^\s*(from (chippy|\.)\S* import|import chippy)", out.read_text(), re.M))
        result = subprocess.run([sys.executable, str(out), "--version"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(C.__version__, result.stdout)

    def test_verify_detects_stale_bundle(self):
        pkg = self.make_package(a="X = 1\n")
        out = build.build(pkg, self.tmp / "out.py")
        self.assertTrue(build.is_up_to_date(pkg, out))
        (pkg / "a.py").write_text("X = 2\n")
        self.assertFalse(build.is_up_to_date(pkg, out))

    def test_rejects_non_ascii_output(self):
        self.assert_build_fails("non-ASCII", a="CHECK = '✓'\n")

    def test_orders_by_dependency_and_hoists_imports(self):
        pkg = self.make_package(
            a="import os\nfrom chippy.b import B\nA = B + 1\n",
            b="import os\nimport os.path\nB = 1\n",
            __main__="from chippy.a import A\nprint(A)\n",
        )
        out = build.build(pkg, self.tmp / "out.py")
        text = out.read_text()
        self.assertEqual(text.count("import os\n"), 1)
        self.assertLess(text.index("B = 1"), text.index("A = B + 1"))
        self.assertEqual(subprocess.run([sys.executable, str(out)], capture_output=True, text=True).stdout, "2\n")

    def test_rejects_aliased_internal_import(self):
        self.assert_build_fails("cannot be aliased", a="X = 1\n", b="from chippy.a import X as Y\n")

    def test_rejects_module_import(self):
        self.assert_build_fails("instead of importing modules", a="X = 1\n", b="import chippy.a\n")

    def test_rejects_name_collisions(self):
        self.assert_build_fails("'helper' is bound in both", a="def helper(): pass\n", b="def helper(): pass\n")

    def test_rejects_import_shadowing_definition(self):
        self.assert_build_fails("'Path' is bound in both", a="from pathlib import Path\n", b="Path = 1\n")

    def test_rejects_cycles(self):
        self.assert_build_fails("Import cycle", a="from chippy.b import B\nA = 1\n", b="from chippy.a import A\nB = 1\n")

    def test_rejects_nested_internal_import(self):
        self.assert_build_fails("top level", a="X = 1\n", b="def f():\n    from chippy.a import X\n")

    def test_rejects_main_guard_outside_entry(self):
        self.assert_build_fails("only __main__.py", a="if __name__ == '__main__':\n    pass\n")

    def test_rejects_missing_name(self):
        self.assert_build_fails("does not define", a="X = 1\n", b="from chippy.a import Y\n")
