import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import ROOT, FakeResponse, api_response, quiet, tool_call


def _load_bench():
    spec = importlib.util.spec_from_file_location("chippy_bench", ROOT / "scripts" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


class BenchTests(unittest.TestCase):
    def test_checks_fail_on_the_untouched_fixture(self):
        for task in bench.TASKS:
            with self.subTest(task=task.name), tempfile.TemporaryDirectory() as tmp:
                bench.write_fixture(Path(tmp))
                passed, detail = task.check(Path(tmp))
                self.assertFalse(passed)
                self.assertTrue(detail)

    def test_run_task_applies_edits_and_checks(self):
        edits = [
            tool_call(f"c{i}", "edit_file", {"file_path": f"shop/{name}.py", "old_string": "calc_total",
                                             "new_string": "compute_total", "replace_all": True})
            for i, name in enumerate(("pricing", "cart", "report"))
        ]
        responses = [FakeResponse(api_response(tool_calls=edits)), FakeResponse(api_response("done", usage={
            "prompt_tokens": 100, "completion_tokens": 3}))]
        settings = bench.Settings(model="m", url="http://llm.invalid", explore_mode="never")
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=responses):
            result = bench.run_task(bench.TASKS[0], settings)
        self.assertTrue(result["passed"], result["detail"])
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["explore"], "never")
