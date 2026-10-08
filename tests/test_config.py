import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import chippy as C, quiet

SETTING_ENVS = ("OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_EXPLORE_MODEL", "LLM_CONTEXT_LIMIT")


class EnvFileTestCase(unittest.TestCase):
    """Runs each test with none of chippy's settings in the environment, and a temp dir for env files."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in SETTING_ENVS:
            os.environ.pop(name, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def env_file(self, text: str, name: str = "chippy.env") -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path


class ParseEnvFileTests(unittest.TestCase):
    def test_formats(self):
        text = (
            "# comment\n"
            "\n"
            "OPENAI_API_KEY=STARK_abc  # trailing comment\r\n"
            "export LLM_MODEL = gemini-2.5-flash\n"
            "LLM_BASE_URL=\"https://api.example/v1/chat/completions\"\n"
            "QUOTED='has # hash'\n"
            "EMPTY=\n"
            "URL_WITH_HASH=https://x/#frag\n"
        )
        self.assertEqual(C.parse_env_file(text), {
            "OPENAI_API_KEY": "STARK_abc",
            "LLM_MODEL": "gemini-2.5-flash",
            "LLM_BASE_URL": "https://api.example/v1/chat/completions",
            "QUOTED": "has # hash",
            "EMPTY": "",
            "URL_WITH_HASH": "https://x/#frag",
        })

    def test_bad_lines_name_the_line(self):
        for text in ("OK=1\njust some text\n", "OK=1\n1BAD=x\n", "OK=1\n=value\n"):
            with self.subTest(text=text), self.assertRaises(C.EnvFileError) as ctx:
                C.parse_env_file(text, "chippy.env")
            self.assertIn("chippy.env:2:", str(ctx.exception))


class LoadEnvFileTests(EnvFileTestCase):
    def test_environment_wins_over_file(self):
        os.environ["LLM_MODEL"] = "from-env"
        os.environ["OPENAI_API_KEY"] = ""  # empty counts as unset
        path = self.env_file("LLM_MODEL=from-file\nOPENAI_API_KEY=k\n")
        loaded, skipped = C.load_env_file(path)
        self.assertEqual((loaded, skipped), (["OPENAI_API_KEY"], ["LLM_MODEL"]))
        self.assertEqual((os.environ["LLM_MODEL"], os.environ["OPENAI_API_KEY"]), ("from-env", "k"))

    def test_missing_file(self):
        self.assertEqual(C.load_env_file(self.dir / "nope.env"), ([], []))
        with self.assertRaises(C.EnvFileError):
            C.load_env_file(self.dir / "nope.env", required=True)

    def test_byte_order_mark_is_ignored(self):
        path = self.dir / "chippy.env"
        path.write_bytes("﻿OPENAI_API_KEY=k\n".encode("utf-8"))
        C.load_env_file(path)
        self.assertEqual(os.environ["OPENAI_API_KEY"], "k")

    def test_default_location_is_beside_the_script(self):
        self.assertEqual(C.default_env_file().name, "chippy.env")
        bundle = os.environ.get("CHIPPY_BUNDLE")
        expected = Path(bundle).resolve().parent if bundle else Path(__file__).resolve().parent.parent
        self.assertEqual(C.default_env_file().parent, expected)


class ParseSettingsTests(EnvFileTestCase):
    def parse(self, *argv):
        return C.parse_settings(["-d", str(self.dir), *argv])

    def test_env_file_supplies_settings(self):
        path = self.env_file("OPENAI_API_KEY=STARK_k\nLLM_BASE_URL=https://genai.example/v1/chat/completions\n"
                             "LLM_MODEL=gemini-2.5-flash\nLLM_CONTEXT_LIMIT=64000\n")
        workspace, settings = self.parse("--env-file", str(path))
        self.assertEqual(workspace, self.dir.resolve())
        self.assertEqual((settings.api_key, settings.api_key_source), ("STARK_k", str(path)))
        self.assertEqual((settings.url, settings.model, settings.context_limit),
                         ("https://genai.example/v1/chat/completions", "gemini-2.5-flash", 64000))

    def test_command_line_wins_and_defaults_apply(self):
        path = self.env_file("LLM_MODEL=from-file\n")
        _, settings = self.parse("--env-file", str(path), "-m", "from-cli")
        self.assertEqual(settings.model, "from-cli")
        self.assertEqual((settings.url, settings.context_limit, settings.explore_model),
                         (C.DEFAULT_URL, C.DEFAULT_CONTEXT_LIMIT, ""))
        self.assertEqual((settings.api_key, settings.api_key_source), ("", ""))

    def test_key_source_reports_environment_override(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        path = self.env_file("OPENAI_API_KEY=file-key\n")
        _, settings = self.parse("--env-file", str(path))
        self.assertEqual(settings.api_key, "env-key")
        self.assertIn("environment variable (overriding", settings.api_key_source)

    def test_bad_input_exits(self):
        cases = (
            ("--env-file", str(self.dir / "missing.env")),
            ("--env-file", str(self.env_file("not a setting\n", "bad.env"))),
            ("--env-file", str(self.env_file("LLM_CONTEXT_LIMIT=lots\n", "limit.env"))),
        )
        for argv in cases:
            with self.subTest(argv=argv), quiet(), self.assertRaises(SystemExit):
                self.parse(*argv)


if __name__ == "__main__":
    unittest.main()
