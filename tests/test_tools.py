import inspect
import os
import stat
from unittest import mock

from support import WorkspaceTestCase, chippy as C, quiet


def approve(*answers):
    return mock.patch("builtins.input", side_effect=list(answers))


class ReadFileTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.write("big.txt", "".join(f"line {i}\n" for i in range(1, 1201)))

    def test_pages_through_long_files(self):
        first = C.tool_read_file(self.ws, "big.txt")
        self.assertEqual((first["start_line"], first["end_line"]), (1, C.READ_MAX_LINES))
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_offset"], C.READ_MAX_LINES + 1)

        last = C.tool_read_file(self.ws, "big.txt", offset=1001)
        self.assertEqual((last["start_line"], last["end_line"]), (1001, 1200))
        self.assertFalse(last["has_more"])
        self.assertTrue(last["content"].startswith("line 1001\n"))
        self.assertTrue(last["content"].endswith("line 1200\n"))

    def test_limit_is_clamped(self):
        result = C.tool_read_file(self.ws, "big.txt", limit=10_000)
        self.assertEqual(result["end_line"], C.READ_MAX_LINES)

    def test_offset_past_end(self):
        result = C.tool_read_file(self.ws, "big.txt", offset=5000)
        self.assertEqual(result["status"], "error")

    def test_string_numbers_accepted(self):
        result = C.tool_read_file(self.ws, "big.txt", offset="3", limit="2")
        self.assertEqual(result["content"], "line 3\nline 4\n")

    def test_binary_refused(self):
        self.write("img.bin", data=b"\x89PNG\x00\x00")
        self.assertEqual(C.tool_read_file(self.ws, "img.bin")["status"], "error")

    def test_long_line_clipped_without_losing_following_lines(self):
        self.write("min.js", "x" * (C.READ_MAX_LINE_CHARS * 3) + "\nnext\n")
        result = C.tool_read_file(self.ws, "min.js")
        first, second = result["content"].splitlines()
        self.assertIn("line clipped", first)
        self.assertLess(len(first), C.READ_MAX_LINE_CHARS + 100)
        self.assertEqual(second, "next")

    def test_crlf_is_normalized(self):
        self.write("win.txt", data=b"a\r\nb\r\n")
        self.assertEqual(C.tool_read_file(self.ws, "win.txt")["content"], "a\nb\n")

    def test_secret_denied(self):
        self.write(".env", "TOKEN=1")
        self.assertEqual(C.tool_read_file(self.ws, ".env")["status"], "denied")


class WriteFileTests(WorkspaceTestCase):
    def test_rejection_leaves_disk_untouched(self):
        with quiet(), approve("n"):
            result = C.tool_write_file(self.ws, "new.txt", "hello\n")
        self.assertEqual(result["status"], "denied")
        self.assertFalse((self.ws / "new.txt").exists())

    def test_approved_write_creates_parent_dirs(self):
        with quiet(), approve("y"):
            result = C.tool_write_file(self.ws, "a/b/new.txt", "hello\n")
        self.assertEqual(result["status"], "success")
        self.assertEqual((self.ws / "a/b/new.txt").read_text(), "hello\n")

    def test_preserves_crlf_and_mode(self):
        path = self.write("run.txt", data=b"a\r\nb\r\n")
        os.chmod(path, 0o755)
        with quiet(), approve("y"):
            C.tool_write_file(self.ws, "run.txt", "a\nc\n")
        self.assertEqual(path.read_bytes(), b"a\r\nc\r\n")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)

    def test_no_change_skips_prompt(self):
        self.write("same.txt", "x\n")
        with approve() as prompt:
            result = C.tool_write_file(self.ws, "same.txt", "x\n")
        self.assertEqual(result["status"], "success")
        prompt.assert_not_called()

    def test_git_dir_denied_without_prompt(self):
        with approve() as prompt:
            result = C.tool_write_file(self.ws, ".git/hooks/pre-commit", "#!/bin/sh\n")
        self.assertEqual(result["status"], "denied")
        prompt.assert_not_called()

    def test_eof_at_prompt_denies(self):
        with quiet(), approve(EOFError()):
            result = C.tool_write_file(self.ws, "new.txt", "x")
        self.assertEqual(result["status"], "denied")

    def test_view_full_diff_then_approve(self):
        content = "".join(f"{i}\n" for i in range(C.APPROVAL_DIFF_PREVIEW_LINES * 2))
        with quiet(), approve("v", "y") as prompt:
            result = C.tool_write_file(self.ws, "long.txt", content)
        self.assertEqual(result["status"], "success")
        self.assertEqual(prompt.call_count, 2)

    def test_approval_output_escapes_terminal_sequences(self):
        with mock.patch("builtins.print") as printed, approve("n"):
            C.tool_write_file(self.ws, "x.txt", "safe\x1b[2Kspoof\n")
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertNotIn("\x1b", output)
        self.assertIn("\\x1b", output)


class EditFileTests(WorkspaceTestCase):
    def test_unique_replacement(self):
        path = self.write("a.py", "x = 1\ny = 2\n")
        with quiet(), approve("y"):
            result = C.tool_edit_file(self.ws, "a.py", "y = 2", "y = 3")
        self.assertEqual(result["status"], "success")
        self.assertEqual(path.read_text(), "x = 1\ny = 3\n")

    def test_not_found_and_ambiguous(self):
        self.write("a.py", "v = 1\nv = 1\n")
        with approve() as prompt:
            self.assertIn("not found", C.tool_edit_file(self.ws, "a.py", "nope", "x")["message"])
            self.assertIn("2 times", C.tool_edit_file(self.ws, "a.py", "v = 1", "v = 2")["message"])
        prompt.assert_not_called()

    def test_replace_all(self):
        path = self.write("a.py", "v = 1\nv = 1\n")
        with quiet(), approve("y"):
            C.tool_edit_file(self.ws, "a.py", "v = 1", "v = 2", replace_all=True)
        self.assertEqual(path.read_text(), "v = 2\nv = 2\n")

    def test_preserves_crlf(self):
        path = self.write("w.txt", data=b"a\r\nb\r\n")
        with quiet(), approve("y"):
            C.tool_edit_file(self.ws, "w.txt", "a\nb", "a\nc")
        self.assertEqual(path.read_bytes(), b"a\r\nc\r\n")

    def test_invalid_utf8_refused(self):
        self.write("latin.txt", data=b"caf\xe9\n")
        self.assertIn("UTF-8", C.tool_edit_file(self.ws, "latin.txt", "caf", "x")["message"])

    def test_missing_file(self):
        self.assertEqual(C.tool_edit_file(self.ws, "nope.txt", "a", "b")["status"], "error")


class ListDirectoryTests(WorkspaceTestCase):
    def test_lists_sorted_entries(self):
        self.write("b.txt", "12")
        (self.ws / "a").mkdir()
        entries = C.tool_list_directory(self.ws, ".")["entries"]
        self.assertEqual([e["name"] for e in entries], ["a", "b.txt"])
        self.assertEqual(entries[1]["size_bytes"], 2)

    def test_broken_and_escaping_symlinks(self):
        self.write("ok.txt", "x")
        os.symlink(self.ws / "missing", self.ws / "broken")
        (self.root / "outside.txt").write_text("secret")
        os.symlink(self.root / "outside.txt", self.ws / "escape")
        result = C.tool_list_directory(self.ws, "")
        self.assertEqual(result["status"], "success")
        by_name = {e["name"]: e for e in result["entries"]}
        self.assertIn("error", by_name["broken"])
        self.assertTrue(by_name["escape"]["outside_workspace"])
        self.assertNotIn("size_bytes", by_name["escape"])

    def test_git_listing_denied(self):
        (self.ws / ".git").mkdir()
        self.assertEqual(C.tool_list_directory(self.ws, ".git")["status"], "denied")


class SearchFilesTests(WorkspaceTestCase):
    def test_finds_matches_with_line_numbers(self):
        self.write("src/a.py", "import os\ndef run():\n    pass\n")
        self.write("src/b.py", "def run_all():\n")
        self.write("notes.txt", "def run\n")
        result = C.tool_search_files(self.ws, r"def run", glob="*.py")
        self.assertEqual(result["matches"], ["src/a.py:2: def run():", "src/b.py:1: def run_all():"])
        self.assertEqual(result["files_matched"], 2)

    def test_path_glob_and_ignore_case(self):
        self.write("src/a.py", "TODO\n")
        self.write("lib/a.py", "todo\n")
        result = C.tool_search_files(self.ws, "todo", glob="lib/*", ignore_case=True)
        self.assertEqual(result["matches"], ["lib/a.py:1: todo"])
        self.assertEqual(C.tool_search_files(self.ws, "TODO", path="src/a.py")["matches"], ["src/a.py:1: TODO"])

    def test_skips_secrets_binary_git_and_escaping_symlinks(self):
        self.write(".env", "needle\n")
        self.write("img.bin", data=b"needle\x00")
        self.write(".git/config", "needle\n")
        (self.root / "outside.txt").write_text("needle\n")
        os.symlink(self.root / "outside.txt", self.ws / "escape.txt")
        result = C.tool_search_files(self.ws, "needle")
        self.assertEqual(result["matches"], [])
        self.assertIn("No matches", result["note"])

    def test_match_cap(self):
        self.write("many.txt", "hit\n" * (C.SEARCH_MAX_MATCHES + 5))
        result = C.tool_search_files(self.ws, "hit")
        self.assertEqual(len(result["matches"]), C.SEARCH_MAX_MATCHES)
        self.assertTrue(result["truncated"])

    def test_invalid_regex_and_escape(self):
        self.assertIn("Invalid regular expression", C.tool_search_files(self.ws, "(")["message"])
        self.assertEqual(C.tool_search_files(self.ws, "x", path="..")["status"], "denied")


class DispatchTests(WorkspaceTestCase):
    def test_allowed_restricts_tools(self):
        args = '{"file_path": "a", "content": "x"}'
        result = C.dispatch_tool("write_file", args, self.ws, allowed=C.READ_ONLY_TOOL_NAMES)
        self.assertIn("Unknown tool", result["message"])
        self.assertFalse((self.ws / "a").exists())

    def test_malformed_json(self):
        result = C.dispatch_tool("read_file", '{"file_path": ', self.ws)
        self.assertEqual(result["status"], "error")
        self.assertIn("JSON", result["message"])

    def test_non_object_arguments(self):
        self.assertEqual(C.dispatch_tool("read_file", "null", self.ws)["status"], "error")
        self.assertEqual(C.dispatch_tool("read_file", "[1]", self.ws)["status"], "error")

    def test_unknown_tool(self):
        self.assertIn("Unknown tool", C.dispatch_tool("run_shell", "{}", self.ws)["message"])

    def test_missing_and_unexpected_arguments(self):
        self.assertIn("Invalid arguments", C.dispatch_tool("read_file", "{}", self.ws)["message"])
        self.assertIn("Invalid arguments", C.dispatch_tool("read_file", '{"file_path": "a", "x": 1}', self.ws)["message"])

    def test_dispatches_valid_call(self):
        self.write("a.txt", "hi\n")
        self.assertEqual(C.dispatch_tool("read_file", '{"file_path": "a.txt"}', self.ws)["content"], "hi\n")

    def test_schemas_match_handler_signatures(self):
        for tool in C.TOOLS:
            schema = tool["function"]
            params = list(inspect.signature(C.TOOL_HANDLERS[schema["name"]]).parameters)[1:]
            with self.subTest(tool=schema["name"]):
                self.assertEqual(sorted(schema["parameters"]["properties"]), sorted(params))
