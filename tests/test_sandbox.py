import os

from support import WorkspaceTestCase, chippy as C


class ResolveScopedPathTests(WorkspaceTestCase):
    def test_relative_path_inside_workspace(self):
        self.assertEqual(C.resolve_scoped_path("a/b.txt", self.ws), self.ws / "a" / "b.txt")

    def test_empty_and_dot_resolve_to_root(self):
        self.assertEqual(C.resolve_scoped_path("", self.ws), self.ws)
        self.assertEqual(C.resolve_scoped_path(".", self.ws), self.ws)

    def test_parent_traversal_denied(self):
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path("../outside.txt", self.ws)
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path("a/../../outside.txt", self.ws)

    def test_absolute_paths(self):
        self.assertEqual(C.resolve_scoped_path(str(self.ws / "x"), self.ws), self.ws / "x")
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path(str(self.root / "x"), self.ws)

    def test_sibling_with_shared_prefix_denied(self):
        (self.root / "ws-evil").mkdir()
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path("../ws-evil/x", self.ws)
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path(str(self.root / "ws-evil" / "x"), self.ws)

    def test_symlink_escape_denied(self):
        (self.root / "secret.txt").write_text("s")
        os.symlink(self.root / "secret.txt", self.ws / "link")
        with self.assertRaises(PermissionError):
            C.resolve_scoped_path("link", self.ws)


class PolicyTests(WorkspaceTestCase):
    def test_git_dir_blocked_for_all_access(self):
        for check in (C.check_listable, C.check_readable, C.check_writable):
            with self.assertRaises(PermissionError):
                check(".git/hooks/pre-commit", self.ws)

    def test_git_dir_block_is_case_insensitive(self):
        with self.assertRaises(PermissionError):
            C.check_writable(".GIT/config", self.ws)

    def test_secret_files_unreadable(self):
        for path in (".env", ".env.local", "keys/server.pem", "id_rsa", "deploy/id_ed25519.pub", ".netrc"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                C.check_readable(path, self.ws)

    def test_example_env_and_normal_files_readable(self):
        for path in (".env.example", "src/app.py", "README.md"):
            with self.subTest(path=path):
                C.check_readable(path, self.ws)

    def test_symlink_to_secret_unreadable(self):
        self.write(".env", "TOKEN=1")
        os.symlink(self.ws / ".env", self.ws / "notes.txt")
        with self.assertRaises(PermissionError):
            C.check_readable("notes.txt", self.ws)

    def test_write_warnings(self):
        for path in ("Makefile", ".github/workflows/ci.yml", ".vscode/tasks.json", "scripts/run.sh", ".env"):
            with self.subTest(path=path):
                self.assertTrue(C.write_warnings(C.resolve_scoped_path(path, self.ws), self.ws))
        self.assertEqual(C.write_warnings(C.resolve_scoped_path("src/app.py", self.ws), self.ws), [])


class SanitizeTests(WorkspaceTestCase):
    def test_escapes_control_characters(self):
        self.assertEqual(C.sanitize("ok\x1b[2Kx\r"), "ok\\x1b[2Kx\\x0d")
        self.assertEqual(C.sanitize("a‮b"), "a\\u202eb")

    def test_keeps_newlines_tabs_and_unicode(self):
        self.assertEqual(C.sanitize("a\n\tb é ✓"), "a\n\tb é ✓")
