import os

from support import WorkspaceTestCase, chippy as C


class AgentsMdTests(WorkspaceTestCase):
    def test_rules_included(self):
        self.write("AGENTS.md", "Use tabs.")
        prompt, status = C.build_system_context(self.ws)
        self.assertEqual(status, "Detected")
        self.assertIn("Use tabs.", prompt)

    def test_long_file_sent_as_outline(self):
        body = "x" * C.AGENTS_MD_INLINE_CHARS
        self.write("AGENTS.md", f"Intro\n# Style\n{body}\n```\n# not a heading\n```\n## Testing\nRun tests.\n")
        prompt, status = C.build_system_context(self.ws)
        self.assertIn("outline of 3 sections", status)
        self.assertNotIn(body, prompt)
        self.assertEqual(C.outline_markdown((self.ws / "AGENTS.md").read_text()),
                         ["L1-1: (text before the first heading)", "L2-6: # Style", "L7-8: ## Testing"])

    def test_long_file_without_headings_truncated(self):
        self.write("AGENTS.md", "rule\n" * C.AGENTS_MD_INLINE_CHARS)
        prompt, status = C.build_system_context(self.ws)
        self.assertIn("truncated", status)
        self.assertIn("read the rest with read_file", prompt)

    def test_missing(self):
        prompt, status = C.build_system_context(self.ws)
        self.assertEqual(status, "None")

    def test_symlink_outside_workspace_ignored(self):
        (self.root / "id_rsa").write_text("PRIVATE KEY MATERIAL")
        os.symlink(self.root / "id_rsa", self.ws / "AGENTS.md")
        prompt, status = C.build_system_context(self.ws)
        self.assertTrue(status.startswith("Ignored"))
        self.assertNotIn("PRIVATE KEY MATERIAL", prompt)


class InventoryTests(WorkspaceTestCase):
    def test_sorted_depth_limited_and_skips_vcs(self):
        for rel in ("b.txt", "a.txt", "src/z.py", "src/a.py", "a/b/c/kept.txt", "a/b/c/d/too_deep.txt", ".git/HEAD"):
            self.write(rel)
        inventory = C.build_inventory(self.ws)
        self.assertEqual(inventory, ["a.txt", "b.txt", "a/b/c/kept.txt", "src/a.py", "src/z.py"])

    def test_truncation_marker_only_when_over_limit(self):
        for i in range(C.INVENTORY_MAX_FILES):
            self.write(f"f{i:03}.txt")
        self.assertEqual(len(C.build_inventory(self.ws)), C.INVENTORY_MAX_FILES)

        self.write("extra.txt")
        inventory = C.build_inventory(self.ws)
        self.assertEqual(len(inventory), C.INVENTORY_MAX_FILES + 1)
        self.assertEqual(inventory[-1], "... [truncated]")
