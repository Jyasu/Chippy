import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

from support import FakeResponse, WorkspaceTestCase, api_response, chippy as C, quiet, tool_call


class RunTurnTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.settings = C.Settings(model="m", url="http://llm.invalid", max_steps=5)
        self.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]

    def respond(self, *bodies):
        return mock.patch("urllib.request.urlopen", side_effect=[FakeResponse(b) for b in bodies])

    def assert_history_valid(self):
        """Every assistant tool call must be answered by a tool message before the next request."""
        expected = []
        for message in self.messages:
            if message["role"] == "tool":
                self.assertEqual(message["tool_call_id"], expected.pop(0))
            else:
                self.assertEqual(expected, [], "unanswered tool calls")
            if message.get("tool_calls"):
                expected = [call["id"] for call in message["tool_calls"]]
        self.assertEqual(expected, [])

    def test_interrupt_mid_batch_keeps_history_valid(self):
        calls = [
            tool_call("c1", "write_file", {"file_path": "one.txt", "content": "1"}),
            tool_call("c2", "write_file", {"file_path": "two.txt", "content": "2"}),
            tool_call("c3", "list_directory", {"directory_path": "."}),
        ]
        with quiet(), self.respond(api_response(tool_calls=calls)), \
                mock.patch("builtins.input", side_effect=["y", KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                C.run_turn(self.messages, self.ws, self.settings)

        self.assert_history_valid()
        statuses = [json.loads(m["content"])["status"] for m in self.messages if m["role"] == "tool"]
        self.assertEqual(statuses, ["success", "cancelled", "cancelled"])
        self.assertTrue((self.ws / "one.txt").exists())

    def test_malformed_tool_arguments_do_not_crash(self):
        bad = api_response(tool_calls=[tool_call("c1", "read_file", '{"file_path": ')])
        with quiet(), self.respond(bad, api_response("done")):
            C.run_turn(self.messages, self.ws, self.settings)
        self.assert_history_valid()
        self.assertEqual(json.loads(self.messages[3]["content"])["status"], "error")
        self.assertEqual(self.messages[-1]["content"], "done")

    def test_step_limit_stops_runaway_loops(self):
        looping = api_response(tool_calls=[tool_call("c", "list_directory", {"directory_path": "."})])
        with quiet(), self.respond(*[looping] * 10) as urlopen:
            C.run_turn(self.messages, self.ws, self.settings)
        self.assertEqual(urlopen.call_count, self.settings.max_steps)
        self.assert_history_valid()

    def test_null_content_is_not_printed_as_none(self):
        out = io.StringIO()
        with redirect_stdout(out), self.respond(api_response(None)):
            C.run_turn(self.messages, self.ws, self.settings)
        self.assertNotIn("None", out.getvalue())

    def test_usage_is_printed(self):
        out = io.StringIO()
        usage = {"prompt_tokens": 1234, "completion_tokens": 5}
        with redirect_stdout(out), self.respond(api_response("hi", usage=usage)):
            C.run_turn(self.messages, self.ws, self.settings)
        self.assertIn("[Usage] 1 model call | prompt 1,234 tokens", out.getvalue())


class ExploreTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.settings = C.Settings(model="main", url="http://llm.invalid", max_steps=5,
                                   explore_model="scout", explore_max_steps=3)
        self.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        self.write("app.py", "def run():\n    pass\n")

    def run_with(self, *bodies, answers=()):
        responses = [b if isinstance(b, Exception) else FakeResponse(b) for b in bodies]
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=responses) as urlopen, \
                mock.patch("builtins.input", side_effect=list(answers)):
            C.run_turn(self.messages, self.ws, self.settings)
        return [json.loads(call.args[0].data) for call in urlopen.call_args_list]

    def explore_result(self):
        return json.loads(next(m for m in self.messages if m.get("name") == "explore")["content"])

    def test_brief_returned_and_explorer_context_isolated(self):
        brief = {"summary": "s", "relevant": [{"file": "app.py", "lines": "1-2", "snippet": "def run():"}],
                 "open_questions": ["Rename or wrap?"]}
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "find run"})]),
            api_response(tool_calls=[tool_call("e1", "search_files", {"pattern": "def run"})]),
            api_response(json.dumps(brief)),
            api_response("done"),
            answers=["wrap it"],
        )
        self.assertEqual([r["model"] for r in requests], ["main", "scout", "scout", "main"])
        self.assertEqual({t["function"]["name"] for t in requests[1]["tools"]}, set(C.READ_ONLY_TOOL_NAMES))
        self.assertIn("app.py:1: def run():", requests[2]["messages"][-1]["content"])

        result = self.explore_result()
        # The explorer's own snippet is replaced by the exact text from disk.
        self.assertEqual(result["brief"]["relevant"][0]["snippet"], "def run():\n    pass\n")
        self.assertEqual(result["brief"]["user_answers"], "wrap it")
        # The search output stayed in the explorer's conversation.
        self.assertNotIn("search_files", json.dumps(self.messages))
        self.assertEqual(self.messages[-1]["content"], "done")

    def test_explorer_cannot_write(self):
        self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response(tool_calls=[tool_call("e1", "write_file", {"file_path": "x.txt", "content": "x"})]),
            api_response('{"summary": "nothing"}'),
            api_response("done"),
        )
        self.assertFalse((self.ws / "x.txt").exists())
        self.assertEqual(self.explore_result()["brief"], {"summary": "nothing"})

    def test_step_limit_forces_a_brief(self):
        looping = api_response(tool_calls=[tool_call("e", "list_directory", {"directory_path": "."})])
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            *[looping] * self.settings.explore_max_steps,
            api_response("plain text brief"),
            api_response("done"),
        )
        final = requests[self.settings.explore_max_steps + 1]
        self.assertEqual(final["tool_choice"], "none")
        self.assertEqual(self.explore_result()["brief"], "plain text brief")

    def test_explorer_api_failure_becomes_tool_error(self):
        error = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad request"))
        self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            error,
            api_response("done"),
        )
        self.assertEqual(self.explore_result()["status"], "error")
        self.assertEqual(self.messages[-1]["content"], "done")

    def test_invalid_snippet_ranges_are_reported(self):
        brief = {"relevant": [{"file": "app.py", "lines": "x"}, {"file": "missing.py", "lines": "1-2"},
                              {"file": "app.py", "lines": "L2"}]}
        self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response(json.dumps(brief)),
            api_response("done"),
        )
        bad_range, missing, single = self.explore_result()["brief"]["relevant"]
        self.assertIn("invalid", bad_range["snippet_error"])
        self.assertIn("does not exist", missing["snippet_error"])
        self.assertEqual(single["snippet"], "    pass\n")

    def test_explorer_reads_are_capped(self):
        self.write("big.py", "".join(f"{i}\n" for i in range(C.EXPLORE_READ_MAX_LINES * 2)))
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response(tool_calls=[tool_call("e1", "read_file", {"file_path": "big.py", "limit": 10_000})]),
            api_response("{}"),
            api_response("done"),
        )
        read = json.loads(requests[2]["messages"][-1]["content"])
        self.assertEqual(read["end_line"], C.EXPLORE_READ_MAX_LINES)

    def test_explorer_reports_early_when_its_context_is_full(self):
        self.settings = C.Settings(model="main", url="http://llm.invalid", context_limit=C.MIN_CONTEXT_LIMIT)
        self.write("big.py", "".join("y" * 150 + "\n" for _ in range(150)))
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response(tool_calls=[tool_call("e1", "read_file", {"file_path": "big.py"})]),
            api_response("{}"),
            api_response("done"),
        )
        self.assertEqual(requests[2]["tool_choice"], "none")

    def test_interrupt_returns_what_was_explored(self):
        responses = [
            FakeResponse(api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"}),
                                                  tool_call("m2", "list_directory", {"directory_path": "."})])),
            FakeResponse(api_response(tool_calls=[tool_call("e1", "search_files", {"pattern": "run"})])),
            KeyboardInterrupt(),
        ]
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=responses), \
                self.assertRaises(KeyboardInterrupt):
            C.run_turn(self.messages, self.ws, self.settings)
        result = self.explore_result()
        self.assertEqual(result["status"], "cancelled")
        self.assertIn("search_files", result["explored"][0])
        self.assertEqual(json.loads(self.messages[-1]["content"])["status"], "cancelled")

    def test_always_mode_forces_explore_first(self):
        self.settings = C.Settings(model="main", url="http://llm.invalid", explore_mode="always")
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response("{}"),
            api_response("done"),
        )
        self.assertEqual(requests[0]["tool_choice"], {"type": "function", "function": {"name": "explore"}})
        self.assertEqual(requests[2]["tool_choice"], "auto")

    def test_never_mode_has_no_explore_tool(self):
        self.settings = C.Settings(model="main", url="http://llm.invalid", explore_mode="never")
        requests = self.run_with(
            api_response(tool_calls=[tool_call("m1", "explore", {"task": "t"})]),
            api_response("done"),
        )
        self.assertNotIn("explore", {t["function"]["name"] for t in requests[0]["tools"]})
        self.assertIn("Unknown tool", self.explore_result()["message"])
        prompt, _ = C.build_system_context(self.ws, "never")
        self.assertNotIn("explore", prompt)


class RepeatedReadTests(WorkspaceTestCase):
    def test_identical_read_of_unchanged_file_is_short(self):
        self.write("a.py", "x = 1\n")
        read = tool_call("r1", "read_file", {"file_path": "a.py"})
        again = tool_call("r2", "read_file", {"file_path": "a.py"})
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        responses = [FakeResponse(api_response(tool_calls=[read])), FakeResponse(api_response(tool_calls=[again])),
                     FakeResponse(api_response("done"))]
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=responses):
            C.run_turn(messages, self.ws, C.Settings(model="m", url="http://llm.invalid"))
        first, second = (json.loads(m["content"]) for m in messages if m["role"] == "tool")
        self.assertEqual(first["content"], "x = 1\n")
        self.assertTrue(second["unchanged"])

    def test_tracker_rereads_after_a_change(self):
        path = self.write("a.py", "x = 1\n")
        tracker = C.ReadTracker()
        tracker.dispatch("read_file", {"file_path": "a.py"}, self.ws)
        path.write_text("x = 22\n")
        self.assertEqual(tracker.dispatch("read_file", {"file_path": "a.py"}, self.ws)["content"], "x = 22\n")
        self.assertTrue(tracker.dispatch("read_file", {"file_path": "a.py"}, self.ws)["unchanged"])
        tracker.clear()
        self.assertIn("content", tracker.dispatch("read_file", {"file_path": "a.py"}, self.ws))


class ContextLimitTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.settings = C.Settings(model="m", url="http://llm.invalid", context_limit=C.MIN_CONTEXT_LIMIT)

    def run_with(self, messages, *bodies):
        responses = [b if isinstance(b, Exception) else FakeResponse(b) for b in bodies]
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            C.run_turn(messages, self.ws, self.settings)
        return [json.loads(call.args[0].data) for call in urlopen.call_args_list]

    def test_auto_compaction_elides_first(self):
        big = json.dumps({"status": "success", "content": "x" * C.MIN_CONTEXT_LIMIT * C.CHARS_PER_TOKEN})
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "first"},
                    {"role": "assistant", "content": None, "tool_calls": [tool_call("a", "read_file", {"file_path": "a"})]},
                    {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": big},
                    {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]
        requests = self.run_with(messages, api_response("done"))
        self.assertEqual(len(requests), 1)  # eliding was enough: no summary call
        self.assertEqual(json.loads(messages[3]["content"])["status"], "elided")

    def test_auto_compaction_summarizes_when_eliding_is_not_enough(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "first " + "x" * C.MIN_CONTEXT_LIMIT * C.CHARS_PER_TOKEN},
                    {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]
        requests = self.run_with(messages, api_response("SUMMARY"), api_response("done"))
        self.assertNotIn("tools", requests[0])
        self.assertIn("SUMMARY", requests[1]["messages"][1]["content"])
        self.assertEqual(requests[1]["messages"][-1], {"role": "user", "content": "second"})
        self.assertEqual(messages[-1]["content"], "done")

    def test_overflow_is_recovered_by_compacting(self):
        overflow = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"error": {"code": "context_length_exceeded"}}'))
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]
        requests = self.run_with(messages, overflow, api_response("SUMMARY"), api_response("done"))
        self.assertEqual(len(requests), 3)
        retried = requests[2]["messages"]
        self.assertEqual([m["role"] for m in retried], ["system", "user"])
        self.assertIn("verbatim:\nsecond", retried[1]["content"])
        self.assertEqual(messages[-1]["content"], "done")


class ElisionTests(unittest.TestCase):
    def tool_exchange(self, call_id, name, content):
        call = tool_call(call_id, name, {"file_path": "big.py"})
        return [{"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}]

    def test_elides_bulky_results_older_than_the_last_request(self):
        bulky = json.dumps({"status": "success", "content": "x" * C.ELIDE_MIN_CHARS})
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "first"},
            *self.tool_exchange("a", "read_file", bulky),
            *self.tool_exchange("b", "explore", bulky),
            *self.tool_exchange("c", "list_directory", '{"status": "success"}'),
            {"role": "user", "content": "second"},
            *self.tool_exchange("d", "read_file", bulky),
        ]
        self.assertEqual(C.elide_stale_tool_results(messages), 1)
        old_read = json.loads(messages[3]["content"])
        self.assertEqual(old_read["status"], "elided")
        self.assertIn("read_file(", old_read["message"])
        self.assertIn("big.py", old_read["message"])
        self.assertEqual(messages[5]["content"], bulky)   # explore brief kept
        self.assertEqual(messages[-1]["content"], bulky)  # most recent request kept
        self.assertEqual(C.elide_stale_tool_results(messages), 0)


class RunAgentTests(WorkspaceTestCase):
    def test_api_failure_returns_to_prompt(self):
        settings = C.Settings(model="m", url="http://llm.invalid")
        error = urllib.error.HTTPError(settings.url, 400, "bad", {}, io.BytesIO(b"bad request"))
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=[error, FakeResponse(api_response("ok"))]) as urlopen, \
                mock.patch("builtins.input", side_effect=["first", "second", EOFError()]):
            C.run_agent(self.ws, settings)
        # The failed message was dropped, so the retry sends system + "second" only.
        sent = json.loads(urlopen.call_args.args[0].data)["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user"])
        self.assertEqual(sent[1]["content"], "second")

    def test_compact_and_usage_commands(self):
        settings = C.Settings(model="m", url="http://llm.invalid")
        responses = [FakeResponse(api_response("hi", usage={"prompt_tokens": 50, "completion_tokens": 2})),
                     FakeResponse(api_response("SUMMARY")), FakeResponse(api_response("ok"))]
        out = io.StringIO()
        with redirect_stdout(out), mock.patch("urllib.request.urlopen", side_effect=responses) as urlopen, \
                mock.patch("builtins.input", side_effect=["hello", "/compact keep names", "/usage", "next", EOFError()]):
            C.run_agent(self.ws, settings)
        summary_request, after = (json.loads(call.args[0].data) for call in urlopen.call_args_list[1:])
        self.assertIn("keep names", summary_request["messages"][0]["content"])
        self.assertEqual([m["role"] for m in after["messages"]], ["system", "user", "assistant", "user"])
        self.assertIn("SUMMARY", after["messages"][1]["content"])
        self.assertIn("[Session] 2 model calls (1 for compaction)", out.getvalue())
        self.assertIn("after compacting", out.getvalue())

    def test_banner_shows_endpoint_and_key_source(self):
        for settings, expected in (
            (C.Settings(model="m", url="http://llm.invalid"), "API Key           : NOT SET"),
            (C.Settings(model="m", url="http://llm.invalid", api_key="k", api_key_source="/opt/chippy.env"),
             "API Key           : from /opt/chippy.env"),
        ):
            out = io.StringIO()
            with redirect_stdout(out), mock.patch("builtins.input", side_effect=EOFError()):
                C.run_agent(self.ws, settings)
            self.assertIn("Endpoint          : http://llm.invalid", out.getvalue())
            self.assertIn(expected, out.getvalue())

    def test_unknown_slash_text_is_sent_to_the_model(self):
        settings = C.Settings(model="m", url="http://llm.invalid")
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=[FakeResponse(api_response("ok"))]) as urlopen, \
                mock.patch("builtins.input", side_effect=["/etc/hosts is what?", EOFError()]):
            C.run_agent(self.ws, settings)
        self.assertEqual(json.loads(urlopen.call_args.args[0].data)["messages"][-1]["content"], "/etc/hosts is what?")
