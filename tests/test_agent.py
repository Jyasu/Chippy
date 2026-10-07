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
        self.assertEqual(result["brief"]["relevant"][0]["file"], "app.py")
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
