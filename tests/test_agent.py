import io
import json
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
