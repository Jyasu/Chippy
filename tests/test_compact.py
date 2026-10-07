import json
import unittest
from unittest import mock

from support import FakeResponse, api_response, chippy as C, quiet, tool_call

SETTINGS = C.Settings(model="m", url="http://llm.invalid")


def exchange(call_id, name, args, result):
    return [{"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, name, args)]},
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": result}]


def conversation():
    """Two requests; the second is in progress with one finished step."""
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first request"},
        *exchange("a", "read_file", {"file_path": "a.py"}, '{"status": "success"}'),
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second request"},
        *exchange("b", "read_file", {"file_path": "b.py"}, '{"status": "success"}'),
    ]


class TailStartTests(unittest.TestCase):
    def test_keeps_last_steps_or_whole_request(self):
        messages = conversation()
        self.assertEqual(C.tail_start(messages, 0), len(messages))
        self.assertEqual(C.tail_start(messages, 1), 5)  # the current request only has one step: keep it all
        messages += exchange("c", "read_file", {"file_path": "c.py"}, "{}")
        self.assertEqual(C.tail_start(messages, 1), 8)
        self.assertEqual(messages[8]["role"], "assistant")


class ElideTests(unittest.TestCase):
    def test_stubs_large_call_arguments_as_valid_json(self):
        content = "x" * C.ELIDE_ARG_MIN_CHARS
        messages = [{"role": "user", "content": "u"},
                    *exchange("w", "write_file", {"file_path": "a.py", "content": content}, '{"status": "success"}')]
        self.assertEqual(C.elide_tool_output(messages, len(messages)), 1)
        args = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["file_path"], "a.py")
        self.assertIn("chars elided", args["content"])
        self.assertEqual(C.elide_tool_output(messages, len(messages)), 0)

    def test_briefs_kept_unless_asked(self):
        bulky = json.dumps({"brief": "x" * C.ELIDE_MIN_CHARS})
        messages = exchange("e", "explore", {"task": "t"}, bulky)
        self.assertEqual(C.elide_tool_output(messages, 2), 0)
        self.assertEqual(C.elide_tool_output(messages, 2, keep_briefs=False), 1)


class CompactHistoryTests(unittest.TestCase):
    def summarize_with(self, *bodies):
        responses = [b if isinstance(b, Exception) else FakeResponse(b) for b in bodies]
        return mock.patch("urllib.request.urlopen", side_effect=responses)

    def test_mid_request_keeps_tail_and_repeats_the_request(self):
        messages = conversation() + exchange("c", "read_file", {"file_path": "c.py"}, "{}")
        with quiet(), self.summarize_with(api_response("SUMMARY")) as urlopen:
            self.assertTrue(C.compact_history(messages, SETTINGS, C.Usage(), keep_steps=1, mid_request=True))
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", sent)
        self.assertIn("first request", sent["messages"][1]["content"])

        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertTrue(messages[1]["content"].startswith(C.COMPACTED_PREFIX))
        self.assertIn("SUMMARY", messages[1]["content"])
        self.assertIn("verbatim:\nsecond request", messages[1]["content"])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "c")

    def test_manual_compaction_replaces_everything(self):
        messages = conversation() + [{"role": "assistant", "content": "second answer"}]
        with quiet(), self.summarize_with(api_response("SUMMARY")):
            C.compact_history(messages, SETTINGS, C.Usage(), keep_steps=0, mid_request=False, instructions="keep the API")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[2]["content"], C.COMPACTED_ACK)
        self.assertNotIn("verbatim", messages[1]["content"])

    def test_failure_leaves_history_untouched(self):
        messages = conversation()
        original = json.dumps(messages)
        with quiet(), self.summarize_with(api_response("")), self.assertRaises(C.LLMError):
            C.compact_history(messages, SETTINGS, C.Usage(), keep_steps=0, mid_request=True)
        self.assertEqual(json.dumps(messages), original)

    def test_recompacting_does_not_repeat_an_old_summary_as_the_request(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": C.COMPACTED_PREFIX + "OLD"},
                    *exchange("a", "read_file", {"file_path": "a.py"}, "{}")]
        with quiet(), self.summarize_with(api_response("NEW")):
            C.compact_history(messages, SETTINGS, C.Usage(), keep_steps=0, mid_request=True)
        self.assertNotIn("verbatim", messages[1]["content"])
        self.assertIn("NEW", messages[1]["content"])

    def test_nothing_to_compact(self):
        self.assertFalse(C.compact_history([{"role": "system", "content": "s"}], SETTINGS, C.Usage(), 0, False))

    def test_transcript_drops_the_middle_when_too_long(self):
        messages = [{"role": "user", "content": f"message {i} " + "x" * 100} for i in range(100)]
        text = C.render_transcript(messages, 2000)
        self.assertLessEqual(len(text), 2100)
        self.assertIn("message 0", text)
        self.assertIn("message 99", text)
        self.assertIn("omitted", text)


class ContextMeterTests(unittest.TestCase):
    def test_reported_size_plus_estimate_for_new_messages(self):
        meter = C.ContextMeter()
        messages = [{"role": "user", "content": "x" * 400}]
        self.assertGreater(meter.estimate(messages, []), 90)
        meter.record(1, {"prompt_tokens": 1000})
        messages.append({"role": "tool", "content": "y" * 4000})
        self.assertAlmostEqual(meter.estimate(messages, []), 2000, delta=50)
        meter.reset()
        self.assertLess(meter.estimate(messages, []), 1200)
