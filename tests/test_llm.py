import io
import json
import unittest
import urllib.error
from unittest import mock

from support import FakeResponse, api_response, chippy as C, quiet

URL = "http://llm.invalid/v1/chat/completions"


def http_error(code, headers=None):
    return urllib.error.HTTPError(URL, code, "error", headers or {}, io.BytesIO(b'{"error": "nope"}'))


class CallLlmApiTests(unittest.TestCase):
    def setUp(self):
        self.settings = C.Settings(model="m", url=URL)
        sleep = mock.patch("time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def call(self, *responses, settings=None):
        with quiet(), mock.patch("urllib.request.urlopen", side_effect=list(responses)) as urlopen:
            try:
                return C.call_llm_api([], [], settings or self.settings), urlopen
            finally:
                self.urlopen = urlopen

    def test_retries_transient_errors(self):
        (message, usage), urlopen = self.call(http_error(503), urllib.error.URLError("reset"), FakeResponse(api_response("hi")))
        self.assertEqual(message["content"], "hi")
        self.assertEqual(usage, {})
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_honours_retry_after(self):
        self.call(http_error(429, {"Retry-After": "7"}), FakeResponse(api_response("hi")))
        self.sleep.assert_called_once_with(7.0)

    def test_client_errors_are_not_retried(self):
        with self.assertRaises(C.LLMError):
            self.call(http_error(400))
        self.assertEqual(self.urlopen.call_count, 1)

    def test_gives_up_after_max_retries(self):
        with self.assertRaises(C.LLMError):
            self.call(*[http_error(503)] * (C.HTTP_MAX_RETRIES + 1))
        self.assertEqual(self.urlopen.call_count, C.HTTP_MAX_RETRIES + 1)

    def test_malformed_responses(self):
        for body in (b"not json", b"{}", b'{"choices": []}', b'{"error": {"message": "bad"}}'):
            with self.subTest(body=body), self.assertRaises(C.LLMError):
                self.call(FakeResponse(body))

    def test_request_without_key_or_temperature(self):
        _, urlopen = self.call(FakeResponse(api_response("hi")))
        request = urlopen.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertNotIn("temperature", json.loads(request.data))
        self.assertEqual(urlopen.call_args.kwargs["timeout"], C.HTTP_TIMEOUT_SECONDS)

    def test_request_with_key_and_temperature(self):
        settings = C.Settings(model="m", url=URL, api_key="k", temperature=0.3)
        _, urlopen = self.call(FakeResponse(api_response("hi")), settings=settings)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer k")
        self.assertEqual(json.loads(request.data)["temperature"], 0.3)

    def test_returns_usage_and_sends_tool_choice(self):
        reported = {"prompt_tokens": 10, "completion_tokens": 2}
        with quiet(), mock.patch("urllib.request.urlopen", return_value=FakeResponse(api_response("hi", usage=reported))) as urlopen:
            _, usage = C.call_llm_api([], [], self.settings, tool_choice="none")
        self.assertEqual(usage, reported)
        self.assertEqual(json.loads(urlopen.call_args.args[0].data)["tool_choice"], "none")


class UsageTests(unittest.TestCase):
    def test_sums_calls_and_tracks_largest_prompt(self):
        usage = C.Usage()
        usage.add({"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 80}})
        usage.add({"prompt_tokens": 300, "completion_tokens": 7}, explore=True)
        usage.add({"prompt_tokens": 200, "completion_tokens": None, "prompt_tokens_details": None})
        self.assertEqual((usage.calls, usage.explore_calls), (3, 1))
        self.assertEqual((usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens), (600, 80, 12))
        self.assertEqual(usage.peak_prompt_tokens, 300)
        self.assertIn("3 model calls (1 by explore)", usage.summary())

    def test_unreported_usage(self):
        usage = C.Usage()
        usage.add({})
        self.assertIn("did not report", usage.summary())
