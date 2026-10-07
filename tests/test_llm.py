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
        message, urlopen = self.call(http_error(503), urllib.error.URLError("reset"), FakeResponse(api_response("hi")))
        self.assertEqual(message["content"], "hi")
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
