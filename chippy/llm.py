"""Zero-dependency client for OpenAI-compatible chat completion endpoints."""

import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from chippy.config import (
    HTTP_MAX_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    Settings,
)

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(Exception):
    """The API call failed and retrying will not help. The conversation is left intact."""


def _backoff_delay(attempt: int, retry_after=None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), HTTP_MAX_BACKOFF_SECONDS)
        except ValueError:
            pass  # HTTP-date form; fall back to exponential backoff
    return min(2 ** attempt, HTTP_MAX_BACKOFF_SECONDS) * (0.5 + random.random() / 2)


def _post_with_retries(url: str, body: bytes, headers: dict) -> bytes:
    for attempt in range(HTTP_MAX_RETRIES + 1):
        retry_after = None
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            if e.code not in RETRYABLE_STATUS_CODES or attempt == HTTP_MAX_RETRIES:
                raise LLMError(f"API error {e.code}: {detail}") from e
            reason = f"HTTP {e.code}"
            retry_after = e.headers.get("Retry-After") if e.headers else None
        except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, dropped connections
            detail = getattr(e, "reason", None) or e
            if attempt == HTTP_MAX_RETRIES:
                raise LLMError(f"Network error: {detail}") from e
            reason = f"network error: {detail}"
        delay = _backoff_delay(attempt, retry_after)
        print(f"[{reason}] retrying in {delay:.1f}s ({attempt + 1}/{HTTP_MAX_RETRIES})", file=sys.stderr)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _parse_response(raw: bytes) -> tuple:
    """Returns (assistant message, usage dict; empty when the API reports none)."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise LLMError(f"API returned invalid JSON: {e}") from e
    if isinstance(data, dict) and data.get("error"):
        raise LLMError(f"API error: {data['error']}")
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected API response: {str(data)[:500]}") from None
    if not isinstance(message, dict):
        raise LLMError(f"Unexpected API response: {str(data)[:500]}")
    usage = data.get("usage")
    return message, usage if isinstance(usage, dict) else {}


def call_llm_api(messages: list, tools: list, settings: Settings, tool_choice: str = "auto") -> tuple:
    """
    Sends the conversation and returns (assistant message, usage dict).
    Raises LLMError on failure.
    """
    payload = {
        "model": settings.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    # Some models (e.g. reasoning models) reject temperature, so it is only sent when set.
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    raw = _post_with_retries(settings.url, json.dumps(payload).encode("utf-8"), headers)
    return _parse_response(raw)


def _token_count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


@dataclass
class Usage:
    """Token counts summed over the model calls of one request, as reported by the API."""
    calls: int = 0
    explore_calls: int = 0
    reported: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    peak_prompt_tokens: int = 0

    def add(self, usage: dict, explore: bool = False) -> None:
        self.calls += 1
        self.explore_calls += explore
        if not usage:
            return
        self.reported += 1
        prompt = _token_count(usage.get("prompt_tokens"))
        details = usage.get("prompt_tokens_details")
        self.prompt_tokens += prompt
        self.cached_tokens += _token_count(details.get("cached_tokens")) if isinstance(details, dict) else 0
        self.completion_tokens += _token_count(usage.get("completion_tokens"))
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt)

    def summary(self) -> str:
        calls = f"{self.calls} model call{'s' if self.calls != 1 else ''}"
        if self.explore_calls:
            calls += f" ({self.explore_calls} by explore)"
        if not self.reported:
            return f"[Usage] {calls}; the API did not report token counts."
        return (f"[Usage] {calls} | prompt {self.prompt_tokens:,} tokens ({self.cached_tokens:,} cached)"
                f" | completion {self.completion_tokens:,} | largest prompt {self.peak_prompt_tokens:,}")
