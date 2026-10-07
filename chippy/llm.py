"""Zero-dependency client for OpenAI-compatible chat completion endpoints."""

import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request

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


def _parse_response(raw: bytes) -> dict:
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
    return message


def call_llm_api(messages: list, tools: list, settings: Settings) -> dict:
    """Sends the conversation and returns the assistant message. Raises LLMError on failure."""
    payload = {
        "model": settings.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    # Some models (e.g. reasoning models) reject temperature, so it is only sent when set.
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    raw = _post_with_retries(settings.url, json.dumps(payload).encode("utf-8"), headers)
    return _parse_response(raw)
