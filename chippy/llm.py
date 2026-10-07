"""Zero-dependency client for OpenAI-compatible chat completion endpoints."""

import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from chippy.config import (
    CHARS_PER_TOKEN,
    HTTP_MAX_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    Settings,
)

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# How OpenAI-compatible servers word "the prompt is larger than the context window".
CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded", "maximum context length", "context length", "context window",
    "too many tokens", "prompt is too long", "input is too long", "reduce the length",
)


class LLMError(Exception):
    """The API call failed and retrying will not help. The conversation is left intact."""


class ContextLengthError(LLMError):
    """The request did not fit in the model's context window."""


def estimate_tokens(value) -> int:
    """Rough token count for messages or tool schemas, for when the API reports none."""
    return len(json.dumps(value, ensure_ascii=False)) // CHARS_PER_TOKEN


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
            if e.code in (400, 413) and any(marker in detail.lower() for marker in CONTEXT_ERROR_MARKERS):
                raise ContextLengthError(f"API error {e.code}: {detail}") from e
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


def _log_call(settings: Settings, source: str, messages: list, usage: dict) -> None:
    """Appends one JSON line per model call to settings.log_path, for comparing runs."""
    if not settings.log_path:
        return
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "model": settings.model,
        "messages": len(messages),
        "usage": usage,
    }
    try:
        with open(settings.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"[Log] Could not write {settings.log_path}: {e}", file=sys.stderr)


def call_llm_api(messages: list, tools: list, settings: Settings, tool_choice="auto", source: str = "main") -> tuple:
    """
    Sends the conversation and returns (assistant message, usage dict). When the API
    reports no usage, the dict holds 'estimated_prompt_tokens' instead. `source` labels
    the call in the log. Raises LLMError (ContextLengthError if the prompt was too big).
    """
    payload = {"model": settings.model, "messages": messages}
    # Servers reject tool_choice without tools.
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    # Some models (e.g. reasoning models) reject temperature, so it is only sent when set.
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    raw = _post_with_retries(settings.url, json.dumps(payload).encode("utf-8"), headers)
    message, usage = _parse_response(raw)
    if not usage:
        usage = {"estimated_prompt_tokens": estimate_tokens(messages) + estimate_tokens(tools)}
    _log_call(settings, source, messages, usage)
    return message, usage


def token_count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def prompt_size(usage: dict) -> int:
    """The prompt's size in tokens: as reported, or estimated when the API reported nothing."""
    return token_count(usage.get("prompt_tokens")) or token_count(usage.get("estimated_prompt_tokens"))


USAGE_SOURCE_LABELS = {"explore": "by explore", "compact": "for compaction"}


@dataclass
class Usage:
    """Token counts summed over model calls, as reported by the API (or estimated when it reports none)."""
    calls: int = 0
    by_source: dict = field(default_factory=dict)
    estimated: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    peak_prompt_tokens: int = 0

    def add(self, usage: dict, source: str = "main") -> None:
        self.calls += 1
        self.by_source[source] = self.by_source.get(source, 0) + 1
        prompt = prompt_size(usage)
        self.estimated += "prompt_tokens" not in usage
        details = usage.get("prompt_tokens_details")
        self.prompt_tokens += prompt
        self.cached_tokens += token_count(details.get("cached_tokens")) if isinstance(details, dict) else 0
        self.completion_tokens += token_count(usage.get("completion_tokens"))
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt)

    def merge(self, other: "Usage") -> None:
        self.calls += other.calls
        for source, count in other.by_source.items():
            self.by_source[source] = self.by_source.get(source, 0) + count
        self.estimated += other.estimated
        self.prompt_tokens += other.prompt_tokens
        self.cached_tokens += other.cached_tokens
        self.completion_tokens += other.completion_tokens
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, other.peak_prompt_tokens)

    def summary(self, label: str = "[Usage]") -> str:
        calls = f"{self.calls} model call{'s' if self.calls != 1 else ''}"
        extra = [f"{count} {USAGE_SOURCE_LABELS.get(source, source)}"
                 for source, count in sorted(self.by_source.items()) if source != "main"]
        if extra:
            calls += f" ({', '.join(extra)})"
        prompt = f"prompt {'~' if self.estimated else ''}{self.prompt_tokens:,} tokens"
        if self.cached_tokens:
            prompt += f" ({self.cached_tokens:,} cached)"
        parts = [calls, prompt]
        if self.estimated < self.calls:
            parts.append(f"completion {self.completion_tokens:,}")
        parts.append(f"largest prompt {self.peak_prompt_tokens:,}")
        if self.estimated:
            parts.append(f"~ estimated for {self.estimated} call{'s' if self.estimated != 1 else ''} without API usage")
        return f"{label} " + " | ".join(parts)
