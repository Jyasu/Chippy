"""
Keeps the conversation inside the model's context window.

Two ways to shrink history, cheapest first:
- elision replaces bulky tool output and large tool-call arguments with short stubs;
- compaction asks the model to summarize older messages and replaces them with the summary.
Every rewrite builds the new history first and swaps it in at the end, so an interrupt or
an API failure part-way leaves the conversation untouched.
"""

import json

from chippy.config import (
    CHARS_PER_TOKEN,
    COMPACT_KEEP_STEPS,
    COMPACT_SUMMARY_INPUT_RATIO,
    COMPACT_TARGET_RATIO,
    COMPACT_TRANSCRIPT_ARG_CHARS,
    COMPACT_TRANSCRIPT_RESULT_CHARS,
    COMPACT_TRIGGER_RATIO,
    ELIDE_ARG_MIN_CHARS,
    ELIDE_MIN_CHARS,
    Settings,
)
from chippy.llm import ContextLengthError, LLMError, Usage, call_llm_api, estimate_tokens, prompt_size
from chippy.terminal import clip
from chippy.tools import EXPLORE_TOOL_NAME

SUMMARY_PROMPT = """You compress the conversation of an engineering agent so it can continue the work with a much smaller context.
Write a concise plain-text summary with these sections:
1. User requests and preferences: what the user asked for, in their own words where it matters.
2. Decisions and answers: choices made, and answers the user gave to questions.
3. Work done: files created or changed, and what changed in each.
4. Key findings: file:line locations, facts about the code, and AGENTS.md rules that apply.
5. Current state and next steps: what was in progress and what remains.
Do not copy file contents except very short snippets that are essential; files can be re-read."""

COMPACTED_PREFIX = ("[Context compacted] The earlier conversation was replaced by this summary to save context. "
                    "Re-read files with the tools when you need their exact contents.\n\n")
COMPACTED_ACK = "Understood. I'll continue from this summary."


def _last_user_index(messages: list):
    return max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)


def tail_start(messages: list, keep_steps: int) -> int:
    """
    Index where the verbatim tail begins: the last `keep_steps` model steps of the current
    request, or the request's own user message if it has no more steps than that.
    A tail never starts with a tool result, so cutting there keeps every tool call answered.
    """
    if keep_steps <= 0:
        return len(messages)
    last_user = _last_user_index(messages)
    first = 0 if last_user is None else last_user + 1
    steps = [i for i in range(first, len(messages)) if messages[i].get("role") == "assistant"]
    if len(steps) > keep_steps:
        return steps[-keep_steps]
    return len(messages) if last_user is None else last_user


def _stub_arguments(raw):
    """Tool-call arguments with long string values replaced by a size note. Returns None if nothing changed."""
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return json.dumps({"elided": f"{len(raw):,} chars of malformed arguments"}) if len(raw) >= ELIDE_ARG_MIN_CHARS else None
    if not isinstance(args, dict):
        return None
    stubbed = {key: f"[{len(value):,} chars elided]" if isinstance(value, str) and len(value) >= ELIDE_ARG_MIN_CHARS else value
               for key, value in args.items()}
    return json.dumps(stubbed) if stubbed != args else None


def elide_tool_output(messages: list, end: int, keep_briefs: bool = True) -> int:
    """
    Stubs bulky tool results and large tool-call arguments in messages[:end].
    Explore briefs are kept unless keep_briefs is False. Returns the number of stubs made.
    """
    calls, elided = {}, 0
    for index in range(min(end, len(messages))):
        message = messages[index]
        if message.get("tool_calls"):
            new_calls = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                calls[call.get("id")] = function
                stubbed = _stub_arguments(function.get("arguments"))
                if stubbed is not None:
                    call = {**call, "function": {**function, "arguments": stubbed}}
                    elided += 1
                new_calls.append(call)
            messages[index] = {**message, "tool_calls": new_calls}
        if message.get("role") != "tool" or (keep_briefs and message.get("name") == EXPLORE_TOOL_NAME):
            continue
        if len(message.get("content") or "") < ELIDE_MIN_CHARS:
            continue
        function = calls.get(message.get("tool_call_id"), {})
        signature = f"{function.get('name') or message.get('name')}({clip(str(function.get('arguments') or ''), 200)})"
        messages[index] = {**message, "content": json.dumps({
            "status": "elided",
            "message": f"Output of {signature} was removed to save context. Call the tool again if you still need it.",
        })}
        elided += 1
    return elided


def elide_stale_tool_results(messages: list) -> int:
    """
    Run when a new request starts: elides tool output from before the most recent request.
    Doing it only at this boundary keeps the API's prompt cache valid within a request.
    """
    last_user = _last_user_index(messages)
    return 0 if last_user is None else elide_tool_output(messages, last_user)


def render_transcript(messages: list, max_chars: int) -> str:
    """Plain-text transcript for the summarizer, with long parts clipped and the middle dropped if needed."""
    parts = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            parts.append(f"USER: {message.get('content') or ''}")
        elif role == "assistant":
            if message.get("content"):
                parts.append(f"ASSISTANT: {message['content']}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                parts.append(f"TOOL CALL: {function.get('name')}"
                             f"({clip(str(function.get('arguments') or ''), COMPACT_TRANSCRIPT_ARG_CHARS)})")
        elif role == "tool":
            parts.append(f"TOOL RESULT ({message.get('name')}): "
                         f"{clip(str(message.get('content') or ''), COMPACT_TRANSCRIPT_RESULT_CHARS)}")
    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    return text[:head] + "\n\n[... middle of the conversation omitted ...]\n\n" + text[-(max_chars - head):]


def _summarize(messages: list, settings: Settings, usage: Usage, instructions: str) -> str:
    max_chars = int(settings.context_limit * CHARS_PER_TOKEN * COMPACT_SUMMARY_INPUT_RATIO)
    system = SUMMARY_PROMPT + (f"\n\nThe user asked the summary to focus on: {instructions}" if instructions else "")
    for _ in range(3):
        request = [
            {"role": "system", "content": system},
            {"role": "user", "content": render_transcript(messages, max_chars)},
        ]
        try:
            response, call_usage = call_llm_api(request, [], settings, source="compact")
        except ContextLengthError:
            max_chars //= 2
            continue
        usage.add(call_usage, "compact")
        summary = (response.get("content") or "").strip()
        if not summary:
            raise LLMError("The model returned an empty summary.")
        return summary
    raise LLMError("The conversation is too large to summarize even after shortening it.")


def compact_history(messages: list, settings: Settings, usage: Usage, keep_steps: int,
                    mid_request: bool, instructions: str = "") -> bool:
    """
    Replaces everything between the system prompt and the verbatim tail with a summary.
    If the current request's message gets summarized mid-request, it is repeated
    verbatim so the model still has the user's exact instructions. Returns False if
    there was nothing to compact. Raises LLMError if summarizing fails.
    """
    start = tail_start(messages, keep_steps)
    if start <= 1:
        return False
    head, tail = messages[1:start], messages[start:]
    summary = COMPACTED_PREFIX + _summarize(head, settings, usage, instructions)
    last_user = _last_user_index(messages)
    if mid_request and last_user is not None and 0 < last_user < start:
        request = messages[last_user].get("content") or ""
        # An earlier summary already carries the request, and the new summary is made from it.
        if not request.startswith(COMPACTED_PREFIX):
            summary += f"\n\nThe user's current request, verbatim:\n{request}"

    new = [messages[0], {"role": "user", "content": summary}]
    if (tail and tail[0].get("role") == "user") or (not tail and not mid_request):
        new.append({"role": "assistant", "content": COMPACTED_ACK})
    messages[:] = new + tail
    return True


class ContextMeter:
    """
    Tracks the conversation's size in tokens: the prompt size of the last model call
    plus an estimate for what was added since. Reset it after history is rewritten.
    """

    def __init__(self):
        self._last = None  # (number of messages sent, prompt tokens)

    def record(self, sent_messages: int, usage: dict) -> None:
        tokens = prompt_size(usage)
        self._last = (sent_messages, tokens) if tokens else None

    def reset(self) -> None:
        self._last = None

    def estimate(self, messages: list, tools: list) -> int:
        if self._last and self._last[0] <= len(messages):
            sent, tokens = self._last
            return tokens + estimate_tokens(messages[sent:])
        return estimate_tokens(messages) + estimate_tokens(tools)


def ensure_room(messages: list, settings: Settings, usage: Usage, meter: ContextMeter, tools: list) -> bool:
    """
    Auto-compaction, run before each model call. Near the context limit, older tool
    output is elided first; if that isn't enough, earlier conversation is summarized.
    Returns True if the history changed.
    """
    limit = settings.context_limit
    before = meter.estimate(messages, tools)
    if before < limit * COMPACT_TRIGGER_RATIO:
        return False
    elided = elide_tool_output(messages, tail_start(messages, COMPACT_KEEP_STEPS), keep_briefs=False)
    meter.reset()
    summarized = False
    if meter.estimate(messages, tools) > limit * COMPACT_TARGET_RATIO:
        print("[Context] Near the context limit; summarizing earlier conversation...")
        try:
            summarized = compact_history(messages, settings, usage, COMPACT_KEEP_STEPS, mid_request=True)
        except LLMError as e:
            # Not fatal: the request goes ahead, and an overflow is still recovered from.
            print(f"[Context] Summarizing failed ({clip(str(e), 200)}); continuing without it.")
    after = meter.estimate(messages, tools)
    if elided or summarized:
        print(f"[Context] Auto-compacted: ~{before:,} -> ~{after:,} tokens (limit {limit:,}).")
    return bool(elided or summarized)


def recover_from_overflow(messages: list, settings: Settings, usage: Usage, meter: ContextMeter) -> bool:
    """
    After the API rejected a request as too long: elides all tool output and summarizes
    everything but the system prompt. Returns True if the history changed.
    """
    elided = elide_tool_output(messages, len(messages), keep_briefs=False)
    meter.reset()
    return bool(compact_history(messages, settings, usage, keep_steps=0, mid_request=True) or elided)
