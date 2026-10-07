"""The explore sub-agent: gathers context in a fresh conversation and returns a compact brief."""

import dataclasses
import json
import re
from pathlib import Path

from chippy.config import (
    EXPLORE_CONTEXT_RATIO,
    EXPLORE_READ_MAX_LINES,
    EXPLORE_SNIPPET_MAX_LINES,
    Settings,
)
from chippy.context import build_explorer_prompt
from chippy.llm import LLMError, Usage, call_llm_api, estimate_tokens
from chippy.terminal import clip, sanitize
from chippy.tools import (
    READ_ONLY_TOOL_NAMES,
    READ_ONLY_TOOLS,
    ReadTracker,
    parse_tool_arguments,
    tool_read_file,
    tool_result_message,
)

FINAL_STEP_PROMPT = "Stop exploring now and reply with the JSON brief, based on what you found so far."
SKIPPED_ANSWER = ("The user skipped these questions. Decide using your best judgement and state your "
                  "assumptions, or ask the user in your reply.")
_LINE_RANGE = re.compile(r"\s*L?(\d+)\s*(?:-\s*L?(\d+))?\s*")


class ExploreInterrupted(KeyboardInterrupt):
    """Ctrl-C during explore. `result` lists what the explorer had looked at, for the caller's history."""

    def __init__(self, result: dict):
        super().__init__()
        self.result = result


def parse_brief(text: str):
    """The brief as a dict, or None if the reply is not a JSON object (an optional code fence is allowed)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        brief = json.loads(text)
    except ValueError:
        return None
    return brief if isinstance(brief, dict) else None


def attach_snippets(brief: dict, workspace: Path) -> None:
    """
    Fills each 'relevant' entry's snippet with the exact text of its line range, read
    from disk by the harness, so snippets are correct by construction and the explorer
    never spends tokens copying code.
    """
    relevant = brief.get("relevant")
    if not isinstance(relevant, list):
        return
    for entry in relevant:
        if not isinstance(entry, dict):
            continue
        entry.pop("snippet", None)
        match = _LINE_RANGE.fullmatch(str(entry.get("lines", "")))
        start = int(match[1]) if match else 0
        end = int(match[2] or match[1]) if match else 0
        if not isinstance(entry.get("file"), str) or start < 1 or end < start:
            entry["snippet_error"] = "No snippet: 'file' or the 'start-end' line range is missing or invalid."
            continue
        result = tool_read_file(workspace, entry["file"], offset=start, limit=min(end - start + 1, EXPLORE_SNIPPET_MAX_LINES))
        if result.get("status") != "success":
            entry["snippet_error"] = f"No snippet: {result.get('message')}"
            continue
        entry["snippet"] = result["content"]
        if result["end_line"] < end:
            entry["snippet_note"] = f"Only lines {start}-{result['end_line']} are included; read the rest with read_file."


def _cap_read(raw_arguments):
    """The explorer's read_file calls are capped at EXPLORE_READ_MAX_LINES, so one read can't fill its context."""
    args, error = parse_tool_arguments(raw_arguments)
    if error:
        return raw_arguments
    try:
        limit = int(args.get("limit", EXPLORE_READ_MAX_LINES))
    except (TypeError, ValueError):
        return raw_arguments
    return {**args, "limit": max(1, min(limit, EXPLORE_READ_MAX_LINES))}


def _ask_open_questions(questions: list):
    """Puts the explorer's blocking questions to the user. Returns their answer, or None if skipped."""
    print("\n[Explore] Questions before the agent continues:")
    for number, question in enumerate(questions, start=1):
        print(f"  {number}. {sanitize(clip(str(question), 500))}")
    try:
        answer = input("Answer (Enter to let the agent decide) > ").strip()
    except EOFError:
        return None
    return answer or None


def _finish(reply: str, steps: int, workspace: Path) -> dict:
    print(f"[Explore] Done in {steps} step{'s' if steps != 1 else ''}.")
    brief = parse_brief(reply)
    if brief is None:
        return {"status": "success", "steps": steps, "brief": reply or "(the explorer returned nothing)"}
    attach_snippets(brief, workspace)
    questions = brief.get("open_questions")
    questions = [q for q in questions if q] if isinstance(questions, list) else []
    if questions:
        brief["user_answers"] = _ask_open_questions(questions) or SKIPPED_ANSWER
    return {"status": "success", "steps": steps, "brief": brief}


def run_explore(raw_arguments, workspace: Path, settings: Settings, usage: Usage) -> dict:
    """
    Runs the explore tool: a read-only sub-agent with its own conversation and step budget.
    Only its brief reaches the caller's context; the files it read do not. It is told to
    report early if its own context reaches EXPLORE_CONTEXT_RATIO of the limit.
    """
    args, error = parse_tool_arguments(raw_arguments)
    if error:
        return error
    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return {"status": "error", "message": "'task' must be a non-empty string."}

    sub_settings = dataclasses.replace(settings, model=settings.explore_model or settings.model)
    budget = settings.context_limit * EXPLORE_CONTEXT_RATIO
    messages = [
        {"role": "system", "content": build_explorer_prompt(workspace)},
        {"role": "user", "content": task},
    ]
    reads, explored = ReadTracker(), []
    print(f"[Explore] Using {sanitize(sub_settings.model)}")
    try:
        for step in range(1, settings.explore_max_steps + 2):
            final = step > settings.explore_max_steps or estimate_tokens(messages) > budget
            if final:
                messages.append({"role": "user", "content": FINAL_STEP_PROMPT})
            response, call_usage = call_llm_api(messages, READ_ONLY_TOOLS, sub_settings,
                                                tool_choice="none" if final else "auto", source="explore")
            usage.add(call_usage, "explore")
            messages.append(response)

            tool_calls = response.get("tool_calls") or []
            if final or not tool_calls:
                return _finish(response.get("content") or "", step, workspace)
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                raw_args = _cap_read(function.get("arguments")) if name == "read_file" else function.get("arguments")
                signature = f"{name}({clip(str(function.get('arguments')), 120)})"
                print(f"  [explore] {sanitize(signature)}")
                result = reads.dispatch(name, raw_args, workspace, allowed=READ_ONLY_TOOL_NAMES)
                explored.append(signature)
                messages.append(tool_result_message(call, result))
    except LLMError as e:
        # Returned rather than raised: the caller's tool call still needs a result.
        return {"status": "error", "message": f"The explore sub-agent failed: {e}. Gather context directly instead."}
    except KeyboardInterrupt:
        raise ExploreInterrupted({
            "status": "cancelled",
            "message": "The user interrupted explore before it finished. It had looked at these, "
                       "so you know where to start if you continue:",
            "explored": explored,
        }) from None
    raise AssertionError("unreachable")
