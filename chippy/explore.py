"""The explore sub-agent: gathers context in a fresh conversation and returns a compact brief."""

import dataclasses
import json
from pathlib import Path

from chippy.config import Settings
from chippy.context import build_explorer_prompt
from chippy.llm import LLMError, Usage, call_llm_api
from chippy.terminal import clip, sanitize
from chippy.tools import (
    READ_ONLY_TOOL_NAMES,
    READ_ONLY_TOOLS,
    dispatch_tool,
    parse_tool_arguments,
    tool_result_message,
)

FINAL_STEP_PROMPT = "Step limit reached. Reply now with the JSON brief, based on what you found so far."
SKIPPED_ANSWER = ("The user skipped these questions. Decide using your best judgement and state your "
                  "assumptions, or ask the user in your reply.")


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


def _finish(reply: str, steps: int) -> dict:
    print(f"[Explore] Done in {steps} step{'s' if steps != 1 else ''}; brief is {len(reply):,} chars.")
    brief = parse_brief(reply)
    if brief is None:
        return {"status": "success", "steps": steps, "brief": reply or "(the explorer returned nothing)"}
    questions = brief.get("open_questions")
    questions = [q for q in questions if q] if isinstance(questions, list) else []
    if questions:
        brief["user_answers"] = _ask_open_questions(questions) or SKIPPED_ANSWER
    return {"status": "success", "steps": steps, "brief": brief}


def run_explore(raw_arguments, workspace: Path, settings: Settings, usage: Usage) -> dict:
    """
    Runs the explore tool: a read-only sub-agent with its own conversation and step budget.
    Only its brief reaches the caller's context; the files it read do not.
    """
    args, error = parse_tool_arguments(raw_arguments)
    if error:
        return error
    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return {"status": "error", "message": "'task' must be a non-empty string."}

    sub_settings = dataclasses.replace(settings, model=settings.explore_model or settings.model)
    messages = [
        {"role": "system", "content": build_explorer_prompt(workspace)},
        {"role": "user", "content": task},
    ]
    print(f"[Explore] Using {sanitize(sub_settings.model)}")
    try:
        for step in range(1, settings.explore_max_steps + 2):
            final = step > settings.explore_max_steps
            if final:
                messages.append({"role": "user", "content": FINAL_STEP_PROMPT})
            response, call_usage = call_llm_api(messages, READ_ONLY_TOOLS, sub_settings,
                                                tool_choice="none" if final else "auto")
            usage.add(call_usage, explore=True)
            messages.append(response)

            tool_calls = response.get("tool_calls") or []
            if final or not tool_calls:
                return _finish(response.get("content") or "", step)
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                raw_args = function.get("arguments")
                print(f"  [explore] {sanitize(name)}({sanitize(clip(str(raw_args), 120))})")
                result = dispatch_tool(name, raw_args, workspace, allowed=READ_ONLY_TOOL_NAMES)
                messages.append(tool_result_message(call, result))
    except LLMError as e:
        # Returned rather than raised: the caller's tool call still needs a result.
        return {"status": "error", "message": f"The explore sub-agent failed: {e}. Gather context directly instead."}
    raise AssertionError("unreachable")
