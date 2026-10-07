"""Interactive loop: sends the conversation to the model and executes its tool calls."""

import json
import sys
from pathlib import Path

from chippy.config import ELIDE_MIN_CHARS, Settings
from chippy.context import build_system_context
from chippy.explore import run_explore
from chippy.llm import LLMError, Usage, call_llm_api
from chippy.terminal import clip, sanitize
from chippy.tools import AGENT_TOOLS, EXPLORE_TOOL_NAME, dispatch_tool, tool_result_message

CANCELLED_RESULT = {"status": "cancelled", "message": "Not run: the user interrupted this turn."}


def elide_stale_tool_results(messages: list) -> int:
    """
    Replaces bulky tool output from before the most recent request with a short stub,
    so files read long ago aren't resent with every call. Explore briefs are kept: they
    are already compact. Meant to run only when a new request starts, since rewriting
    history mid-request would defeat the API's prompt caching. Returns the number elided.
    """
    user_indexes = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indexes:
        return 0
    calls, elided = {}, 0
    for message in messages[:user_indexes[-1]]:
        for call in message.get("tool_calls") or []:
            calls[call.get("id")] = call.get("function") or {}
        if message.get("role") != "tool" or message.get("name") == EXPLORE_TOOL_NAME:
            continue
        if len(message.get("content") or "") < ELIDE_MIN_CHARS:
            continue
        function = calls.get(message.get("tool_call_id"), {})
        signature = f"{function.get('name') or message.get('name')}({clip(str(function.get('arguments') or ''), 200)})"
        message["content"] = json.dumps({
            "status": "elided",
            "message": f"Output of {signature} from an earlier request was removed to save context. "
                       "Call the tool again if you still need it.",
        })
        elided += 1
    return elided


def _run_steps(messages: list, workspace: Path, settings: Settings, usage: Usage) -> None:
    for _ in range(settings.max_steps):
        response_msg, call_usage = call_llm_api(messages, AGENT_TOOLS, settings)
        usage.add(call_usage)
        messages.append(response_msg)

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        if not tool_calls:
            print(f"\nAgent > {sanitize(content) or '(no response)'}\n")
            return
        if content:
            print(f"\nAgent > {sanitize(content)}\n")

        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments")
            print(f"[Tool Call] {sanitize(name)}({sanitize(clip(str(raw_args), 200))})")
            try:
                if name == EXPLORE_TOOL_NAME:
                    result = run_explore(raw_args, workspace, settings, usage)
                else:
                    result = dispatch_tool(name, raw_args, workspace)
            except KeyboardInterrupt:
                for pending in tool_calls[index:]:
                    messages.append(tool_result_message(pending, CANCELLED_RESULT))
                raise
            if result.get("status") != "success":
                print(f"  -> {result.get('status')}: {sanitize(clip(str(result.get('message', '')), 200))}")
            messages.append(tool_result_message(call, result))

    print(f"\n[Stopped] Reached the limit of {settings.max_steps} steps for one request. "
          "Send another message to let the agent continue.\n")


def run_turn(messages: list, workspace: Path, settings: Settings) -> None:
    """
    Alternates model calls and tool calls until the model answers without tools,
    or the step limit is reached. On KeyboardInterrupt every tool call the model
    already requested gets a result, so the history stays valid for the next request.
    Token usage for the request is printed at the end, however it ends.
    """
    usage = Usage()
    try:
        _run_steps(messages, workspace, settings, usage)
    finally:
        if usage.calls:
            print(usage.summary() + "\n")


def _discard_unanswered(messages: list) -> bool:
    """Drops the user's message if nothing happened after it, so retrying doesn't send it twice."""
    if messages and messages[-1].get("role") == "user":
        messages.pop()
        return True
    return False


def run_agent(workspace: Path, settings: Settings) -> None:
    system_prompt, agents_status = build_system_context(workspace)
    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 60)
    print(f"Directory Sandbox : {workspace.resolve()}")
    print("Shell Disabled    : True (Pure Python file APIs only)")
    print(f"AGENTS.md         : {agents_status}")
    print(f"Model             : {settings.model}")
    print(f"Explore Model     : {settings.explore_model or settings.model}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        elided = elide_stale_tool_results(messages)
        if elided:
            print(f"[Context] Elided {elided} tool result{'s' if elided != 1 else ''} from older requests.")
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(messages, workspace, settings)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except LLMError as e:
            print(f"\n[Error] {e}", file=sys.stderr)
        else:
            continue
        if _discard_unanswered(messages):
            print("Your last message was not sent; enter it again to retry.\n")
        else:
            print("Progress so far is kept; send a message to continue.\n")
