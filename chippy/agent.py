"""Interactive loop: sends the conversation to the model and executes its tool calls."""

import json
import sys
from pathlib import Path

from chippy.config import Settings
from chippy.context import build_system_context
from chippy.llm import LLMError, call_llm_api
from chippy.terminal import clip, sanitize
from chippy.tools import TOOLS, dispatch_tool

CANCELLED_RESULT = {"status": "cancelled", "message": "Not run: the user interrupted this turn."}


def _tool_message(call: dict, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.get("id"),
        "name": (call.get("function") or {}).get("name") or "",
        "content": json.dumps(result),
    }


def run_turn(messages: list, workspace: Path, settings: Settings) -> None:
    """
    Alternates model calls and tool calls until the model answers without tools,
    or the step limit is reached. On KeyboardInterrupt every tool call the model
    already requested gets a result, so the history stays valid for the next request.
    """
    for _ in range(settings.max_steps):
        response_msg = call_llm_api(messages, TOOLS, settings)
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
                result = dispatch_tool(name, raw_args, workspace)
            except KeyboardInterrupt:
                for pending in tool_calls[index:]:
                    messages.append(_tool_message(pending, CANCELLED_RESULT))
                raise
            if result.get("status") != "success":
                print(f"  -> {result.get('status')}: {sanitize(clip(str(result.get('message', '')), 200))}")
            messages.append(_tool_message(call, result))

    print(f"\n[Stopped] Reached the limit of {settings.max_steps} steps for one request. "
          "Send another message to let the agent continue.\n")


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
