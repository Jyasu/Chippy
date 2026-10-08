"""Interactive loop: sends the conversation to the model and executes its tool calls."""

import sys
from pathlib import Path

from chippy.compact import (
    ContextMeter,
    compact_history,
    elide_stale_tool_results,
    ensure_room,
    recover_from_overflow,
)
from chippy.config import API_KEY_ENV, Settings
from chippy.context import build_system_context
from chippy.envfile import ENV_FILE_NAME
from chippy.explore import run_explore
from chippy.llm import ContextLengthError, LLMError, Usage, call_llm_api
from chippy.terminal import clip, sanitize
from chippy.tools import AGENT_TOOLS, EXPLORE_TOOL_NAME, TOOLS, ReadTracker, tool_result_message

CANCELLED_RESULT = {"status": "cancelled", "message": "Not run: the user interrupted this turn."}
COMMANDS_HELP = """Commands:
  /compact [focus]  Summarize the conversation so far to free context (optionally say what to keep)
  /usage            Token usage for this session and the current context size
  /help             Show this help
  exit, quit        Leave"""


def agent_tools(settings: Settings) -> list:
    return TOOLS if settings.explore_mode == "never" else AGENT_TOOLS


def _call_model(messages: list, tools: list, settings: Settings, usage: Usage, meter: ContextMeter,
                tool_choice, reads: ReadTracker) -> dict:
    """One model call. If the API rejects the prompt as too long, compacts everything and retries once."""
    try:
        response, call_usage = call_llm_api(messages, tools, settings, tool_choice)
    except ContextLengthError:
        print("[Context] The request exceeded the model's context window; compacting and retrying.")
        if not recover_from_overflow(messages, settings, usage, meter):
            raise
        reads.clear()
        response, call_usage = call_llm_api(messages, tools, settings, tool_choice)
    meter.record(len(messages), call_usage)
    usage.add(call_usage)
    return response


def _run_steps(messages: list, workspace: Path, settings: Settings, usage: Usage, meter: ContextMeter) -> None:
    tools = agent_tools(settings)
    reads = ReadTracker()
    for step in range(settings.max_steps):
        if ensure_room(messages, settings, usage, meter, tools):
            reads.clear()
        tool_choice = "auto"
        if step == 0 and settings.explore_mode == "always":
            tool_choice = {"type": "function", "function": {"name": EXPLORE_TOOL_NAME}}
        response_msg = _call_model(messages, tools, settings, usage, meter, tool_choice, reads)
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
                if name == EXPLORE_TOOL_NAME and settings.explore_mode != "never":
                    result = run_explore(raw_args, workspace, settings, usage)
                else:
                    result = reads.dispatch(name, raw_args, workspace)
            except KeyboardInterrupt as interrupt:
                # An interrupted explore still reports what it had looked at.
                messages.append(tool_result_message(call, getattr(interrupt, "result", None) or CANCELLED_RESULT))
                for pending in tool_calls[index + 1:]:
                    messages.append(tool_result_message(pending, CANCELLED_RESULT))
                raise
            if result.get("status") != "success":
                print(f"  -> {result.get('status')}: {sanitize(clip(str(result.get('message', '')), 200))}")
            messages.append(tool_result_message(call, result))

    print(f"\n[Stopped] Reached the limit of {settings.max_steps} steps for one request. "
          "Send another message to let the agent continue.\n")


def run_turn(messages: list, workspace: Path, settings: Settings, meter: ContextMeter = None,
             session: Usage = None) -> None:
    """
    Alternates model calls and tool calls until the model answers without tools,
    or the step limit is reached. On KeyboardInterrupt every tool call the model
    already requested gets a result, so the history stays valid for the next request.
    Token usage for the request is printed at the end, however it ends, and added to `session`.
    """
    usage = Usage()
    try:
        _run_steps(messages, workspace, settings, usage, meter or ContextMeter())
    finally:
        if usage.calls:
            print(usage.summary() + "\n")
        if session is not None:
            session.merge(usage)


def _discard_unanswered(messages: list, user_input: str) -> bool:
    """
    Drops the user's message if nothing happened after it, so retrying doesn't send it twice.
    Compares content too: after compaction the last message may be a summary, which must stay.
    """
    if messages and messages[-1] == {"role": "user", "content": user_input}:
        messages.pop()
        return True
    return False


def _context_line(messages: list, settings: Settings, meter: ContextMeter) -> str:
    size = meter.estimate(messages, agent_tools(settings))
    return f"[Context] ~{size:,} of {settings.context_limit:,} tokens ({size * 100 // settings.context_limit}%)"


def run_command(line: str, messages: list, settings: Settings, meter: ContextMeter, session: Usage) -> bool:
    """Handles a /command typed at the prompt. Returns False if the line is not a known command."""
    command, _, argument = line.partition(" ")
    command = command.lower()
    if command == "/help":
        print(COMMANDS_HELP + "\n")
    elif command == "/usage":
        print(session.summary("[Session]") if session.calls else "[Session] No model calls yet.")
        print(_context_line(messages, settings, meter) + "\n")
    elif command == "/compact":
        usage = Usage()
        before = _context_line(messages, settings, meter)
        try:
            compacted = compact_history(messages, settings, usage, keep_steps=0, mid_request=False,
                                        instructions=argument.strip())
        except KeyboardInterrupt:
            print("\n[Interrupted] The conversation was not changed.\n")
            return True
        except LLMError as e:
            print(f"[Error] Compaction failed, the conversation was not changed: {e}\n", file=sys.stderr)
            return True
        finally:
            session.merge(usage)
        if compacted:
            meter.reset()
            print(f"{before}\n -> after compacting: {_context_line(messages, settings, meter)[len('[Context] '):]}\n")
        else:
            print("Nothing to compact yet.\n")
    else:
        return False
    return True


def run_agent(workspace: Path, settings: Settings) -> None:
    system_prompt, agents_status = build_system_context(workspace, settings.explore_mode)
    messages = [{"role": "system", "content": system_prompt}]
    meter, session = ContextMeter(), Usage()

    print("=" * 60)
    print(f"Directory Sandbox : {workspace.resolve()}")
    print("Shell Disabled    : True (Pure Python file APIs only)")
    print(f"AGENTS.md         : {agents_status}")
    print(f"Endpoint          : {settings.url}")
    print(f"Model             : {settings.model}")
    if settings.api_key:
        print(f"API Key           : from {settings.api_key_source or API_KEY_ENV}")
    else:
        print(f"API Key           : NOT SET - requests are sent without one. Set {API_KEY_ENV}, "
              f"or put it in {ENV_FILE_NAME} next to chippy.py (see --help).")
    print(f"Context Limit     : {settings.context_limit:,} tokens (auto-compacts near the limit)")
    explore = "off" if settings.explore_mode == "never" else f"{settings.explore_mode}, {settings.explore_model or settings.model}"
    print(f"Explore           : {explore}")
    if settings.log_path:
        print(f"Usage Log         : {settings.log_path}")
    print("Type /help for commands.")
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
        if user_input.startswith("/") and run_command(user_input, messages, settings, meter, session):
            continue

        elided = elide_stale_tool_results(messages)
        if elided:
            meter.reset()
            print(f"[Context] Elided {elided} bulky tool output{'s' if elided != 1 else ''} from older requests.")
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(messages, workspace, settings, meter, session)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except LLMError as e:
            print(f"\n[Error] {e}", file=sys.stderr)
        else:
            continue
        if _discard_unanswered(messages, user_input):
            print("Your last message was not sent; enter it again to retry.\n")
        else:
            print("Progress so far is kept; send a message to continue.\n")

    if session.calls:
        print(session.summary("[Session]"))
