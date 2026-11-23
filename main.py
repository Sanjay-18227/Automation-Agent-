"""Interactive terminal interface for AutonomousAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from agent import AgentRunResult, AutonomousAgent
from providers import ProviderEngine
from tools import ToolRegistry


def _approval_prompt(action: str, details: Mapping[str, object]) -> bool:
    if action == "run_terminal_command":
        print("\nApproval required before terminal execution:")
        print("  Command: {}".format(details.get("command", "")))
        print("  Working directory: {}".format(details.get("cwd", "")))
        print("  Timeout: {} seconds".format(details.get("timeout", "")))
    else:
        print("\nApproval required before modifying an existing file:")
        print("  Action: {}".format(action))
        print("  Path: {}".format(details.get("path", "")))
        if "matches" in details:
            print("  Text matches to replace: {}".format(details["matches"]))
    try:
        answer = input("Approve? [y/N]: ").strip().casefold()
    except EOFError:
        print()
        return False
    return answer in {"y", "yes"}


def _show_progress(event: Mapping[str, object]) -> None:
    event_name = event.get("event")
    if event_name == "provider":
        print(
            "[provider] {} ({})".format(event.get("provider"), event.get("model"))
        )
    elif event_name == "plan":
        print("[plan]")
        for index, step in enumerate(event.get("steps", ()), start=1):
            print("  {}. {}".format(index, step))
    elif event_name == "tool_start":
        print("[tool] {}".format(event.get("tool")))
    elif event_name == "tool_result":
        print("[tool result] {}".format(json.dumps(event.get("result"), ensure_ascii=True)))
    elif event_name == "protocol_error":
        print("[protocol] {}".format(event.get("error")))
    elif event_name == "provider_error":
        print("[provider error] {}".format(event.get("error")))


def _print_status(agent: AutonomousAgent, project_root: Path) -> None:
    print("Project: {}".format(project_root))
    print(
        "Provider fallback: {}".format(
            " -> ".join(agent.provider.fallback_order)
        )
    )
    if agent.last_result is None:
        print("Agent: idle; no task has completed")
        return
    result = agent.last_result
    print(
        "Last task: {} ({}; {} iteration(s))".format(
            "completed" if result.completed else result.stop_reason,
            result.provider or "no provider",
            result.iterations,
        )
    )
    if result.plan:
        print("Planned steps: {}".format(len(result.plan)))


def main() -> int:
    project_root = Path(__file__).resolve().parent
    try:
        provider = ProviderEngine()
        tools = ToolRegistry(
            project_root=project_root,
            approval_callback=_approval_prompt,
        )
        agent = AutonomousAgent(
            provider=provider,
            tools=tools,
            max_iterations=12,
            progress_callback=_show_progress,
        )
    except (ValueError, OSError) as exc:
        print("Could not initialize AutonomousAgent: {}".format(exc))
        return 1

    print("AutonomousAgent ready. Type /help for commands; Ctrl+C stops a task.")
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nExiting.")
            return 0

        if not user_input:
            continue
        if user_input == "/help":
            print(
                "Enter a task to chat with the agent.\n"
                "/help   Show this help\n"
                "/status Show the project and last task status\n"
                "/exit   Exit AutonomousAgent\n"
                "Ctrl+C  Stop the current task"
            )
            continue
        if user_input == "/status":
            _print_status(agent, project_root)
            continue
        if user_input == "/exit":
            return 0
        if user_input.startswith("/"):
            print("Unknown command. Type /help for available commands.")
            continue

        try:
            result = agent.run(user_input)
        except KeyboardInterrupt:
            agent.stop()
            print("\nTask interrupted.")
            continue
        _print_result(result)


def _print_result(result: AgentRunResult) -> None:
    if result.completed:
        print("\nAgent> {}".format(result.response))
    elif result.error:
        print("\nAgent stopped ({}): {}".format(result.stop_reason, result.error))
    else:
        print("\nAgent stopped: {}".format(result.stop_reason))


if __name__ == "__main__":
    raise SystemExit(main())