import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent import AgentError, AutonomousAgent
from providers import ProviderEngine, ProviderError, ProviderResponse
from tools import ApprovalDenied, ToolError, ToolRegistry


def _response(value, provider="ollama"):
    return ProviderResponse(provider=provider, model="test-model", text=value)


class ToolRegistryTests(unittest.TestCase):
    def test_create_read_and_edit_file(self):
        approvals = []
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                directory,
                approval_callback=lambda action, details: (
                    approvals.append((action, details)) or True
                ),
            )
            created = registry.execute(
                "create_file", {"path": "notes.txt", "content": "first line"}
            )
            read = registry.execute("read_file", {"path": "notes.txt"})
            edited = registry.execute(
                "edit_file",
                {
                    "path": "notes.txt",
                    "old_text": "first",
                    "new_text": "updated",
                },
            )

            self.assertTrue(created["result"]["success"])
            self.assertEqual(read["result"]["content"], "first line")
            self.assertTrue(edited["result"]["success"])
            self.assertEqual(
                Path(directory, "notes.txt").read_text(encoding="utf-8"),
                "updated line",
            )
            self.assertEqual([action for action, _ in approvals], ["edit_file"])

    def test_existing_file_overwrite_requires_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "existing.txt")
            target.write_text("original", encoding="utf-8")
            registry = ToolRegistry(directory)

            with self.assertRaises(ApprovalDenied):
                registry.execute(
                    "create_file",
                    {"path": "existing.txt", "content": "replacement"},
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_file_contents_redact_credentials_before_returning_to_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "settings.py").write_text(
                'OPENROUTER_API_KEY = "test-api-key-value"\n'
                '{"GEMINI_API_KEY": "test-gemini-key-value"}\n',
                encoding="utf-8",
            )
            registry = ToolRegistry(directory)

            result = registry.execute("read_file", {"path": "settings.py"})

            content = result["result"]["content"]
            self.assertNotIn("test-api-key-value", content)
            self.assertNotIn("test-gemini-key-value", content)
            self.assertIn("[REDACTED]", content)

    def test_path_traversal_and_protected_paths_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(directory)
            for path in (
                "../outside.txt",
                r"C:\outside.txt",
                ".env",
                ".env.local",
                ".venv\\Scripts\\python.exe",
                ".git\\config",
                "credentials.json",
            ):
                with self.subTest(path=path):
                    with self.assertRaises(ToolError):
                        registry.execute("read_file", {"path": path})

    def test_terminal_command_requires_approval_and_scrubs_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                directory,
                approval_callback=lambda action, details: True,
            )
            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "do-not-pass",
                    "MY_CUSTOM_SECRET": "also-do-not-pass",
                },
                clear=False,
            ), patch("tools.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=["python"],
                    returncode=0,
                    stdout="tests passed",
                    stderr="",
                )
                result = registry.execute(
                    "run_terminal_command",
                    {"command": "python -m unittest", "timeout": 5},
                )

            self.assertTrue(result["result"]["success"])
            run.assert_called_once()
            options = run.call_args.kwargs
            self.assertFalse(options["shell"])
            self.assertEqual(options["timeout"], 5)
            self.assertNotIn("OPENROUTER_API_KEY", options["env"])
            self.assertNotIn("MY_CUSTOM_SECRET", options["env"])

    def test_terminal_command_is_not_launched_when_approval_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(directory)
            with patch("tools.subprocess.run") as run:
                with self.assertRaises(ApprovalDenied):
                    registry.execute(
                        "run_terminal_command",
                        {"command": "python -m unittest"},
                    )
            run.assert_not_called()

    def test_terminal_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                directory,
                approval_callback=lambda action, details: True,
            )
            with patch(
                "tools.subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    "python", 1, output=b"partial output"
                ),
            ):
                result = registry.execute(
                    "run_terminal_command",
                    {"command": "python -m unittest", "timeout": 1},
                )

            self.assertTrue(result["result"]["timed_out"])
            self.assertFalse(result["result"]["success"])
            self.assertEqual(result["result"]["stdout"], "partial output")

    def test_shell_and_background_commands_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                directory,
                approval_callback=lambda action, details: True,
            )
            for command in (
                "python -m unittest &",
                "powershell -Command Get-Process",
                "python --detach",
            ):
                with self.subTest(command=command):
                    with patch("tools.subprocess.run") as run:
                        with self.assertRaises(ToolError):
                            registry.execute(
                                "run_terminal_command", {"command": command}
                            )
                    run.assert_not_called()

    def test_terminal_command_with_secret_literal_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                directory,
                approval_callback=lambda action, details: True,
            )
            with patch("tools.subprocess.run") as run:
                with self.assertRaises(ToolError):
                    registry.execute(
                        "run_terminal_command",
                        {
                            "command": (
                                'python -c "print(\'sk-12345678901234567890\')"'
                            )
                        },
                    )
            run.assert_not_called()


class AutonomousAgentTests(unittest.TestCase):
    def _agent(self, provider, directory, **kwargs):
        return AutonomousAgent(
            provider=provider,
            tools=ToolRegistry(directory),
            **kwargs
        )

    def test_integrates_with_provider_engine_and_creates_task_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"plan","steps":["Inspect task"]}'),
                _response('{"type":"final","message":"Ready"}'),
            ]
            agent = self._agent(provider, directory)

            result = agent.run("Help me understand the project")

            self.assertTrue(result.completed)
            self.assertEqual(result.plan, ("Inspect task",))
            self.assertEqual(result.response, "Ready")
            self.assertEqual(result.provider, "ollama")
            self.assertEqual(provider.generate.call_count, 2)

    def test_conversation_context_is_preserved_between_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"plan","steps":["Answer first task"]}'),
                _response('{"type":"final","message":"First answer"}'),
                _response('{"type":"plan","steps":["Use earlier context"]}'),
                _response('{"type":"final","message":"Second answer"}'),
            ]
            agent = self._agent(provider, directory)

            agent.run("First conversation task")
            result = agent.run("Follow up on the first task")

            self.assertTrue(result.completed)
            second_task_context = provider.generate.call_args_list[2].args[0]
            self.assertTrue(
                any("First conversation task" in item.content for item in second_task_context)
            )

    def test_selects_tool_and_verifies_its_result(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "sample.txt").write_text("data", encoding="utf-8")
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"plan","steps":["List project files"]}'),
                _response(
                    '{"type":"tool_call","tool":"list_files",'
                    '"arguments":{"path":".","max_entries":10}}'
                ),
                _response('{"type":"final","message":"I found sample.txt"}'),
            ]
            agent = self._agent(provider, directory)

            result = agent.run("List the project files")

            self.assertTrue(result.completed)
            self.assertEqual(len(result.tool_results), 1)
            self.assertTrue(result.tool_results[0]["ok"])
            self.assertTrue(result.tool_results[0]["verified"])
            listed = result.tool_results[0]["result"]["entries"]
            self.assertIn("sample.txt", [entry["path"] for entry in listed])

    def test_tool_error_is_returned_to_model_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"plan","steps":["Read the requested file"]}'),
                _response(
                    '{"type":"tool_call","tool":"unknown_tool","arguments":{}}'
                ),
                _response('{"type":"final","message":"I could not use that tool"}'),
            ]
            agent = self._agent(provider, directory)

            result = agent.run("Inspect a file")

            self.assertTrue(result.completed)
            self.assertFalse(result.tool_results[0]["ok"])
            final_context = provider.generate.call_args_list[-1].args[0]
            self.assertTrue(
                any("Unknown tool" in item.content for item in final_context)
            )

    def test_iteration_limit_stops_repeated_invalid_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.return_value = _response("not json")
            agent = self._agent(provider, directory, max_iterations=10)

            result = agent.run("Do a bounded task")

            self.assertFalse(result.completed)
            self.assertEqual(result.stop_reason, "protocol_error")
            self.assertEqual(result.iterations, 3)
            self.assertEqual(provider.generate.call_count, 3)
            self.assertIn("2 protocol repair attempts", result.error)
            correction_messages = [
                message.content
                for message in provider.generate.call_args_list[-1].args[0]
                if "previous response was rejected" in message.content
            ]
            self.assertEqual(len(correction_messages), 2)

    def test_parses_valid_final_protocol_json(self):
        message = AutonomousAgent._parse_protocol_message(
            '{"type":"final","message":"Hello"}'
        )

        self.assertEqual(message, {"type": "final", "message": "Hello"})

    def test_parses_valid_tool_call_protocol_json(self):
        message = AutonomousAgent._parse_protocol_message(
            '{"type":"tool_call","tool":"list_files","arguments":{"path":"."}}'
        )

        self.assertEqual(
            message,
            {
                "type": "tool_call",
                "tool": "list_files",
                "arguments": {"path": "."},
            },
        )

    def test_rejects_invalid_protocol_json(self):
        with self.assertRaisesRegex(AgentError, "valid JSON"):
            AutonomousAgent._parse_protocol_message("not json")

    def test_rejects_duplicate_protocol_json_keys(self):
        with self.assertRaisesRegex(
            AgentError, "valid JSON without duplicate keys"
        ):
            AutonomousAgent._parse_protocol_message(
                '{"type":"tool_call","tool":"list_files",'
                '"tool":"run_terminal_command","arguments":{}}'
            )

    def test_simple_greeting_can_return_final_without_a_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.return_value = _response(
                '{"type":"final","message":"Hi! How can I help?"}'
            )
            agent = self._agent(provider, directory)

            result = agent.run("hi")

            self.assertTrue(result.completed)
            self.assertEqual(result.response, "Hi! How can I help?")
            self.assertEqual(result.plan, ())
            self.assertEqual(provider.generate.call_count, 1)
            schema = provider.generate.call_args.kwargs["response_schema"]
            self.assertEqual(
                {
                    variant["properties"]["type"]["enum"][0]
                    for variant in schema["oneOf"]
                },
                {"plan", "tool_call", "final"},
            )

    def test_duplicate_key_tool_call_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"plan","steps":["List files"]}'),
                _response(
                    '{"type":"tool_call","tool":"list_files",'
                    '"tool":"run_terminal_command","arguments":{}}'
                ),
                _response('{"type":"final","message":"Finished safely"}'),
            ]
            agent = self._agent(provider, directory)

            result = agent.run("List the project files")

            self.assertTrue(result.completed)
            self.assertEqual(result.tool_results, ())

    def test_stop_request_halts_at_next_loop_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.return_value = _response(
                '{"type":"plan","steps":["Review task"]}'
            )
            agent = self._agent(provider, directory)
            agent.progress_callback = lambda event: (
                agent.stop() if event["event"] == "plan" else None
            )

            result = agent.run("Stop after planning")

            self.assertFalse(result.completed)
            self.assertEqual(result.stop_reason, "user_stopped")
            self.assertEqual(provider.generate.call_count, 1)

    def test_provider_failure_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = ProviderError("all providers unavailable")
            agent = self._agent(provider, directory)

            result = agent.run("Do a task")

            self.assertFalse(result.completed)
            self.assertEqual(result.stop_reason, "provider_error")
            self.assertIn("unavailable", result.error)

    def test_invalid_protocol_is_corrected_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(spec=ProviderEngine)
            provider.generate.side_effect = [
                _response('{"type":"tool_call","tool":"run_terminal_command","arguments":{}}'),
                _response('{"type":"plan","steps":["Review task"]}'),
                _response('{"type":"final","message":"No changes needed"}'),
            ]
            agent = self._agent(provider, directory)

            result = agent.run("Review this project")

            self.assertTrue(result.completed)
            self.assertEqual(result.plan, ("Review task",))
            self.assertEqual(result.tool_results, ())


if __name__ == "__main__":
    unittest.main()
