"""Workspace-bounded tools and approval-gated terminal execution."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Callable, Dict, List, Mapping, Optional, Sequence


class ToolError(RuntimeError):
    """Raised when a tool request is invalid or cannot be completed safely."""


class ApprovalDenied(ToolError):
    """Raised when a user declines an approval request."""


ApprovalCallback = Callable[[str, Mapping[str, object]], bool]

_BLOCKED_EXECUTABLES = {
    "bash",
    "cmd",
    "cmd.exe",
    "cscript",
    "cscript.exe",
    "mshta",
    "mshta.exe",
    "nohup",
    "javaw",
    "javaw.exe",
    "nodew",
    "nodew.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "pythonw",
    "pythonw.exe",
    "setsid",
    "sh",
    "start",
    "start-process",
    "wscript",
    "wscript.exe",
}
_BLOCKED_COMMAND_FLAGS = {
    "--background",
    "--daemon",
    "--detach",
    "-windowstyle",
    "-windowstyle=hidden",
    "/b",
    "/min",
}
_INLINE_EVAL_EXECUTABLES = {
    "node",
    "node.exe",
    "perl",
    "perl.exe",
    "py",
    "py.exe",
    "python",
    "python.exe",
    "ruby",
    "ruby.exe",
}
_INLINE_EVAL_FLAGS = {"-c", "-e", "--eval"}
_SECRET_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)
_SAFE_ENV_NAMES = {
    "HOMEDRIVE",
    "HOMEPATH",
    "PATHEXT",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
    "WINDIR",
}
_MAX_FILE_BYTES = 1024 * 1024
_MAX_COMMAND_OUTPUT = 16 * 1024
_SECRET_ASSIGNMENT = re.compile(
    r'(?im)(\b[\w.-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)'
    r'[\w.-]*["\']?\s*[:=]\s*)(?:"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s,;#}]+)'
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_KNOWN_SECRET_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)


def redact_sensitive_text(text: str) -> str:
    """Redact common credential assignments and recognizable key formats."""
    redacted = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    redacted = _BEARER_SECRET.sub("Bearer [REDACTED]", redacted)
    return _KNOWN_SECRET_TOKEN.sub("[REDACTED]", redacted)


class ToolRegistry:
    """Validate and execute a small set of tools rooted in one project."""

    def __init__(
        self,
        project_root: str | os.PathLike[str],
        approval_callback: Optional[ApprovalCallback] = None,
        command_timeout: int = 30,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise ValueError("project_root must be an existing directory")
        if type(command_timeout) is not int or not 1 <= command_timeout <= 120:
            raise ValueError("command_timeout must be between 1 and 120 seconds")
        self.approval_callback = approval_callback
        self.command_timeout = command_timeout

    @staticmethod
    def describe_tools() -> List[Dict[str, object]]:
        return [
            {
                "name": "list_files",
                "description": "List safe project files and directories.",
                "arguments": {
                    "path": "Optional relative project path; defaults to '.'",
                    "recursive": "Optional boolean; defaults to false",
                    "max_entries": "Optional integer from 1 to 500; defaults to 100",
                },
            },
            {
                "name": "read_file",
                "description": "Read a UTF-8 text file inside the project.",
                "arguments": {
                    "path": "Required relative project file path",
                    "max_bytes": "Optional integer from 1 to 1048576",
                },
            },
            {
                "name": "create_file",
                "description": "Create a new UTF-8 file; overwrites require approval.",
                "arguments": {
                    "path": "Required relative project file path",
                    "content": "Required text content",
                },
            },
            {
                "name": "edit_file",
                "description": "Replace exact text in an existing UTF-8 file.",
                "arguments": {
                    "path": "Required relative project file path",
                    "old_text": "Required non-empty exact text to find",
                    "new_text": "Required replacement text",
                    "replace_all": "Optional boolean; defaults to false",
                },
            },
            {
                "name": "run_terminal_command",
                "description": (
                    "Run one foreground command in the project; user approval "
                    "is always required. Shell operators and shell interpreters "
                    "are blocked."
                ),
                "arguments": {
                    "command": "Required single command without shell operators",
                    "timeout": "Optional integer seconds from 1 to 120",
                },
            },
        ]

    def execute(
        self, tool_name: str, arguments: Mapping[str, object]
    ) -> Dict[str, object]:
        if not isinstance(tool_name, str):
            raise ToolError("Tool name must be a string")
        if not isinstance(arguments, Mapping):
            raise ToolError("Tool arguments must be a JSON object")

        validators = {
            "list_files": (
                {"path", "recursive", "max_entries"},
                set(),
                self._list_files,
            ),
            "read_file": ({"path", "max_bytes"}, {"path"}, self._read_file),
            "create_file": ({"path", "content"}, {"path", "content"}, self._create_file),
            "edit_file": (
                {"path", "old_text", "new_text", "replace_all"},
                {"path", "old_text", "new_text"},
                self._edit_file,
            ),
            "run_terminal_command": (
                {"command", "timeout"},
                {"command"},
                self._run_terminal_command,
            ),
        }
        if tool_name not in validators:
            raise ToolError("Unknown tool: {}".format(tool_name))

        allowed, required, handler = validators[tool_name]
        supplied = set(arguments)
        if supplied - allowed:
            raise ToolError("Unexpected arguments for {}".format(tool_name))
        if required - supplied:
            raise ToolError("Missing required arguments for {}".format(tool_name))
        self._validate_argument_types(tool_name, arguments)
        result = handler(arguments)
        if not isinstance(result, dict):
            raise ToolError("{} returned an invalid result".format(tool_name))
        return {"ok": True, "tool": tool_name, "result": result}

    @staticmethod
    def _validate_argument_types(
        tool_name: str, arguments: Mapping[str, object]
    ) -> None:
        text_fields = {
            "read_file": ("path",),
            "create_file": ("path", "content"),
            "edit_file": ("path", "old_text", "new_text"),
            "run_terminal_command": ("command",),
        }
        for field in text_fields.get(tool_name, ()):
            if not isinstance(arguments.get(field), str):
                raise ToolError("{} must be a string".format(field))

        if tool_name == "list_files":
            if "path" in arguments and not isinstance(arguments["path"], str):
                raise ToolError("path must be a string")
            if "recursive" in arguments and type(arguments["recursive"]) is not bool:
                raise ToolError("recursive must be a boolean")
            if "max_entries" in arguments:
                limit = arguments["max_entries"]
                if type(limit) is not int or not 1 <= limit <= 500:
                    raise ToolError("max_entries must be an integer from 1 to 500")
        if tool_name == "read_file" and "max_bytes" in arguments:
            limit = arguments["max_bytes"]
            if type(limit) is not int or not 1 <= limit <= _MAX_FILE_BYTES:
                raise ToolError("max_bytes must be an integer from 1 to 1048576")
        if tool_name == "edit_file" and "replace_all" in arguments:
            if type(arguments["replace_all"]) is not bool:
                raise ToolError("replace_all must be a boolean")
        if tool_name == "run_terminal_command" and "timeout" in arguments:
            timeout = arguments["timeout"]
            if type(timeout) is not int or not 1 <= timeout <= 120:
                raise ToolError("timeout must be an integer from 1 to 120")

    @staticmethod
    def _is_blocked_component(component: str) -> bool:
        name = component.casefold()
        return (
            name.startswith(".env")
            or name in {".venv", ".git", ".aws", ".ssh"}
            or "credential" in name
            or "secret" in name
            or name in {"id_rsa", "id_ed25519"}
        )

    def _resolve_path(self, relative_path: str) -> Path:
        if not relative_path or "\0" in relative_path or ":" in relative_path:
            raise ToolError("Path must be a non-empty relative project path")
        windows_path = PureWindowsPath(relative_path)
        local_path = Path(relative_path)
        if (
            windows_path.is_absolute()
            or windows_path.drive
            or local_path.is_absolute()
            or any(part == ".." for part in windows_path.parts)
        ):
            raise ToolError("Absolute paths and path traversal are not allowed")

        components = [part for part in windows_path.parts if part not in ("", ".")]
        if any(self._is_blocked_component(part) for part in components):
            raise ToolError("Access to protected project paths is blocked")

        target = (self.project_root / local_path).resolve()
        try:
            common = os.path.commonpath(
                (str(self.project_root), str(target))
            )
        except ValueError:
            raise ToolError("Path is outside the configured project") from None
        if os.path.normcase(common) != os.path.normcase(str(self.project_root)):
            raise ToolError("Path is outside the configured project")

        resolved_parts = target.relative_to(self.project_root).parts
        if any(self._is_blocked_component(part) for part in resolved_parts):
            raise ToolError("Access to protected project paths is blocked")
        return target

    def _list_files(self, arguments: Mapping[str, object]) -> Dict[str, object]:
        relative = arguments.get("path", ".")
        recursive = arguments.get("recursive", False)
        max_entries = arguments.get("max_entries", 100)
        target = self._resolve_path(relative)
        if not target.is_dir():
            raise ToolError("The requested path is not a directory")

        entries: List[Dict[str, object]] = []
        pending = [target]
        truncated = False
        while pending:
            current = pending.pop(0)
            try:
                children = sorted(os.scandir(current), key=lambda entry: entry.name.casefold())
            except OSError as exc:
                raise ToolError("Could not list project directory ({})".format(type(exc).__name__)) from None

            subdirectories = []
            for entry in children:
                if entry.is_symlink() or self._is_blocked_component(entry.name):
                    continue
                child = Path(entry.path)
                try:
                    safe_child = self._resolve_path(
                        str(child.relative_to(self.project_root))
                    )
                except (ToolError, ValueError):
                    continue

                if entry.is_dir(follow_symlinks=False):
                    kind = "directory"
                    if recursive:
                        subdirectories.append(safe_child)
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    continue
                entries.append(
                    {
                        "path": str(safe_child.relative_to(self.project_root)),
                        "type": kind,
                    }
                )
                if len(entries) >= max_entries:
                    truncated = (
                        bool(pending or subdirectories) or entry is not children[-1]
                    )
                    break

            if len(entries) >= max_entries:
                break
            if recursive:
                pending.extend(subdirectories)

        return {
            "path": str(target.relative_to(self.project_root)) or ".",
            "entries": entries,
            "truncated": truncated,
        }

    def _read_file(self, arguments: Mapping[str, object]) -> Dict[str, object]:
        relative = arguments["path"]
        max_bytes = arguments.get("max_bytes", 64 * 1024)
        target = self._resolve_path(relative)
        if not target.is_file():
            raise ToolError("The requested path is not a file")
        try:
            with target.open("rb") as stream:
                raw_content = stream.read(max_bytes + 1)
            if len(raw_content) > max_bytes:
                raise ToolError("File exceeds the configured read limit")
            content = raw_content.decode("utf-8")
        except ToolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ToolError("Could not read file ({})".format(type(exc).__name__)) from None
        return {
            "path": str(target.relative_to(self.project_root)),
            "content": redact_sensitive_text(content),
        }

    def _create_file(self, arguments: Mapping[str, object]) -> Dict[str, object]:
        relative = arguments["path"]
        content = arguments["content"]
        if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ToolError("File content exceeds the 1 MiB limit")

        target = self._resolve_path(relative)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = self._resolve_path(relative)
            if target.exists():
                self._request_approval(
                    "create_file",
                    {"path": str(target.relative_to(self.project_root)), "overwrite": True},
                )
                self._write_existing(target, content)
            else:
                try:
                    with target.open("x", encoding="utf-8", newline="") as stream:
                        stream.write(content)
                except FileExistsError:
                    self._request_approval(
                        "create_file",
                        {
                            "path": str(target.relative_to(self.project_root)),
                            "overwrite": True,
                        },
                    )
                    self._write_existing(target, content)
            if target.read_text(encoding="utf-8") != content:
                raise ToolError("File verification failed after creation")
        except ToolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ToolError("Could not create file ({})".format(type(exc).__name__)) from None
        return {
            "path": str(target.relative_to(self.project_root)),
            "success": True,
            "bytes_written": len(content.encode("utf-8")),
        }

    def _edit_file(self, arguments: Mapping[str, object]) -> Dict[str, object]:
        relative = arguments["path"]
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        replace_all = arguments.get("replace_all", False)
        if not old_text:
            raise ToolError("old_text must not be empty")
        if len(new_text.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ToolError("Replacement text exceeds the 1 MiB limit")

        target = self._resolve_path(relative)
        if not target.is_file():
            raise ToolError("The requested path is not an existing file")
        try:
            if target.stat().st_size > _MAX_FILE_BYTES:
                raise ToolError("File exceeds the 1 MiB edit limit")
            original = target.read_text(encoding="utf-8")
        except ToolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ToolError("Could not read file for editing ({})".format(type(exc).__name__)) from None

        count = original.count(old_text)
        if count == 0:
            raise ToolError("old_text was not found in the file")
        if count > 1 and not replace_all:
            raise ToolError("old_text occurs more than once; set replace_all to true")
        updated = (
            original.replace(old_text, new_text)
            if replace_all
            else original.replace(old_text, new_text, 1)
        )
        if len(updated.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ToolError("Edited file would exceed the 1 MiB limit")
        self._request_approval(
            "edit_file",
            {
                "path": str(target.relative_to(self.project_root)),
                "matches": count,
            },
        )
        try:
            self._write_existing(target, updated)
            if target.read_text(encoding="utf-8") != updated:
                raise ToolError("File verification failed after editing")
        except ToolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ToolError("Could not edit file ({})".format(type(exc).__name__)) from None
        return {
            "path": str(target.relative_to(self.project_root)),
            "success": True,
            "replacements": count if replace_all else 1,
        }

    def _write_existing(self, target: Path, content: str) -> None:
        relative = str(target.relative_to(self.project_root))
        verified_target = self._resolve_path(relative)
        if verified_target != target or not target.is_file():
            raise ToolError("File changed before the approved write")
        target.write_text(content, encoding="utf-8", newline="")

    def _request_approval(
        self, action: str, details: Mapping[str, object]
    ) -> None:
        if self.approval_callback is None:
            raise ApprovalDenied("User approval is required for {}".format(action))
        if not self.approval_callback(action, details):
            raise ApprovalDenied("User declined {}".format(action))

    def _run_terminal_command(
        self, arguments: Mapping[str, object]
    ) -> Dict[str, object]:
        command = arguments["command"]
        timeout = arguments.get("timeout", self.command_timeout)
        args = self._parse_command(command)
        self._request_approval(
            "run_terminal_command",
            {"command": command, "cwd": str(self.project_root), "timeout": timeout},
        )

        safe_env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _SAFE_ENV_NAMES
            and not any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
        }
        safe_env["PYTHONUTF8"] = "1"
        try:
            completed = subprocess.run(
                args,
                cwd=str(self.project_root),
                env=safe_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            timed_out = False
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = self._decode_output(exc.stdout)
            stderr = self._decode_output(exc.stderr)
            timed_out = True
            exit_code = None
        except OSError as exc:
            raise ToolError(
                "Could not start terminal command ({})".format(type(exc).__name__)
            ) from None

        return {
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "success": not timed_out and exit_code == 0,
            "stdout": self._truncate(redact_sensitive_text(stdout)),
            "stderr": self._truncate(redact_sensitive_text(stderr)),
        }

    @staticmethod
    def _parse_command(command: str) -> List[str]:
        if not command.strip():
            raise ToolError("Command must not be empty")
        if any(operator in command for operator in (";", "&", "|", "<", ">", "`", "\n", "\r")):
            raise ToolError("Shell operators and multiline commands are blocked")
        lowered = command.casefold()
        if any(
            token in lowered
            for token in (".env", ".venv", ".git", "credential", "secret")
        ):
            raise ToolError("Commands referencing protected paths are blocked")
        if (
            _SECRET_ASSIGNMENT.search(command)
            or _BEARER_SECRET.search(command)
            or _KNOWN_SECRET_TOKEN.search(command)
        ):
            raise ToolError("Commands containing secret-like values are blocked")
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            raise ToolError("Command quoting is invalid") from None
        args = [
            part[1:-1]
            if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"')
            else part
            for part in parts
        ]
        if not args:
            raise ToolError("Command must not be empty")

        executable = PureWindowsPath(args[0]).name.casefold()
        if (
            executable in _BLOCKED_EXECUTABLES
            or executable.endswith((".bat", ".cmd", ".ps1", ".pyw"))
        ):
            raise ToolError("Shells, script wrappers, and background launchers are blocked")
        if any(arg.casefold() in _BLOCKED_COMMAND_FLAGS for arg in args[1:]):
            raise ToolError("Hidden or background command options are blocked")
        if (
            executable in _INLINE_EVAL_EXECUTABLES
            and any(arg.casefold() in _INLINE_EVAL_FLAGS for arg in args[1:])
        ):
            raise ToolError("Inline interpreter commands are blocked")
        return args

    @staticmethod
    def _decode_output(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _truncate(value: str) -> str:
        if len(value) <= _MAX_COMMAND_OUTPUT:
            return value
        return value[:_MAX_COMMAND_OUTPUT] + "\n[output truncated]"