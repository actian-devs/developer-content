"""Project-root-confined coding tools for the controlled agent demonstration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PureWindowsPath
from time import monotonic
from typing import Any


class CodingToolError(RuntimeError):
    """Raised when a coding tool cannot safely complete an operation."""


class PathSafetyError(CodingToolError):
    """Raised when a requested path violates workspace confinement."""


class CommandSafetyError(CodingToolError):
    """Raised when a requested test command is not explicitly allowed."""


BLOCKED_PARTS = frozenset(
    {
        ".git",
        ".env",
        "credentials",
        "credentials.json",
        "evidence",
        "secrets",
    }
)
ALLOWED_TEXT_SUFFIXES = frozenset({".py", ".toml", ".txt", ".md", ".json", ".yaml", ".yml"})

CODING_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "name": "list_project_files",
        "description": "List readable source and test files in the selected workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read one text file inside the selected workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_project_files",
        "description": "Find literal text in readable files inside the selected workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query", "max_results"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_tests",
        "description": "Run one predefined test suite in the selected workspace.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "enum": ["pytest", "unittest"]}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apply_edit",
        "description": "Replace one exact text occurrence in an existing workspace file.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
)


class SafeCodingTools:
    """Execute a minimal allowlist of file and test operations inside one root."""

    def __init__(
        self,
        project_root: Path,
        *,
        command_timeout_seconds: float = 30.0,
        maximum_file_bytes: int = 100_000,
    ) -> None:
        root = project_root.resolve(strict=True)
        if not root.is_dir():
            raise PathSafetyError("project_root must be an existing directory")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if maximum_file_bytes <= 0:
            raise ValueError("maximum_file_bytes must be positive")
        self.project_root = root
        self.command_timeout_seconds = command_timeout_seconds
        self.maximum_file_bytes = maximum_file_bytes
        self._commands = {
            "pytest": (sys.executable, "-m", "pytest", "-q"),
            "unittest": (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"),
        }

    @staticmethod
    def _validate_relative_path(path: object) -> str:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise PathSafetyError("path must be a non-empty relative string")
        value = path.strip().replace("\\", "/")
        candidate = Path(value)
        if candidate.is_absolute() or PureWindowsPath(path).is_absolute():
            raise PathSafetyError("absolute paths are forbidden")
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise PathSafetyError("path traversal is forbidden")
        lowered = [part.casefold() for part in candidate.parts]
        if any(
            part in BLOCKED_PARTS
            or part.startswith(".env")
            or "credential" in part
            or "secret" in part
            for part in lowered
        ):
            raise PathSafetyError("sensitive or internal paths are forbidden")
        return value

    def _resolve_existing_file(self, path: object) -> Path:
        relative = self._validate_relative_path(path)
        try:
            resolved = (self.project_root / relative).resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise PathSafetyError("requested file does not exist") from error
        if not resolved.is_relative_to(self.project_root):
            raise PathSafetyError("path or symlink escapes the selected workspace")
        if not resolved.is_file():
            raise PathSafetyError("requested path is not a file")
        if resolved.suffix.casefold() not in ALLOWED_TEXT_SUFFIXES:
            raise PathSafetyError("file type is not allowed")
        if resolved.stat().st_size > self.maximum_file_bytes:
            raise PathSafetyError("file exceeds the safe read-size limit")
        return resolved

    def list_project_files(self) -> dict[str, object]:
        files: list[str] = []
        for candidate in self.project_root.rglob("*"):
            try:
                relative = candidate.relative_to(self.project_root).as_posix()
                resolved = self._resolve_existing_file(relative)
            except PathSafetyError:
                continue
            files.append(resolved.relative_to(self.project_root).as_posix())
        return {"files": sorted(set(files))}

    def read_file(self, path: object) -> dict[str, object]:
        resolved = self._resolve_existing_file(path)
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise CodingToolError("file is not valid UTF-8 text") from error
        return {
            "path": resolved.relative_to(self.project_root).as_posix(),
            "content": content,
        }

    def search_project_files(self, query: object, max_results: object) -> dict[str, object]:
        if not isinstance(query, str) or not query:
            raise CodingToolError("query must be a non-empty string")
        if type(max_results) is not int or not 1 <= max_results <= 50:
            raise CodingToolError("max_results must be an integer between 1 and 50")
        matches: list[dict[str, object]] = []
        for relative in self.list_project_files()["files"]:
            file_result = self.read_file(relative)
            for line_number, line in enumerate(str(file_result["content"]).splitlines(), start=1):
                if query in line:
                    matches.append({"path": relative, "line": line_number, "text": line[:500]})
                    if len(matches) >= max_results:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def apply_edit(self, path: object, old_text: object, new_text: object) -> dict[str, object]:
        if not isinstance(old_text, str) or not old_text:
            raise CodingToolError("old_text must be a non-empty string")
        if not isinstance(new_text, str):
            raise CodingToolError("new_text must be a string")
        resolved = self._resolve_existing_file(path)
        content = resolved.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise CodingToolError(
                f"old_text must occur exactly once; found {occurrences} occurrences"
            )
        updated = content.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > self.maximum_file_bytes:
            raise CodingToolError("edited file would exceed the safe size limit")
        resolved.write_text(updated, encoding="utf-8", newline="\n")
        return {
            "path": resolved.relative_to(self.project_root).as_posix(),
            "replacements": 1,
        }

    def run_tests(self, command: object) -> dict[str, object]:
        if not isinstance(command, str) or command not in self._commands:
            raise CommandSafetyError("test command is not in the explicit allowlist")
        argv = self._commands[command]
        started = monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CodingToolError(
                f"test command exceeded {self.command_timeout_seconds:g} seconds"
            ) from error
        result: dict[str, object] = {
            "command": command,
            "argv": list(argv[1:]),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-20_000:],
            "stderr": completed.stderr[-20_000:],
            "duration_seconds": round(monotonic() - started, 6),
        }
        if completed.returncode == 0:
            result["confirmed_outcome"] = f"{command} passed with exit code 0"
        return result
