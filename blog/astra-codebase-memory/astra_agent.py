"""Offline-first Responses API orchestration for the Astra coding agent."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

import requests

from coding_tools import CODING_TOOL_DEFINITIONS, SafeCodingTools
from cost_guard import BudgetGuard
from vectoraidb_memory_tools import MEMORY_TOOL_DEFINITIONS, MemoryService

RESPONSES_URL = "https://api.openai.com/v1/responses"
SYSTEM_INSTRUCTIONS = """You are fixing a bug in a confined demonstration workspace.
Call search_codebase_memory using only observed failure symptoms before stating a diagnosis or
editing code. Wait for its function output. An empty result still completes the required search.
Use only the supplied coding tools. Store a fix only after run_tests returns exit code 0, and use
the exact confirmed_outcome string returned by that test call. Do not assume memory will help."""


class AgentError(RuntimeError):
    """Base class for safe orchestration failures."""


class LiveModeDisabledError(AgentError):
    """Raised before network activity when live mode is not explicitly enabled."""


class ResponseValidationError(AgentError):
    """Raised when a Responses payload is malformed."""


class ToolArgumentsError(AgentError):
    """Raised when a function call contains malformed or unexpected arguments."""


class UnknownToolError(AgentError):
    """Raised when the model requests a tool outside the registry."""


class MemoryGateError(AgentError):
    """Raised when diagnosis or editing is attempted before memory output is delivered."""


class ConfirmedFixGateError(AgentError):
    """Raised when memory storage lacks current passing-test evidence."""


class AgentLimitError(AgentError):
    """Raised when a configured model-turn or local-tool limit is reached."""


class ResponsesTransport(Protocol):
    """Minimal transport boundary shared by fake and real Responses clients."""

    request_count: int

    async def create(
        self, payload: dict[str, Any], *, estimated_input_tokens: int
    ) -> dict[str, Any]: ...


class DeterministicFakeResponsesTransport:
    """Return checked-in or test-built Responses payloads without network access."""

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        if not responses:
            raise ValueError("at least one fake response is required")
        self._responses = [copy.deepcopy(dict(response)) for response in responses]
        self.requests: list[dict[str, Any]] = []
        self.responses_returned: list[dict[str, Any]] = []
        self.request_count = 0

    async def create(
        self, payload: dict[str, Any], *, estimated_input_tokens: int
    ) -> dict[str, Any]:
        if estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens cannot be negative")
        self.requests.append(copy.deepcopy(payload))
        self.request_count += 1
        if not self._responses:
            raise ResponseValidationError("fake response sequence is exhausted")
        response = self._responses.pop(0)
        self.responses_returned.append(copy.deepcopy(response))
        return response


class RealHTTPResponsesTransport:
    """Raw HTTP Responses transport that is inert unless live mode is explicit."""

    total_network_requests = 0

    def __init__(
        self,
        budget_guard: BudgetGuard,
        *,
        enabled: bool = False,
        api_key: str | None = None,
        processing_tier: str = "standard",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.budget_guard = budget_guard
        self.enabled = enabled
        self._api_key = api_key
        self.processing_tier = processing_tier
        self.timeout_seconds = timeout_seconds
        self.request_count = 0

    async def create(
        self, payload: dict[str, Any], *, estimated_input_tokens: int
    ) -> dict[str, Any]:
        if not self.enabled:
            raise LiveModeDisabledError("real Responses transport is disabled by default")
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise LiveModeDisabledError("OPENAI_API_KEY is required for explicit live mode")
        if payload.get("model") != "gpt-6-astra":
            raise ResponseValidationError("live transport permits only gpt-6-astra")
        max_output_tokens = payload.get("max_output_tokens")
        if type(max_output_tokens) is not int:
            raise ResponseValidationError("max_output_tokens must be explicit")
        self.budget_guard.check_request(
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
            processing_tier=self.processing_tier,
        )
        self.request_count += 1
        RealHTTPResponsesTransport.total_network_requests += 1
        response = await asyncio.to_thread(
            requests.post,
            RESPONSES_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ResponseValidationError("Responses API returned a non-object payload")
        return result


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_type: str
    tool_name: str | None
    session_id: str
    response_id: str | None
    call_id: str | None
    status: str
    duration_seconds: float
    timestamp: str


class EventLogger:
    """Record metadata-only JSONL events without arguments, results, or secrets."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[AgentEvent] = []

    def record(self, event: AgentEvent) -> None:
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(asdict(event), sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    session_id: str
    final_text: str
    final_response_id: str
    model_request_count: int
    local_tool_call_count: int
    events: tuple[AgentEvent, ...]


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_turns: int = 20
    max_tool_calls: int = 50
    max_output_tokens: int = 2_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_turns <= 100:
            raise ValueError("max_turns must be between 1 and 100")
        if not 1 <= self.max_tool_calls <= 500:
            raise ValueError("max_tool_calls must be between 1 and 500")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


class AstraAgent:
    """Execute complete function-call loops through a supplied Responses transport."""

    def __init__(
        self,
        transport: ResponsesTransport,
        memory_service: MemoryService,
        coding_tools: SafeCodingTools,
        *,
        limits: AgentLimits | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.transport = transport
        self.memory_service = memory_service
        self.coding_tools = coding_tools
        self.limits = limits or AgentLimits()
        self.event_logger = event_logger or EventLogger()
        self.model_request_count = 0
        self.local_tool_call_count = 0

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(tool) for tool in (*MEMORY_TOOL_DEFINITIONS, *CODING_TOOL_DEFINITIONS)
        ]

    async def run(self, prompt: str, *, session_id: str | None = None) -> AgentRunResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        run_session_id = session_id or str(uuid4())
        previous_response_id: str | None = None
        next_input: object = [{"role": "user", "content": prompt.strip()}]
        memory_result_delivered = False
        pending_memory_delivery = False
        last_confirmed_outcome: str | None = None
        start_event_index = len(self.event_logger.events)
        run_model_requests = 0
        run_local_tools = 0

        while True:
            if run_model_requests >= self.limits.max_turns:
                raise AgentLimitError("model turn limit reached")
            payload: dict[str, Any] = {
                "model": "gpt-6-astra",
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": next_input,
                "tools": self.tool_definitions,
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
                "max_output_tokens": self.limits.max_output_tokens,
            }
            if previous_response_id is not None:
                payload["previous_response_id"] = previous_response_id
            if pending_memory_delivery:
                memory_result_delivered = True
                pending_memory_delivery = False

            response = await self._request(payload, run_session_id)
            run_model_requests += 1
            response_id, output = self._validate_response(response)
            function_calls = [item for item in output if item.get("type") == "function_call"]
            if not function_calls:
                if not memory_result_delivered:
                    raise MemoryGateError(
                        "agent produced a diagnosis before receiving memory output"
                    )
                final_text = self._extract_text(output)
                return AgentRunResult(
                    session_id=run_session_id,
                    final_text=final_text,
                    final_response_id=response_id,
                    model_request_count=run_model_requests,
                    local_tool_call_count=run_local_tools,
                    events=tuple(self.event_logger.events[start_event_index:]),
                )

            outputs: list[dict[str, str]] = []
            memory_ready_at_response_start = memory_result_delivered
            search_completed = False
            for item in function_calls:
                if run_local_tools >= self.limits.max_tool_calls:
                    raise AgentLimitError("local tool-call limit reached")
                tool_output, confirmed_outcome, was_search = await self._dispatch(
                    item,
                    session_id=run_session_id,
                    response_id=response_id,
                    memory_ready=memory_ready_at_response_start,
                    last_confirmed_outcome=last_confirmed_outcome,
                )
                run_local_tools += 1
                if item.get("name") == "run_tests":
                    last_confirmed_outcome = confirmed_outcome
                if item.get("name") == "apply_edit":
                    last_confirmed_outcome = None
                search_completed = search_completed or was_search
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(item["call_id"]),
                        "output": tool_output,
                    }
                )
            if search_completed:
                pending_memory_delivery = True
            previous_response_id = response_id
            next_input = outputs

    async def _request(self, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
        started = monotonic()
        status = "ok"
        response_id: str | None = None
        try:
            serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            estimated_input_tokens = math.ceil(len(serialized.encode("utf-8")) / 4)
            response = await self.transport.create(
                payload, estimated_input_tokens=estimated_input_tokens
            )
            response_id_value = response.get("id")
            response_id = response_id_value if isinstance(response_id_value, str) else None
            return response
        except Exception:
            status = "error"
            raise
        finally:
            self.model_request_count += 1
            self.event_logger.record(
                AgentEvent(
                    event_type="model_request",
                    tool_name=None,
                    session_id=session_id,
                    response_id=response_id,
                    call_id=None,
                    status=status,
                    duration_seconds=round(monotonic() - started, 6),
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

    @staticmethod
    def _validate_response(response: object) -> tuple[str, list[dict[str, Any]]]:
        if not isinstance(response, Mapping):
            raise ResponseValidationError("response must be an object")
        response_id = response.get("id")
        output = response.get("output")
        if not isinstance(response_id, str) or not response_id:
            raise ResponseValidationError("response id is missing")
        if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
            raise ResponseValidationError("response output must be an array of objects")
        return response_id, output

    @staticmethod
    def _extract_text(output: Sequence[Mapping[str, Any]]) -> str:
        pieces: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        if not pieces:
            raise ResponseValidationError("terminal response contains no output_text")
        return "\n".join(pieces)

    async def _dispatch(
        self,
        item: Mapping[str, Any],
        *,
        session_id: str,
        response_id: str,
        memory_ready: bool,
        last_confirmed_outcome: str | None,
    ) -> tuple[str, str | None, bool]:
        name = item.get("name")
        call_id = item.get("call_id")
        if not isinstance(name, str) or not isinstance(call_id, str) or not call_id:
            raise ResponseValidationError("function call lacks name or call_id")
        started = monotonic()
        status = "ok"
        try:
            arguments = self._parse_arguments(item.get("arguments"))
            if name == "search_codebase_memory":
                self._require_keys(arguments, {"query", "error_type", "top_k"})
                result: object = await self.memory_service.search_codebase_memory(**arguments)
                was_search = True
                confirmed_outcome = None
            elif name == "store_fix_memory":
                self._require_keys(
                    arguments, {"fix_description", "error_type", "file_paths", "outcome"}
                )
                if not memory_ready:
                    raise MemoryGateError("memory storage attempted before memory result delivery")
                if last_confirmed_outcome is None or arguments["outcome"] != last_confirmed_outcome:
                    raise ConfirmedFixGateError(
                        "store_fix_memory requires the latest passing-test confirmed_outcome"
                    )
                self.memory_service.confirm_outcome(last_confirmed_outcome)
                result = {"memory_id": await self.memory_service.store_fix_memory(**arguments)}
                was_search = False
                confirmed_outcome = last_confirmed_outcome
            elif name == "list_project_files":
                self._require_keys(arguments, set())
                result = self.coding_tools.list_project_files()
                was_search = False
                confirmed_outcome = None
            elif name == "read_file":
                self._require_keys(arguments, {"path"})
                result = self.coding_tools.read_file(**arguments)
                was_search = False
                confirmed_outcome = None
            elif name == "search_project_files":
                self._require_keys(arguments, {"query", "max_results"})
                result = self.coding_tools.search_project_files(**arguments)
                was_search = False
                confirmed_outcome = None
            elif name == "run_tests":
                self._require_keys(arguments, {"command"})
                result = self.coding_tools.run_tests(**arguments)
                value = result.get("confirmed_outcome")
                confirmed_outcome = value if isinstance(value, str) else None
                was_search = False
            elif name == "apply_edit":
                self._require_keys(arguments, {"path", "old_text", "new_text"})
                if not memory_ready:
                    raise MemoryGateError("edit attempted before memory result delivery")
                result = self.coding_tools.apply_edit(**arguments)
                was_search = False
                confirmed_outcome = None
            else:
                raise UnknownToolError(f"unknown tool: {name}")
            return self._serialize_tool_output(result), confirmed_outcome, was_search
        except Exception:
            status = "rejected" if name not in {"search_codebase_memory", "run_tests"} else "error"
            raise
        finally:
            self.local_tool_call_count += 1
            self.event_logger.record(
                AgentEvent(
                    event_type="local_tool",
                    tool_name=name,
                    session_id=session_id,
                    response_id=response_id,
                    call_id=call_id,
                    status=status,
                    duration_seconds=round(monotonic() - started, 6),
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

    @staticmethod
    def _parse_arguments(value: object) -> dict[str, Any]:
        if not isinstance(value, str):
            raise ToolArgumentsError("function arguments must be a JSON string")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ToolArgumentsError("function arguments are not valid JSON") from error
        if not isinstance(parsed, dict):
            raise ToolArgumentsError("function arguments must decode to an object")
        return parsed

    @staticmethod
    def _require_keys(arguments: Mapping[str, Any], expected: set[str]) -> None:
        actual = set(arguments)
        if actual != expected:
            raise ToolArgumentsError(
                f"tool arguments must contain exactly {sorted(expected)!r}; got {sorted(actual)!r}"
            )

    @staticmethod
    def _serialize_tool_output(value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)
