"""Shared, offline-only support for controlled demo runs and evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from astra_agent import AgentEvent, AgentRunResult, DeterministicFakeResponsesTransport
from vectoraidb_memory_tools import Embedder, MemoryBackend, MemoryService

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PROJECT_ROOT / "demo_projects" / "templates"
RUN_ROOT = PROJECT_ROOT / "runs"
SCENARIOS = ("scenario_alpha", "scenario_beta")
DEMO_PROMPT = (
    "Investigate the failing acceptance test in this project. Use the available tools, "
    "make the smallest safe repair, run the tests, and store the result only if confirmed."
)
SIMULATION_LABEL = "OFFLINE DETERMINISTIC SIMULATION - NOT ASTRA OUTPUT OR BENCHMARK EVIDENCE"


@dataclass(frozen=True, slots=True)
class ToolTotals:
    model_requests: int
    memory_searches: int
    memory_stores: int
    file_calls: int
    test_calls: int
    edit_calls: int
    total_local_tool_calls: int


@dataclass(frozen=True, slots=True)
class SessionMeasurement:
    session_id: str
    scenario: str
    memory_condition: str
    memory_retrieved: bool
    elapsed_seconds: float
    passed: bool
    totals: ToolTotals


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def template_hash(scenario: str) -> str:
    root = checked_template(scenario)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checked_template(scenario: str) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    root = (TEMPLATE_ROOT / scenario).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("scenario template is not a directory")
    return root


def reset_workspace(run_id: str, scenario: str) -> Path:
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in run_id
    ):
        raise ValueError(
            "run_id must contain only lowercase letters, digits, hyphens, or underscores"
        )
    source = checked_template(scenario)
    run_parent = RUN_ROOT.resolve() / run_id
    destination = run_parent / scenario
    if not destination.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("run workspace escaped the project run root")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def measure_events(events: tuple[AgentEvent, ...]) -> ToolTotals:
    model_requests = sum(event.event_type == "model_request" for event in events)
    tools = Counter(
        event.tool_name for event in events if event.event_type == "local_tool" and event.tool_name
    )
    return ToolTotals(
        model_requests=model_requests,
        memory_searches=tools["search_codebase_memory"],
        memory_stores=tools["store_fix_memory"],
        file_calls=sum(
            tools[name] for name in ("list_project_files", "read_file", "search_project_files")
        ),
        test_calls=tools["run_tests"],
        edit_calls=tools["apply_edit"],
        total_local_tool_calls=sum(tools.values()),
    )


def assert_measurement_matches(result: AgentRunResult, totals: ToolTotals) -> None:
    if totals.model_requests != result.model_request_count:
        raise AssertionError("model-request total does not match event log")
    if totals.total_local_tool_calls != result.local_tool_call_count:
        raise AssertionError("local-tool total does not match event log")


def function_call(
    response_id: str, call_id: str, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": response_id,
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments, sort_keys=True),
            }
        ],
    }


def message(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def scenario_responses(scenario: str, session_id: str) -> list[dict[str, Any]]:
    prefix = session_id.replace("-", "_")
    if scenario == "scenario_alpha":
        old_text = 'return tuple(raw.split(","))'
        new_text = 'return tuple(value.strip() for value in raw.split(","))'
        description = (
            "A comma-separated environment value retained boundary whitespace on individual "
            "entries. Normalize each parsed entry at the configuration boundary before consumers "
            "compare exact values."
        )
        error_type = "configuration_boundary_canonicalization"
        query = "second configured origin is rejected and the expected response header is absent"
    elif scenario == "scenario_beta":
        old_text = 'return tuple(value.lower() for value in raw.split(","))'
        new_text = (
            'return tuple(value.strip().lower().replace("-", "_") for value in raw.split(","))'
        )
        description = (
            "Environment-provided feature names did not match the consumer's lowercase snake_case "
            "identifiers. Canonicalize each parsed entry at the configuration boundary before "
            "exact membership checks."
        )
        error_type = "configuration_boundary_canonicalization"
        query = "configured capability is reported disabled and no active handlers are selected"
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    outcome = "pytest passed with exit code 0"
    return [
        function_call(
            f"resp_{prefix}_01",
            f"call_{prefix}_search",
            "search_codebase_memory",
            {"query": query, "error_type": None, "top_k": 3},
        ),
        function_call(f"resp_{prefix}_02", f"call_{prefix}_list", "list_project_files", {}),
        function_call(
            f"resp_{prefix}_03", f"call_{prefix}_test_before", "run_tests", {"command": "pytest"}
        ),
        function_call(
            f"resp_{prefix}_04",
            f"call_{prefix}_read_test",
            "read_file",
            {"path": "tests/test_service.py"},
        ),
        function_call(
            f"resp_{prefix}_05", f"call_{prefix}_read_app", "read_file", {"path": "app/service.py"}
        ),
        function_call(
            f"resp_{prefix}_06",
            f"call_{prefix}_edit",
            "apply_edit",
            {"path": "app/service.py", "old_text": old_text, "new_text": new_text},
        ),
        function_call(
            f"resp_{prefix}_07", f"call_{prefix}_test_after", "run_tests", {"command": "pytest"}
        ),
        function_call(
            f"resp_{prefix}_08",
            f"call_{prefix}_store",
            "store_fix_memory",
            {
                "fix_description": description,
                "error_type": error_type,
                "file_paths": ["app/service.py"],
                "outcome": outcome,
            },
        ),
        message(f"resp_{prefix}_09", "The acceptance tests pass after the confirmed repair."),
    ]


def render_session_transcript(
    *,
    measurement: SessionMeasurement,
    transport: DeterministicFakeResponsesTransport,
    result: AgentRunResult,
) -> str:
    lines = [
        SIMULATION_LABEL,
        f"session_id={measurement.session_id}",
        f"scenario={measurement.scenario}",
    ]
    for index, (request, response) in enumerate(
        zip(transport.requests, transport.responses_returned, strict=True), start=1
    ):
        lines.append(f"\nMODEL REQUEST {index}")
        lines.append(json.dumps(request, indent=2, sort_keys=True))
        lines.append(f"FAKE RESPONSE {index}")
        lines.append(json.dumps(response, indent=2, sort_keys=True))
    lines.extend(
        (
            "\nSESSION RESULT",
            json.dumps(
                {
                    "final_text": result.final_text,
                    "measurement": {**asdict(measurement), "totals": asdict(measurement.totals)},
                },
                indent=2,
                sort_keys=True,
            ),
        )
    )
    return "\n".join(lines) + "\n"


class RunMemoryTracker:
    """Capture exact memory IDs stored by one project-owned run."""

    def __init__(self) -> None:
        self.memory_ids: list[str] = []

    def remember(self, memory_id: str) -> None:
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)


class TrackedMemoryService(MemoryService):
    """Memory service that records IDs created by its own run."""

    def __init__(
        self,
        backend: MemoryBackend,
        embedder: Embedder,
        *,
        session_id: str,
        tracker: RunMemoryTracker,
    ) -> None:
        super().__init__(backend, embedder, session_id=session_id)
        self.tracker = tracker

    async def store_fix_memory(
        self,
        *,
        fix_description: str,
        error_type: str,
        file_paths: Sequence[str],
        outcome: str,
        timestamp: datetime | None = None,
    ) -> str:
        memory_id = await super().store_fix_memory(
            fix_description=fix_description,
            error_type=error_type,
            file_paths=file_paths,
            outcome=outcome,
            timestamp=timestamp,
        )
        self.tracker.remember(memory_id)
        return memory_id


def elapsed_since(started: float) -> float:
    return round(monotonic() - started, 6)
