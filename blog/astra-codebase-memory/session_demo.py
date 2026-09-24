"""Run the two-session demonstration offline, or with explicitly approved live transport."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any

from dotenv import load_dotenv

from astra_agent import (
    AgentLimits,
    AstraAgent,
    DeterministicFakeResponsesTransport,
    EventLogger,
    RealHTTPResponsesTransport,
    ResponsesTransport,
)
from coding_tools import SafeCodingTools
from cost_guard import BudgetGuard, TokenUsage, UsageLedger, estimate_cost
from demo_support import (
    DEMO_PROMPT,
    PROJECT_ROOT,
    RUN_ROOT,
    SIMULATION_LABEL,
    RunMemoryTracker,
    SessionMeasurement,
    TrackedMemoryService,
    assert_measurement_matches,
    elapsed_since,
    measure_events,
    render_session_transcript,
    reset_workspace,
    scenario_responses,
    template_hash,
    utc_now,
)
from settings import Settings
from vectoraidb_memory_tools import (
    ActianVectorAIBackend,
    Embedder,
    MemoryBackend,
    SentenceTransformerEmbedder,
)

LIVE_MAX_TURNS = 9
LIVE_MAX_OUTPUT_TOKENS = 1_000
LIVE_ESTIMATED_INPUT_TOKENS = 6_000
LIVE_SESSION_AUTHORIZATION_USD = Decimal("1.125")


@dataclass(frozen=True, slots=True)
class DemoResult:
    run_id: str
    label: str
    started_at: str
    completed_at: str
    session_alpha: SessionMeasurement
    session_beta: SessionMeasurement
    template_hashes_before: dict[str, str]
    template_hashes_after: dict[str, str]
    session_ids_differ: bool
    first_requests_are_new_sessions: bool
    beta_retrieved_alpha_memory: bool
    cleaned_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveDemoResult:
    run_id: str
    alpha_session_id: str
    beta_session_id: str
    alpha_model_requests: int
    beta_model_requests: int
    alpha_local_tool_calls: int
    beta_local_tool_calls: int
    beta_retrieved_alpha_memory: bool
    first_requests_are_new_sessions: bool
    cleaned_memory_ids: tuple[str, ...]
    cumulative_project_cost_usd: str


def require_live_approval(
    *, live: bool, approve_live_cost: bool, api_key: str | None, processing_tier: str
) -> None:
    """Reject every incomplete live invocation before constructing an HTTP transport."""

    if not live or not approve_live_cost:
        raise SystemExit("Live mode requires both --live and --approve-live-cost.")
    if not api_key or not api_key.strip():
        raise SystemExit("Live mode requires OPENAI_API_KEY.")
    if processing_tier != "standard":
        raise SystemExit("Live mode requires Standard processing.")


class RecordedLiveTransport:
    """Persist prompt-free usage and private raw evidence after each live response."""

    def __init__(
        self,
        transport: ResponsesTransport,
        *,
        ledger: UsageLedger,
        guard: BudgetGuard,
        session_id: str,
        transcript_path: Path,
        show_response_ids: bool = True,
        show_request_costs: bool = True,
    ) -> None:
        self.transport = transport
        self.ledger = ledger
        self.guard = guard
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.show_response_ids = show_response_ids
        self.show_request_costs = show_request_costs
        self.requests: list[dict[str, Any]] = []
        self.memory_search_call_ids: set[str] = set()
        self.request_count = 0
        self.session_actual_cost_usd = Decimal("0")

    async def create(
        self, payload: dict[str, Any], *, estimated_input_tokens: int
    ) -> dict[str, Any]:
        max_output_tokens = payload.get("max_output_tokens")
        if type(max_output_tokens) is not int:
            raise RuntimeError("live request requires explicit max_output_tokens")
        maximum = self.guard.check_request(
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
            processing_tier="standard",
        )
        if self.session_actual_cost_usd + maximum > LIVE_SESSION_AUTHORIZATION_USD:
            raise RuntimeError("next request exceeds the $1.125 session authorization")
        self.requests.append(payload)
        response = await self.transport.create(
            payload, estimated_input_tokens=estimated_input_tokens
        )
        self.request_count += 1
        for item in response.get("output", []):
            if isinstance(item, dict) and item.get("name") == "search_codebase_memory":
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    self.memory_search_call_ids.add(call_id)
        usage_data = response.get("usage")
        response_id = response.get("id")
        if not isinstance(usage_data, dict) or not isinstance(response_id, str):
            raise RuntimeError("live response lacks usage or response ID; stop for review")
        usage = TokenUsage.from_response_usage(usage_data)
        estimate = estimate_cost(usage)
        self.guard.record_unpersisted_cost(estimate.total_usd)
        self.ledger.append(
            run_id=self.session_id,
            response_id=response_id,
            model=str(response.get("model", "gpt-6-astra")),
            usage=usage,
            estimate=estimate,
        )
        self.guard.mark_cost_persisted(estimate.total_usd)
        self.session_actual_cost_usd += estimate.total_usd
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("a", encoding="utf-8", newline="\n") as output:
            record = {"request": payload, "response": response}
            output.write(json.dumps(record, sort_keys=True) + "\n")
        if self.show_response_ids:
            print(f"response_id={response_id} estimated_cost_usd={estimate.total_usd}")
        elif self.show_request_costs:
            print(f"estimated_request_cost_usd={estimate.total_usd}")
        return response


def _returned_memory_text(transport: RecordedLiveTransport) -> str:
    return "\n".join(
        str(item.get("output", ""))
        for request in transport.requests
        for item in request.get("input", [])
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") in transport.memory_search_call_ids
    )


async def run_live_two_session_demo(
    *,
    run_id: str,
    backend: ActianVectorAIBackend,
    embedder: Embedder,
    evidence_dir: Path,
    ledger: UsageLedger,
    guard: BudgetGuard,
    transport_factory: Callable[[], ResponsesTransport],
) -> LiveDemoResult:
    """Run two fresh response chains; remove only their confirmed memories on success."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in run_id
    ):
        raise ValueError("run_id contains an unsupported character")
    if (RUN_ROOT / run_id).exists():
        raise RuntimeError("live run ID already exists; choose a fresh --run-id")
    schema = await backend.get_collection_schema()
    if schema.get("points_count") != 0:
        raise RuntimeError("the project collection is not empty; refusing live demo")
    maximum = (
        guard.worst_case_request_cost(
            estimated_input_tokens=LIVE_ESTIMATED_INPUT_TOKENS,
            max_output_tokens=LIVE_MAX_OUTPUT_TOKENS,
        )
        * LIVE_MAX_TURNS
    )
    if maximum > LIVE_SESSION_AUTHORIZATION_USD:
        raise RuntimeError("session authorization exceeds its approved maximum")
    if guard.current_spending_usd + maximum * 2 > guard.project_limit_usd:
        raise RuntimeError("two-session authorization exceeds the project budget")
    print(f"maximum_authorized_cost_per_session_usd={maximum}")
    tracker = RunMemoryTracker()
    before = {name: template_hash(name) for name in ("scenario_alpha", "scenario_beta")}
    results: list[Any] = []
    transports: list[RecordedLiveTransport] = []
    for scenario, suffix in (("scenario_alpha", "alpha"), ("scenario_beta", "beta")):
        session_id = f"{run_id}-session-{suffix}"
        workspace = reset_workspace(run_id, scenario)
        coding_tools = SafeCodingTools(workspace)
        if coding_tools.run_tests("pytest")["exit_code"] == 0:
            raise RuntimeError(f"{scenario} acceptance tests did not initially fail")
        transport = RecordedLiveTransport(
            transport_factory(),
            ledger=ledger,
            guard=guard,
            session_id=session_id,
            transcript_path=evidence_dir / "private" / f"{run_id}-{suffix}-transcript.jsonl",
        )
        service = TrackedMemoryService(backend, embedder, session_id=session_id, tracker=tracker)
        agent = AstraAgent(
            transport,
            service,
            coding_tools,
            limits=AgentLimits(max_turns=LIVE_MAX_TURNS, max_output_tokens=LIVE_MAX_OUTPUT_TOKENS),
            event_logger=EventLogger(evidence_dir / "private" / f"{run_id}-{suffix}-events.jsonl"),
        )
        result = await agent.run(DEMO_PROMPT, session_id=session_id)
        if coding_tools.run_tests("pytest")["exit_code"] != 0:
            raise RuntimeError(f"{scenario} final acceptance tests failed")
        if len(tracker.memory_ids) != len(results) + 1:
            raise RuntimeError(f"{scenario} did not store exactly one confirmed memory")
        results.append(result)
        transports.append(transport)
        if suffix == "alpha" and "No relevant confirmed fixes found." not in _returned_memory_text(
            transport
        ):
            raise RuntimeError("Scenario Alpha did not receive an empty memory result")
        if suffix == "beta" and f"Session: {results[0].session_id}" not in _returned_memory_text(
            transport
        ):
            raise RuntimeError("Scenario Beta did not retrieve the Session 1 memory")
    if before != {name: template_hash(name) for name in before}:
        raise RuntimeError("an immutable scenario template changed")
    new_chains = all(
        bool(transport.requests) and "previous_response_id" not in transport.requests[0]
        for transport in transports
    )
    if not new_chains:
        raise RuntimeError("a session reused previous_response_id")
    owned_ids = tuple(tracker.memory_ids)
    for memory_id in owned_ids:
        await backend.delete(memory_id)
    return LiveDemoResult(
        run_id=run_id,
        alpha_session_id=results[0].session_id,
        beta_session_id=results[1].session_id,
        alpha_model_requests=results[0].model_request_count,
        beta_model_requests=results[1].model_request_count,
        alpha_local_tool_calls=results[0].local_tool_call_count,
        beta_local_tool_calls=results[1].local_tool_call_count,
        beta_retrieved_alpha_memory=True,
        first_requests_are_new_sessions=True,
        cleaned_memory_ids=owned_ids,
        cumulative_project_cost_usd=str(guard.current_spending_usd),
    )


def _memory_output(transport: DeterministicFakeResponsesTransport) -> str:
    if len(transport.requests) < 2:
        raise AssertionError("memory output was not returned to the fake model")
    item = transport.requests[1]["input"][0]
    if item.get("type") != "function_call_output":
        raise AssertionError("first tool output was not returned as function_call_output")
    return str(item.get("output", ""))


async def run_offline_session(
    *,
    run_id: str,
    scenario: str,
    session_id: str,
    memory_condition: str,
    backend: MemoryBackend,
    embedder: Embedder,
    tracker: RunMemoryTracker,
    event_path: Path,
) -> tuple[SessionMeasurement, DeterministicFakeResponsesTransport, Any, str]:
    workspace = reset_workspace(run_id, scenario)
    transport = DeterministicFakeResponsesTransport(scenario_responses(scenario, session_id))
    service = TrackedMemoryService(
        backend,
        embedder,
        session_id=session_id,
        tracker=tracker,
    )
    agent = AstraAgent(
        transport,
        service,
        SafeCodingTools(workspace),
        event_logger=EventLogger(event_path),
    )
    started = monotonic()
    result = await agent.run(DEMO_PROMPT, session_id=session_id)
    elapsed = elapsed_since(started)
    totals = measure_events(result.events)
    assert_measurement_matches(result, totals)
    memory_output = _memory_output(transport)
    measurement = SessionMeasurement(
        session_id=session_id,
        scenario=scenario,
        memory_condition=memory_condition,
        memory_retrieved="Match 1" in memory_output,
        elapsed_seconds=elapsed,
        passed="acceptance tests pass" in result.final_text.casefold(),
        totals=totals,
    )
    return measurement, transport, result, memory_output


async def run_two_session_demo(
    *,
    run_id: str,
    backend: MemoryBackend,
    embedder: Embedder,
    evidence_dir: Path,
) -> DemoResult:
    """Run two independent fake-response sessions and clean only their memories."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    event_path = evidence_dir / "phase4_two_session_events.jsonl"
    event_path.unlink(missing_ok=True)
    transcript_path = evidence_dir / "phase4_two_session_simulation.txt"
    before = {scenario: template_hash(scenario) for scenario in ("scenario_alpha", "scenario_beta")}
    tracker = RunMemoryTracker()
    started_at = utc_now()
    transcript_parts: list[str] = []
    alpha_session_id = f"{run_id}-session-alpha"
    beta_session_id = f"{run_id}-session-beta"
    try:
        alpha, alpha_transport, alpha_result, alpha_memory_output = await run_offline_session(
            run_id=run_id,
            scenario="scenario_alpha",
            session_id=alpha_session_id,
            memory_condition="empty relevant memory",
            backend=backend,
            embedder=embedder,
            tracker=tracker,
            event_path=event_path,
        )
        if "No relevant confirmed fixes found." not in alpha_memory_output:
            raise RuntimeError("Scenario Alpha did not start with an empty relevant memory result")
        transcript_parts.append(
            render_session_transcript(
                measurement=alpha,
                transport=alpha_transport,
                result=alpha_result,
            )
        )

        beta, beta_transport, beta_result, beta_memory_output = await run_offline_session(
            run_id=run_id,
            scenario="scenario_beta",
            session_id=beta_session_id,
            memory_condition="confirmed Scenario Alpha memory available",
            backend=backend,
            embedder=embedder,
            tracker=tracker,
            event_path=event_path,
        )
        retrieved = alpha_session_id in beta_memory_output and "Match 1" in beta_memory_output
        if not retrieved:
            raise RuntimeError("Scenario Beta did not retrieve the Scenario Alpha memory")
        transcript_parts.append(
            render_session_transcript(
                measurement=beta,
                transport=beta_transport,
                result=beta_result,
            )
        )
        first_requests_are_new = all(
            "previous_response_id" not in transport.requests[0]
            for transport in (alpha_transport, beta_transport)
        )
        if not first_requests_are_new:
            raise AssertionError("an independent session reused previous_response_id")
        after = {
            scenario: template_hash(scenario) for scenario in ("scenario_alpha", "scenario_beta")
        }
        if before != after:
            raise AssertionError("an immutable scenario template changed during the run")
        result = DemoResult(
            run_id=run_id,
            label=SIMULATION_LABEL,
            started_at=started_at,
            completed_at=utc_now(),
            session_alpha=alpha,
            session_beta=beta,
            template_hashes_before=before,
            template_hashes_after=after,
            session_ids_differ=alpha_session_id != beta_session_id,
            first_requests_are_new_sessions=first_requests_are_new,
            beta_retrieved_alpha_memory=retrieved,
            cleaned_memory_ids=tuple(tracker.memory_ids),
        )
        transcript_parts.append(
            "\nDEMO SUMMARY\n" + json.dumps(asdict(result), indent=2, sort_keys=True)
        )
        transcript_path.write_text("\n".join(transcript_parts) + "\n", encoding="utf-8")
        return result
    finally:
        for memory_id in tracker.memory_ids:
            await backend.delete(memory_id)


async def _main_async(run_id: str) -> DemoResult:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_environment()
    backend = ActianVectorAIBackend(
        settings.vectorai_url,
        collection_name=settings.vectorai_collection,
        grpc_url=settings.vectorai_grpc_url,
    )
    await backend.ensure_collection()
    schema = await backend.get_collection_schema()
    if schema.get("points_count") != 0:
        raise RuntimeError(
            "the project collection is not empty; refusing to alter or hide existing records"
        )
    return await run_two_session_demo(
        run_id=run_id,
        backend=backend,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        evidence_dir=PROJECT_ROOT / "evidence",
    )


async def _live_main_async(run_id: str, settings: Settings) -> LiveDemoResult:
    evidence_dir = PROJECT_ROOT / "evidence"
    ledger = UsageLedger(evidence_dir / "usage.jsonl")
    guard = BudgetGuard(
        project_limit_usd=settings.project_budget_usd,
        max_input_tokens=settings.max_input_tokens,
        ledger=ledger,
    )
    backend = ActianVectorAIBackend(
        settings.vectorai_url,
        collection_name=settings.vectorai_collection,
        grpc_url=settings.vectorai_grpc_url,
    )
    await backend.ensure_collection()
    return await run_live_two_session_demo(
        run_id=run_id,
        backend=backend,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        evidence_dir=evidence_dir,
        ledger=ledger,
        guard=guard,
        transport_factory=lambda: RealHTTPResponsesTransport(
            guard,
            enabled=True,
            api_key=settings.api_key,
            processing_tier=settings.processing_tier,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="offline-two-session")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use real Astra Responses transport after explicit cost approval",
    )
    parser.add_argument("--approve-live-cost", action="store_true")
    args = parser.parse_args()
    if args.live or args.approve_live_cost:
        if not args.live or not args.approve_live_cost:
            raise SystemExit("Live mode requires both --live and --approve-live-cost.")
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        settings = Settings.from_environment()
        require_live_approval(
            live=args.live,
            approve_live_cost=args.approve_live_cost,
            api_key=settings.api_key,
            processing_tier=settings.processing_tier,
        )
        result = asyncio.run(_live_main_async(args.run_id, settings))
    else:
        result = asyncio.run(_main_async(args.run_id))
        print(SIMULATION_LABEL)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
