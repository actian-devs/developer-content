"""Present the real two-session Astra memory workflow as a readable terminal narrative."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv

from astra_agent import (
    AgentLimits,
    AgentRunResult,
    AstraAgent,
    DeterministicFakeResponsesTransport,
    RealHTTPResponsesTransport,
    ResponsesTransport,
)
from coding_tools import SafeCodingTools
from cost_guard import BudgetGuard, UsageLedger
from demo_support import (
    DEMO_PROMPT,
    PROJECT_ROOT,
    RUN_ROOT,
    RunMemoryTracker,
    TrackedMemoryService,
    checked_template,
    reset_workspace,
    scenario_responses,
    template_hash,
)
from session_demo import (
    LIVE_ESTIMATED_INPUT_TOKENS,
    LIVE_MAX_OUTPUT_TOKENS,
    LIVE_MAX_TURNS,
    LIVE_SESSION_AUTHORIZATION_USD,
    RecordedLiveTransport,
    require_live_approval,
)
from settings import Settings
from vectoraidb_memory_tools import (
    EMBEDDING_DIMENSION,
    ActianVectorAIBackend,
    Embedder,
    InMemoryMemoryBackend,
    MemoryBackend,
    SentenceTransformerEmbedder,
)

PRIVATE_IDENTIFIER = re.compile(r"\b(?:resp|call)_[A-Za-z0-9_-]+\b")
SECRET_VALUE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
MAX_DISPLAY_CHARS = 700


class OfflineEmbedder:
    """Deterministic local vector used only by the fake video preview."""

    async def embed(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1)


def _safe_text(value: object, *, limit: int = MAX_DISPLAY_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = PRIVATE_IDENTIFIER.sub("[private-id]", text)
    text = SECRET_VALUE.sub("[redacted-secret]", text)
    if len(text) > limit:
        return text[: limit - 20].rstrip() + " ... [shortened]"
    return text


class VideoPresenter:
    """Render only events and results observed from the active run."""

    def __init__(self, output: TextIO = sys.stdout, *, verbose: bool = False) -> None:
        self.output = output
        self.verbose = verbose
        self.tool_names: list[str] = []
        self.search_outputs: list[str] = []
        self.stored_memory_ids: list[str] = []

    def line(self, value: str = "") -> None:
        print(value, file=self.output, flush=True)

    def title(self, *, live: bool, run_id: str) -> None:
        mode = "LIVE PAID ASTRA RUN" if live else "OFFLINE FAKE-TRANSPORT PREVIEW"
        self.line("=" * 72)
        self.line("ASTRA CROSS-SESSION CODEBASE MEMORY DEMONSTRATION")
        self.line("=" * 72)
        self.line(f"Mode: {mode}")
        self.line(f"Run ID: {run_id}")
        self.line(
            "Session 1 will solve Scenario Alpha and store a tested fix. Session 2 "
            "will start a separate response chain and search for that fix before editing."
        )
        self.line(
            "Memory retrieval is evidence of cross-session availability; untouched "
            "acceptance tests are the correctness evidence."
        )

    def session_start(self, number: int, scenario: str, session_id: str) -> None:
        self.tool_names = []
        self.search_outputs = []
        self.stored_memory_ids = []
        self.line()
        self.line("-" * 72)
        self.line(f"SESSION {number}: {scenario.replace('_', ' ').title()}")
        self.line("-" * 72)
        self.line(f"Session ID: {session_id}")
        self.line("Response chain: NEW (first request omits previous_response_id)")
        self.line("Task sent to Astra:")
        self.line(_safe_text(DEMO_PROMPT))

    def baseline_test(self, result: Mapping[str, object]) -> None:
        self.line(f"Baseline acceptance tests: FAIL as expected (exit {result.get('exit_code')})")
        self._test_details(result)

    def model_request(self, number: int, *, continues_chain: bool) -> None:
        if not self.verbose:
            return
        chain = "continuation" if continues_chain else "new chain"
        self.line(f"[Astra request {number}] {chain}")

    def tool_call(self, name: str, arguments: Mapping[str, Any]) -> None:
        self.tool_names.append(name)
        if not self.verbose:
            if name == "search_codebase_memory":
                self.line("Memory search requested.")
            elif name == "list_project_files":
                self.line("Listing project files.")
            elif name == "read_file":
                self.line(f"Inspecting file: {_safe_text(arguments.get('path', 'unknown'))}")
            elif name == "search_project_files":
                self.line(
                    "Searching project files for: "
                    + _safe_text(arguments.get("query", "unknown"), limit=120)
                )
            elif name == "apply_edit":
                path = _safe_text(arguments.get("path", "unknown"), limit=120)
                old = _safe_text(arguments.get("old_text", ""), limit=100)
                new = _safe_text(arguments.get("new_text", ""), limit=100)
                self.line(f"Editing {path}: {old} -> {new}")
            elif name == "run_tests":
                self.line("Running the original acceptance tests.")
            elif name == "store_fix_memory":
                self.line("Storing the confirmed fix after passing tests.")
            else:
                self.line(f"Tool call: {name}")
            return
        self.line(f"  -> Tool call: {name}")
        if name == "apply_edit":
            edit = {
                "path": arguments.get("path"),
                "replace": arguments.get("old_text"),
                "with": arguments.get("new_text"),
            }
            self.line("     " + _safe_text(edit))
        elif arguments:
            self.line("     " + _safe_text(arguments))

    def tool_result(self, name: str, output: str) -> None:
        try:
            parsed: object = json.loads(output)
        except json.JSONDecodeError:
            parsed = output
        if not self.verbose:
            self._concise_tool_result(name, parsed)
        elif name == "run_tests" and isinstance(parsed, Mapping):
            self.line(f"  <- Tool result: {name}")
            self._test_details(parsed)
        else:
            self.line(f"  <- Tool result: {name}")
            self.line("     " + _safe_text(parsed))
        if name == "search_codebase_memory":
            self.search_outputs.append(output)
        if name == "store_fix_memory" and isinstance(parsed, Mapping):
            memory_id = parsed.get("memory_id")
            if isinstance(memory_id, str):
                self.stored_memory_ids.append(memory_id)
                prefix = "     " if self.verbose else ""
                self.line(prefix + "Confirmed tested fix stored for the next session.")

    def _concise_tool_result(self, name: str, result: object) -> None:
        if name == "search_codebase_memory":
            text = str(result)
            if text == "No relevant confirmed fixes found.":
                self.line("Memory search result: no relevant confirmed fix found.")
                return
            lines = [line for line in text.splitlines() if line.strip()]
            summary = [line for line in lines if line.startswith(("Match ", "Fix: ", "Session: "))]
            self.line("Memory search result:")
            for line in summary:
                self.line(f"  {line}")
        elif name == "list_project_files" and isinstance(result, Mapping):
            files = result.get("files", [])
            if isinstance(files, list):
                self.line("Project files: " + ", ".join(str(path) for path in files))
        elif name == "read_file" and isinstance(result, Mapping):
            self.line("File inspection completed.")
        elif name == "search_project_files" and isinstance(result, Mapping):
            matches = result.get("matches", [])
            count = len(matches) if isinstance(matches, list) else 0
            self.line(f"Project search completed: {count} matching lines.")
        elif name == "apply_edit" and isinstance(result, Mapping):
            self.line(
                f"Edit applied to {result.get('path', 'file')} "
                f"({result.get('replacements', 0)} replacement)."
            )
        elif name == "run_tests" and isinstance(result, Mapping):
            self._test_details(result)
        elif name != "store_fix_memory":
            self.line(f"Tool result received: {name}.")

    def _test_details(self, result: Mapping[str, object]) -> None:
        if not self.verbose:
            stdout = str(result.get("stdout", ""))
            passed = sum(int(value) for value in re.findall(r"(\d+) passed", stdout))
            failed = sum(int(value) for value in re.findall(r"(\d+) failed", stdout))
            counts: list[str] = []
            if passed:
                counts.append(f"{passed} passed")
            if failed:
                counts.append(f"{failed} failed")
            count_text = ", ".join(counts) if counts else "counts unavailable"
            self.line(f"Tests: {count_text} (exit {result.get('exit_code')}).")
            return
        self.line(
            "     "
            + _safe_text(
                {
                    "exit_code": result.get("exit_code"),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                }
            )
        )

    def final_response(self, text: str) -> None:
        self.line("Astra final response:")
        self.line(_safe_text(text, limit=1_500))

    def session_summary(
        self,
        *,
        result: AgentRunResult,
        fresh_audit: Mapping[str, object],
        session_cost: Decimal,
        retrieval_statement: str,
        excluded_test_edits: Sequence[str],
    ) -> None:
        self.line(retrieval_statement)
        if excluded_test_edits:
            self.line(
                "Astra edited "
                + ", ".join(excluded_test_edits)
                + "; the test edit was excluded from the fresh acceptance audit."
            )
        stdout = str(fresh_audit.get("stdout", ""))
        passed = sum(int(value) for value in re.findall(r"(\d+) passed", stdout))
        if fresh_audit.get("exit_code") == 0 and passed:
            self.line(f"Fresh audit with the original untouched tests: {passed} passed")
        else:
            self.line("Fresh audit with the original untouched tests: FAILED")
            self._test_details(fresh_audit)
        self.line(
            f"Session totals: {result.model_request_count} API/model requests, "
            f"{result.local_tool_call_count} local tool calls, "
            f"estimated cost ${session_cost:.7f}"
        )

    def success_summary(self, *, total_cost: Decimal, cleaned_records: int) -> None:
        self.line()
        self.line("=" * 72)
        self.line("DEMONSTRATION COMPLETE")
        self.line("=" * 72)
        self.line("Verified: both sessions began independent response chains.")
        self.line("Verified: Session 2 received Session 1's confirmed fix from VectorAI DB.")
        self.line("Verified: both fixes passed the original untouched acceptance tests.")
        self.line(
            "Conclusion: the run proves cross-session fix availability and separate test-based "
            "correctness. It does not prove that retrieval caused the repair or reduced work."
        )
        self.line(f"Run-owned memory records cleaned: {cleaned_records}")
        self.line(f"Total estimated API cost: ${total_cost:.7f}")

    def failure(self, error: BaseException) -> None:
        self.line()
        self.line("DEMONSTRATION FAILED")
        self.line(_safe_text(str(error)))
        self.line("No successful cross-session handoff is claimed.")


class PresentedTransport:
    """Display actual response tool calls and the tool outputs returned on the next request."""

    def __init__(self, transport: ResponsesTransport, presenter: VideoPresenter) -> None:
        self.transport = transport
        self.presenter = presenter
        self.request_count = 0
        self.pending_tools: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.edit_arguments: list[dict[str, Any]] = []

    async def create(
        self, payload: dict[str, Any], *, estimated_input_tokens: int
    ) -> dict[str, Any]:
        self.request_count += 1
        self.requests.append(payload)
        inputs = payload.get("input")
        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            outputs = [
                item
                for item in inputs
                if isinstance(item, Mapping) and item.get("type") == "function_call_output"
            ]
            for index, item in enumerate(outputs):
                name = self.pending_tools[index] if index < len(self.pending_tools) else "tool"
                self.presenter.tool_result(name, str(item.get("output", "")))
        self.presenter.model_request(
            self.request_count, continues_chain="previous_response_id" in payload
        )
        response = await self.transport.create(
            payload, estimated_input_tokens=estimated_input_tokens
        )
        self.pending_tools = []
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping) or item.get("type") != "function_call":
                    continue
                name = item.get("name")
                arguments = item.get("arguments")
                if not isinstance(name, str):
                    continue
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else {}
                except json.JSONDecodeError:
                    parsed = {"malformed_arguments": True}
                if not isinstance(parsed, dict):
                    parsed = {"arguments": parsed}
                self.presenter.tool_call(name, parsed)
                if name == "apply_edit":
                    self.edit_arguments.append(parsed)
                self.pending_tools.append(name)
        return response


def _fresh_acceptance_audit(
    *, scenario: str, workspace: Path, run_path: Path, edited_paths: Sequence[str]
) -> dict[str, object]:
    """Test edited application files against a fresh copy of the immutable tests."""

    audit_root = run_path / f"fresh-audit-{scenario}"
    shutil.copytree(checked_template(scenario), audit_root)
    application_paths = sorted({path for path in edited_paths if not path.startswith("tests/")})
    for relative in application_paths:
        source = (workspace / relative).resolve(strict=True)
        destination = (audit_root / relative).resolve()
        if audit_root.resolve() not in destination.parents:
            raise RuntimeError("fresh audit path escaped its workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return SafeCodingTools(audit_root).run_tests("pytest")


@dataclass(frozen=True, slots=True)
class VideoRunResult:
    run_id: str
    first_requests_are_new_sessions: bool
    beta_retrieved_alpha_memory: bool
    alpha_model_requests: int
    beta_model_requests: int
    alpha_local_tool_calls: int
    beta_local_tool_calls: int
    total_cost_usd: str
    cleaned_memory_ids: tuple[str, ...]


def _fake_transport_factory() -> Callable[[str, str], ResponsesTransport]:
    def create(scenario: str, session_id: str) -> ResponsesTransport:
        source_responses = scenario_responses(scenario, session_id)
        if scenario == "scenario_beta":
            source_responses[5]["output"].append(
                {
                    "type": "function_call",
                    "call_id": f"call_{session_id.replace('-', '_')}_edit_test",
                    "name": "apply_edit",
                    "arguments": json.dumps(
                        {
                            "path": "tests/test_service.py",
                            "old_text": "assert len(configured_values(ENVIRONMENT)) == 2",
                            "new_text": (
                                "assert configured_values(ENVIRONMENT) == "
                                '("audit_log", "email_digest")'
                            ),
                        },
                        sort_keys=True,
                    ),
                }
            )
        responses = [
            {
                **response,
                "model": "gpt-6-astra",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            for response in source_responses
        ]
        return DeterministicFakeResponsesTransport(responses)

    return create


async def run_video_demo(
    *,
    run_id: str,
    backend: MemoryBackend,
    embedder: Embedder,
    transport_factory: Callable[[str, str], ResponsesTransport],
    presenter: VideoPresenter,
    live: bool,
    ledger: UsageLedger | None = None,
    guard: BudgetGuard | None = None,
    evidence_dir: Path | None = None,
) -> VideoRunResult:
    """Run the two real workflows while presenting observed events without private IDs."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in run_id
    ):
        raise ValueError("run_id contains an unsupported character")
    run_path = RUN_ROOT / run_id
    if run_path.exists():
        raise RuntimeError("run ID already exists; choose a fresh --run-id")
    before = {name: template_hash(name) for name in ("scenario_alpha", "scenario_beta")}
    if isinstance(backend, ActianVectorAIBackend):
        await backend.health_check()
        await backend.ensure_collection()
        schema = await backend.get_collection_schema()
        if schema.get("points_count") != 0:
            raise RuntimeError("the project collection is not empty; refusing the video demo")
    if live:
        if ledger is None or guard is None or evidence_dir is None:
            raise ValueError("live video mode requires ledger, guard, and evidence directory")
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
        presenter.line("PAID LIVE MODE: explicit approval received; API charges may be incurred.")
        presenter.line(f"Maximum authorized cost per session: ${maximum}")
        presenter.line(f"Maximum authorized cost for both sessions: ${maximum * 2}")

    if live and isinstance(embedder, SentenceTransformerEmbedder):
        if presenter.verbose:
            await embedder.embed("video demonstration embedding warmup")
        else:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                await embedder.embed("video demonstration embedding warmup")

    presenter.title(live=live, run_id=run_id)
    tracker = RunMemoryTracker()
    results: list[AgentRunResult] = []
    first_requests: list[dict[str, Any]] = []
    total_cost = Decimal("0")
    alpha_memory_id: str | None = None
    video_result: VideoRunResult | None = None
    try:
        for number, (scenario, suffix) in enumerate(
            (("scenario_alpha", "alpha"), ("scenario_beta", "beta")), start=1
        ):
            session_id = f"{run_id}-session-{suffix}"
            presenter.session_start(number, scenario, session_id)
            workspace = reset_workspace(run_id, scenario)
            tools = SafeCodingTools(workspace)
            baseline = tools.run_tests("pytest")
            if baseline.get("exit_code") == 0:
                raise RuntimeError(f"{scenario} acceptance tests did not initially fail")
            presenter.baseline_test(baseline)

            raw_transport = transport_factory(scenario, session_id)
            recorded: RecordedLiveTransport | None = None
            if live:
                assert ledger is not None and guard is not None and evidence_dir is not None
                recorded = RecordedLiveTransport(
                    raw_transport,
                    ledger=ledger,
                    guard=guard,
                    session_id=session_id,
                    transcript_path=evidence_dir
                    / "private"
                    / f"video-{run_id}-{suffix}-transcript.jsonl",
                    show_response_ids=False,
                    show_request_costs=presenter.verbose,
                )
                active_transport: ResponsesTransport = recorded
            else:
                active_transport = raw_transport
            presented = PresentedTransport(active_transport, presenter)
            service = TrackedMemoryService(
                backend, embedder, session_id=session_id, tracker=tracker
            )
            agent = AstraAgent(
                presented,
                service,
                tools,
                limits=AgentLimits(
                    max_turns=LIVE_MAX_TURNS, max_output_tokens=LIVE_MAX_OUTPUT_TOKENS
                ),
            )
            result = await agent.run(DEMO_PROMPT, session_id=session_id)
            presenter.final_response(result.final_text)
            final_test = tools.run_tests("pytest")
            if final_test.get("exit_code") != 0:
                raise RuntimeError(f"{scenario} final workspace acceptance tests failed")
            edited_paths = [
                str(request.get("path"))
                for request in presented.edit_arguments
                if isinstance(request.get("path"), str)
            ]
            fresh_audit = _fresh_acceptance_audit(
                scenario=scenario,
                workspace=workspace,
                run_path=run_path,
                edited_paths=edited_paths,
            )
            if fresh_audit.get("exit_code") != 0:
                raise RuntimeError(f"{scenario} failed the fresh untouched-test audit")
            if not presented.requests or "previous_response_id" in presented.requests[0]:
                raise RuntimeError(f"{scenario} did not begin a new response chain")
            first_requests.append(presented.requests[0])
            if len(tracker.memory_ids) != number:
                raise RuntimeError(f"{scenario} did not store exactly one confirmed fix")
            if number == 1:
                alpha_memory_id = tracker.memory_ids[0]
                if (
                    not presenter.search_outputs
                    or "No relevant confirmed fixes found." not in presenter.search_outputs[0]
                ):
                    raise RuntimeError("Session 1 did not receive an empty memory-search result")
                retrieval = "Session 1 memory search: no relevant confirmed fix found."
            else:
                retrieved = bool(
                    alpha_memory_id
                    and results
                    and presenter.search_outputs
                    and f"Session: {results[0].session_id}" in presenter.search_outputs[0]
                )
                if not retrieved:
                    raise RuntimeError("Session 2 did not retrieve the Session 1 memory record")
                retrieval = "Session 2 memory search: retrieved the Session 1 confirmed-fix record."
            session_cost = recorded.session_actual_cost_usd if recorded else Decimal("0")
            total_cost += session_cost
            presenter.session_summary(
                result=result,
                fresh_audit=fresh_audit,
                session_cost=session_cost,
                retrieval_statement=retrieval,
                excluded_test_edits=sorted(
                    path for path in edited_paths if path.startswith("tests/")
                ),
            )
            results.append(result)
        if before != {name: template_hash(name) for name in before}:
            raise RuntimeError("an immutable scenario template changed")
        video_result = VideoRunResult(
            run_id=run_id,
            first_requests_are_new_sessions=all(
                "previous_response_id" not in request for request in first_requests
            ),
            beta_retrieved_alpha_memory=True,
            alpha_model_requests=results[0].model_request_count,
            beta_model_requests=results[1].model_request_count,
            alpha_local_tool_calls=results[0].local_tool_call_count,
            beta_local_tool_calls=results[1].local_tool_call_count,
            total_cost_usd=str(total_cost),
            cleaned_memory_ids=tuple(tracker.memory_ids),
        )
    except Exception as error:
        presenter.failure(error)
        raise
    finally:
        for memory_id in tracker.memory_ids:
            await backend.delete(memory_id)
        if run_path.exists():
            resolved_run = run_path.resolve()
            resolved_root = RUN_ROOT.resolve()
            if resolved_run.parent != resolved_root:
                raise RuntimeError("refusing to clean a workspace outside the run root")
            shutil.rmtree(resolved_run)
    if video_result is None:
        raise RuntimeError("video demonstration ended without a result")
    presenter.success_summary(total_cost=total_cost, cleaned_records=len(tracker.memory_ids))
    return video_result


async def _offline_main(run_id: str, presenter: VideoPresenter) -> VideoRunResult:
    return await run_video_demo(
        run_id=run_id,
        backend=InMemoryMemoryBackend(),
        embedder=OfflineEmbedder(),
        transport_factory=_fake_transport_factory(),
        presenter=presenter,
        live=False,
    )


async def _live_main(run_id: str, settings: Settings, presenter: VideoPresenter) -> VideoRunResult:
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

    def factory(_scenario: str, _session_id: str) -> ResponsesTransport:
        return RealHTTPResponsesTransport(
            guard,
            enabled=True,
            api_key=settings.api_key,
            processing_tier=settings.processing_tier,
        )

    return await run_video_demo(
        run_id=run_id,
        backend=backend,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        transport_factory=factory,
        presenter=presenter,
        live=True,
        ledger=ledger,
        guard=guard,
        evidence_dir=evidence_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="fresh identifier for isolated workspaces")
    parser.add_argument("--live", action="store_true", help="enable paid Astra API transport")
    parser.add_argument("--approve-live-cost", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="show detailed tool arguments and shortened output"
    )
    args = parser.parse_args()
    presenter = VideoPresenter(verbose=args.verbose)
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
        asyncio.run(_live_main(args.run_id, settings, presenter))
    else:
        asyncio.run(_offline_main(args.run_id, presenter))


if __name__ == "__main__":
    main()
