"""Astra token-cost accounting, budget enforcement, and append-only usage logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ONE_MILLION = Decimal(1_000_000)
DEFAULT_PROJECT_LIMIT_USD = Decimal("8.50")
MAX_INPUT_TOKENS = 272_000


class UsageValidationError(ValueError):
    """Raised when response usage fields are missing or internally inconsistent."""


class BudgetExceededError(RuntimeError):
    """Raised before a request that could exceed a project safety limit."""


@dataclass(frozen=True, slots=True)
class AstraPricing:
    """Standard-processing prices in US dollars per million tokens."""

    input_per_million: Decimal = Decimal("10")
    cached_input_per_million: Decimal = Decimal("1")
    cache_write_per_million: Decimal = Decimal("12.50")
    output_per_million: Decimal = Decimal("50")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Validated usage categories returned by one Responses API response."""

    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if type(value) is not int or value < 0:
                raise UsageValidationError(f"{name} must be a non-negative integer")
        if self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            raise UsageValidationError(
                "cached and cache-write input tokens cannot exceed total input tokens"
            )

    @property
    def ordinary_input_tokens(self) -> int:
        """Return input tokens billed at the uncached input rate."""

        return self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens

    @classmethod
    def from_response_usage(cls, usage: Mapping[str, Any]) -> TokenUsage:
        """Parse the usage object from a Responses API payload."""

        details = usage.get("input_tokens_details", {})
        if not isinstance(details, Mapping):
            raise UsageValidationError("input_tokens_details must be an object")
        try:
            return cls(
                input_tokens=usage["input_tokens"],
                cached_input_tokens=details.get("cached_tokens", 0),
                cache_write_input_tokens=details.get("cache_write_tokens", 0),
                output_tokens=usage["output_tokens"],
            )
        except KeyError as error:
            raise UsageValidationError(f"missing usage field: {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Traceable cost breakdown for one response."""

    ordinary_input_usd: Decimal
    cached_input_usd: Decimal
    cache_write_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return (
            self.ordinary_input_usd + self.cached_input_usd + self.cache_write_usd + self.output_usd
        )


def estimate_cost(usage: TokenUsage, pricing: AstraPricing | None = None) -> CostEstimate:
    """Calculate standard-tier cost without prematurely rounding any category."""

    pricing = pricing or AstraPricing()

    return CostEstimate(
        ordinary_input_usd=(
            Decimal(usage.ordinary_input_tokens) * pricing.input_per_million / ONE_MILLION
        ),
        cached_input_usd=(
            Decimal(usage.cached_input_tokens) * pricing.cached_input_per_million / ONE_MILLION
        ),
        cache_write_usd=(
            Decimal(usage.cache_write_input_tokens) * pricing.cache_write_per_million / ONE_MILLION
        ),
        output_usd=(Decimal(usage.output_tokens) * pricing.output_per_million / ONE_MILLION),
    )


@dataclass(slots=True)
class BudgetGuard:
    """Reject oversized or potentially over-budget requests before transmission."""

    committed_cost_usd: Decimal = Decimal("0")
    project_limit_usd: Decimal = DEFAULT_PROJECT_LIMIT_USD
    max_input_tokens: int = MAX_INPUT_TOKENS
    ledger: UsageLedger | None = None

    def __post_init__(self) -> None:
        if self.committed_cost_usd < 0:
            raise ValueError("committed_cost_usd cannot be negative")
        if self.project_limit_usd <= 0:
            raise ValueError("project_limit_usd must be positive")
        if not 0 < self.max_input_tokens <= MAX_INPUT_TOKENS:
            raise ValueError("max_input_tokens must be between 1 and 272000")

    @property
    def remaining_usd(self) -> Decimal:
        return max(self.project_limit_usd - self.current_spending_usd, Decimal("0"))

    @property
    def current_spending_usd(self) -> Decimal:
        """Return explicit committed spending plus persisted ledger spending."""

        ledger_cost = self.ledger.cumulative_cost_usd() if self.ledger else Decimal("0")
        return self.committed_cost_usd + ledger_cost

    def worst_case_request_cost(
        self,
        *,
        estimated_input_tokens: int,
        max_output_tokens: int,
        possible_cache_write_tokens: int | None = None,
    ) -> Decimal:
        """Price a request without assuming any cached-input discount."""

        if type(estimated_input_tokens) is not int or estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be a non-negative integer")
        if estimated_input_tokens > self.max_input_tokens:
            raise BudgetExceededError(
                f"request has {estimated_input_tokens} estimated input tokens; "
                f"limit is {self.max_input_tokens}"
            )
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if possible_cache_write_tokens is None:
            possible_cache_write_tokens = estimated_input_tokens
        if (
            type(possible_cache_write_tokens) is not int
            or not 0 <= possible_cache_write_tokens <= estimated_input_tokens
        ):
            raise ValueError(
                "possible_cache_write_tokens must be between zero and estimated input tokens"
            )

        ordinary_tokens = estimated_input_tokens - possible_cache_write_tokens
        input_cost = (
            Decimal(ordinary_tokens) * AstraPricing().input_per_million
            + Decimal(possible_cache_write_tokens) * AstraPricing().cache_write_per_million
        ) / ONE_MILLION
        output_cost = Decimal(max_output_tokens) * AstraPricing().output_per_million / ONE_MILLION
        return input_cost + output_cost

    def check_request(
        self,
        *,
        estimated_input_tokens: int,
        max_output_tokens: int,
        processing_tier: str,
        possible_cache_write_tokens: int | None = None,
    ) -> Decimal:
        """Return worst-case cost or raise before a request violates a control."""

        if processing_tier != "standard":
            raise BudgetExceededError("only standard processing is allowed; Fast is forbidden")
        maximum_cost_usd = self.worst_case_request_cost(
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
            possible_cache_write_tokens=possible_cache_write_tokens,
        )
        projected = self.current_spending_usd + maximum_cost_usd
        if projected > self.project_limit_usd:
            raise BudgetExceededError(
                f"projected cost ${projected} exceeds ${self.project_limit_usd} project limit"
            )
        return maximum_cost_usd

    def record_unpersisted_cost(self, actual_cost_usd: Decimal) -> None:
        """Track a response cost until the same amount is persisted to the ledger."""

        if actual_cost_usd < 0:
            raise ValueError("actual_cost_usd cannot be negative")
        projected = self.current_spending_usd + actual_cost_usd
        if projected > self.project_limit_usd:
            raise BudgetExceededError(f"actual cumulative cost ${projected} exceeds project limit")
        self.committed_cost_usd += actual_cost_usd

    def mark_cost_persisted(self, actual_cost_usd: Decimal) -> None:
        """Remove an in-memory amount after it has been appended to the ledger."""

        if actual_cost_usd < 0:
            raise ValueError("actual_cost_usd cannot be negative")
        if actual_cost_usd > self.committed_cost_usd:
            raise ValueError("persisted cost exceeds unpersisted spending")
        self.committed_cost_usd -= actual_cost_usd

    def commit(self, actual_cost_usd: Decimal) -> None:
        """Backward-compatible alias for recording an unpersisted response cost."""

        self.record_unpersisted_cost(actual_cost_usd)


class UsageLedger:
    """Append and read prompt-free usage records in JSON Lines format."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        run_id: str,
        response_id: str,
        model: str,
        usage: TokenUsage,
        estimate: CostEstimate,
        recorded_at: datetime | None = None,
    ) -> None:
        """Append one machine-readable record without storing prompt or response text."""

        if not run_id.strip() or not response_id.strip() or not model.strip():
            raise ValueError("run_id, response_id, and model must be non-empty")
        timestamp = recorded_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")

        record = {
            "recorded_at": timestamp.isoformat(),
            "run_id": run_id,
            "response_id": response_id,
            "model": model,
            "usage": {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_write_input_tokens": usage.cache_write_input_tokens,
                "ordinary_input_tokens": usage.ordinary_input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "estimated_cost_usd": str(estimate.total_usd),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as ledger_file:
            ledger_file.write(json.dumps(record, sort_keys=True) + "\n")

    def cumulative_cost_usd(self) -> Decimal:
        """Sum existing records, rejecting malformed cost values."""

        if not self.path.exists():
            return Decimal("0")
        total = Decimal("0")
        with self.path.open(encoding="utf-8") as ledger_file:
            for line_number, line in enumerate(ledger_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    total += Decimal(record["estimated_cost_usd"])
                except (json.JSONDecodeError, KeyError, InvalidOperation) as error:
                    raise UsageValidationError(
                        f"invalid usage ledger record on line {line_number}"
                    ) from error
        return total
