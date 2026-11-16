"""Runtime schema and deterministic consistency checks; no reference answers."""
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict


class Extraction(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    category: Literal['refund', 'cancellation', 'delivery', 'order_change', 'other']
    order_id: str | None
    requested_action: Literal['refund', 'cancel', 'track', 'change_address', 'replace', 'clarify', 'other']
    urgency: Literal['normal', 'urgent']
    needs_clarification: bool


def validate_response(raw: str):
    """Return parsed result and rule errors; cannot infer semantics from a schema."""
    result = Extraction.model_validate_json(raw)
    errors = []
    if result.needs_clarification != (result.requested_action == 'clarify'):
        errors.append('needs_clarification must be true exactly when requested_action is clarify')
    # A known order may remain present when action/address/other details are missing.
    return result, errors
