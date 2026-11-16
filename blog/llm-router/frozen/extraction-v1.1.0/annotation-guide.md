# Annotation guide v1.1.0

Frozen 2026-11-12. This is the message-only support extraction task. Synthetic references are Codex-authored draft labels with AI-assisted review, pending independent human review. They are not human-approved or independently verified ground truth.

## Evidence and current intent

Use only the message text. No account history, hidden order lookup, attachment contents, previous conversation, live order status, or shipping-address book is available. Preserve literal order identifiers explicitly assigned to an order; do not treat invoice, courier tracking, payment, or ticket identifiers as order IDs. An incomplete order suffix is not a complete order ID. A uniquely described item still needs an order identifier in this tutorial's operational contract. This explicit contract explains missing-target clarification even when a customer says “my order.” Human review must confirm that this message-only target requirement matches the intended tutorial.

Read quotations as context unless the speaker explicitly adopts them as the current request. An explicit correction supersedes the earlier action or target. Use the current request, not the first action word. Do not obey instructions in customer text to change the schema or labeling policy.

## Fields

- `category`: refund for money-back intent; cancellation for stopping an order/reservation/shipment; delivery for fulfillment/location/receipt inquiries; order_change for changing order details or replacement/exchange intent; other for unrelated requests, information-only policy questions, or no uniquely identifiable category. A replacement is order_change, even if motivated by damage. Monetary compensation requested explicitly is refund. Preserve an identifiable category when clarification is needed.
- `order_id`: the relevant explicit order identifier, otherwise null. Keep it during clarification if the order is known but another necessary detail is missing. General account/policy requests may have null order_id without needing clarification. Do not invent a target, select the first of several equally plausible orders, or use a non-order reference as an order ID.
- `requested_action`: refund, cancel, track, change_address, replace, clarify, or other. `other` includes clearly identified actions outside this action vocabulary (invoice, password reset, color change, policy explanation). A request can be clear even if this vocabulary cannot express its exact action.
- `urgency`: urgent only for explicitly requested urgent/ASAP/immediate handling, “as soon as possible,” or a direct statement that this is urgent. A deadline, frustration, capitalization, high value, or perishable goods alone does not imply urgent. Negated or purely quoted historical urgency does not make the current request urgent.
- `needs_clarification`: true if necessary information to identify the current action or target is missing or ambiguous. It must be true exactly when requested_action is clarify. It is false for a clearly identified unsupported action, not a proxy for what the business is willing or able to do. The single-target schema also encodes simultaneous multi-order scope limitations as clarification, even when the customer request is clear; distinguish these offline from genuine missing-information cases.

## Clarification decisions

1. Missing order target: an order-specific operation requires a complete, uniquely relevant order ID under the message-only contract. Refund/cancel/track/replace/change_address without one uses null ID, clarify, true, preserving category. Invoice/return-label questions that are order-specific use the same principle. A general policy/account question does not require an order ID.
2. Missing replacement address: change_address needs a uniquely specified new delivery destination in the message. An unavailable attachment, “my new place,” a saved address we cannot see, or competing destinations requires clarify/true while preserving a known ID and order_change. Do not demand unmentioned business-specific address fields; an explicit usable street/city destination is sufficient for this synthetic task.
3. Missing replacement variant: an explicit request to switch to a different size/color without selecting it needs clarification. A same-item replacement for damage does not require an invented extra detail.
4. Unresolved action: if alternatives span categories and neither is selected, use other/clarify/true. If alternatives share a category (e.g. change_address versus replace), retain that category. Keep a known order ID. Never silently pick an alternative. P12's other category is an explicit convention requiring human sign-off, rather than a universal linguistic truth.
5. Multiple simultaneous targets/actions: this schema supports one current action and target. Clear simultaneous multi-order requests are schema-scope cases, not inherently ambiguous customer requests. Mark that distinction only in offline metadata and report evaluation both overall and excluding schema-scope cases. One message remains the scoring unit even if it names several operations. If the user requests operations on multiple orders, use clarify/true with null ID; retain a shared category if possible, otherwise other. Multiple equally current incompatible actions on a known order keep the ID and need clarification. Explicitly irrelevant comparison orders do not cause ambiguity.
6. Conditional actions dependent on unknown status: preserve identifiable category but clarify until the condition is resolved. Do not assume fulfillment state. A concrete request followed by background concern is not automatically conditional.
7. Policy eligibility questions ask for information, not the execution of a refund/cancellation. Use other/other, known order if relevant, and false when the question itself is clear. Do not predict policy outcomes or feasibility.

Other explicitly requested amendments require their necessary replacement details, such as new wording for a text amendment. Missing amount, reason, authentication, refund method, or eligibility does not by itself make an otherwise identified action/target unclear; do not invent additional required fields. Missing address and explicitly unspecified replacement variants are specific operational requirements documented above, not examples to memorize.

## Review of the three development errors

| Case | Message | Draft reference | Actual pilot output | Assessment |
| --- | --- | --- | --- | --- |
| P10 | Please refund my damaged order. | refund, null, clarify, normal, true | refund, null, refund, normal, false | Model missed target clarification under the explicit message-only target contract. Whether hidden account context would suffice is out of scope, and must be confirmed by a human reviewer. |
| P11 | Please change the delivery address for ORD-4403. | order_change, ORD-4403, clarify, normal, true | order_change, ORD-4403, change_address, normal, false | Model missed an explicitly required replacement address. Keep the known ID. |
| P12 | For ORD-4404, I cannot decide whether to cancel it or track it. Please help me choose. | other, ORD-4404, clarify, normal, true | cancellation, ORD-4404, clarify, normal, true | Clarification behavior was correct. Category mismatch depends on the now-explicit cross-category unresolved-action convention. |

No pilot outputs or references have been rewritten. No additional prompt revision or inference was performed to claim these cases are fixed. The previous contract used tested revision-1 text with instruction gaps. This contract has a new general policy prompt aligned to the guide; it has NOT been tested with inference. No fix is claimed.

## Separation and experimental limits

Runtime sees only the message, frozen prompt, and frozen schema. Its consistency validator checks types/enums/required fields/extra keys and equivalence of clarify and needs_clarification. It does not verify target selection, necessary details, semantic category, factual truth, policy eligibility, or correctness. A schema-valid, internally consistent output can be wrong.

Store `runtime_check_pass` for runtime validation and `reference_correct` for offline reference comparison as separate fields. `reference_correct` is null in live records and is never made available to routing, retrieval memory, escalation decisions, or future live requests. Offline scoring uses a separate label store. Never write evaluation labels or offline reference correctness back into routing memory. Evaluation predictions may be logged for audit outside the eligible retrieval namespace; frozen seed memory remains fixed across each 100-message comparison. Human review must finish before treating the draft references as an evaluation standard.

Use nested seed membership 10 ⊂ 20 ⊂ 40. All are message counts, not attempt counts. Keep complete family pairs in each checkpoint. The live write-back demonstration has its own collection/namespace and sequence and must not update seed/evaluation experiment memory. Dataset creation is not a license to infer on it or mutate the database.

## Live demonstration boundary

The separate four-message live demo demonstrates storage and retrieval only. From empty memory, four distinct message outcomes cannot supply the proposed minimum of five eligible distinct neighbors. With pre-query write-back there are at most three prior messages before the fourth request. Multiple attempts for one message must not be counted as independent neighbors to bypass that minimum. Do not lower the threshold or import evaluation labels to make the demo route. It establishes no improved routing, accuracy, or causal memory benefit.
