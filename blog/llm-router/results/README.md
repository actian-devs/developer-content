# Recorded experiment evidence

This is a minimal export of the existing article experiment, not a new evaluation. It contains 100 per-message baseline/decision rows and 40 cloud-seeding costs. Preparation checked the original input/attempt bindings, saved parsed responses against unchanged draft references, replay response reuse, native decision logs, and settled cost records. The original records were read only and remain unchanged.

## Verify without a network or installed model packages

From `blog/llm-router/`, using Python 3.12:

```powershell
python -I -S results/verify.py
# Optional deterministic tests, using the project's dependency environment:
python -B -m unittest tests.test_results_verification -v
python -B -m unittest discover -s tests -v
```

`verify.py` uses only Python's standard library. It reads `recorded_results.json`, validates IDs/types, recomputes counts and response-reuse/escalation paths, sums integer micro-USD costs, calculates savings/break-even, and fails if the reported figures disagree. It does not import the application, contact a database/API, generate embeddings, run models, or read a private journal. One micro-USD is $0.000001; integer arithmetic avoids accumulating binary floating-point cost errors.

## Reported totals reproduced from the rows

| Setup | Complete / planned | Exact draft-reference agreement | Local selections | Escalations | API cost |
|---|---:|---:|---:|---:|---:|
| Always Qwen | 100 / 100 | 58/100 (58.00%) | 100 baseline calls | Not applicable | $0 API; operating cost excluded |
| Always cloud | 99 / 100 | 95/99 (95.96%) | 0 | Not applicable | $0.287472 actual recorded charge |
| Router replay, 10 seeds | 99 / 100 | 95/99 initially and finally | 0 | 0 | $0.287472 counterfactual |
| Router replay, 20 seeds | 99 / 100 | 95/99 initially and finally | 0 | 0 | $0.287472 counterfactual |
| Router replay, 40 seeds | 99 / 100 | 95/99 initially and finally | 2 | 0 | $0.281686 counterfactual |

The two checkpoint-40 local selections are **E-041 and E-043**. Both have recorded runtime-check passes and exact draft-reference matches. Every other query selected cloud. Checkpoint selections use the preserved `semantic-exploratory-0.5-v1.0.0` policy, Qwen profile `qwen-memory-v2`, extraction contract `extraction-v1.1.0`, and validator `runtime-validator-v2.0.0`. The starting gate was cosine minimum 0.5, five distinct messages, and 80% historical draft agreement, with up to ten neighbors. The boundary failed its seed calibration acceptance criterion and was retained for exploratory evaluation; it is not a validated reliability threshold.

The saved repaired-native decisions and earlier replay choices agreed on all 300 checkpoint decisions. This export copies those recorded choices, then reuses each selected saved baseline response. A local runtime rejection would require the saved cloud response and count both attempts' API costs; the verifier implements and fixture-tests that path. No selected local response in these recorded checkpoints required escalation.

### Why 100 versus 99

The Qwen baseline has 100 usable returned responses, including preserved responses from memory-interrupted orchestration. Its exact-match rate uses all 100, including incorrect answers. The cloud baseline has no usable **E-001** response. The original cloud request remains unresolved; absence is not evidence of a free or unbilled request. The export represents its cloud response as JSON `null`, not a fabricated answer, failure label, zero charge, or known cost.

All three checkpoints selected cloud for E-001, so each replay is incomplete for that message. Cloud and replay agreement/cost comparisons use the same 99 completed cases. Unknown E-001 cost/exposure is not included in the $0.287472 confirmed cost of the 99 usable cloud responses. The separate experiment ledger retains its unresolved exposure. No E-001 retry or reconciliation was performed for this publication.

## Actual charges versus counterfactual replay

The 99 cloud baseline responses were generated once. Their actual recorded usage-priced costs sum to **$0.287472**. The 40 completed cloud seed responses sum separately to **$0.116129**. These two exported actual-cost categories total $0.403601. This is not a complete experiment ledger: pilots, the separate four-message demo, and unresolved exposure are outside that subtotal. Cost values were checked against settled original ledger entries during export; billing/reservation/provider identifiers are not published. They are recorded API cost accounting, not independently audited provider invoices, taxes, or a current pricing quote.

Replay checkpoints make **no additional cloud calls**. Their costs are the hypothetical API charges of the selected saved baseline responses, including both attempts if a runtime rejection escalates. They must not be added to actual spending or multiplied as if each checkpoint were billed again. Both baselines were already generated to perform this comparison; there was no realized spending saving from running the replay.

For checkpoint 40, on the same 99 cases:

```text
Same-case cloud API cost             $0.287472
Counterfactual routed API cost      -$0.281686
Counterfactual API difference         $0.005786
Relative difference                  0.005786 / 0.287472 = 2.01271776...%
```

Report this as approximately **2.01% counterfactual API savings**, with no measured quality improvement. Checkpoints 10 and 20 have zero calculated API savings.

### API-only seeding break-even

The requested calculation charges all 40 cloud seed responses as an upfront cost:

```text
Counterfactual saving per subsequent request = $0.005786 / 99
                                             = $0.0000584444...
API-only break-even = $0.116129 / ($0.005786 / 99)
                   = 1,986.998098859... subsequent requests
                   = 1,987 whole requests, rounding upward
```

This is an amortization calculation, **not an observed production break-even**. It assumes the same traffic mix, local-selection behavior, API prices and per-request savings persist. The cloud seeds were collected to complete the paired-model seeding requirement; they were not used by the already-frozen local-evidence gate and did not cause the measured replay choices. Charging that cost here does not establish that cloud seeding improves routing.

The calculation excludes local generation/embedding compute, GPU/CPU memory and energy, VectorAI DB/server operation, storage, development, evaluation-baseline generation, pilots, feedback checking, human review and other deployment costs. Actual total-cost break-even can be later or nonexistent. No general savings, speed, live-router latency benefit, or zero local compute cost is established.

## What the minimal export can and cannot audit

`recorded_results.json` contains:

- Public configuration/version identifiers and missing-cloud explanation.
- For E-001 through E-100, each available baseline's **recorded exact-match boolean**, runtime-check boolean and integer API cost, plus the recorded local/cloud choice at each checkpoint.
- For S-001 through S-040, each cloud seed's recorded API cost.

There are no credentials, private paths, request/billing IDs, raw journals, provider response bodies, unrelated runs or internal notes. The export contains no evaluation messages, expected JSON, model output JSON, embeddings, neighbor lists or seed-feedback snapshots.

Consequently, readers can independently reproduce **the arithmetic over the recorded flags/costs and replay choices**. They **cannot independently regrade exact agreement or semantic correctness**, judge whether the draft labels were appropriate, reproduce vector retrieval decisions, or audit provider invoices from these files. Those limitations are deliberate and should not be hidden. The local preparation checks against original responses/references are provenance, not public independent verification of those judgments. Repository history preserves the published export; the verifier is not cryptographic proof that a recorded match flag is true.

All correctness flags are agreement with **synthetic Codex-authored draft labels with AI-assisted review, pending independent human review**. They are not verified ground truth. Runtime validation checks schema and limited consistency, not correctness. The Qwen results are **resumed diagnostic execution on an already inspected evaluation set**, not fresh held-out confirmation. No prompt, threshold or label was tuned during this evidence export, and no inference/evaluation request was repeated.

This results directory is offline evidence only. Do not import its evaluation correctness flags into live routing or feedback memory. The fresh reader implementation and later feedback-loop changes have not received a new held-out evaluation. Historical replay timings must not be described as fresh end-to-end router latency.
