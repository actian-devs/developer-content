| Test | What it stresses | False-positive rate (before → after mitigation) | Mitigation |
|---|---|---|---|
| 1. Threshold sweep | One global threshold: hit rate and wrong answers rise together | 6.4% (hit rate 79.2% at 0.8) → 4.8% | Threshold set against a 5.0% FP target (0.83, hit rate 70.1%) |
| 2. Paraphrase against negation | Negated questions that score close to the cached original | 65.0% (98.7% of flips served) → 29.0% | Negation check: reject a hit when one side is negated and the other is not |
| 3. Numbers against dates | Questions that differ only in an amount, a count or a date | 47.5% (100.0% of flips served) → 0.0% | Exact-match check on numbers, amounts and dates before serving a hit |
| 4. Partitions and per-intent thresholds | Hits crossing tenant and intent boundaries under one global threshold | 6.4% (plus 67.0% of hits cross-tenant) → 9.6% | MUST filters on tenant and predicted intent in VectorAI DB, per-intent thresholds fitted to the correctness target |
| 5. Local verification with decider-4b | Wrong hits that pass the threshold; cost of checking them | 6.0% → 2.8% | decider-4b yes/no check on each hit (tau 0.5), p50 106 ms on Apple MPS (arm64) |
| 6. Short against long queries | One threshold across query lengths | 9.4% (short 9.4%, long 4.6%) → 2.0% | Length-aware thresholds fitted per bucket to the correctness target |
| 7. Stale answers after the source changes | Hits on answers whose source changed (20.0% of sources) | 24.9% → 11.4% | Filtered set_payload marks a changed source's entries stale; lookups add MUST NOT is_stale |
