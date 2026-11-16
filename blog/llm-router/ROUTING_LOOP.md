# Routing loop

```mermaid
flowchart TD
    Q[New support message] --> E[EmbeddingGemma embedding]
    E --> A[Native points.search over eligible checked outcomes\nIncludes matches and failures]
    E --> S[Native points.search with FilterBuilder\nmust ID allowlist AND passed_check true]
    A --> G[Distinct-message, model-specific evidence\nCosine minimum 0.5 · up to 10 neighbors\nAt least 5 messages · at least 80% matches]
    S --> G
    G --> M{Cheapest qualifying tier}
    M -->|Local qualifies| L[Qwen via Ollama]
    M -->|Insufficient local support| C[Cloud fallback\nExplicit paid permission and ledger reservation]
    L --> V[Runtime schema and consistency validation]
    C --> V
    V --> W[Upsert separate attempt UUID\nRaw answer, usage, runtime result, cost\nSemantic outcome initially unverified]
    W --> F{Explicit semantic feedback available?}
    F -->|No| U[Keep unverified\nDo not count as semantic success or support]
    F -->|Yes| B[Check exact attempt and input binding\nSeparate checker, no evaluation labels]
    B --> P[Upsert immutable versioned outcome feedback\npassed_check true OR false\nPreserve source attempt unchanged]
    P --> N[Later query uses updated evidence]
    N --> A
    V -->|Local runtime rejection| C
    B -->|Immediate local semantic rejection\nOne cloud escalation at most| C
```

The successful-outcome query proposes eligible successful tiers. The separate all-outcome query supplies the denominator, including failures. Unknown live outcomes supply neither successes nor reviewed support. Model choice, native neighbors, both checks, and cost are logged.

The bounded demonstration replays existing answers. Its checker compares the saved demo output with an existing synthetic draft reference in a separate offline process. That is disclosed draft-reference agreement, not independent human review or guaranteed correctness. Without an explicit check, a schema-valid answer or a cloud response stays unverified. Feedback does not guarantee a model switch or improved routing.

The database performs vector search and scoring. Native retrieval over-fetches to the collection point count with explicit search breadth, verifies expected IDs, and stops on incomplete results. No Python cosine ranking is used by this loop. See README.md for the bounded collection-size limitation.
