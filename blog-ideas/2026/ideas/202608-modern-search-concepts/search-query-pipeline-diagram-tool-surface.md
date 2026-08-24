# search-query-pipeline-diagram-tool-surface

Template 3 — Full surface — from search-query-pipeline-diagram-tool.html:2186, flattened into an indented outline. Row layout in the tool is [left aside | centre spine | right control], so asides and controls are shown as sub-items of the spine step they sit beside.

NOTE: This is not the source of truth. It is a point in time extraction at 25 Aug 2026.

Tiers: [core] [rec] = recommended [full] = optional/full-surface only

1. Query inputs
   ├─ Text query .......................... [core] 0 ms
   │ the raw string
   ├─ (left) Image query .................. [full] 0 ms
   │ validated upload or image reference
   │ └─ feeds → Image-to-image vector retrieval; steers → Retrieval routing
   └─ (control) Session & context ......... [full] 2–10 ms
   prior turns, clicks, anaphora
   └─ steers → Query understanding, Query rewriting, Retrieval routing,
   Personalisation

2. Query processing
   ├─ Query normalisation ................. [core] 1–5 ms
   │ NFKC, case, punctuation, tokenisation
   ├─ Query understanding ................. [core] 4–15 ms
   │ intent, entities, constraint extraction
   │ └─ (control) Constraint confidence ... [full] 0–3 ms
   │ per-constraint extraction confidence
   │ └─ steers → Metadata pre-filter, Constraint relaxation
   ├─ Query rewriting ..................... [rec] 5–25 ms
   │ spelling, synonyms, constraint-stripped query
   └─ Query expansion ..................... [full] 20–120 ms
   synonyms, PRF, HyDE, LLM variants

3. Constraint handling (entry)
   └─ Metadata pre-filter ................. [core] 1–5 ms
   compile constraints and mandatory policy
   └─ (control) Retrieval routing ........ [rec] 0–3 ms
   which legs and gated rerankers fire
   └─ steers → all 7 retrieval legs, Candidate budget allocation,
   and the gated rerankers (semantic, late-interaction,
   VLM, Learning-to-Rank)

4. Retrieval components ── parallel group ──
   legs run concurrently · the group costs its slowest member
   ├─ Lexical & metadata retrieval ........ [core] 10–40 ms
   │ BM25 over text fields + filters
   │ └─ serves from: Text & metadata inverted index
   ├─ Text vector retrieval ............... [core] 10–40 ms
   │ ANN over text embeddings
   │ └─ serves from: Document-embedding ANN index
   ├─ Text-to-image vector retrieval ...... [rec] 10–40 ms GATED: visual intent
   │ text-tower ANN over image embeddings
   │ └─ serves from: Image-embedding ANN index (shared)
   ├─ Image-to-image vector retrieval ..... [full] 10–40 ms GATED: image query
   │ image-tower ANN over image embeddings
   │ └─ serves from: Image-embedding ANN index (shared)
   ├─ Learned sparse retrieval ............ [full] 20–60 ms GATED: route selected
   │ SPLADE-style term expansion
   │ └─ serves from: Learned-sparse postings index
   ├─ Multi-vector / passage retrieval .... [full] 15–50 ms GATED: route selected
   │ chunk hits rolled up to documents
   │ └─ serves from: Passage-embedding ANN index
   ├─ Late-interaction retrieval .......... [full] 30–90 ms GATED: route selected
   │ ColBERT-style token-level MaxSim
   │ └─ serves from: Token-vector MaxSim index
   │ └─ alternative placement of: Late-interaction rerank
   ├─ (control) Candidate budget allocation [full] 0–2 ms
   │ per-leg k under a latency budget → steers all 7 legs
   └─ (control) Degradation controller ... [rec] 0–2 ms
   deadlines, partial results
   └─ steers → all 7 legs, Late-interaction rerank, Cross-encoder rerank,
   VLM rerank

5. Candidate generation
   ├─ Candidate union ..................... [core] 2–8 ms
   │ merge by doc_id, keep per-leg ranks
   ├─ Result sufficiency check (decision) . [rec] 1–3 ms
   │ enough candidates above the quality floor?
   │ ├─ next: "sufficient" → Candidate pruning
   │ ├─ branch "too few" → Constraint relaxation
   │ └─ branch "zero" → Zero-result fallback
   └─ Candidate pruning ................... [rec] 1–5 ms
   cap per leg, score floor, cut to rerank depth
   └─ (left asides, recovery paths)
   ├─ Constraint relaxation ...... [rec] 5–20 ms (conditional)
   │ ordered ladder, bounded passes
   │ └─ returns "relaxed predicate" → Metadata pre-filter
   └─ Zero-result fallback ....... [rec] 10–40 ms (conditional)
   drop hard filters, semantic-only, suggestions
   └─ returns "recovery mode" → Metadata pre-filter

6. Fusion
   └─ Fusion .............................. [core] 1–5 ms
   Reciprocal Rank Fusion by default
   └─ (control) Fusion policy ........... [full] 0–2 ms
   strategy and weights per query class
   └─ steers → Fusion; trained by Behavioural event log

7. Selective rerank cascade
   gated passes · skipped stages pass candidates through
   ├─ Semantic rerank ..................... [full] 5–20 ms GATED: optional pass
   │ optional bi-encoder budget-protection tier
   ├─ Late-interaction rerank ............. [full] 20–80 ms GATED: route selected
   │ MaxSim over a fused shortlist
   │ └─ alternative placement of: Late-interaction retrieval
   ├─ Cross-encoder rerank ................ [rec] 50–150 ms
   │ joint query-document scoring
   ├─ VLM rerank .......................... [full] 80–500 ms GATED: visual intent
   │ query vs document images, top-M only (conditional)
   └─ Learning-to-Rank .................... [full] 5–20 ms GATED: optional pass
   GBDT over retrieval, behaviour, quality features
   └─ trained by Behavioural event log

8. Final ranking
   ├─ Business ranking .................... [rec] 1–5 ms
   │ quality tiers, promotions, soft policy
   ├─ Freshness & temporal ................ [rec] 1–3 ms
   │ recency decay or time-window boost
   ├─ Personalisation ..................... [full] 3–15 ms
   │ runtime profile lookup + incremental updates
   │ └─ steered by Session & context; profile updated by Behavioural event log
   ├─ Diversity ........................... [rec] 2–8 ms
   │ MMR or per-attribute capping
   └─ Deduplication ....................... [rec] 2–10 ms
   near-duplicate collapse

9. Results
   └─ Results assembly .................... [core] 15–40 ms
   hydrate fields, images, highlights, paging
   └─ serves from: Document / source store
   └─ (control) Behavioural event log ... [rec] 0 ms
   impressions joined to clicks, saves and reformulations
   ├─ fed by Results assembly (impression context)
   ├─ trains → Fusion policy, Learning-to-Rank (offline)
   └─ updates → Personalisation (asynchronous profile update)
