# Plan 006 — Physical Architecture pipeline view

**Target:** `search-query-pipeline-diagram-tool.html` — plus
`search-query-pipeline-diagram-tool-architecture.md` and the About tab in Phase 6.
**Status:** **Phase 1 done** (`bda600e`, 25 Aug 2026); Phases 2–6 outstanding.
Written 25 Aug 2026.
**Scope:** a fourth tab, **Physical architecture**, sitting between _Build order_ and _About_,
rendering the AWS + OpenSearch + Qdrant realisation of the same three levels.
**Decisions:** D1 blended best-of-breed engines · D2 model-parameterised render engine ·
D3 query path plus a collapsed consistency plane · D4 managed-first on ECS Fargate ·
D5 configured numbers only, never measured ones · D6 physical inherits the logical defaults ·
D7 cross-view fidelity is a validator, not a convention · D8 no physical build order in pass one.

Line numbers below are as at `777849e`, **before Phase 1**, and are now stale — the file is
4971 lines at `bda600e`. Re-grep before acting on any of them. Phase 1's own section has been
rewritten to describe what was built; §2.1 of that section records the four places where the
built shape differs from the shape planned here, and what each one means for Phases 2–5.

---

## 0. What this is, in one paragraph

The existing tool answers _how should these capabilities work together_. This plan adds a tab
that answers _how is that concretely implemented_, using the vocabulary from
`architecture-mental-model.md` §1: same abstraction ladder, one rung down. The physical view
takes the **suggested default from every "What you are choosing" panel in the logical view**
(D6) and shows the AWS topology that those defaults imply — where each logical component
lands, which ones merge, which one splits, which move into an engine, and what happens to a
single request as it crosses the wire six times.

The opinionated part is stated up front and defended in the panels: **OpenSearch owns
everything sparse and every filter it can push down; Qdrant owns everything dense; fusion
happens in neither, because neither can see the other's ranks.** Everything else — Fargate,
SageMaker, ElastiCache, Kinesis, DynamoDB, AppConfig — is scaffolding around that one split.

---

## 1. Decisions

### D1 — Blended best-of-breed, one topology

| Logical concern                             | Engine                                          | Why this one                                                                                                                                  |
| ------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| BM25 over analysed text, field weighting    | **OpenSearch**                                  | Analyser parity with index time, per-field boosts, explain output. Qdrant has no lexical story.                                               |
| Learned sparse (SPLADE-style)               | **OpenSearch**                                  | `rank_features` / neural-sparse postings live in the same inverted structure, so it rides the _same query_ as BM25 — one round trip, not two. |
| Hard predicate pushdown for the sparse legs | **OpenSearch**                                  | `bool.filter` is cacheable and executes inside the search operation.                                                                          |
| Dense text ANN                              | **Qdrant**                                      | Named vectors, filterable HNSW with mandatory payload indexes, quantization with rescore.                                                     |
| Cross-modal text→image and image→image ANN  | **Qdrant**                                      | Second and third named vector in the same collection and the same point — one call, shared payload filter, shared point id.                   |
| Multi-vector / late interaction             | **Qdrant**                                      | Native multivector MaxSim, so late-interaction rerank collapses _into retrieval_ instead of being a GPU service.                              |
| Hydration of the response                   | **OpenSearch** (+ DynamoDB for volatile fields) | The doc already exists there with `_source` and doc values; a third store on the critical path buys nothing.                                  |
| Fusion                                      | **Neither — the orchestrator**                  | RRF needs both rank lists in one process. This is the single most important physical departure.                                               |

**The load-bearing consequence:** the same document exists as an OpenSearch `_doc` and a
Qdrant point, and the _same hard predicate must be executable in both_. That forces metadata
duplication into the Qdrant payload, payload indexes on every filtered field, and a shared
canonical id. Every dual-store gotcha in Appendix B descends from this one fact.

### D2 — Model-parameterised render engine

The renderer currently reads module globals (`S`, `TPL`, `REL`, `DATA_SOURCES`, `SERVING_REL`,
`GATED_EXECUTION`, `state.tpl`, `canvas`, `wires`, `rowsEl`). Phase 1 turns those into an
explicit **pipeline model** passed as an argument. Logical and physical then become two
instances of one type, and a third view (deployment topology, runtime/latency) costs data
rather than code. This is the largest phase and it changes **no rendered output** — that is
the property that makes it safe to land on its own.

### D3 — Query path, plus a collapsed consistency plane

The request path is the subject. A second, separately-toggled band shows source of truth →
change stream → indexer → both engines, so the panels can honestly explain freshness skew,
payload parity and reindex without inventing a full ingestion pipeline. Off by default,
rendered in the data-plane style, three nodes.

### D4 — Managed-first on ECS Fargate

CloudFront + WAF → ALB → search orchestrator on ECS Fargate → Amazon OpenSearch Service
managed domain + Qdrant Cloud Hybrid (BYOC, in the customer VPC) → SageMaker real-time
endpoints for embedding and cross-encoder. ElastiCache (Valkey), DynamoDB, Kinesis Firehose →
S3, Glue/Athena, AppConfig, ADOT → X-Ray/CloudWatch around them. API Gateway, EKS,
OpenSearch Serverless and Lambda are each recorded in the relevant panel's **Alternatives**
section with the condition that would select them.

### D5 — Configured numbers only, never measured ones

Plan 003 stripped every millisecond and percentile from the tool on the grounds that _"the
tool describes a logical query-side search pipeline; quantified serving characteristics are
claims the diagram cannot support"_. The physical view partially re-admits numbers, under a
rule that keeps plan 003's intent intact:

> **Allowed** — values _you set_: deadlines, timeouts, retry and hedge policy, `k` and
> rerank depth, shard and replica counts, HNSW `m` / `ef_construct` / `ef`, quantization
> mode, oversampling factor, batch size, instance family, cache TTL, connection-pool size.
> These are configuration, and configuration is exactly what a physical diagram is for.
>
> **Forbidden** — values you would have to _measure_: no "p99 is 45 ms", no throughput
> claims, no cost figures in dollars, no "this is 3× faster". Where the reader needs a sense
> of proportion, use the existing relative-cost adjectives.

The distinction is greppable and gets a validator (Phase 3, §3.5). A deadline of 120 ms on
the cross-encoder is a budget the deployment _enforces_; a p99 of 45 ms is a claim the diagram
cannot support. Only the first appears.

### D6 — Physical inherits the logical defaults

Every "Default:" in the logical panels is treated as chosen. The physical view is what those
choices cost in infrastructure. Worked through in Appendix C; the load-bearing ones are:

| Logical default                                                           | Physical consequence                                                                                                                                        |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Pre-filter, always" + "Native filtered ANN, verified by benchmark"       | Qdrant payload indexes are mandatory, not an optimisation; filtered HNSW must be benchmarked at production selectivity.                                     |
| "Asymmetric: full text lexically, constraint-stripped to the vector legs" | Two query strings on the wire, two cache keys, and the embedding cache keys on the _stripped_ text — a higher hit rate than the raw query.                  |
| "RRF with k ≈ 60" / "Set 60 and move on"                                  | Fusion is rank-only, so raw scores never cross the engine boundary and no score normalisation service is needed. Removes a whole class of physical problem. |
| "100, after measuring 25/50/100/200" (rerank depth)                       | Sizes the cross-encoder endpoint: 100 pairs per request is the batch the GPU must clear inside its deadline.                                                |
| "Start with rules plus a small classifier" (not an LLM)                   | Query understanding is an in-process ONNX model, not a Bedrock hop. Bedrock QU appears only at Full, behind a cache.                                        |
| "Whole document to start" (embedding granularity)                         | One dense vector per point at Core; the passage collection is a Full-level addition with its own point count.                                               |
| "Stored compressed representations" (late-interaction docs)               | Multivectors live in Qdrant with quantization, so late-interaction rerank is a Qdrant rescore, not a GPU service.                                           |
| "From day one" (behavioural events)                                       | Firehose → S3 is in the **Recommended** level, not Full.                                                                                                    |

### D7 — Cross-view fidelity is a validator

Every physical component declares `realises:[logicalIds]`. A validator asserts the mapping is
total and that every non-1:1 mapping carries a written `departure`. See §3.4. This is how
"high fidelity, departures stated if needed" stops being an aspiration.

### D8 — No physical build order in pass one

`STEPS` stays logical-only. The physical model declares `steps:null` and the Build order tab
keeps reading the logical model. Revisit once the physical inventory settles.

---

## 2. Phase summary

| Phase | Change                                                               | Output changes? | Fixture?               | Size             | Risk     | Status |
| ----- | -------------------------------------------------------------------- | --------------- | ---------------------- | ---------------- | -------- | ------ |
| **1** | Extract the pipeline-model type; a view per level                    | **no**          | reshaped, same content | +666 / −433      | **high** | **done** `bda600e` |
| **2** | Physical component registry, data plane, consistency plane, panels   | new tab renders | new                    | ~1400 new lines  | medium   | todo   |
| **3** | Physical templates, relations, gates, baseline, two invariants       | new tab correct | new                    | ~350 lines       | medium   | todo   |
| **4** | Tab, LHS options panel, RHS panel spec, hover card, cross-view links | new tab usable  | no                     | ~450 lines + CSS | medium   | todo   |
| **5** | "Follow one request" stepper — the query dataflow narrative          | new             | no                     | ~200 lines       | low      | todo   |
| **6** | About tab, architecture doc, TODO reconciliation                     | prose           | no                     | 3 files          | lowest   | todo   |

**Order: 1 → 2 → 3 → 4 → 5 → 6.** One commit per phase. Phase 1 left the tool
pixel-identical; Phases 2–3 must leave it rendering with the new tab reachable but possibly
rough; Phase 4 is where it becomes usable.

Phase 1 was the only phase that could break the existing view, and the only one with a free
correctness oracle. It is done and verified (§1.6). From Phase 2 on, the existing view is
protected by the same oracle for free — **any change to the logical render path must still
reproduce `163328:4705a20e` at 1280×900**, and a Phase 2–5 change that moves it is a bug in the
shared layer, not a physical-view decision.

---

## Phase 1 — The pipeline-model type — **DONE** (`bda600e`)

### 1.1 The problem it solved

Nine of the functions that would need to serve two views closed over module-level constants.
`logicalDependencies` named `TPL`, `REL`, `SERVING_REL` and `COMPONENT_RELATION_KINDS`
directly; `renderCanvas` named `state.tpl`, `rowsEl`, `canvas` and `S`; `openDrawer` named `S`,
`RREL`, `RSERVING`, `DATA_SOURCES` and eight more. Nothing was injectable. A second view built
on that either duplicates the renderer or mutates the globals — both drift.

### 1.2 What was built

**A level is now a value.** `PIPELINE_MODELS` holds one entry per abstraction level.
`defModel(spec)` derives the reverse relation index, the reverse serving index, the data-source
tiers and `maxCx`, then runs the validators and registers the model — so no model can reach a
renderer half-built, and a level that cannot pass its own rules never becomes reachable.

`Model` is a namespace of the four pure queries a model answers — `stages`, `gate`,
`relationDetail`, `dependencies`. They take the model explicitly because they are the only ones
both the validators and the view need. Every other model-aware function opens with a
destructuring preamble instead:

```js
function validateRelationshipModel(model){
  const {components:S, templates:TPL, relations:REL, gates:GATED_EXECUTION, …} = model;
```

**The view is a closure, not a parameter list.** `createPipelineView(model, slice, root)`
returns a frozen handle; the ~40 render functions inside it close over the destructured model,
the state slice and the root element. This is the one material departure from §1.4 as planned
(see §2.1 below) and it is what kept the change to 666 insertions rather than ~230 call-site
rewrites.

Model fields as built, and what each replaced:

| Field | Replaced | Note |
| --- | --- | --- |
| `id`, `label`, `abstraction` | — | `abstraction` is `"logical"`; panels and validators will branch on it in Phase 4. |
| `components`, `templates`, `relations`, `relationDetails`, `gates`, `requires` | `S`, `TPL`, `REL`, `RELATION_DETAILS`, `GATED_EXECUTION`, `REQUIRES` | |
| `dataSources`, `serving` | `DATA_SOURCES`, `SERVING_REL` | |
| `phases`, `capabilities`, `facets` | `PHASES`, `CAPS`, `FACETS` | `PHASES` moved up out of the view region into the model. |
| `parallelGroups`, `stageGroups` | `PARALLEL_GROUPS`, `STAGE_GROUPS` | Also moved up; they are level vocabulary, not renderer vocabulary. |
| `steps`, `ladderExempt` | `STEPS`, `LADDER_UNBUILT` | `steps:null` makes `validateBuildLadder` a no-op, which is D8. |
| `baseline`, `expectedCounts` | `DEPENDENCY_BASELINE`, the two hardcoded count maps | |
| `planes` | the boolean `state.showServing` | `[{id, label, serves, defaultOn}]`. A plane marked `serves` is what admits the serving edges. |
| `example` | a string literal in the wiring | The worked query. |
| `collapseGroups` | `LEGS` / `GATED_RERANKERS` inlined in `compactDependencyNames` | Sets the narrow layout names collectively. |
| `invariants` | four blocks inside `validateRelationshipModel` | See §1.3. |
| **derived by `defModel`** | | `reverseRelations`, `reverseServing`, `maxCx`, `servesPlanes`, `validation` |

`RELATION_TYPES`, `GATE_KINDS`, `GATE_SCOPE_LABEL`, `TIER_LABEL` and `COMPONENT_RELATION_KINDS`
stayed module-global and shared, as planned. Two levels must not be able to disagree about what
"steers" means.

### 1.3 Invariants — an unplanned split that Phase 3 should reuse

`validateRelationshipModel` was checking two different kinds of thing: rules every level must
satisfy (endpoints exist, relations are unique, reverse indexes are not stale, every used kind
has a complete visual definition) and rules only *this* level can state (the rerank cascade is
all Reranking components; the gated-reranker set is exactly what routing steers; every feedback
relation carries a transmission detail). The second kind is now `model.invariants` — an array of
`(model, {componentRelations, servingRelations, usedKinds})` functions run at the end of the
shared validator.

**This is where `validateRealisationMap` (§3.4) belongs.** It is not a fourth top-level
validator; it is the physical model's first invariant. Same for the `MEASURED_CLAIM` regex
check (§3.5). Phase 3 gets smaller because of this.

### 1.4 State and DOM

```js
const state = {
  view:"pipe",
  models:Object.fromEntries(Object.values(PIPELINE_MODELS).map(model=>[model.id,{
    tpl:1, prevTpl:null,
    planes:Object.fromEntries(model.planes.map(plane=>[plane.id,plane.defaultOn])),
    off:{1:new Set(),2:new Set(),3:new Set()},
    collapsedGroups:new Set(), sel:null, focus:null
  }]))
};
```

Slices are built from the registered models, so Phase 2 gets its slice for free the moment
`defModel` runs on the physical level — no edit to `state` at all.

The DOM contract came out **better than the `#phys-` id-prefix scheme in §1.5 as planned.**
Ids belong to the document; per-view handles are `data-el` attributes, and inside a view `$` and
`$$` are shadowed by root-scoped versions with `ui(name)` for element lookups:

```js
const $  = (sel,scope)=> (scope||root).querySelector(sel);
const $$ = (sel,scope)=> Array.from((scope||root).querySelectorAll(sel));
const ui = name => root.querySelector('[data-el="'+name+'"]');
```

25 ids in the pipeline-view markup became `data-el`. **Phase 4's markup is therefore a copy of
the pipeline section with `id="view-phys"` and nothing else renamed.** The few handles the
document itself needs are generated per view (`rowsEl.id = model.id+"-rows"`, the template-tip
ids) so two views cannot collide.

No CSS selector in the file uses an id, which is what made this safe; the class contract
(`.node[data-id]`, `.row[data-row]`, `.cell.center`, …) is unchanged and shared.

### 1.5 Shared page furniture

The drawer, the scrim and the hover card are document singletons, so they were hoisted out of
the view: a `DRAWER` bundle (`root`, `scrim`, `title`, `tags`, `purpose`, `chips`, `gate`,
`body`, `toggle`), `drawerSect`, `hoverCard`, `finePointer`, `isNarrowLayout` and
`escapeAttribute`. `closeDrawer()` is module-level and delegates to `drawerOwner.clearSelection()`
— whichever view filled the drawer in clears its own selection when it shuts, even if the reader
has since changed tabs. Each view keeps its own `ResizeObserver` on its own canvas (a hidden
canvas never resizes); the window `resize` and `document.fonts.ready` handlers are global and
dispatch to `activeView()`.

### 1.6 Acceptance — met

`window.dependencyDiagnostics` keeps its public shape, so it remained a usable oracle. Verified
against the pre-refactor build **at a pinned 1280×900 viewport** — the viewport must be pinned
on both sides or the comparison is meaningless, which cost one false alarm:

- every wire path's full markup (class, `d`, style, aria-label, edge index) and every node and
  data-source card's full HTML, across three levels crossed with the data-source plane —
  identical (`163328:4705a20e`);
- all three validation summaries — identical;
- nine dependency inventories, including two with disabled components — identical;
- eight component drawers spanning inputs, legs, gated rerankers, controls and evaluation, and
  all six source drawers — identical;
- all four focus-chip kinds (facet, capability, phase, gate condition), the sidebar, the
  component toggle and reset paths — identical;
- the hover card, the narrow-layout mobile selection panel, the Escape and close handlers, tab
  switching and the Build-tab hand-off — all exercised, all correct.

### 1.7 Where the built shape departs from this plan, and what it means downstream

| # | Planned | Built | Consequence |
| --- | --- | --- | --- |
| 1 | Thread `(model, slice, dom)` through ~40 signatures (§1.4) | One closure, `createPipelineView(model, slice, root)`; destructuring preamble at the top | ~230 references stayed untouched. **Phases 2–5 must not add a `model` parameter to a render function** — it is already in scope. Anything genuinely shared between levels goes *outside* the factory, not into a parameter. |
| 2 | `DOM` bundle of `#phys-`-prefixed ids (§1.5) | `data-el` handles, root-scoped `$`/`$$`, `ui(name)` | Phase 4's markup is a copy of the pipeline section with one changed id. Cheaper than planned. |
| 3 | `validateRealisationMap` as a fourth top-level validator (§3.4) | `model.invariants` exists and is the right home | Phase 3 shrinks: realisation-map and measured-claim checks are invariants on the physical model, not new plumbing. |
| 4 | `planes` drives a toggle per plane (§1.2, §4.2) | `planes` is declared and drives the dependency set, but the LHS still renders **one** control, `renderPlaneToggle`, bound to `model.planes[0]` | **Phase 4 must generalise this before the physical view's two planes work.** It is the one piece of Phase 1 that is deliberately provisional; it is marked as such in the code. |

Two smaller notes for later phases:

- `Model.dependencies` takes `{planes, off}` where `planes` is the slice's plane map. The
  physical model declaring `planes:[{id:"data", serves:true, …}, {id:"consistency", …}]` gets
  serving edges from the `data` plane with no code change; the `consistency` plane will need its
  own edge source in Phase 3.
- `layoutSweep` iterates `Object.keys(model.templates)` crossed with the **first** plane only.
  When the physical model declares two planes, either extend the sweep to the cross product or
  state in the diagnostics why it does not.

---

## Phase 2 — The physical registry

### 2.1 The component record

Physical components reuse the logical record's spine — `id, name, group, tier, intro, sub,
purpose, cx, decisions, contract, dials, notes, failures, example, caps` — so the node card,
the hover card and half the drawer work with no new code. Six fields are added, and one
logical field is reinterpreted:

| Field          | Shape                                  | Purpose                                                                                                                                                                                             |
| -------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `realises`     | `string[]` (logical ids)               | The fidelity link (D7). Drives the "Realises" panel section, the reverse "Implemented by" section on logical components, and the completeness validator.                                            |
| `departure`    | `string`                               | Required whenever `realises.length !== 1` or the component is physical-only. States what moved and why. Printed in its own panel section with a warning rail.                                       |
| `runtime`      | `RUNTIME` key                          | Where the code actually executes. Drives the node's runtime chip, the deployment-unit legend, and the network-hop metric.                                                                           |
| `substrate`    | `{service, unit, sizing[]}`            | The AWS/engine identity: `{service:"Amazon OpenSearch Service", unit:"managed domain", sizing:["3 data nodes","2 replicas","20–40 GB shards"]}`.                                                    |
| `budget`       | `{deadline, timeout, onExpiry, retry}` | Configured only (D5). `onExpiry` names the degraded behaviour, which is the field that makes the deadline meaningful.                                                                               |
| `gotchas`      | `{g, why, fix}[]`                      | Distinct from `failures`. A failure is a way the system is wrong; a gotcha is a way _you_ will get it wrong. Appendix B is the source.                                                              |
| `optimisation` | `{o, when}[]`                          | The levers, each with the condition that makes it worth pulling.                                                                                                                                    |
| `scaling`      | `{axis, response}[]`                   | The DDIA section: what you change when corpus, QPS or fan-out grows. Partitioning, replication, and what each costs.                                                                                |
| `cost`         | `string`                               | The dominant cost driver, qualitative only (D5).                                                                                                                                                    |
| `alternatives` | `{a, when}[]`                          | The road not taken, with its selecting condition.                                                                                                                                                   |
| `cx`           | `1..5`                                 | **Reinterpreted**: operational burden — what it costs to run, patch, scale and page on — not build effort. Stated in the panel note so the two views' complexity indexes are not silently compared. |

`RUNTIME` is a new shared registry, and it is what makes the physical view legible at a glance:

```js
const RUNTIME = Object.freeze({
  edge: { label: "Edge", hop: true, chip: "edge", color: "var(--rt-edge)" },
  inproc: {
    label: "In the orchestrator",
    hop: false,
    chip: "in-process",
    color: "var(--rt-inproc)",
  },
  service: {
    label: "Container service",
    hop: true,
    chip: "Fargate",
    color: "var(--rt-service)",
  },
  gpu: {
    label: "GPU endpoint",
    hop: true,
    chip: "SageMaker",
    color: "var(--rt-gpu)",
  },
  foundation: {
    label: "Foundation model",
    hop: true,
    chip: "Bedrock",
    color: "var(--rt-fm)",
  },
  engine: {
    label: "Search engine",
    hop: true,
    chip: "engine",
    color: "var(--rt-engine)",
  },
  store: {
    label: "Managed store",
    hop: true,
    chip: "store",
    color: "var(--rt-store)",
  },
  stream: {
    label: "Stream / async",
    hop: false,
    chip: "async",
    color: "var(--rt-stream)",
  },
  control: {
    label: "Control plane",
    hop: false,
    chip: "config",
    color: "var(--rt-ctrl)",
  },
});
```

`hop:true` feeds the **"network hops in the request path"** metric — the physical analogue of
the logical view's "models in the request path", and the number this view exists to make
visible. `stream:false` because Firehose is fire-and-forget off the critical path; that is
itself a design claim the panel defends.

### 2.2 The inventory

Thirty-two request-path components, five data-plane sources, three consistency-plane nodes.
Full detail per component in **Appendix A**; the shape is:

**Band 1 — Edge & entry** (`Edge`)
`px-client` _core_ · `px-image` _full_ · `px-waf` _rec, control_ · `px-edge` _rec_ · `px-alb` _core_

**Band 2 — Orchestration** (`Orchestration`)
`px-orch` _core_ — the Fargate task; every `runtime:"inproc"` component below runs inside it.

**Band 3 — Query understanding** (`Query understanding`)
`px-qu` _core_ · `px-bedrock-qu` _full_ · `px-rewrite` _rec_ · `px-expand` _full_ ·
`px-session` _full, control_

**Band 4 — Predicate & routing** (`Constraint handling`)
`px-predicate` _core_ — the dual-target filter compiler · `px-router` _rec, control_

**Band 5 — Encoding** (`Encoding`)
`px-embed-cache` _rec_ · `px-embed-text` _core_ · `px-embed-image` _full_ · `px-embed-sparse` _full_

**Band 6 — Retrieval** (`Candidate retrieval`, parallel group)
`px-os-search` _core_ · `px-qdrant-search` _core_ · `px-deadline` _rec, control_

**Band 7 — Merge** (`Candidate generation`)
`px-sufficiency` _rec, decision_ · `px-recovery` _rec_ · `px-fuse` _core_

**Band 8 — Rerank** (`Reranking`, stage group)
`px-semrerank` _full, config-gated_ · `px-rescore-late` _full_ · `px-crossenc` _rec_ ·
`px-vlm` _full_ · `px-ltr` _full_

**Band 9 — Final ranking** (`Final ranking`)
`px-personal` _full_ · `px-final` _rec_

**Band 10 — Response** (`Results`)
`px-hydrate` _core_ · `px-assemble` _core_

**Band 11 — Feedback & control** (`Operations`)
`px-appconfig` _rec, control_ · `px-experiment` _rec, control_ · `px-otel` _rec, control_ ·
`px-events` _rec_ · `px-lake` _rec, control_ · `px-train` _full, control_

**Data plane** (toggle `data`)
`pxd-os-index` · `pxd-qdrant-collection` · `pxd-cache` · `pxd-ddb` · `pxd-s3`

**Consistency plane** (toggle `consistency`, off by default)
`pxi-source` — Aurora / DynamoDB source of truth ·
`pxi-log` — DynamoDB Streams / DMS → Kinesis Data Streams ·
`pxi-indexer` — indexing consumer that writes **both** engines from the one ordered log

### 2.3 The merges, the split, and the moves

This is the section the reader came for, and it is what `departure` records per component.

**Merged — many logical, one physical**

| Physical           | Realises                                               | Why they merge                                                                                                                                                                                         |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `px-qu`            | `normalise`, `understand`                              | Both are pure CPU on the same string in the same process; splitting them costs a hop and buys nothing. Normalisation stays a _distinct step_ inside it because it is the cache key.                    |
| `px-predicate`     | `prefilter`, `confidence`                              | One AST compiled twice, to `bool.filter` and to a Qdrant `Filter`. Confidence is a field on each predicate node, not a separate pass.                                                                  |
| `px-os-search`     | `lexical`, `sparse`                                    | One `_msearch`. BM25 `multi_match` and the neural-sparse clause share the same `filter` and the same round trip. Two logical legs, one network call — the largest single latency saving in the design. |
| `px-qdrant-search` | `textvec`, `textimagevec`, `imageimagevec`, `multivec` | One Query API request with up to four `prefetch` branches over named vectors on the same points. Four logical legs, one network call.                                                                  |
| `px-fuse`          | `union`, `prune`, `fusion`                             | Three in-process passes over one array. Drawing three boxes for three `for` loops would be dishonest at this abstraction level.                                                                        |
| `px-recovery`      | `relax`, `zerofallback`                                | One bounded controller with a pass counter; the two are branches of the same loop.                                                                                                                     |
| `px-final`         | `freshness`, `dedup`, `diversity`, `business`          | Four in-process re-sorts in the order plan 005 fixed, in one pass. The panel keeps the order explicit because the audit trail depends on it.                                                           |
| `px-lake`          | `offlineeval`                                          | Firehose lands Parquet; Glue catalogues; Athena replays. One logical component, one physical _stack_, drawn as one node.                                                                               |

**Split — one logical, many physical**

| Logical                   | Physical                                                                  | Why                                                                                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `understand`              | `px-qu` (rules + ONNX classifier) **and** `px-bedrock-qu` (Full only)     | The logical default is "rules plus a small classifier"; the LLM tier is a different _substrate_ with a different failure mode, a different cache requirement and a different bill. Drawing them as one node would hide the only decision that matters here. |
| `textvec` etc. → encoding | `px-embed-text` / `px-embed-image` / `px-embed-sparse` + `px-embed-cache` | Logical retrieval legs assume a query vector exists. Physically, producing it is a separate GPU hop with its own deadline and its own cache — and on a cache hit the whole hop disappears. That is invisible in the logical view and unmissable here.       |
| `assembly`                | `px-hydrate` + `px-assemble`                                              | Fetching the fields is a network call to two stores; shaping the response is not.                                                                                                                                                                           |

**Moved — the logical stage lands somewhere unexpected**

| Logical                                  | Lands in                                                 | Departure                                                                                                                                                                                                                                                      |
| ---------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `latererank`                             | **Inside Qdrant** (`px-rescore-late`)                    | Late-interaction rerank is a multivector MaxSim rescore over points already retrieved. It is a `query.rescore` clause, not a service. It therefore runs _before_ fusion, not after — a real ordering departure from the logical cascade, stated on both cards. |
| `ltr`                                    | **In-process** (`px-ltr`), not the OpenSearch LTR plugin | The plugin can only score documents OpenSearch retrieved. Half the candidates came from Qdrant. Blended retrieval rules out engine-side LTR — a direct, non-obvious consequence of D1.                                                                         |
| `routing`, `degradation`, `fusionpolicy` | `px-router`, `px-deadline`, `px-appconfig`               | Three logical control components become two in-process controllers reading one config store. The control plane physically _is_ AppConfig plus two objects.                                                                                                     |
| `q-image`                                | `px-image` — presigned S3 PUT, then a reference          | The browser does not post image bytes through the search API. The upload is a separate, earlier round trip and the query carries a reference. Pure physical concern, invisible logically.                                                                      |

**Physical-only — realises nothing**

`px-client`, `px-waf`, `px-edge`, `px-alb`, `px-orch`, `px-embed-cache`, `px-otel`,
`pxd-cache`, `pxd-ddb`, `pxd-s3`, and all three consistency-plane nodes. Each carries
`realises:[]` plus a `departure` explaining why the logical view was right to omit it. The
validator requires the explanation (§3.4).

**Unrealised logical components** — none. Every one of the 38 has a home; `expand` and
`session` land at Full only, matching their logical `intro`.

### 2.4 Worked panel — `px-qdrant-search`

One filled-in example, to fix the tone and prove the field set carries the weight.

```js
defPhys({
  id: "px-qdrant-search",
  name: "Qdrant query",
  group: "Candidate retrieval",
  tier: "core",
  intro: 1,
  parallel: "retrieval",
  runtime: "engine",
  cx: 4,
  sub: "one Query API call · named vectors · prefetch branches",
  realises: ["textvec", "textimagevec", "imageimagevec", "multivec"],
  departure:
    "Four logical retrieval legs become one network call. Qdrant's Query API takes " +
    "several prefetch branches in one request, each over a different named vector on the " +
    "same points, sharing one payload filter. The legs are still four distinct queries — " +
    "they just stop being four round trips. The panel's branch table keeps them visible.",
  substrate: {
    service: "Qdrant Cloud Hybrid (BYOC, in-VPC)",
    unit: "collection",
    sizing: ["shards = 2× nodes", "replication factor 2", "gRPC, not REST"],
  },
  byLevel: {
    1: { sub: "one branch · text_dense", chips: ["1 prefetch branch"] },
    2: {
      sub: "two branches · text_dense, image_clip",
      chips: ["2 prefetch branches"],
    },
    3: {
      sub: "four branches · + i2i, colbert",
      chips: ["4 prefetch branches"],
    },
  },
  decisions: [
    {
      q: "One call with prefetch branches, or one call per leg?",
      opts: [
        "One call per leg",
        "One call, N prefetch branches",
        "Server-side fusion in Qdrant",
      ],
      def: "One call, N prefetch branches, fused in the orchestrator",
      why:
        "Per-leg calls multiply the round trip and the tail: the leg you wait on is the " +
        "slowest of N independent network calls, not the slowest of N concurrent index " +
        "scans. Qdrant can also fuse the branches server-side with RRF, and you still " +
        "should not let it — the OpenSearch ranks are not in that request, so a " +
        "server-side fusion would produce a ranking over half the candidates.",
    },
    {
      q: "How is the hard predicate applied?",
      opts: [
        "Payload filter with payload indexes",
        "Post-filter the top-k",
        "Pre-fetch ids from OpenSearch",
      ],
      def: "Payload filter, with a payload index on every filtered field",
      why:
        "Without a payload index Qdrant cannot use the filterable-HNSW path and degrades " +
        "toward a scan; with one it prunes inside the graph traversal. Post-filtering a " +
        "truncated top-k is the failure the logical view already warns about, arriving here " +
        "as a config omission rather than a design choice.",
    },
  ],
  contract: {
    in: "query vectors + compiled Qdrant Filter + per-branch limits",
    out: "points with id, score, branch, payload subset",
  },
  dials: [
    {
      n: "HNSW m / ef_construct",
      v: "16 / 128",
      note: "Raise m for high-dimensional or high-recall collections; it costs memory permanently.",
    },
    {
      n: "ef (search)",
      v: "128",
      note: "The recall/work dial. Tune against exact search until the logical target — recall@100 ≥ 0.95 — holds at production filter selectivity, not on an unfiltered benchmark.",
    },
    { n: "Quantization", v: "scalar int8, rescore on, oversampling 2×" },
    { n: "Vectors on disk", v: "mmap, quantized vectors resident" },
    { n: "Transport", v: "gRPC" },
  ],
  budget: {
    deadline:
      "the retrieval band's allocation, shared with the OpenSearch call",
    timeout: "deadline + one RTT",
    onExpiry: "drop the branches the router marked optional, keep text_dense",
    retry:
      "none inside the band — a retry cannot fit in a deadline that already expired",
  },
  optimisation: [
    {
      o: "Payload index on every field the predicate can name",
      when: "Always. This is not optional; it is what makes filtered ANN work.",
    },
    {
      o: "Scalar quantization with rescore and 2× oversampling",
      when: "Once the collection no longer fits comfortably in RAM.",
    },
    {
      o: "Return ids and score only; hydrate elsewhere",
      when: "Always. Payload in the search response is bytes on the critical path you are about to overwrite.",
    },
    { o: "gRPC over REST", when: "Always, for a service-to-service caller." },
    {
      o: "Set indexing_threshold to 0 during bulk load, restore after",
      when: "Backfill and reindex only.",
    },
  ],
  gotchas: [
    {
      g: "Filter parity drift between the two engines",
      why: "The OpenSearch filter and the Qdrant filter are compiled from one AST but executed by two engines with different type coercion, different null semantics and different range-boundary handling. They diverge silently, and the symptom is a fusion ranking over two differently-filtered candidate sets.",
      fix: "Golden-file the compiler's two outputs. Assert equal result-id sets for a filter-only query with no scoring, in CI, against a fixture corpus.",
    },
    {
      g: "Recall measured unfiltered",
      why: "Filtered HNSW recall at 1% selectivity is a different number from unfiltered recall, and it is the one production experiences.",
      fix: "Benchmark ef against exact search at the selectivity your predicates actually produce.",
    },
    {
      g: "Point id space diverges from OpenSearch _id",
      why: "RRF fuses rank lists by document id. If the engines disagree about identity, fusion silently ranks two half-populated lists.",
      fix: "One canonical id, assigned upstream in the indexer, used verbatim as both the Qdrant point id and the OpenSearch _id.",
    },
  ],
  scaling: [
    {
      axis: "Corpus grows",
      response:
        "More shards, and quantization before more RAM. Resharding is a rebuild, so overshard slightly at the start.",
    },
    {
      axis: "QPS grows",
      response:
        "Replication factor, then read spread. Replicas cost the full index in memory each.",
    },
    {
      axis: "Branches grow",
      response:
        "Cost is roughly additive per branch within one call; the call itself does not get more expensive.",
    },
  ],
  cost: "Memory-dominated. Vector count × dimensions × replication, mitigated by quantization.",
  alternatives: [
    {
      a: "OpenSearch k-NN for the dense legs too, retiring Qdrant",
      when: "Corpus small enough that one engine's operational simplicity outweighs ANN control, or the team cannot run a second stateful system. Removes the entire filter-parity class of gotcha — the strongest argument against this plan's own D1, and it is stated here on purpose.",
    },
    {
      a: "Separate collections per modality instead of named vectors",
      when: "The modalities have genuinely different point populations, e.g. far more images than documents.",
    },
  ],
  failures: [
    {
      f: "Filtered ANN degrades to a scan under a selective predicate",
      s: "Retrieval deadline expires on exactly the queries with the most specific constraints",
      m: "deadline-expiry rate segmented by predicate selectivity",
    },
    {
      f: "Quantized search returns without rescore",
      s: "Ranking quality drops with no error and no latency change",
      m: "recall@100 against exact search, on the CI fixture",
    },
  ],
  example:
    'Carries the constraint-stripped text — "quiet waterfront cabin with a wood ' +
    'fireplace" — as text_dense, the same text against image_clip, and the compiled ' +
    "beds/price/geo predicate as the payload filter on both branches.",
});
```

The corresponding `px-os-search` panel is the mirror image: analyser parity, `bool.filter`
cacheability, `search_after` instead of `from`+`size`, shard sizing, `preference` for cache
locality, and the neural-sparse clause riding the same request.

---

## Phase 3 — Templates, relationships, gates and the validator wall

### 3.1 The three physical templates

Same rule as the logical view: **only the centre spine generates flow edges**; a component in
`l` or `r` sits on the row where it is produced and reaches consumers through `relations`.

**Physical Core (P1)** — 13 spine nodes, 2 data sources. Nothing recommended, nothing full.

```
px-client → px-alb → px-orch → px-qu → px-predicate → px-embed-text
  → [px-os-search ‖ px-qdrant-search] → px-fuse → px-hydrate → px-assemble
```

Data plane: `pxd-os-index`, `pxd-qdrant-collection`.
The tagline: _"Two engines, one process, six hops."_ Core deliberately has **no cache, no
rerank, no events** — it is the smallest thing that serves a hybrid query, and its point is
that it is already a distributed system with a consistency problem.

**Physical Recommended (P2)** — the production target. Adds, in row order:
`px-waf` + `px-edge` at the top; `px-experiment` and `px-otel` beside the orchestrator;
`px-rewrite`; `px-router` beside the predicate compiler; `px-embed-cache` as a left aside on
the encoding row; `px-deadline` beside retrieval; `px-sufficiency` → `px-recovery`;
`px-crossenc`; `px-final`; `px-appconfig`, `px-events` and `px-lake` on the right.
Data plane gains `pxd-cache` and `pxd-s3`.

**Physical Full (P3)** — the menu. Adds `px-image` and `px-session`; `px-bedrock-qu` and
`px-expand` as left asides; `px-embed-image` and `px-embed-sparse`; the remaining four
cascade tiers; `px-personal`; `px-train`; `pxd-ddb`.

Row layout for P3, in the `{l, c, r}` shape `templates` expects:

| #   | l                | c                                                    | r                         |
| --- | ---------------- | ---------------------------------------------------- | ------------------------- |
| 0   | `px-image`       | `px-client`                                          | `px-waf`                  |
| 1   |                  | `px-edge`                                            |                           |
| 2   |                  | `px-alb`                                             | `px-experiment`           |
| 3   |                  | `px-orch`                                            | `px-appconfig`, `px-otel` |
| 4   | `px-bedrock-qu`  | `px-qu`                                              | `px-session`              |
| 5   | `px-expand`      | `px-rewrite`                                         |                           |
| 6   |                  | `px-predicate`                                       | `px-router`               |
| 7   | `px-embed-cache` | `px-embed-text`, `px-embed-image`, `px-embed-sparse` |                           |
| 8   |                  | `px-os-search`, `px-qdrant-search`                   | `px-deadline`             |
| 9   |                  | `px-sufficiency` _(nextLabel: "sufficient")_         |                           |
| 10  | `px-recovery`    | `px-fuse`                                            |                           |
| 11  |                  | `px-semrerank`                                       |                           |
| 12  |                  | `px-rescore-late`                                    |                           |
| 13  |                  | `px-crossenc`                                        |                           |
| 14  |                  | `px-vlm`                                             |                           |
| 15  |                  | `px-ltr`                                             |                           |
| 16  |                  | `px-personal`                                        |                           |
| 17  |                  | `px-final`                                           |                           |
| 18  | `px-hydrate`     | `px-assemble`                                        | `px-events`               |
| 19  |                  |                                                      | `px-lake`, `px-train`     |

Row 7 is a second `parallel` group (the three encoders fire concurrently when all three legs
are routed); row 8 is the retrieval group; rows 11–15 are the `stage-group` cascade. All
three reuse existing geometry — **no new layout code in this plan.**

`extra` (branch and loop edges):

```js
[
  ["px-sufficiency", "px-recovery", "too few / zero", "branch"],
  ["px-recovery", "px-predicate", "recompiled predicate", "loop"],
];
```

Note the loop target. In the logical view relaxation returns to `prefilter`. Physically it
must return to the **predicate compiler**, because a relaxed constraint has to be recompiled
into _both_ engine dialects before either can be re-queried — you cannot loosen the
OpenSearch filter and not the Qdrant filter without fusing two incomparable candidate sets.
This is one of the clearest "physical is not just logical with logos on it" moments in the
diagram, and it gets a `RELATION_DETAILS` entry saying so.

### 3.2 Relationships

```js
relations:{
  "px-image":     {feeds:["px-embed-image"], steers:["px-router"]},
  "px-session":   {steers:["px-qu","px-rewrite","px-router","px-personal"]},
  "px-appconfig": {steers:["px-router","px-deadline","px-fuse","px-final"]},
  "px-router":    {steers:["px-os-search","px-qdrant-search","px-embed-text","px-embed-image",
                           "px-embed-sparse","px-semrerank","px-rescore-late","px-vlm","px-ltr"]},
  "px-deadline":  {steers:["px-os-search","px-qdrant-search","px-crossenc","px-vlm","px-rescore-late"]},
  "px-embed-cache":{feeds:["px-embed-text","px-embed-image","px-embed-sparse"]},
  "px-expand":    {feeds:["px-os-search"]},
  "px-experiment":{steers:["px-router","px-fuse"], feeds:["px-assemble"]},
  "px-otel":      {observes:[...]},                       // new kind — see below
  "px-assemble":  {feeds:["px-events"]},
  "px-events":    {updates:["px-personal"], feeds:["px-lake"]},
  "px-lake":      {feeds:["px-train"]},
  "px-train":     {trains:["px-ltr","px-crossenc","px-fuse"]},  // Full only
  "pxi-log":      {feeds:["pxi-indexer"]},
  "pxi-indexer":  {updates:["pxd-os-index","pxd-qdrant-collection"]}
}
```

**One new relation kind**, added to the shared `RELATION_TYPES` (2425) with the full visual
and copy contract the file requires:

```js
observes:{
  fwd:"Observes", rev:"Observed by",
  description:"Trace and metric collection. Carries no request data and can fail without "+
              "failing the request.",
  aria:"Telemetry collection",
  color:"var(--wire-observe)", dash:"2 6", linecap:"round", marker:"observe",
  opacity:.55, width:1.2, route:"side", curve:.5, componentRelation:true,
  abstraction:["physical"],                 // new: restricts the kind to one model
  legend:{visible:true, label:"observes — telemetry"}
}
```

It needs a colour token in both theme blocks, and it is drawn faintest of all the kinds on
purpose — telemetry touches everything and must not become the loudest thing on the canvas.
Consider defaulting it to hidden behind a "show telemetry" checkbox if the first render is
noisy; the plane mechanism from §1.2 already supports that.

`pxi-indexer → pxd-*` uses `updates`, which already means "asynchronous state change" — an
exact fit for a derived index being written from a log, and no new kind needed.

### 3.3 Gates

```js
gates:{
  "px-embed-image":  defGate(3,"intent","visual intent"),
  "px-embed-sparse": defGate(3,"route","route selected"),
  "px-bedrock-qu":   defGate(3,"config","optional pass"),
  "px-expand":       defGate(3,"route","route selected"),
  "px-semrerank":    defGate(3,"config","optional pass"),
  "px-rescore-late": defGate(3,"route","route selected"),
  "px-vlm":          defGate(3,"intent","visual intent"),
  "px-ltr":          defGate(3,"config","optional pass"),
  "px-personal":     defGate(3,"config","optional pass")
}
```

`GATE_KINDS` (2313) needs no new kind — but `decidedBy:"routing"` is a _logical_ id. Make
`decidedBy` per model: the logical model resolves it to `routing`, the physical model to
`px-router`. Cleanest is to move `decidedBy` out of `GATE_KINDS` into the model as
`model.gateOwner = {route:"px-router", intent:"px-router", config:null}` and have `defGate`
take the model. `defGate` currently resolves at definition time (2344) and throws on an
unknown kind — preserve that; it is the property the architecture doc singles out.

Note `px-os-search` and `px-qdrant-search` are **not** gated even though four of the six
logical legs they realise are. The branches inside them are gated; the calls are not, because
the call still happens for the ungated branches. The panels' branch tables carry the gates,
and `px-router` steers both nodes. This is a real fidelity loss from merging, and it belongs
in each node's `departure` text.

### 3.4 The cross-view fidelity validator (D7)

> **Revised after Phase 1.** This is not a fourth top-level validator. `model.invariants`
> already exists and runs at the end of `validateRelationshipModel` with
> `(model, {componentRelations, servingRelations, usedKinds})`. Ship the eight rules below as
> `PHYSICAL_INVARIANTS[0]`, using the module-level `assertComponent(model,id,context)` helper.
> The logical model's four invariants are the worked example of the shape.

New, and the most valuable thing in this phase:

```js
function validateRealisationMap(logical, physical) {
  // 1. every `realises` id exists in the logical model
  // 2. every logical component appears in at least one `realises` array,
  //    unless it is listed in physical.unrealised with a written reason
  // 3. a physical component whose realises.length !== 1 must carry `departure`
  // 4. a logical component realised by more than one physical component must
  //    be named in each of their `departure` texts (substring check on name)
  // 5. a physical component with realises:[] must carry `departure`
  // 6. the intro level never regresses: for every realisation edge,
  //    physical.intro >= logical.intro — a physical component may not appear
  //    at a level before the logical capability it implements
  // 7. every physical component declares a `runtime` present in RUNTIME
  // 8. every physical component on the request path declares `budget`
}
```

Rule 6 is the one that will actually catch mistakes. It is the physical analogue of the
serving layer's derived tier, and it stops the physical view from quietly promising a
capability the logical view has not introduced yet.

### 3.5 The numbers rule validator (D5)

> **Revised after Phase 1.** Ship as `PHYSICAL_INVARIANTS[1]`, for the same reason as §3.4.

```js
const MEASURED_CLAIM =
  /\bp\d{2}\b|\b\d+\s?ms\b|\bmilliseconds?\b|\blatenc|\bthroughput\b|\bQPS is\b|\$\d/i;
```

Walk every string field of every physical component **except** the fields where a configured
value is the point (`dials`, `budget`, `substrate.sizing`). Throw on a match. The exemption
list is short and explicit, which is what keeps the rule honest: a millisecond number is
allowed only where the schema says "this is a setting".

Expect one legitimate collision — the worked-example query contains `$300`. Scope the `\$\d`
branch to exclude `example`, or drop it and rely on review.

### 3.6 Baseline and counts

`physical.baseline` follows `DEPENDENCY_BASELINE`'s exact shape (2556) — per template, per
kind, arrays of `[from, to, label?]`. It is hand-maintained and it will throw on the first
reload after every structural change. That is the design, and the architecture doc already
warns about it; the physical fixture is simply a second wall of the same kind.

**Generate it, don't hand-write it.** After Phase 3's data lands but before the fixture
exists, stub `physical.baseline = null`, make `validateDependencyBaseline` skip a null
fixture, load the page, and run:

```js
window.dependencyDiagnostics.emitBaseline("physical");
```

— a new diagnostic that prints `dependencies()` for all three templates × all plane
combinations, already formatted as a fixture literal. **Read it before pasting it.** The
fixture's value is that a human confirmed each edge once; generating it and reviewing it
gets that value at a fraction of the cost of typing it. Add `emitBaseline` for the logical
model too — it is useful there and it proves the generator agrees with the existing fixture.

Plane combinations multiply: logical had 2 (serving on/off), physical has 4 (data ×
consistency). Either assert all 4 or declare that `consistency` only ever adds `updates` and
`feeds` edges among `pxi-*` and `pxd-*` nodes, and assert that behavioural invariant instead
of enumerating. **Prefer the invariant** — it is the same trick the existing baseline uses
for the serving layer ("showing sources adds _only_ `serves` edges") and it keeps the fixture
from doubling.

---

## Phase 4 — The view

### 4.1 The tab

Insert between Build order and About (786–788):

```html
<button
  role="tab"
  id="tab-phys"
  aria-controls="view-phys"
  aria-selected="false"
  data-view="phys"
>
  Physical architecture
</button>
```

`setView` (4588) is already a chain of `classList.toggle` calls; generalise it to iterate the
view ids. Switching to `phys` must `requestAnimationFrame(drawWires)` on the physical canvas,
exactly as `pipe` does — wires are measured from live DOM and a hidden canvas measures as zero.

### 4.2 LHS options panel

> **Carried over from Phase 1.** `renderPlaneToggle` currently renders exactly one control,
> bound to `model.planes[0]`, and is marked provisional in the code. Generalising it to one
> control per declared plane is the **first** thing Phase 4 must do — the physical view's `data`
> and `consistency` planes do not work until it is done. The plane declarations, the state
> slice and `Model.dependencies` are already plane-driven; only the control is not.

Same furniture as the logical sidebar, different contents.

**Level selector** — Core / Recommended / Full, reusing `.seg`. The tooltips carry the
physical taglines. Independent of the logical view's level (per-model state, §1.3), because
comparing "logical Full" against "physical Core" is a legitimate thing to want to do.

**Plane toggles** — two `.dep-toggle` buttons: _Show data plane_ (engines, caches, stores) and
_Show write path_ (the consistency plane). Rendered from `model.planes`, so a third view adds
toggles by declaring them.

**Deployment shape** (replaces "Pipeline shape"):

| Metric                           | Computation                                                                                                  | Why it earns the space                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Network hops in the request path | count of enabled components with `RUNTIME[runtime].hop`, counting a parallel group once at its widest member | The headline number. Core is 6; Full is 11. Watching it move as you toggle components is the whole argument of the tab. |
| Distinct deployment units        | unique `substrate.service` over enabled components                                                           | What you actually have to run, patch and page on.                                                                       |
| GPU endpoints in the path        | `runtime === "gpu"` count                                                                                    | The cost and cold-start axis, and the physical echo of "models in the request path".                                    |
| Operational burden               | sum of `cx`, banded like the logical one                                                                     | With an explicit note that `cx` means something different here (§2.1) so the two indexes are not compared.              |

**Concerns coverage** (replaces "Capability coverage") — the physical taxonomy, same
click-to-trace behaviour, using the existing `caps` field and `setFocus("cap", …)`:

```js
const CONCERNS = [
  { id: "parity", label: "Filter parity" },
  { id: "freshness", label: "Index freshness" },
  { id: "tail", label: "Tail latency control" },
  { id: "degrade", label: "Graceful degradation" },
  { id: "blast", label: "Blast-radius containment" },
  { id: "cache", label: "Cache coherence" },
  { id: "replay", label: "Deterministic replay" },
  { id: "cost", label: "Cost control" },
  { id: "observe", label: "Traceability" },
  { id: "identity", label: "Document identity" },
  { id: "pii", label: "PII boundary" },
  { id: "rollback", label: "Reversible deploys" },
];
```

Twelve concerns, and every one of them is a thing the logical view is right not to model and
the physical view is wrong to omit. `px-fuse` covers `identity`; `px-predicate` covers
`parity`; `px-deadline` covers `tail` and `degrade`; `pxi-indexer` covers `freshness` and
`parity`; and so on. The uncovered ones at Core are the argument for Recommended.

**Legend** — runtime chips (from `RUNTIME`), relation kinds (auto-generated from the shared
registry, so `observes` appears for free), gate conditions (auto-generated), plane keys.

**Components list** — unchanged mechanism.

### 4.3 RHS panel — the section spec

Replace the hardcoded chain at 4194–4226 with an ordered spec per model:

```js
const PANEL_SECTIONS = {
  // key -> {title, render(component, model, slice)}
  realises: { title: "Realises", render: renderRealises },
  departure: {
    title: "Departure from logical",
    render: renderDeparture,
    rail: "warn",
  },
  decisions: { title: "What you are choosing", render: renderDecisions },
  substrate: { title: "Substrate", render: renderSubstrate },
  contract: { title: "Wire contract", render: renderContract },
  budget: { title: "Deadline & failure", render: renderBudget },
  dials: { title: "Configuration", render: renderDials },
  optimisation: { title: "Optimisation", render: renderOptimisation },
  gotchas: { title: "Gotchas", render: renderGotchas },
  scaling: { title: "Scaling", render: renderScaling },
  failures: { title: "Failure modes", render: renderFailures },
  alternatives: { title: "Alternatives", render: renderAlternatives },
  cost: { title: "Cost driver", render: renderCost },
  notes: { title: "Notes", render: renderNotes },
  example: { title: "In the worked example", render: renderExample },
  relations: { title: null, render: renderRelationRows },
  serving: { title: "Served by", render: renderServedBy },
  appears: { title: "Appears in", render: renderAppearsIn },
  implementedBy: { title: "Implemented by", render: renderImplementedBy }, // derived, logical only
};

logical.panel = [
  "decisions",
  "contract",
  "dials",
  "notes",
  "failures",
  "example",
  "refs",
  "relations",
  "serving",
  "implementedBy",
  "appears",
];
physical.panel = [
  "realises",
  "departure",
  "decisions",
  "substrate",
  "contract",
  "budget",
  "dials",
  "optimisation",
  "gotchas",
  "scaling",
  "failures",
  "alternatives",
  "cost",
  "notes",
  "example",
  "refs",
  "relations",
  "serving",
  "appears",
];
```

A section renders only when its field is present, exactly as today. This turns "add a field
to the drawer" from an edit inside a 60-line function into a one-line spec change — worth
doing on its own merits even without the physical view.

**The cross-view links** are the payoff:

- On a physical component, _Realises_ lists logical components as buttons. Clicking one
  switches to the Pipelines tab, sets that view's level to the component's `intro`, and opens
  its drawer.
- On a logical component, a derived _Implemented by_ section does the reverse. It is computed
  from the physical model's `realises` arrays at load — no second hand-maintained mapping.

Round-tripping between the two views on one component is the feature that makes this a
single tool rather than two diagrams in a trench coat.

### 4.4 Hover card

`HOVER_RELATIONS` (4344) becomes per model. The physical card's four fields:

```
name · runtime chip · substrate.service
gate line (unchanged mechanism)
purpose
Realises — logical component names
Deadline — budget.deadline · on expiry: budget.onExpiry
first gotcha, if any
```

The gotcha on hover is deliberate: it is the highest-value sentence on most of these cards
and the reason to hover rather than click.

### 4.5 CSS

Additive only; no existing rule changes.

- `--rt-*` runtime colour tokens in both theme blocks, plus `--wire-observe`.
- `.node .rtchip` — runtime chip in the node's `.meta` row, alongside the existing tierchip
  and groupchip. Check the three-chip case does not wrap on the narrowest node.
- `.sect.warn` — the departure section's rail.
- `.plane-band` — a subtle background band behind consistency-plane rows so the write path
  reads as a different plane rather than more pipeline.
- `.branch-table` — the per-branch tables inside `px-os-search` and `px-qdrant-search`.

### 4.6 Narrow screens

Below 700px the tool renders dependency chips instead of wires (`mobileTransition`,
`mobileRecovery`, `renderMobileSelection`, 3218–3384). Those functions take `logical` — the
dependency array — and are otherwise generic, so parameterising them in Phase 1 covers the
physical view too. **Verify it rather than assuming it**: the architecture doc's standing
warning is that relationship display usually needs doing twice.

---

## Phase 5 — Follow one request

The tab's stated purpose is _"how the query dataflow happens from start to results"_, and a
static component diagram does not show that. Reuse the `.trace` row in the context bar
(797–803): keep facet tracing, add a stepper.

```js
physical.trace = [
  {
    at: "px-alb",
    wire: "HTTPS POST /search",
    carries: "raw query, UI filters, session id, JWT",
    note:
      "The UI filters arrive separate from the query text and stay separate — the logical " +
      "q-text default made that choice; here it is a field on the request body.",
  },
  {
    at: "px-qu",
    wire: "in-process",
    carries:
      "normalised string, intent class, constraint list with confidences",
    note: "Cache lookup on the normalised string first. Deterministic, so the TTL is long.",
  },
  {
    at: "px-predicate",
    wire: "in-process",
    carries:
      "one predicate AST → an OpenSearch bool.filter and a Qdrant Filter",
    note: "Compiled twice from one source. The two outputs are golden-filed against each other in CI.",
  },
  {
    at: "px-embed-text",
    wire: "gRPC → SageMaker",
    carries: "constraint-stripped query text → one dense vector",
    note:
      "Cache hit here removes the hop entirely. Keys on the stripped text, not the raw " +
      "query, which is why the asymmetric-rewrite default raises the hit rate.",
  },
  {
    at: "px-os-search",
    wire: "HTTPS POST /_msearch",
    carries: "BM25 clause + neural-sparse clause + shared filter → ranked ids",
    note: "One round trip for two logical legs.",
  },
  {
    at: "px-qdrant-search",
    wire: "gRPC Query",
    carries:
      "N prefetch branches + payload filter → ranked point ids per branch",
    note: "Concurrent with the OpenSearch call. The band costs its slower member, not their sum.",
  },
  {
    at: "px-fuse",
    wire: "in-process",
    carries: "per-leg rank lists → one RRF-ordered candidate list",
    note:
      "Ranks only. No score ever crosses the engine boundary, which is why k=60 and nothing " +
      "needs normalising.",
  },
  {
    at: "px-crossenc",
    wire: "gRPC → SageMaker",
    carries: "top 100 (query, title+field) pairs → 100 scores",
    note:
      "One batched call, not 100. Deadline-checked before dispatch: if the budget is " +
      "already spent, the fused order ships.",
  },
  {
    at: "px-final",
    wire: "in-process",
    carries: "scored candidates → the delivered order",
    note:
      "Freshness, dedup, diversity, then business last — the order plan 005 fixed so the " +
      "audit trail describes the page the user sees.",
  },
  {
    at: "px-hydrate",
    wire: "HTTPS _mget + DynamoDB BatchGetItem",
    carries:
      "20 ids → fields, images, snippets, plus volatile price and availability",
    note:
      "Volatile fields come from DynamoDB, not the index, because the index is a derived " +
      "dataset and is allowed to be stale. This is the freshness-skew mitigation.",
  },
  {
    at: "px-assemble",
    wire: "HTTPS response",
    carries: "results + relaxation, degradation, route and experiment state",
    note:
      "The disclosure fields are what make a degraded response debuggable instead of " +
      "mysterious.",
  },
  {
    at: "px-events",
    wire: "async → Firehose",
    carries: "impression context, positions, route, variant",
    note: "After the response is written. Nothing on the critical path waits for this.",
  },
];
```

Rendering: a `‹ step n of 12 ›` control plus the step's `wire` / `carries` / `note`. Stepping
sets `focus = {kind:"trace", index:n}`, which dims every node except the current one and its
immediate predecessor and highlights the edge between them. `applyFocus` (4081) already does
dimming by predicate; this is a new predicate, not new machinery.

Steps whose `at` is disabled or absent from the current level are skipped, so the trace
shortens as the level drops — twelve steps at Full, seven at Core. That contraction is a
better argument for the tab than any prose.

---

## Phase 6 — Documentation

1. **About tab** — a fourth section explaining the abstraction ladder (borrowing
   `architecture-mental-model.md` §1), what "physical" adds, and a guardrail paragraph
   mirroring the existing one at 1005–1010: _this is one opinionated topology on AWS with two
   named engines, not a recommendation, a capacity model or a cost estimate; the numbers are
   settings, not measurements._ State D1 and D5 explicitly — a reader who disagrees with the
   engine split should be able to see that it was a decision, not an assumption.
2. **`search-query-pipeline-diagram-tool-architecture.md`** — the file regions table, the
   "five sources of truth" section (now _models_, and a model is the source of truth), the
   layers mermaid, the recipes, and a new **"Add an abstraction level"** recipe. The doc's own
   header says a point-in-time snapshot is not worth maintaining; this change invalidates
   enough of it that it needs a real pass rather than a patch.
3. **`TODO.md`** — tick the physical-implementations feature; add the follow-ups from §7.

---

## 7. Deliberately not in this pass

| Deferred                                               | Why                                                                                                                                                                                    | Cost when it comes back                                             |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Physical build order (D8)                              | The physical inventory will move once it has been read by someone other than its author. A ladder over unstable rungs is churn.                                                        | A `steps` array and the existing validator.                         |
| Deployment / topology view (AZs, subnets, node counts) | It is a third abstraction _viewpoint_, not a third level, and it wants a different layout entirely.                                                                                    | A third model — which is exactly what Phase 1 exists to make cheap. |
| Second substrate (OpenSearch-only, retiring Qdrant)    | Doubles the physical data before the first version has been reviewed. It is written up as the `px-qdrant-search` **alternative** instead, which carries the argument without the data. | A `substrate` axis on the physical model, orthogonal to level.      |
| Full ingest pipeline                                   | Three consistency-plane nodes carry the argument; connectors, backfill and alias swaps would outweigh the query path.                                                                  | A fourth plane, or its own model.                                   |
| URL state, mermaid export                              | Already on `TODO.md`, and both get harder if built before the model refactor and trivial after.                                                                                        | Unchanged — but do them after Phase 1, not before.                  |
| Cost modelling                                         | D5 forbids dollar figures, and a cost model is a different tool.                                                                                                                       | —                                                                   |

---

## 8. Risks

| Risk                                                                                     | Likelihood | Mitigation                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ~~Phase 1 changes rendered output subtly~~ — **retired**                                 | —          | Did not happen. Verified byte-identical (§1.6). The snapshot stays useful: it now guards the shared render path against Phases 2–5.                                                                                  |
| A Phase 2–5 change to the shared render path breaks the logical view                     | medium     | Re-run the §1.6 comparison before each commit; the logical view must still hash to `163328:4705a20e` at 1280×900. A move means the change belongs inside the physical model, not in the shared layer.                |
| The physical view's two planes do not work because the toggle is still singular          | high       | Known and recorded (§1.7 row 4, §4.2). It is the first task of Phase 4 and it is small — the declarations, the state slice and `Model.dependencies` are already plane-driven.                                        |
| The physical baseline fixture becomes a maintenance tax                                  | high       | Generate it with `emitBaseline` and review; assert the consistency plane as an invariant rather than enumerating it (§3.6).                                                                                          |
| Two parallel groups plus a stage group in one template exceeds what the geometry handles | medium     | Rows 7, 8 and 11–15 use only existing mechanisms, but they have never coexisted. Build P3's rows _first_, before writing any panel prose, so a layout problem surfaces while the data is still cheap to move.        |
| The retrieval band collapsing 6 logical legs into 2 nodes reads as a loss of fidelity    | medium     | It is the point, and it is defended in `departure` and in the branch tables — but if reviewers reject it, splitting the two nodes back into per-leg nodes is a template-and-baseline change only, not a code change. |
| Panel prose drifts from AWS and engine reality                                           | medium     | Every `substrate`, `dials` and `gotchas` entry needs a citation in `REFS` under the same id, resolved and title-checked, as the existing REFS banner (1856) requires.                                                |
| The tab is simply too dense                                                              | medium     | The plane toggles and the level selector are the pressure valve. Physical Core is 13 nodes — smaller than logical Core+recommended. Land it, look at Core, and cut from Full if Full is unreadable.                  |
| `observes` wires bury the canvas                                                         | low        | Ship it behind a plane toggle if the first render is noisy (§3.2).                                                                                                                                                   |

---

## Appendix A — Component inventory

`R` = realises. `rt` = runtime. Level is the physical `intro`.

### Request path

| id                 | name                     | lvl | rt         | substrate                       | R                                                      |
| ------------------ | ------------------------ | --- | ---------- | ------------------------------- | ------------------------------------------------------ |
| `px-client`        | Search client            | 1   | edge       | Web / mobile app                | —                                                      |
| `px-image`         | Image upload             | 3   | edge       | S3 presigned PUT                | `q-image`                                              |
| `px-waf`           | AWS WAF                  | 2   | edge       | WAF web ACL, rate rules         | —                                                      |
| `px-edge`          | CloudFront               | 2   | edge       | Distribution, no search caching | —                                                      |
| `px-alb`           | ALB + authorizer         | 1   | edge       | ALB, OIDC/Cognito, TLS          | —                                                      |
| `px-orch`          | Search orchestrator      | 1   | service    | ECS Fargate service, ≥2 AZ      | —                                                      |
| `px-qu`            | Query understanding      | 1   | inproc     | Rules + ONNX classifier         | `normalise`, `understand`                              |
| `px-bedrock-qu`    | LLM query understanding  | 3   | foundation | Bedrock, cached, gated          | `understand`                                           |
| `px-rewrite`       | Per-leg rewriting        | 2   | inproc     | Dictionary + rules              | `rewrite`                                              |
| `px-expand`        | Query expansion          | 3   | inproc     | PRF over the lexical leg        | `expand`                                               |
| `px-session`       | Session context          | 3   | store      | DynamoDB + ElastiCache          | `session`                                              |
| `px-predicate`     | Predicate compiler       | 1   | inproc     | One AST → two dialects          | `prefilter`, `confidence`                              |
| `px-router`        | Retrieval router         | 2   | inproc     | Rules from AppConfig            | `routing`                                              |
| `px-embed-cache`   | Embedding cache          | 2   | store      | ElastiCache (Valkey)            | —                                                      |
| `px-embed-text`    | Text embedding           | 1   | gpu        | SageMaker, Triton batching      | _(query side of)_ `textvec`                            |
| `px-embed-image`   | Image embedding          | 3   | gpu        | SageMaker, CLIP/SigLIP tower    | _(query side of)_ `imageimagevec`                      |
| `px-embed-sparse`  | Sparse encoder           | 3   | gpu        | SPLADE, or OpenSearch-side      | _(query side of)_ `sparse`                             |
| `px-os-search`     | OpenSearch query         | 1   | engine     | `_msearch`, one round trip      | `lexical`, `sparse`                                    |
| `px-qdrant-search` | Qdrant query             | 1   | engine     | Query API, N prefetch           | `textvec`, `textimagevec`, `imageimagevec`, `multivec` |
| `px-deadline`      | Deadline controller      | 2   | inproc     | Budget, hedging, breakers       | `degradation`                                          |
| `px-sufficiency`   | Sufficiency check        | 2   | inproc     | Per-leg floors                  | `sufficiency`                                          |
| `px-recovery`      | Recovery controller      | 2   | inproc     | Bounded, 2 passes               | `relax`, `zerofallback`                                |
| `px-fuse`          | Union, prune, RRF        | 1   | inproc     | k=60, ranks only                | `union`, `prune`, `fusion`                             |
| `px-semrerank`     | Semantic rerank          | 3   | inproc     | ONNX bi-encoder, off by default | `semrerank`                                            |
| `px-rescore-late`  | Late-interaction rescore | 3   | engine     | Qdrant multivector MaxSim       | `latererank`                                           |
| `px-crossenc`      | Cross-encoder rerank     | 2   | gpu        | SageMaker, top 100 batched      | `crossenc`                                             |
| `px-vlm`           | VLM rerank               | 3   | foundation | Bedrock vision, top 10–20       | `vlmrerank`                                            |
| `px-ltr`           | Learning-to-Rank         | 3   | inproc     | LightGBM via ONNX               | `ltr`                                                  |
| `px-personal`      | Personalisation          | 3   | store      | DynamoDB profile read           | `personalisation`                                      |
| `px-final`         | Final ranking passes     | 2   | inproc     | Four ordered passes             | `freshness`, `dedup`, `diversity`, `business`          |
| `px-hydrate`       | Hydration                | 1   | engine     | `_mget` + DDB BatchGetItem      | _(data side of)_ `assembly`                            |
| `px-assemble`      | Response assembly        | 1   | inproc     | Paging token, disclosure        | `assembly`                                             |

### Control, feedback, data and write planes

| id                      | name                      | lvl | rt      | substrate                        | R                                              |
| ----------------------- | ------------------------- | --- | ------- | -------------------------------- | ---------------------------------------------- |
| `px-appconfig`          | Config & flags            | 2   | control | AWS AppConfig                    | `fusionpolicy`                                 |
| `px-experiment`         | Experiment assignment     | 2   | inproc  | Hash + AppConfig                 | `experiment`                                   |
| `px-otel`               | Tracing & metrics         | 2   | control | ADOT → X-Ray, CloudWatch         | —                                              |
| `px-events`             | Impression events         | 2   | stream  | Kinesis Firehose → S3            | `behavioural`                                  |
| `px-lake`               | Evaluation store          | 2   | control | S3 + Glue + Athena               | `offlineeval`                                  |
| `px-train`              | Training jobs             | 3   | control | SageMaker + Step Functions       | —                                              |
| `pxd-os-index`          | OpenSearch index          | 1   | engine  | Shards, replicas, analysers      | `ds-text`, `ds-sparse`, `ds-doc-store`         |
| `pxd-qdrant-collection` | Qdrant collection         | 1   | engine  | Named vectors, payload indexes   | `ds-doc-ann`, `ds-image-ann`, `ds-passage-ann` |
| `pxd-cache`             | Cache tier                | 2   | store   | ElastiCache (Valkey)             | —                                              |
| `pxd-ddb`               | Profile & volatile fields | 3   | store   | DynamoDB                         | —                                              |
| `pxd-s3`                | Data lake                 | 2   | store   | S3 (Parquet) + Glue              | —                                              |
| `pxi-source`            | Source of truth           | 2   | store   | Aurora / DynamoDB                | —                                              |
| `pxi-log`               | Change log                | 2   | stream  | Streams / DMS → Kinesis          | —                                              |
| `pxi-indexer`           | Dual-target indexer       | 2   | service | ECS consumer, idempotent upserts | —                                              |

The data-plane rows are worth reading twice: **six logical data sources collapse into two
physical stores.** `ds-text`, `ds-sparse` and `ds-doc-store` are three indexes and a store in
the logical view and one OpenSearch index in this one; `ds-doc-ann`, `ds-image-ann` and
`ds-passage-ann` are three ANN indexes logically and three named vectors on one Qdrant
collection here. The logical view is right to keep them separate — they have different
analysis, granularity and update requirements, exactly as `ds-text`'s note (2216) says — and
the physical view is right to co-locate them. Showing both readings on one screen, with the
`realises` link between them, is the single best justification for the tab existing.

---

## Appendix B — The gotchas catalogue

Source material for the `gotchas` and `optimisation` fields. Grouped by where the trap lives,
not by which component happens to own it — several span two.

### B1 — The dual-store problem (the tab's central subject)

Everything here follows from one document existing as an OpenSearch `_doc` and a Qdrant point.

1. **Document identity.** RRF fuses rank lists by id. If the OpenSearch `_id` and the Qdrant
   point id are assigned independently, fusion silently ranks two half-populated lists and
   the symptom is mediocre relevance, not an error. → One canonical id assigned in
   `pxi-indexer`, used verbatim in both. Qdrant point ids are UUID or unsigned integer, so
   a string business key needs a deterministic UUIDv5 derivation, not a hash you cannot invert.
2. **Filter parity.** The same predicate compiled to two dialects diverges on type coercion,
   null and missing-field semantics, range inclusivity, and geo units. → Golden-file both
   compiler outputs; assert equal id sets for a filter-only, scoring-free query in CI.
3. **Freshness skew.** Two engines acknowledge a write at different times. OpenSearch has a
   refresh interval; Qdrant has its own indexing threshold and optimiser. A document is
   briefly retrievable by one leg and not the other, so it enters fusion with one rank
   instead of two and is systematically under-ranked. → Accept it, bound it, and keep volatile
   fields (price, availability) out of both indexes and in DynamoDB at hydration.
4. **Never dual-write from the request path or from two independent jobs.** The DDIA framing:
   both indexes are _derived data_, and the only thing that keeps derived datasets consistent
   is deriving them from **one ordered log** with idempotent, version-stamped writes. Two
   writers, two orderings, permanent divergence with no way to detect it. → `pxi-log` is the
   ordering authority; `pxi-indexer` is the only writer of both.
5. **Reindex asymmetry.** Re-embedding is expensive and re-analysing is cheap, so the two
   engines want different rebuild cadences. An encoder version bump rebuilds Qdrant entirely;
   an analyser change rebuilds OpenSearch entirely. → Version the encoder in the collection
   name and alias-swap; the two rebuilds must be independently runnable.
6. **Deletion and suppression.** A document removed from one engine and not the other still
   appears in fused results with a single rank. → Deletions ride the same log; hydration
   revalidates eligibility as defence in depth, which is what `ds-doc-store`'s filtering
   contract (2255) already asks for.

### B2 — OpenSearch

- **Analyser parity.** Query-time and index-time analysis must match, or exact-title queries
  stop returning rank 1. The logical `normalise` failure mode ("exact-match probe suite, hard
  CI gate") is the mitigation; here it is a concrete `_analyze` assertion in CI.
- **`filter` versus `must`.** Hard predicates belong in `bool.filter`: no scoring, and the
  filter cache applies. A predicate in `must` is scored, uncached, and pollutes BM25.
- **Deep paging.** `from` + `size` re-sorts the whole window on every shard. Use
  `search_after`, and a PIT when the ranking must be stable across pages — which it must be
  here, because the fused order is not reproducible from the index alone.
- **Shard sizing.** Roughly 20–40 GB per shard, and shard count is fixed at creation.
  Oversharding multiplies per-query coordination for no gain; undersharding forces a reindex.
- **`preference`.** Routing a user's successive requests to the same shard copies improves
  cache locality — free, and easy to forget.
- **The request cache is `size:0` only.** It caches aggregations, not hits. Assuming it caches
  search results is a common and expensive misreading.
- **`_msearch` fails as a unit under a bad clause.** One malformed sub-query does not fail the
  batch, but one _slow_ sub-query holds the response. Per-clause `timeout` matters.
- **Neural sparse rides the same request** — that is the whole reason sparse is on this engine.
  But it inflates the query's term count, so `terminate_after` and clause limits need checking.
- **UltraWarm and cold tiers are not for the query path.** They exist; they are not where a
  p99-sensitive search request should land.

### B3 — Qdrant

- **Payload indexes are mandatory, not an optimisation.** Without one on a filtered field,
  filtered search degrades toward a scan. The failure is invisible until a selective predicate
  arrives, which is exactly when the query mattered most.
- **Filtered recall is a different number.** Benchmark `ef` against exact search _at production
  selectivity_. The logical `textvec` default — recall@100 ≥ 0.95 versus exact — is only
  meaningful measured under filters.
- **Quantization without rescore silently loses quality.** Scalar int8 with rescore and ~2×
  oversampling is the safe default; quantization alone changes ranking with no error and no
  latency signal.
- **`m` is permanent.** HNSW `m` is set at collection creation and costs memory forever.
  `ef` is the per-query dial; `ef_construct` is the build-time one.
- **Named vectors versus separate collections.** Named vectors share the point, the payload
  and the filter — a large win when the modalities describe the same entity. They are wrong
  when the populations differ sharply (far more images than documents), because every point
  carries every named vector's overhead.
- **Multivector cost is per token.** Late-interaction storage scales with tokens, not
  documents. Quantize it, keep it on disk, and treat it as a rescore over an already-narrow
  candidate set — never as a first-stage retriever.
- **gRPC, not REST**, for a service-to-service caller. REST is for the console.
- **`indexing_threshold` during bulk load.** Set to 0 to defer index building, restore
  afterwards. Forgetting the restore leaves the collection unindexed and fast to write,
  slow to query — a delightful way to lose a day.
- **Resharding is a rebuild.** Overshard slightly at creation.

### B4 — Fan-out and the tail

- **The band costs its slowest member, and its p99 is the _max_ of the legs' p99s.** Two legs
  each with a rare stall produce a band that stalls roughly twice as often. Adding a leg
  makes the tail worse even when it makes the mean better.
- **Deadlines must be propagated, not just set.** A deadline the orchestrator holds and the
  engine does not know about produces a cancelled client and a still-running query — the
  engine keeps burning the CPU you are no longer waiting for. Send the engine's own timeout
  too, and make sure cancellation actually reaches it.
- **Hedged requests are for tail control, not error handling**, and only for idempotent reads
  with a low hedge rate. They cost real load; budget them.
- **Retry is usually wrong inside a band.** If the deadline is spent, a retry cannot fit.
  Retry belongs at the connection level for genuine connect failures, not for slow responses.
- **Circuit breakers protect the _whole_ request path**, not the failing dependency. An open
  breaker on the cross-encoder must produce the fused ordering, not an error.
- **Connection pools and DNS.** Both engines behind AWS-managed endpoints will re-resolve;
  a client that caches DNS forever will pin itself to a replaced node.

### B5 — Inference

- **Batch the cross-encoder call.** 100 pairs in one request, not 100 requests. Triton dynamic
  batching gets this right by default; a naive loop does not, and the difference is the
  difference between shipping and not.
- **Check the deadline before dispatch, not after.** A rerank started with no budget left is
  pure waste, and it is the single easiest degradation win.
- **Cold starts are real.** SageMaker autoscaling that scales to zero will occasionally serve
  a request behind a model load. Provisioned minimum capacity, or accept the tail.
- **Cache the embedding, key on the _stripped_ text.** The logical `rewrite` default sends
  constraint-stripped text to the vector legs; that text repeats far more often than raw
  queries do, so the asymmetric rewrite is quietly also a cache-hit-rate optimisation.
- **Encoder and index version together.** A query embedded with v2 against an index built
  with v1 returns confident nonsense — no error, plausible-looking results. Stamp the version
  in the collection name and refuse the mismatch loudly.
- **The LLM in the query path is a different animal.** Bedrock QU is gated to Full, cached,
  and has a hard fallback to the rules path. Non-determinism in the request path also breaks
  the deterministic replay the logical `offlineeval` default demands — which is the real
  reason it is Full-only, not the cost.

### B6 — Caching

- **Do not cache the search response at CloudFront.** It is personalised, experiment-stamped
  and freshness-sensitive. Cache static assets and autocomplete; nothing else.
- **Three distinct caches, three TTLs.** Query understanding (deterministic, long TTL);
  embeddings (versioned by encoder, long TTL, invalidated by version); results (short TTL,
  and only if the response is genuinely anonymous).
- **The experiment variant is part of every cache key** that can affect ranking, or the
  experiment measures the cache.
- **A cache hit changes the shape of the request**, removing a whole hop. That is worth
  showing in the trace, and it is why `px-embed-cache` is a Recommended-level component
  rather than an implementation detail.

### B7 — Correctness of the merged ranking

- **Fusion needs ranks, not scores** — which is what makes RRF the right default across two
  engines with incomparable score scales. This is the single physical argument for the
  logical "RRF with k ≈ 60, set it and move on" default, and it deserves saying plainly.
- **A missing leg changes every rank below it.** When degradation drops a leg, the fused
  ordering is not "the same list minus some documents" — it is a different ranking. Disclose
  it in the response (the `assembly` default already asks for exactly this).
- **Engine-side LTR is unavailable by construction.** OpenSearch's LTR plugin can only rerank
  what OpenSearch retrieved. Half the candidates came from Qdrant. Blended retrieval forces
  LTR in-process — a direct consequence of D1 that nobody discovers until they try.
- **Late-interaction rescore runs inside retrieval, before fusion.** The logical cascade puts
  it after. Both are defensible; only one is buildable here, and the ordering difference is
  worth a `departure` note on both cards.

### B8 — Security, tenancy and operations

- Both engines in private subnets; Qdrant Cloud Hybrid keeps the data plane in the customer
  VPC, which is the reason to choose it over fully-managed. No public endpoint on either.
- Authorization predicates are compiled by the same predicate compiler as user constraints
  and are **non-relaxable** — `px-recovery` must not be able to loosen them. Mark them in the
  AST; assert it in a test. This is where a bounded-relaxation ladder becomes a security
  control rather than a relevance feature.
- Query strings are user text and reach two query DSLs. The logical `understand` note already
  says to treat extracted structure as untrusted and validate against an allow-list; here it
  is the difference between a filter and an injection.
- Trace id propagated to both engines, so a slow request can be attributed to a leg rather
  than guessed at.
- Blue-green on the index alias, not on the cluster. Rollback is an alias swap.

---

## Appendix C — The logical defaults, priced

D6 says the physical view assumes every logical default was taken. This is that reading,
end to end — it doubles as the reviewer's checklist that the physical view is faithful.

| Logical component | Default taken                                                              | What it costs or saves physically                                                                                                                                           |
| ----------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q-text`          | String + explicit UI filters, kept separate                                | Two fields on the request body; the compiler marks UI-sourced predicates non-relaxable.                                                                                     |
| `q-image`         | Extra retrieval leg, text still drives constraints                         | A presigned S3 upload round trip _before_ the search, plus a fourth Qdrant prefetch branch — not a different pipeline.                                                      |
| `normalise`       | Normalise first; moderate, stemming left to the index                      | Normalisation is the cache key, so it must be deterministic and versioned. Stemming stays in the OpenSearch analyser, which is what keeps query/index parity checkable.     |
| `understand`      | Rules + small classifier; split constraints by confidence                  | An ONNX model in-process, no hop. Confidence becomes a field on each predicate node and the input to the relaxation order.                                                  |
| `rewrite`         | Asymmetric: full text lexically, stripped to the vector legs               | Two query strings on the wire and two cache keys — and a materially better embedding-cache hit rate.                                                                        |
| `expand`          | None by default; PRF first, into the lexical and sparse legs               | Full-level, route-gated, and it means a _second_ OpenSearch round trip when it fires. That cost is why it is off by default.                                                |
| `session`         | Inherit only on a refinement                                               | A DynamoDB read plus a cache, gated on a classifier output — not on every request.                                                                                          |
| `confidence`      | Heuristic by type, then calibrate                                          | No component of its own; a field on the predicate AST.                                                                                                                      |
| `prefilter`       | Pre-filter always; native filtered ANN, benchmarked                        | Qdrant payload indexes are mandatory, and the benchmark is a CI job, not a one-off.                                                                                         |
| `sufficiency`     | After union, before pruning; per-leg calibrated floors                     | In-process; the floors live in AppConfig so they can be tuned without a deploy.                                                                                             |
| `relax`           | Domain-specific ladder, 2 passes hard-capped                               | The loop returns to the _predicate compiler_, and the cap is what bounds the worst-case request.                                                                            |
| `zerofallback`    | Back into retrieval                                                        | A second full retrieval band inside one request — the reason the deadline controller must reserve budget for it rather than spend everything on the first pass.             |
| `routing`         | Static rules on intent class                                               | A config object in AppConfig, not a model. Rollback is a config push.                                                                                                       |
| `lexical`         | Per-field weighting, title-heavy                                           | Field boosts in the `multi_match`; changing them is a query change, not a reindex.                                                                                          |
| `sparse`          | Alongside dense, only if the mismatch tail is large                        | Rides the same `_msearch` — the marginal network cost is zero, which is why it is on OpenSearch and not its own service.                                                    |
| `textvec`         | Whole document; recall@100 ≥ 0.95 vs exact                                 | One dense named vector per point at Core, and `ef` tuned until the target holds _under filters_.                                                                            |
| `textimagevec`    | Max rollup; down-weighted to start                                         | Rollup happens in the orchestrator, because the branch returns image points and fusion needs documents. The down-weight is an RRF branch weight in config.                  |
| `imageimagevec`   | Extra leg; max rollup                                                      | Third named vector, same points, same filter. Gated on an image being supplied.                                                                                             |
| `multivec`        | Structural chunks with overlap; max rollup plus a small multi-hit bonus    | A separate point population with a parent id — the one place the "one point per document" model breaks, and the reason the collection's point count is not the corpus size. |
| `union`           | Keep all per-leg ranks, scores and evidence                                | The candidate object carries per-branch provenance to the end, which is what makes a degraded response explainable.                                                         |
| `prune`           | Top-N per leg plus conservative floors                                     | In-process, and it is what sizes the cross-encoder batch.                                                                                                                   |
| `fusion`          | RRF, k = 60, do not tune                                                   | **No score normalisation anywhere.** Ranks only across the engine boundary — the largest single simplification in the physical design.                                      |
| `fusionpolicy`    | Rules first, learned later                                                 | An AppConfig object at Recommended; a trained model at Full.                                                                                                                |
| `semrerank`       | Skip it if the cross-encoder affords full depth                            | Present at Full, deployment-gated, off. The default is the absence of a component.                                                                                          |
| `latererank`      | Compressed stored representations                                          | Collapses into Qdrant as a multivector rescore. No service, no GPU, and an ordering departure.                                                                              |
| `crossenc`        | Depth 100; title + most informative field, truncated                       | Sizes the endpoint: one batch of 100 pairs inside the deadline, with truncation bounding the sequence length.                                                               |
| `vlmrerank`       | Top 10–20, gated on visual intent                                          | Bedrock, tiny depth, hard deadline, and the first thing degradation drops.                                                                                                  |
| `ltr`             | After the neural rerankers; relevance and soft signals only                | In-process ONNX. Hard rules stay outside the model — which is also what keeps them auditable.                                                                               |
| `business`        | Hard rules outside the model, soft preferences as features                 | Last pass before assembly, and `applied_rules[]` describes the delivered order.                                                                                             |
| `freshness`       | Decay multiplier per intent class                                          | In-process, and it needs a trustworthy timestamp — which is a hydration concern, not an index one.                                                                          |
| `personalisation` | Adjust a few slots, reserve unpersonalised ones                            | One DynamoDB read; the reserved slots also bound the blast radius of a bad profile.                                                                                         |
| `diversity`       | Per-attribute capping                                                      | In-process, needs the attribute at ranking time, so hydration cannot be the only place it appears.                                                                          |
| `dedup`           | Hash at union; SimHash or embedding threshold here                         | The near-duplicate key must be computed at index time and carried on the candidate — a physical requirement the logical view does not surface.                              |
| `assembly`        | Results plus relaxation, degradation and route state                       | The disclosure fields, and a paging token that pins the ranking snapshot.                                                                                                   |
| `degradation`     | Fixed order: optional verification, rerank depth, slowest nonessential leg | Encodes directly into the deadline controller, and the order lives in AppConfig so it can be tested.                                                                        |
| `behavioural`     | From day one                                                               | Firehose is a **Recommended**-level component, not a Full one.                                                                                                              |
| `experiment`      | Unit = user; variant logged with the impression                            | An in-process hash, and the variant joins every ranking-affecting cache key.                                                                                                |
| `offlineeval`     | Labelled set first, debiased behaviour later; deterministic replay         | Athena over S3 — and the determinism requirement is what keeps the LLM out of the Core and Recommended request path.                                                        |

---

## Appendix D — Suggested checkpoints

The plan is long; these are the four places to stop and look at the screen rather than the code.

1. **After Phase 1** — the diagnostics diff is clean and the existing three tabs are
   unchanged. Nothing else matters yet.
2. **After Phase 3, P3 rows only, no panel prose** — does the layout hold with two parallel
   groups and a stage group? This is the cheapest possible moment to discover it does not.
3. **After Phase 4, looking at Physical Core** — 13 nodes, six hops. If Core is not
   immediately legible to someone who has not read this plan, the merges are wrong, not the
   CSS.
4. **After Phase 5** — step through the trace at Core and at Full. If the twelve-to-seven
   contraction does not tell the story on its own, the trace copy needs work before anything
   else does.
