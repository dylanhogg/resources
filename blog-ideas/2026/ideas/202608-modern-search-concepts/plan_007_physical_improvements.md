# Plan 007 — Physical architecture corrections

Target file: `search-query-pipeline-diagram-tool.html` (single file, zero build).
Companion reading: `search-query-pipeline-diagram-tool-architecture.md` (point-in-time,
25 Aug 2026 — the region table has drifted since plan 006 added the physical level, but the
**layer model, the validator-wall contract and the recipes are still accurate**).

**Status:** **Not started — no phase has been begun** (as at 26 Aug 2026). The target file
stands at `f8d8d29` with plan 006 phases 1–4 landed; nothing below has been applied. Update
this line as each phase lands, in the style of plan 006 (`Phase N done (<sha>, <date>)`).

### Phase summary

| Phase | What it does | Fixture | Depends on | Status |
| --- | --- | --- | --- | --- |
| **1 — Registry groundwork** | Installs vocabulary only: `writes` relation kind, `updates` restored to one meaning, `confidence` gate kind, an `"optional stage"` config condition, the `configured` marker. No node moves, no edge added. | Re-emit (retags move edges between kinds, count unchanged) | — | Not started |
| **2 — Split candidate generation** | Splits `px-fuse` into `px-union` + `px-fuse` and puts `px-sufficiency` between them, so the spine reads `engines → union → sufficiency → fuse` as its own card claims. Count-neutral on hops and units; +1 component. | Re-emit — biggest churn | 1 | Not started |
| **3 — The missing request-path inputs** | The user-visible fix: adds `px-qu → px-bedrock-qu`, gives the encoder band an input, and adds `PX_REL_BY_LEVEL` for level-scoped relations. Two of four orphans; two held. | Re-emit | 2 (order only) | Not started |
| **4 — Gate reassignment** | `px-bedrock-qu` becomes a `confidence` gate; `px-personal` gains a config gate plus conditional chip. No structural churn, but edge kinds are asserted. | Re-emit | 1.3, 1.4 | Not started |
| **5 — Coverage notes** | Adds a derived "drawn to N of M" line to the three partial control-plane relations (`px-otel observes`, `px-deadline steers`, `px-appconfig steers`), computed from a scope predicate so it cannot drift. | No change | 1.5 | Not started |
| **6 — Cross-view fidelity** | Level-filters the REALISES list, records departures for the two retargets, and applies/documents the four proposed logical-side changes (L1 required, L2 applied, L3 documented, L4 deferred). | Logical baseline re-emitted for L2 | 2, 3, 4 | Not started |
| **7 — Invariants** | Three validators that would have caught this whole class of defect — every request-path stage has an input, a departure is owed both ways, coverage notes must describe something — each throwing when it becomes unnecessary. | No change | everything | Not started |
| **8 — Verification & documentation** | All three templates × both planes, orphan sweep, correspondence sweep, narrow screens, dark theme, then the architecture doc and `TODO.md`. | No change | 1–7 | Not started |

Two items are **held** pending a decision and are deliberately not scheduled above: H1
(`px-expand` and the pseudo-relevance-feedback loop) and H2 (`ridesCall` draws nothing).
They are the two entries `PX_UNREACHED` is expected to hold at the end of Phase 7.

## What this plan fixes

The sanity check found the physical diagram renders correctly — 9 wires at Core, 66 at Full,
all with markers and ARIA labels — but that several request-path stages have **no inbound
edge at all**, so the wires that exist do not tell a complete story. Alongside that sit a
handful of direction errors, gate-copy contradictions and one ordering contradiction where a
component's card and its position on the spine disagree.

The unifying principle for every change below:

> **A component's card and its wires must say the same thing.**
> Every defect fixed here is a place where a record asserts an input, an owner, a
> position or a cadence that the diagram does not draw — or draws backwards.

### Decisions taken (from review)

| #   | Question                               | Decision                                                                                                                            |
| --- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `px-qu` → `px-bedrock-qu`              | **Complementary.** `px-qu` runs first and always; the LLM is a confidence-gated enhancement whose output overrides selected fields. |
| 2   | Encoder inputs / cache direction       | **Implementer's call** — see Phase 3, "The cache-aside reading".                                                                    |
| 3   | `px-sufficiency` vs `px-fuse` ordering | **Split** `px-fuse` into `px-union` + `px-fuse`, sufficiency between them.                                                          |
| 4   | `px-expand` PRF loop                   | **HELD** — see "Held items" below.                                                                                                  |
| 5   | `ridesCall` visual treatment           | **HELD** — see "Held items" below.                                                                                                  |
| 6   | Control-plane completeness             | **Keep the curated subsets**, add a derived "drawn to N of M" note.                                                                 |
| 7   | Logical-side changes                   | **Propose them** — Phase 6.                                                                                                         |

### Working rules

- **The baseline fixture is the wall, and it throws on the first reload after every
  structural change. That is the point.** Every phase that adds, removes or retargets an
  edge ends with `dependencyDiagnostics.emitBaseline("physical")`, a _read_ of the emitted
  fixture, and a paste back into `PHYSICAL_BASELINE`. Emitting without reading defeats the
  fixture — a human confirming every edge once is the whole value.
- **Never assert against the SVG.** Several logical edges collapse into one drawn path at
  group boundaries. Assert against `Model.dependencies(...)` / the fixture.
- **Derive, do not duplicate.** Where this plan adds a registry, it adds a _predicate_ or a
  _marker_ and computes the number, so the second hand-maintained list that would drift
  never exists. This is the file's existing ethos (`RSERVING`, `REALISED_BY`, `record.hop`,
  `record.handles` are all derived) and every new structure below follows it.
- **A validator that can go stale is worse than none.** Each new invariant added in Phase 7
  also throws when it becomes _unnecessary_ (e.g. a coverage note on a relation that is now
  complete), so the note cannot outlive the condition it describes.
- Run the page after every phase. A validator throw kills the script and the page renders as
  a dead shell — check the browser console first when it looks empty.

### Serving the file locally

A dev server is already configured in `.claude/launch.json` (`diagram`, port 8731). Use the
preview tooling rather than `python3 -m http.server` directly.

---

## Phase 1 — Registry groundwork

_No node moves and no edge is added yet. This phase installs the vocabulary the later phases
need, so that the structural churn happens once._

### 1.1 New relation kind: `writes`

**Problem.** `serves` is defined as _"a logical index or store queried or hydrated by the
target component"_ — a **read**. Two places use it, or its neighbour `updates`, for a
**write**:

- `PX_SERVING["pxd-s3"]` includes `px-events`. `px-events` does not read the lake; it
  **writes** impressions to it via Firehose. (`px-lake` and `px-train` reads are correct.)
- `PXI_REL["pxi-indexer"] = {updates:["pxd-os-index","pxd-qdrant-collection"]}` uses
  `updates`, whose description is profile-specific: _"logged events update durable profile
  state for future requests"_. That copy renders on the write-path wires.

**Fix.** Add one kind to `RELATION_TYPES` and both problems resolve together.

```js
writes:{
  fwd:"Writes to", rev:"Written by",
  description:"A component writing durable state into a store, off the request path.",
  aria:"Durable write",
  color:"var(--wire-data)", dash:"1 5", linecap:"round", marker:"data",
  opacity:.9, width:1.5,
  route:"source", componentRelation:false,
  legend:{visible:true,label:"writes — durable state"}
}
```

`route:"source"` reuses the existing store-edge routing; the arrow runs component → store,
the opposite direction from `serves`. Note the marker/colour are shared with `serves` on
purpose — same plane, opposite direction — but `opacity` and the legend label separate them.

Then:

- Move `px-events` out of `PX_SERVING["pxd-s3"]` into a new `PX_WRITES` registry:
  ```js
  const PX_WRITES = {
    "px-events": ["pxd-s3"],
    "pxi-indexer": ["pxd-os-index", "pxd-qdrant-collection"],
  };
  ```
- Drop the `updates` entry from `PXI_REL`; `PXI_REL` retains only the two `feeds` edges
  (`pxi-source → pxi-log → pxi-indexer`).
- Register `PX_WRITES` on the physical model in `defModel` (`writes:PX_WRITES`) and teach
  the dependency generator to emit it exactly the way it emits `serving` — same plane
  gating, same "a store is active only while a consumer is" rule, with the endpoints
  swapped.
- Move the four `pxi-indexer|updates|*` entries in `PX_RELATION_DETAILS` to the `writes`
  key form (`"pxi-indexer|writes|pxd-os-index"`, etc.). Their `label`/`payload`/`cadence`
  text is already correct and needs no rewording.
- Add a `PX_RELATION_DETAILS` entry for `"px-events|writes|pxd-s3"`:
  `label:"impression records"`, `payload:"the impressions the stream buffered, as partitioned Parquet"`,
  `cadence:"Firehose delivery, off the request path"`.

### 1.2 Restore `updates` to its one meaning

With the indexer edges gone, `updates` again describes only `px-events → px-personal` (and
its logical twin `behavioural → personalisation`). Leave the description as it is — it is
now accurate.

Add a `RELATION_DETAILS` clarification to the existing `"px-events|updates|px-personal"`
entry noting that the write physically lands in the profile store:

> `payload:"clicks, saves and enquiries, keyed on the subject — written into the profile store px-personal reads"`

**Why not retarget it to `pxd-ddb`?** Because the component-level edge is the one that tells
the story (_behaviour changes ranking_), it mirrors the logical level exactly, and replacing
it with `px-events writes pxd-ddb` would make the loop visible only when the data plane is
switched on. The detail line carries the physical truth without costing the narrative.
_Flagged as an alternative if the write-path overlay later becomes the primary reading._

### 1.3 New gate kind: `confidence`

**Problem.** `px-bedrock-qu` is gated `config` / `"optional pass"`, which renders:

> _"A cascade tier kept only when the corpus or available training data justifies it — a
> deployment choice rather than a per-query decision. Skipped, it passes candidates through."_

Four fields on the same card say the opposite: purpose _"Gated on the classifier's own
uncertainty"_; decision default _"Only when the classifier is uncertain"_; dial
_"Invocation gate: classifier confidence below the floor"_; contract
`in:"normalised query + the classifier's low-confidence output"`. It is also not a cascade
tier and it passes no candidates.

**Fix.** Add to `GATE_KINDS`:

```js
confidence: Object.freeze({
  decidedAt: "request",
  color: "var(--gate-request)",
  conditions: Object.freeze({
    "classifier uncertain":
      "Runs only when the in-process classifier's confidence in its own extraction falls below the floor.",
  }),
});
```

`gatesDecidedBy()` throws if any kind lacks a declared owner in **every** level's owner map,
so both must be extended:

- `LOGICAL_GATE_OWNER` → `confidence:"understand"` (query understanding is what produces the
  confidence, logically)
- `PX_GATE_OWNER` → `confidence:"px-qu"`

`requireComponent()` validates both ids exist in their own registry — `understand` and
`px-qu` both do. No new colour token is needed: `decidedAt:"request"` reuses
`var(--gate-request)`, and the legend picks the kind up automatically via
`gateConditionsFor()`.

The logical level declares an owner without yet using the kind. That is correct and
intentional — the owner map is the level's claim about _who could decide_, and Phase 6
records why the logical view does not currently gate on it.

### 1.4 Second `config` condition: `"optional stage"`

`px-personal` is also gated `config` / `"optional pass"` and inherits the same wrong copy —
it is in Final ranking, not the cascade, and passes no candidates through. Add a sibling
condition under the existing `config` kind:

```js
"optional stage":"A stage kept only when the product calls for it — a deployment choice rather than a per-query decision."
```

`"optional pass"` stays exactly as it is and continues to serve the genuine cascade tiers
(`px-semrerank`, `px-ltr`, and their logical twins).

### 1.5 The `configured` marker

Phase 5 needs a drift-proof denominator for `px-appconfig`'s coverage. Add
`configured:true` to every component whose behaviour is set by remote configuration — a real
structural property, one word per record, sitting alongside the existing `control`,
`decision`, `io`, `parallel` markers:

| Component        | Evidence in its own record                     |
| ---------------- | ---------------------------------------------- |
| `px-router`      | `"static rules from AppConfig"`                |
| `px-deadline`    | budget dials read from configuration           |
| `px-fuse`        | `k`, per-leg top-N and leg weights             |
| `px-final`       | pass order and business rules                  |
| `px-sufficiency` | `unit:"floors loaded from AppConfig"`          |
| `px-recovery`    | `sizing:["relaxation ladder in AppConfig", …]` |

The first four are drawn today; the last two are the omissions the sanity check found. Do
**not** mark `px-appconfig` or `px-experiment` — they mention AppConfig because they _are_
the control plane, not because they read it.

### Exit criteria — Phase 1

- Page loads; `dependencyDiagnostics.modelValidation` reports the new kinds in use.
- Legend shows `writes` alongside `serves`, and the `confidence` gate condition appears in
  the gate legend.
- `emitBaseline("physical")` re-read and pasted; `PHYSICAL_COUNTS.planeEdges` re-checked
  (the `updates` → `writes` retag moves edges between kinds without changing the count; the
  `px-events` move from `serves` to `writes` likewise).

---

## Phase 2 — Split candidate generation, and put the sufficiency check where its card says it is

### The defect

`px-sufficiency` states its own position in four places:

- `sub:"per-leg floors, **after union and before pruning**"`
- `contract.in:"unioned candidates with per-leg ranks"`
- `dials:[…{n:"Position", v:"after union, before pruning — pruning would hide the shortfall"}]`
- `purpose:"Applies calibrated per-leg floors to the **unioned** set"`

The diagram places it **before** `px-fuse`, and `px-fuse` realises `union` + `prune` +
`fusion`. So as drawn, the floors are applied to two unmerged per-engine responses — which
is exactly the failure its own gotcha warns about (_"A dense leg always returns k results and
a filtered lexical leg may return three"_).

The logical order is `union → sufficiency → prune → fusion`. Merging three logical stages
into one physical box forced sufficiency to one side or the other, and it went to the wrong
side.

### The fix

Split into two in-process components. **The split is count-neutral**: both are
`runtime:"inproc"` with `substrate:{on:[]}`, so `hopMetric` and `unitMetric` are unmoved and
only the component count and `cxMetric` shift.

**`px-union` — "Candidate union"** · `group:"Candidate generation"` · `tier:"core"` ·
`intro:1` · `runtime:"inproc"` · `cx:2` · `realises:["union"]`

- `sub:"one join, on the canonical id"`
- `purpose:` unions the two engines' rank lists on a shared document id, retaining per-leg
  ranks and provenance.
- `contract:{in:"two engines' rank lists", out:"one candidate set with per-leg ranks and provenance"}`
- Inherits from today's `px-fuse`: the decision **"What joins the two rank lists?"**, the
  gotcha **"Document identity diverges between the engines"**, and the failure mode
  **"Id mismatch between the engines"**. All three are about the _join_, not the _fusion_.
- `caps:["identity","observe"]`
- `realises.length === 1`, so no departure is required — but write one anyway
  (see 7.2: this plan makes many-to-one departures mandatory, and `union` is now realised by
  exactly one component, so no departure is owed. Skip it.)

**`px-fuse` — rename to "Prune and fuse"** · `realises:["prune","fusion"]`

- `sub:` unchanged (`"RRF, k = 60 · ranks only, never scores"`)
- `contract.in:` → `"the unioned candidate set with per-leg ranks and evidence"`
- Retains: the decision **"Ranks or scores across the engine boundary?"**, the gotchas
  **"A missing leg treated as a shorter list"** and **"Fusion treated as a failure when only
  one leg answers"**, the failure mode **"One leg dominates every query"**, the optimisation,
  the scaling axis, the alternatives.
- `caps:["degrade","replay","observe"]`
- **Rewrite the departure.** It currently opens _"Three logical stages, one physical
  component"_. It becomes two, and the new text should say what the split bought:

  > Two logical stages, one physical component: two passes over one array in one process.
  > They merge because pruning and fusion are the same traversal; the union does not, because
  > the sufficiency check has to see the joined set before anything is pruned from it.
  > Fusion runs here rather than in either engine, because reciprocal rank fusion needs both
  > rank lists in one process and neither engine can see the other's.

  Keep the intro-mismatch licence intact — `prune` is logical level 2 while `px-fuse` is
  level 1, and the departure is what permits that.

### Template rows

`PX_TPL` — insert `px-union` on its own spine row, immediately after the engines row:

| Level | Before                                                                                            | After                                                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1     | engines → `{c:["px-fuse"]}`                                                                       | engines → `{c:["px-union"]}` → `{c:["px-fuse"]}`                                                                       |
| 2     | engines → `{c:["px-sufficiency"], nextLabel:"sufficient"}` → `{l:["px-recovery"], c:["px-fuse"]}` | engines → `{c:["px-union"]}` → `{c:["px-sufficiency"], nextLabel:"sufficient"}` → `{l:["px-recovery"], c:["px-fuse"]}` |
| 3     | same as 2                                                                                         | same as 2                                                                                                              |

At Core the two boxes sit adjacent with no sufficiency between them. That is honest — union
and fusion _are_ distinct passes — and it means the reader who later switches to Recommended
sees the check appear in the gap rather than the spine re-ordering under them.

### Relationship retargets

| Relation                  | Today     | After          | Why                                                                                                                                                                                                                               |
| ------------------------- | --------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `px-rescore-late feeds …` | `px-fuse` | **`px-union`** | Its payload is _"the dense candidate list already reordered by MaxSim"_ — a rank list, and rank lists enter at the join. Its `RELATION_DETAILS` cadence (_"before fusion rather than after it"_) becomes literally true as drawn. |
| `px-appconfig steers …`   | `px-fuse` | `px-fuse`      | `k`, leg weights and per-leg top-N are fusion dials.                                                                                                                                                                              |
| `px-experiment steers …`  | `px-fuse` | `px-fuse`      | The variant changes fusion behaviour.                                                                                                                                                                                             |
| `px-train trains …`       | `px-fuse` | `px-fuse`      | Fusion weights.                                                                                                                                                                                                                   |

Also update the `PX_RELATION_DETAILS` key `"px-rescore-late|feeds|px-fuse"` →
`"px-rescore-late|feeds|px-union"`.

### Exit criteria — Phase 2

- Spine at Recommended and Full reads `engines → union → sufficiency → fuse`, matching both
  `px-sufficiency`'s dial and the logical `union → sufficiency → prune → fusion`.
- `px-union` carries the identity gotcha; `px-fuse` carries the degradation gotchas.
- `hopMetric` and `unitMetric` unchanged at every level; component count +1.
- Baseline re-emitted, **read**, pasted. Expect churn in all three levels' `flow` lists and
  in `steers`/`trains`/`feeds` where `px-fuse` appears.

---

## Phase 3 — The missing request-path inputs

This is the phase that fixes the user-visible complaint. Two of the four orphans found are
addressed here; two are held (see "Held items").

### 3.1 `px-qu → px-bedrock-qu`

The classifier that gates the LLM pass **lives inside `px-qu`** —
`sub:"normalise, then rules + a small classifier"`, `unit:"rules + ONNX Runtime classifier"`.
Not the orchestrator. Add:

```js
"px-qu": {feeds:["px-bedrock-qu"]},
```

with a `PX_RELATION_DETAILS` entry that carries the decision:

```js
"px-qu|feeds|px-bedrock-qu":Object.freeze({
  label:"low-confidence extraction",
  payload:"the normalised query and the classifier's own uncertain output",
  cadence:"only where confidence falls below the floor"
})
```

`px-bedrock-qu` remains level 3, so the edge only exists where both endpoints are active —
the dependency generator already filters on the active set, so no level guard is needed.

The drawer will now render a **`Fed by`** section on `px-bedrock-qu`, which it has never
had. That alone answers _"where does it get the query text?"_.

**Complementary, not alternative** — record this explicitly, since it is the question the
diagram was failing to answer. Add to `px-bedrock-qu.notes`:

> `px-qu` runs first and always. This is an enhancement layered on its output, not a
> replacement for it: on a hit it overrides selected extracted fields, and on any failure or
> timeout the rules output stands unchanged. That is why both components feed the rewriter —
> the second one only sometimes, and only some of the fields.

### 3.2 The encoder band's input

At Full, `PX_TPL[3]` row 8 is `{l:["px-embed-cache","px-embed-text","px-embed-image","px-embed-sparse"]}`
— an aside row with an **empty centre**, so it generates no spine edge and nothing reaches
it. Meanwhile `px-rewrite`'s own worked example reads:

> _"Sends the full six-facet string to OpenSearch and 'quiet waterfront cabin with a wood
> fireplace' to both encoders."_

And its optimisation: _"Key the embedding cache on the stripped text, not the raw query."_
And its `contract.out` is `{ lexical_text, vector_text, expansion_terms? }`. The producer is
named three times and drawn zero.

**The cache-aside reading (decision for question 2).** The truthful picture is:

```
px-rewrite ──vector_text──▶ px-embed-cache ──on a miss──▶ px-embed-* ──vector──▶ engines
                                  └──────────on a hit───────────────────────────▶ engines
```

**Draw the first two legs, not the third.** Add `px-rewrite feeds px-embed-cache`, keep the
existing `px-embed-cache feeds px-embed-*` and **relabel it** as the miss path. Do _not_ add
`px-embed-cache → px-qdrant-search` / `→ px-os-search` hit-path wires.

Rationale: the cache-aside lookup genuinely _is_ the first thing that happens, so
`cache → encoder` is the miss path rather than a wrong arrow — it was only ever missing its
own input. Adding the hit path would put two more long wires across the busiest band at Full
to state something `px-embed-cache`'s departure already says in words (_"on a hit, the
request has one fewer hop, and the shape of the request changes"_). A `RELATION_DETAILS`
label carries it at zero geometric cost:

```js
"px-embed-cache|feeds|px-embed-text":Object.freeze({
  label:"on a miss",
  payload:"the stripped text and the encoder version that keyed the lookup",
  cadence:"only when the lookup misses — on a hit the encoder is skipped and the vector goes straight to the engine"
})
```

…and the same for `px-embed-image` and `px-embed-sparse`.

_Follow-up if the band later reads cleanly: draw the hit path explicitly. Noted, not done._

### 3.3 Level-scoped relations — `PX_REL_BY_LEVEL`

At Core there is no `px-rewrite` and no `px-embed-cache`; the encoder's input comes from
`px-qu`. `PX_REL` is level-independent, so declaring both sources would draw two edges at
Recommended and Full.

Add one small, well-named registry — the model's first level-scoped relation table:

```js
/* Relations whose *source* moves as a level inserts a stage in front of it.
   The encoder band is fed by the rewriter once one exists, and by query
   understanding before that. Declaring both in PX_REL would draw both at the
   levels where both are active, so the source is named per template and merged
   over PX_REL when the level's dependencies are resolved. */
const PX_REL_BY_LEVEL = {
  1: { "px-qu": { feeds: ["px-embed-text"] } },
  2: { "px-rewrite": { feeds: ["px-embed-cache"] } },
  3: { "px-rewrite": { feeds: ["px-embed-cache"] } },
};
```

Wire it into `defModel` as `relationsByLevel:PX_REL_BY_LEVEL` and merge it in
`Model.dependencies(model, templateId, …)` — a shallow per-kind concat over `model.relations`,
before the active-set filter. Keep the merge in **one** place so `RREL`, the drawer's
dependency sections, the hover card and the wires all follow, exactly as they do for
`PX_REL` today.

Validate in Phase 7 that every level key names a real template and every endpoint is active
at that level.

_Note for the reviewer:_ the alternative was to promote `px-rewrite` to Core, but logical
`rewrite` is level 2 and that would either violate the level-regression invariant or need a
departure written to excuse it — a worse trade than one small registry.

### Exit criteria — Phase 3

- At Core: `px-embed-text` has an inbound edge from `px-qu`.
- At Recommended and Full: `px-embed-cache` has an inbound edge from `px-rewrite`.
- At Full: `px-bedrock-qu` has an inbound edge from `px-qu` and its drawer shows `Fed by`.
- The only remaining request-path components with no data input are the two held items and
  the legitimate I/O and control nodes.
- Baseline re-emitted, read, pasted.

---

## Phase 4 — Gate reassignment

Uses the vocabulary installed in Phase 1. **No structural churn — `PX_GATES` changes gate
_kinds_, and edge kinds are part of the asserted inventory, so the baseline still moves.**

### 4.1 `px-bedrock-qu`

```js
"px-bedrock-qu": defPhysGate(3,"confidence","classifier uncertain"),
```

The drawer's Gate block will now read _"CLASSIFIER UNCERTAIN · request decision · decided by
Query understanding"_, agreeing with the purpose line, the design decision, the dial and the
wire contract instead of contradicting all four.

### 4.2 `px-personal`

```js
"px-personal": defPhysGate(3,"config","optional stage"),
```

Then add the per-request half, which the `config` gate deliberately does not cover
(`GATE_KINDS`' own comment: _"a gate says whether a stage may run at all, `conditional` says
it runs on only some of the requests that reach it"_):

```js
conditional:true,
conditionalOn:"Runs only where the request carries a known subject — an anonymous search skips it entirely.",
```

The validator at the gate/conditional block already enforces the `conditional`/`conditionalOn`
pair in both directions. Two small follow-ons:

- `PHYSICAL.hover` is `["runtimePills","gate","purpose","crossView","budget","gotcha"]` — add
  `"conditional"` after `"gate"` so the hover card carries it, matching the logical hover.
- `px-personal` is the first physical component to use `conditional`, so check the node chip
  renders (the chip site is shared between levels, so it should — verify, do not assume).

`px-session` already steers `px-personal`, which is what tells the request whether a subject
is known. The two now agree.

### Exit criteria — Phase 4

- `px-bedrock-qu`'s drawer no longer describes a cascade tier passing candidates through.
- `px-personal` shows a config gate _and_ a conditional chip.
- Gate legend lists four conditions across three kinds.
- Baseline re-emitted, read, pasted.

---

## Phase 5 — Control-plane coverage: "drawn to N of M"

**Decision taken: keep the curated subsets.** They are the right editorial call — the
`px-otel` banner already argues it well (_"a diagram buried under its own tracing is exactly
the failure this kind is meant to describe rather than commit"_). What is missing is that the
reader is never told the subset is a subset, so the omissions read as oversights.

Three relations are partial, and each one's card is contradicted by its wires:

| Relation              | Drawn at Full | Should be scoped to                     | Omissions worth naming                                                                                                                                                                |
| --------------------- | ------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `px-otel observes`    | 5             | every network hop (14)                  | `px-bedrock-qu` and `px-vlm` — the two most expensive hops at Full — while `px-crossenc` is included. The list is the Recommended set, unchanged.                                     |
| `px-deadline steers`  | 6             | every remote call under the budget (13) | `px-bedrock-qu`, whose budget says `timeout:"the deadline, not the model's default"`; `px-embed-image`; `px-embed-sparse`; `px-hydrate`; `px-personal` (`"a small slice, skippable"`) |
| `px-appconfig steers` | 4             | every `configured` component (6)        | `px-sufficiency`, `px-recovery` — both name AppConfig in their own records                                                                                                            |

### The mechanism

Declare the **scope predicate**, not the count. The count is then computed per active
template, so it moves as the reader climbs the levels and can never drift.

```js
/* A control relation drawn to a chosen few rather than to everything it
   touches. The denominator is a predicate over the registry, not a list — a
   second hand-maintained list is a second thing to disagree with the first.
   Rendered per template, so the note reads "5 of 6" at Core and "5 of 14" at
   Full without either number being written down. */
const PX_COVERAGE = Object.freeze({
  "px-otel|observes": Object.freeze({
    scope: (record) => record.hop && !record.origin,
    of: "network hops in the request path",
    why: "Tracing instruments every hop in a real deployment. Drawn to the ones that dominate the critical band, because a diagram buried under its own instrumentation is the failure this kind describes rather than one it should commit.",
  }),
  "px-deadline|steers": Object.freeze({
    scope: (record) => record.hop && !!record.budget,
    of: "remote calls under the request budget",
    why: "The deadline is enforced on every call the orchestrator makes. Drawn to the calls whose expiry changes what the user sees; the rest carry their budget on the card.",
  }),
  "px-appconfig|steers": Object.freeze({
    scope: (record) => !!record.configured,
    of: "components configured at runtime",
    why: "Every configured component reads this at start and on change. Drawn to the ones whose configuration changes a ranking.",
  }),
});
```

Register as `coverage:PX_COVERAGE` in `defModel`, and render in `relationSection()` — the
existing renderer that already prints `relation.description` as `<p class="relnote">`. Append
a sibling line for the `fwd` direction only:

> **Drawn to 5 of 14** network hops in the request path. Tracing instruments every hop in a
> real deployment…

Compute the denominator from components **active at the current template and not switched
off**, reusing whatever `enabled` accessor the sidebar metrics already read — do not
introduce a second notion of "active".

### Exit criteria — Phase 5

- Opening `px-otel`, `px-deadline` or `px-appconfig` shows the coverage line, and the numbers
  change between Core, Recommended and Full.
- Nothing else in the drawer changes; no new wires.
- No baseline change — this phase adds no edges.

---

## Phase 6 — Cross-view fidelity, in both directions

### 6.1 Level-filter the REALISES list

At **Core**, clicking "Qdrant query" lists four logical legs under REALISES — Text vector,
Text-to-image, Image-to-image, Multi-vector — three of which do not exist in the logical Core
view. Same pattern for `px-os-search` → `sparse`, `px-predicate` → `confidence`,
`px-fuse` → `prune`, `pxd-os-index` → `ds-sparse`, `pxd-qdrant-collection` →
`ds-image-ann`/`ds-passage-ann`.

This is the honest side-effect of merging — the engine really can do all four at Core, you
just have not asked it to yet — but as rendered it reads as the two levels disagreeing.

Fix in `DRAWER_SECTIONS.crossView`: mark links whose target is not active at the current
template, mirroring the existing `" · off"` treatment in `relationSection()`. Suggested
copy: `" · not yet at this level"`. Keep them clickable — following one to the logical view
is exactly the round trip the section exists for.

### 6.2 Departures for the two retargets

Neither `px-experiment` nor `px-events` carries a departure today, because
`realises.length === 1` does not require one. Both retarget a logical relationship:

- Logical `experiment steers fusionpolicy`; physical `px-experiment steers px-fuse`. The
  physical collapses an indirection — `px-appconfig` realises `fusionpolicy` and holds the
  weights, but the variant is applied at fusion time. Add a departure saying so.
- Logical `behavioural trains fusionpolicy, ltr`; physical `px-train trains px-crossenc,
px-ltr, px-fuse`. Physical adds cross-encoder training and retargets fusionpolicy to
  `px-fuse` for the same reason. Add a departure on `px-train` (it has one — extend it).

### 6.3 Proposed logical-side changes

Listed as proposals; each is a small, self-contained edit to the logical registry, and each
one moves the two levels closer.

| #   | Change                                                       | Rationale                                                                                                                                                                                                                                                           | Cost                                                                                    |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| L1  | `LOGICAL_GATE_OWNER` gains `confidence:"understand"`         | **Required** by Phase 1.3 — `gatesDecidedBy()` throws if any kind lacks an owner at every level.                                                                                                                                                                    | One line. No behaviour change; the logical level declares an owner it does not yet use. |
| L2  | Add `crossenc` to `behavioural.trains`                       | The physical `px-train trains px-crossenc` asserts a training loop the logical view does not have. If the physical is right — and it is; a cross-encoder is fine-tuned on judgements — the logical view is missing an edge.                                         | `REL` + `DEPENDENCY_BASELINE[3].trains`.                                                |
| L3  | Leave `personalisation` ungated logically                    | The physical `config` gate is a deployment fact ("do we ship personalisation"), not a retrieval decision. The logical view is right to omit it; Phase 4's `conditional` marker carries the per-request half.                                                        | None — document the asymmetry in `px-personal`'s notes.                                 |
| L4  | Consider `experiment steers fusion` alongside `fusionpolicy` | Would make the logical view agree with the physical retarget in 6.2 rather than needing a departure to excuse it. **Recommend deferring** — the indirection through the policy component is defensible at the logical level, and a departure is the cheaper record. | Deferred.                                                                               |

Note what Phase 2 does _for free_ here: splitting `px-fuse` **reduces** divergence. The
physical departure shrinks from three logical stages to two, and the sufficiency check's
position now matches the logical order exactly. That is the correspondence check paying for
itself.

### Exit criteria — Phase 6

- At Core, `px-qdrant-search`'s REALISES list marks three of four links as not yet reached.
- `px-experiment` and `px-train` carry departures naming their retargets.
- L1 applied (required); L2 applied; L3 documented; L4 recorded as deferred.
- Logical baseline re-emitted for L2.

---

## Phase 7 — The invariants that would have caught all of this

Three new invariants. Each is written so it also throws when it becomes **unnecessary**, so a
note or an exemption cannot outlive the condition it describes.

### 7.1 Every request-path stage has an input

The invariant that catches the entire class of defect this plan fixes.

```js
/* A stage the request passes through must be reachable. A component with no
   inbound data-carrying edge is a box the reader cannot get to, and every one
   of them found so far was a real omission rather than a real orphan. Control
   nodes steer without being fed and inputs arrive from outside, so both are
   out of scope by their own markers rather than by a list. */
```

- **Scope**: for each template, every active component that is not `io`, not `control`, and
  not a data source.
- **Rule**: at least one inbound edge of kind `flow | gated | branch | loop | feeds`.
- **Exemptions**: a named `PX_UNREACHED` register, each entry carrying a _reason string_,
  and the invariant throws if an id in the register **now has** an inbound edge — so an
  exemption cannot go stale.

The two held items become declared exemptions rather than silent holes:

```js
/* Known-incomplete, deliberately. Each entry is a decision not yet taken, not
   a component that genuinely has no input — see plan 007, "Held items". */
const PX_UNREACHED = Object.freeze({
  "px-expand":
    "Fed by the rewriter and by the first lexical pass. The second is a loop back from px-os-search and changes the shape of the retrieval band, so it is held pending a decision.",
  "px-rescore-late":
    "Rides the Qdrant call (`ridesCall`), which today draws nothing. Held pending a decision on how a ridden call should be shown.",
});
```

### 7.2 A departure is owed in both directions

`defPhysical` requires a departure when `realises.length !== 1`, so a **many-physical-to-one-logical**
split escapes it entirely. Today `px-qu`+`px-bedrock-qu` → `understand` and
`px-hydrate`+`px-assemble` → `assembly` both happen to have departures; nothing enforces it.

Add to `PHYSICAL_INVARIANTS` — it needs the whole inventory, so it belongs there rather than
in `defPhysical`:

> For every logical id realised by more than one physical record, **every** one of those
> records must carry a departure.

### 7.3 Coverage notes must describe something

For each `PX_COVERAGE` entry: every drawn target must be inside the scope predicate (catches
a typo), and the drawn set must be a **strict** subset of the scope at the fullest template.
If someone later completes the relation, the note becomes a lie and the validator says so.

Also validate `PX_REL_BY_LEVEL`: every key is a real template id, every endpoint exists, and
every endpoint is active at that level.

### Exit criteria — Phase 7

- Temporarily delete one Phase 3 edge and confirm 7.1 throws naming the component; restore.
- Temporarily complete `px-appconfig steers` and confirm 7.3 throws; restore.
- `PX_UNREACHED` contains exactly the two held items.

---

## Phase 8 — Verification and documentation

1. **All three templates, both planes.** For each of Core / Recommended / Full, with data
   stores and write path both off, both on, and each alone: page loads, no console error,
   `dependencyDiagnostics.validation` clean.
2. **Orphan sweep.** Re-run the data-edge orphan analysis over `PHYSICAL_BASELINE`. Expected
   remaining "no data in" at Full: `px-image` (I/O), the control nodes, and the two
   `PX_UNREACHED` entries. Nothing else.
3. **Correspondence sweep.** Re-run the per-level coverage check: every active logical
   component realised by an active physical one at every level, `q-text` excepted.
4. **Narrow screens.** Below 700px the tool renders textual dependency chips instead of
   wires (`mobileTransition`, `mobileRecovery`, `renderMobileSelection`). Changes to
   relationship display usually need doing twice — check the `writes` kind, the coverage
   note and the new `Fed by` sections all appear there.
5. **Dark theme.** No new colour tokens are introduced (`writes` reuses `--wire-data`,
   `confidence` reuses `--gate-request`) — confirm rather than assume.
6. **Documentation.**
   - Update `search-query-pipeline-diagram-tool-architecture.md`: the five sources of truth
     become seven (`PX_WRITES`, `PX_REL_BY_LEVEL`), the relation-kind count moves from nine
     to ten, and the validator wall gains three entries. The region table needs regenerating
     for the physical level regardless — it predates plan 006.
   - `TODO.md`: strike _"Review physical architecture 'Query understanding' and 'LLM query
     understanding' components"_ (Phase 3.1/4.1 answers it — they are **complementary**, and
     the classifier is inside `px-qu`). Note that _"What are the 'Write paths'"_ is partly
     answered by the `writes` kind. Leave _"Add link from Text query to Query expansion?"_
     open and cross-reference the held item below — the physical reading says the answer is
     a loop from the lexical leg, not a link from the query.

---

## Held items — noted, paused for a decision

### H1 · `px-expand` and the pseudo-relevance-feedback loop

**What is drawn:** one edge, `px-expand → px-os-search`. Nothing feeds `px-expand`.

**Why that is wrong.** `px-expand`'s contract is
`in:"lexical text + **the first pass's top documents**"`. It is pseudo-relevance feedback: it
runs a lexical query, takes terms from the top of _that result set_, and re-queries. So it
needs two inputs — the text (from `px-rewrite`, whose `contract.out` already carries
`expansion_terms?`) and the first pass's results (from `px-os-search`). The second one is a
**loop back up the diagram**.

**Why it matters more than a missing arrow.** `px-expand`'s own gotcha is the whole point of
drawing it:

> _"Unlike every other optional stage here, this one cannot be run concurrently — the second
> query depends on the first result set. It is the only component in the request path that
> lengthens the critical path rather than widening it."_

That claim is invisible as drawn. With only `expand → os-search`, it reads as a term source
that costs nothing, which is precisely the misreading the card argues against.

**The decision needed:** draw `px-os-search → px-expand → px-os-search` as a loop (honest,
makes the serial second round trip visible, but puts a loop in the middle of the retrieval
band alongside the recovery loop that is already there), or leave it and add a note to the
card. Related and worth deciding together — your `TODO.md` asks _"Review Query Expansion —
what does it feed exactly? Anything missing? Is it really required?"_, and `px-expand`'s own
alternatives answer part of it:

> _"Learned sparse retrieval instead… almost always the better answer — the sparse encoder
> does term expansion in the model, inside the same round trip, **which is exactly why
> px-embed-sparse and this component are alternatives rather than companions**."_

If that is right, `px-expand` and `px-embed-sparse` should probably be mutually exclusive at
Full, which is a third thing the diagram does not currently say.

### H2 · `ridesCall` draws nothing

**What is drawn:** `px-rescore-late` sits beside the engines row with one outbound feed into
candidate generation, and no visible relationship to `px-qdrant-search`.

**Why that is wrong.** The record carries `ridesCall:"px-qdrant-search"`, but grep shows
`ridesCall` is used in exactly one place — computing `record.hop`:

```js
record.hop = RUNTIME[runtime].hop && !ridesCall;
```

It draws no edge, no containment, no chip. So the node whose entire argument is _"it is a
clause in the query, not a service… no service, no GPU, and no extra round trip"_ appears as
a free-standing stage that emits candidates from nowhere. Its own note says the combination
is the interesting part:

> _"This is a remote component that costs no round trip, because it rides the retrieval call
> rather than following it. That combination — engine runtime, no hop — is why the hop count
> cannot simply be read off the runtime."_

The same applies to `px-waf`, which rides `px-edge` — though `px-waf` is less affected,
because it also `steers` `px-edge` and so at least has a wire.

**The decision needed:** a containment box (drawn like the existing `stage-group` boundaries
around the rerank cascade), a tenth relation kind (`rides`), or a chip on the card plus a
note. A boundary box is probably cheapest and reads best — it says "inside" rather than
"connected to", which is the actual relationship — but it competes with the parallel-group
box already drawn around the two engines.

---

## Phase dependency order

```
Phase 1  Registry groundwork ────────┬──▶ Phase 2  union / fuse split
  writes · confidence · configured   │      (structural, biggest fixture churn)
                                     │              │
                                     ├──▶ Phase 4  Gate reassignment
                                     │      (needs 1.3 and 1.4)
                                     │              │
                                     └──▶ Phase 5  Coverage notes
                                            (needs 1.5)          │
                                                                 │
Phase 3  Missing inputs  ◀── independent of 1, after 2 ──────────┤
  (fixture churn; do after 2 so the fixture is re-emitted once)  │
                                                                 ▼
                                            Phase 6  Cross-view fidelity
                                                     (needs 2, 3, 4)
                                                          │
                                                          ▼
                                            Phase 7  Invariants
                                              (needs everything; the
                                               exemption register records
                                               H1 and H2)
                                                          │
                                                          ▼
                                            Phase 8  Verification & docs
```

Phases 2 and 3 both move the baseline fixture. Run them back to back and re-emit once at the
end of each rather than mid-phase — the fixture is hand-confirmed, and confirming it twice
for one structural change is the cost the design deliberately accepts, not one to pay twice
over.
