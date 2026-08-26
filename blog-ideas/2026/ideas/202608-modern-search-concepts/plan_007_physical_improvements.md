# Plan 007 — Physical architecture corrections

Target file: `search-query-pipeline-diagram-tool.html` (single file, zero build).
Companion reading: `search-query-pipeline-diagram-tool-architecture.md` (point-in-time,
25 Aug 2026 — the region table has drifted since plan 006 added the physical level, but the
**layer model, the validator-wall contract and the recipes are still accurate**).

**Status:** **Phases 1–2 done** (`7d093e6`, 26 Aug 2026), **Phase 3 done** (`1b46328`,
26 Aug 2026), **Phase 4 done** (uncommitted, 26 Aug 2026) — the vocabulary is installed
(`writes`, `confidence`, `"optional stage"`, `configured`), candidate generation is split so
the spine reads `engines → union → sufficiency → fuse`, every request-path stage but the two
held items now has a visible input, and the two contradicted gates now carry the kinds their
own cards claim. The fixture was re-emitted, diffed and read at each phase; hops, units and
`cx` did not move in Phase 3, and Phase 4 moved no edge at all (**F23**). Phases 5–8
outstanding. Written on top of `f8d8d29` (plan 006 phases 1–4). Update this line as each
phase lands, in the style of plan 006 (`Phase N done (<sha>, <date>)`).

Where implementing a phase turns up something this plan did not predict, the finding is
recorded in that phase under **"What Phase N found"** — the later phases are written against
those, not against the original guess.

### Phase summary

| Phase                                   | What it does                                                                                                                                                                                                                   | Fixture                                                        | Depends on     | Status      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | -------------- | ----------- |
| **1 — Registry groundwork**             | Installs vocabulary only: `writes` relation kind, `updates` restored to one meaning, `confidence` gate kind, an `"optional stage"` config condition, the `configured` marker. No node moves, no edge added.                    | Re-emitted and read — 8 edges retagged, 158 total unchanged    | —              | **Done**    |
| **2 — Split candidate generation**      | Splits `px-fuse` into `px-union` + `px-fuse` and puts `px-sufficiency` between them, so the spine reads `engines → union → sufficiency → fuse` as its own card claims. Count-neutral on hops and units; +1 component.          | Re-emitted and read — 7 out, 10 in, 158 → 161                  | 1              | **Done**    |
| **3 — The missing request-path inputs** | The user-visible fix: adds `px-qu → px-bedrock-qu`, gives the encoder band an input, and adds `PX_REL_BY_LEVEL` for level-scoped relations. Two of four orphans; two held.                                                     | Re-emitted and compared byte-for-byte — 0 out, 4 in, 161 → 165 | 2 (order only) | **Done**    |
| **4 — Gate reassignment**               | `px-bedrock-qu` becomes a `confidence` gate; `px-personal` gains a config gate plus conditional chip. The gate-authority check widens to admit a decider that feeds what it gates, and the physical legend stops naming one owner. | Re-emitted and compared character-for-character — no change    | 1.3, 1.4       | **Done**    |
| **5 — Coverage notes**                  | Adds a derived "drawn to N of M" line to the three partial control-plane relations (`px-otel observes`, `px-deadline steers`, `px-appconfig steers`), computed from a scope predicate so it cannot drift.                      | No change                                                      | 1.5            | Not started |
| **6 — Cross-view fidelity**             | Level-filters the REALISES list, records departures for the two retargets, and applies/documents the four proposed logical-side changes (L1 required, L2 applied, L3 documented, L4 deferred).                                 | Logical baseline re-emitted for L2                             | 2, 3, 4        | Not started |
| **7 — Invariants**                      | Three validators that would have caught this whole class of defect — every request-path stage has an input, a departure is owed both ways, coverage notes must describe something — each throwing when it becomes unnecessary. | No change                                                      | everything     | Not started |
| **8 — Verification & documentation**    | All three templates × both planes, orphan sweep, correspondence sweep, narrow screens, dark theme, then the architecture doc and `TODO.md`.                                                                                    | No change                                                      | 1–7            | Not started |

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
- **Re-emitting the physical fixture takes one extra step.** `defModel` validates before it
  registers a model, so while the old fixture is throwing, `PHYSICAL` never reaches
  `PIPELINE_MODELS` and `dependencyDiagnostics.emitBaseline("physical")` cannot be called at
  all. Set `baseline:null` in the `PHYSICAL` spec, reload, emit, paste the fixture in and
  restore `baseline:PHYSICAL_BASELINE` in the same edit. **Diff the emission against the
  outgoing fixture rather than re-reading 158 pairs** — flatten both to `level|kind|from|to|label`
  and list what was added and removed. That is the human confirmation the fixture is for, and
  it is the only form in which a two-edge change is actually reviewable. **Then prove the
  paste rather than trusting the transcription**: have the page fetch its own source, slice
  out the `PHYSICAL_BASELINE` literal and compare it character-for-character with
  `emitBaseline("physical")`. The set-based `assertDependencySet` on reload cannot see a
  re-ordering, and this can (F17).
- **`dependencyDiagnostics.modelValidation` is the _logical_ summary.** The physical one is
  `PIPELINE_MODELS.physical.validation.relationships`; its `dependencies` key existing at all
  is proof the fixture matched.
- **Never assert against the SVG.** Several logical edges collapse into one drawn path at
  group boundaries. Assert against `Model.dependencies(...)` / the fixture. When you do want
  to read the drawn wires — to check an ARIA label or a detail payload — note that they are
  drawn on a layout pass the freshly-switched tab has not run yet: `[data-edge-kind]` comes
  back empty until a `resize` or a scroll. Dispatch one and wait before querying.
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

- Move `px-events` out of `PX_SERVING["pxd-s3"]` into a new `PX_WRITES` registry — and see
  F4 below: this move is what pulls three view-layer helpers with it:
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

### What Phase 1 found

**F1 · `writes` must be `abstraction:["physical"]`.** The snippet above omits it. Without it
the kind is shared vocabulary, and the relation legend — unlike the gate legend — is _not_
filtered by use: it lists every kind the level may draw. The logical tab would gain a
phantom "writes — durable state" key for a wire it never draws. The argument is exactly
`observes`': a write path is a property of a deployment. The logical vocabulary therefore
stays at nine kinds and the physical goes from ten to eleven, which is what Phase 8's
documentation note needs to say.

**F2 · There are two `pxi-indexer|updates|*` details, not four.** `pxd-os-index` and
`pxd-qdrant-collection`. Both retagged verbatim, as planned.

**F3 · One `writes` registry, two kinds of writer.** `px-events` is a component and
`pxi-indexer` is a source, so the generator admits the writer end by whichever rule owns it
(`active()` for a stage, `shown()` for a plane node) and the store end always by `shown()`.
That is why `PX_WRITES` is one registry rather than a component one and a plane one.

**F4 · A registry move is never only a wire change — this is the finding Phases 3 and 5
should read first.** Three view-layer helpers read `PX_SERVING` and `PXI_REL` directly, for
_placement and liveness_ rather than for wires, and all three broke silently:

| Helper                  | Read          | What breaks without the fix                                                   |
| ----------------------- | ------------- | ----------------------------------------------------------------------------- |
| `writersOf`             | `RPLANE_REL`  | the whole `pxi-source → pxi-log → pxi-indexer` chain vanishes from the canvas |
| `sourceIdsForConsumers` | `SERVING_REL` | `pxd-s3` slides one row down, off the `px-events` row it belongs beside       |
| `sourceLive`            | `PLANE_REL`   | a store greys out while something is still writing into it                    |
| `dataSourceEl` · drawer | `PLANE_REL`   | `pxi-indexer`'s "Writes …" line and the source drawer's "Derives" chip empty  |

Fixed by giving the view two derived notions instead of letting it read registries:
`sourceUsers` (readers **plus** component writers — what places, lights and highlights a
store) beside `sourceConsumers` (readers only — what a store's card counts and names), and
`derives` / `derivedFrom` (the write path in both directions, across both registries).
`sourceIdsForConsumers` is now `sourceIdsForUsers`. Net visual result: **nothing moved**,
which was the phase's contract.

**F5 · The gate legend is already usage-filtered, so declaring `confidence` changes nothing
until Phase 4 assigns it.** `renderGateLegend` drops any condition with no gated stage in the
active template, so neither view shows the new kind yet and the logical level's owner
declaration is inert exactly as intended. `gateConditionsFor` does carry it — both models
now list six conditions across four kinds.

**F6 · The plane counts did not move.** `PHYSICAL_COUNTS` untouched: 5/7/8 nodes and 7/10/16
edges, and the fixture stayed at 158 pairs. The diff was exactly eight edges — the indexer's
two writes retagged at all three levels, and `pxd-s3 serves px-events` reversed into
`px-events writes pxd-s3` at Recommended and Full.

**F7 · Observed, not fixed.** The relation legend lists every kind a level may draw, used or
not — at physical Core `updates` now has no edge behind it, where before it had the indexer's
two. This is pre-existing and level-wide (logical Core lists nine keys and draws one kind),
so it is an editorial call about the legend rather than a defect this phase introduced.
Noted for Phase 7 to accept or Phase 8 to record.

### Exit criteria — Phase 1

- ~~Page loads; `dependencyDiagnostics.modelValidation` reports the new kinds in use.~~
  Done — `PIPELINE_MODELS.physical.validation.relationships` reports
  `relationTypes:11 · writeRelations:3 · planeRelations:2 · reverseIndexes:4`.
- ~~Legend shows `writes` alongside `serves`~~ — done, on the physical tab only (F1), and
  the three wires carry their ARIA labels and relation details. The `confidence` condition
  correctly does **not** appear yet (F5).
- ~~`emitBaseline("physical")` re-read and pasted; `PHYSICAL_COUNTS.planeEdges` re-checked~~
  — done, and the count was unchanged at every level as predicted (F6).
- Also verified: the four affected drawers (`px-events` "Writes to", `pxd-s3` and
  `pxd-os-index` "Written by", `pxi-indexer` "Fed by" + "Writes to"), node placement
  unchanged at all three levels × both planes, and the narrow-screen chips
  ("Writes to: …" / "Written by: …") and kind list.

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

### What Phase 2 found

**F8 · The retarget table over-predicted the churn.** `steers`, `trains` and the other
`feeds` did not move at all: `px-appconfig`, `px-experiment` and `px-train` all keep
`px-fuse` as their target, exactly as the table said they should, so the only registry line
that changed is `px-rescore-late`. The fixture diff is therefore **7 removed, 10 added, 158 →
161** — three new spine hops, one per level, and the rescore retarget. Nothing else in the
diagram noticed the split.

**F9 · The spine rewires itself; the retrieval band's fan-in follows.** Flow edges are
generated from adjacent template rows, so inserting one row was the whole structural change.
The parallel group's egress re-aimed itself with no edit — the drawn edge is now
`parallel group: OpenSearch query, Qdrant query → Candidate union` — and `PX_PARALLEL_GROUPS`
needed nothing. Same for `PX_BRANCHES`, `PX_REQUIRES` and `PX_PHASES` (phases band by group,
and `px-union` is already in "Candidate generation").

**F10 · The split is count-neutral on hops and units, but not on `cx`.** As predicted:
hops 5/8/14 and units 6/15/17 are unmoved, because `px-union` is `runtime:"inproc"` with
`substrate:{on:[]}`. Operational burden goes 32/70/108 → **34/72/110** with `cx:2`, and the
bands (`Lean` / `Moderate` / `Heavy`) do not change, so only the number after the band label
moves. `model.maxCx` is the sum over all components and is the sidebar bar's denominator, so
it moves with it — the Full bar stays exactly full and the other two shift by under a
percent.

**F11 · `px-union` takes no `configured` marker, so Phase 5's denominator is unchanged.**
The join has no dial: `k`, the per-leg top-N and the leg weights are all still `px-fuse`'s,
and AppConfig still steers `px-fuse`. This is the obvious thing to get wrong — the "drawn to
N of M" numbers Phase 5 computes are exactly what they were before the split.

**F12 · Phase 6.1's `px-fuse → prune` example survives, and 6.3's note is now true.**
`px-fuse` still realises `prune` (logical level 2) while sitting at physical Core, so the
REALISES list still needs the "not yet at this level" marking. What changed is the size of
the divergence the departure has to excuse: three logical stages became two, and the
sufficiency check's position now matches the logical `union → sufficiency → prune → fusion`
exactly. `px-union` realises `union` one-to-one at the same level and owes no departure.

**F13 · `TODO.md` line 7 is now stale, and Phase 8 should retire it.** It asks whether the
_logical_ sufficiency check should move after pruning and fusion, and gives as its reason
that the physical view groups "Union, prune and fuse". That grouping no longer exists: this
phase moved the physical to match the logical order rather than the other way round, so the
question it poses has been answered from the other end.

### Exit criteria — Phase 2

- ~~Spine at Recommended and Full reads `engines → union → sufficiency → fuse`~~ — done, and
  verified by node order in the DOM at all three levels. At Core it reads
  `engines → union → fuse`, the two boxes adjacent as intended.
- ~~`px-union` carries the identity gotcha; `px-fuse` carries the degradation gotchas.~~
  Done, along with the matching decision, failure mode and worked example on each. `px-fuse`
  was renamed "Prune and fuse", its departure rewritten, and its `contract.in`, `purpose`,
  `substrate.unit` and `caps` brought in line.
- ~~`hopMetric` and `unitMetric` unchanged at every level; component count +1.~~ Done —
  5/8/14 hops, 6/15/17 units, 38 → 39 components (F10).
- ~~Baseline re-emitted, **read**, pasted.~~ Done, and diffed: the churn was smaller than
  predicted (F8).
- Also verified: both models validate on reload, the physical drawn edge count at Full went
  66 → 67, `px-rescore-late feeds px-union` carries its detail payload (its cadence "before
  fusion rather than after it" is now literally true as drawn), the `px-union` and `px-fuse`
  drawers render their new sections, `REALISED_BY` reads `union → px-union` and
  `prune, fusion → px-fuse`, and the narrow-screen chip on `px-union` reads
  "Flows to Prune and fuse".

---

## Phase 3 — The missing request-path inputs

This is the phase that fixes the user-visible complaint. Two of the four orphans found are
addressed here; two are held (see "Held items").

> **Read Phase 1's F4 first.** `PX_REL_BY_LEVEL` changes which relations exist per level,
> and the view places, lights and highlights nodes from the relationship registries rather
> than from the dependency set. Grep every read of the registry being changed before
> assuming a level-scoped relation is only a wire change — F4 is three helpers that broke
> silently on a move half this size.

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

### What Phase 3 found

**F14 · The registry move cost the view almost nothing — because F4 had already taken the
registries away from it.** F4 predicted this would be the expensive part and it was not, for
a reason worth recording rather than for luck: the three helpers F4 fixed read `PX_SERVING`
and `PXI_REL`, not `PX_REL`. `PX_REL`/`RREL` turned out to have exactly **three** readers —
the drawer's relation sections and the hover card's two `steers` rows — and all three became
level-aware by swapping a destructured constant for a two-line accessor:

```js
const relationsOf = (id) => Model.relations(model, slice.tpl)[id] || {};
const reverseRelationsOf = (id) =>
  Model.reverseRelations(model, slice.tpl)[id] || {};
```

The generalisable form of F4 is therefore not "grep before you move a registry" but **the
view should not be able to name a registry at all**. Where a lookup can be level-dependent,
give it a `Model` query that takes the template id, and the question stops being possible to
get wrong.

**F15 · The base reverse index was deleted, not shadowed.** Keeping `model.reverseRelations`
beside a per-level one would have left two indexes with the same meaning and different
truths — the exact bug F4 describes, installed deliberately. So the base is gone:
`Model.reverseRelations(model,templateId)` is now the only reverse index for components, and
`validateRelationshipModel` checks it once per level rather than once. Three incidental
cleanups fell out of the same edit: `reverseRelationIndex()` now serves the component, plane
and per-level indexes (three hand-rolled loops down to one), `requireKinds()` replaces two
copies of the kind check, and `expectedReverse()` replaces two copies of the expectation
builder. The mechanism is **net smaller** than what it replaced, excluding comments.

**F16 · Four edges in, none out.** 161 → 165, and the diff is exactly the intended set:

| Level | Edge                                | Why                                  |
| ----- | ----------------------------------- | ------------------------------------ |
| 1     | `feeds px-qu → px-embed-text`       | 3.3 — Core has no rewriter           |
| 2     | `feeds px-rewrite → px-embed-cache` | 3.2 — the cache-aside lookup's input |
| 3     | `feeds px-qu → px-bedrock-qu`       | 3.1                                  |
| 3     | `feeds px-rewrite → px-embed-cache` | 3.2                                  |

Plane counts, hops (5/8/14), units (6/15/17) and `cx` (34/72/110) are all untouched: this
phase adds wires, not components. Drawn paths at Full go 67 → 69, both additions primary.
`componentRelations` 52 → 53, `levelRelations` 0 → 3, `annotatedRelations` 11 → 17.

**F17 · The fixture was verified by comparing text, not by transcribing it.** Phase 2's
hand-transcription plus a Python cross-check was replaced by something both cheaper and
stronger: splice the four pairs into the outgoing fixture programmatically, then have the
page fetch its own source, slice out the `PHYSICAL_BASELINE` literal and compare it to
`emitBaseline("physical")` character-for-character. It returned `identical:true`, which
`assertDependencySet` alone could not have told us — that assertion is set-based and blind to
ordering, so a fixture that no longer matches what the generator prints would still pass and
would produce a spurious diff on the _next_ phase's re-emit. Promoted to a Working rule.

**F18 · Three encoder-miss annotations wanted three different payloads.** The plan's snippet
proposed one label repeated for `px-embed-text`, `px-embed-image` and `px-embed-sparse`, but
the image tower is keyed on an image reference rather than on stripped text, and the sparse
encoder is handed the text it will expand. A one-line `encoderMiss(key)` factory carries the
shared label and cadence and takes the input per encoder, so the three differ where they
should and cannot drift where they should not.

**F19 · The level-scoped drawer works, and the unscoped half of the drawer is pre-existing.**
Confirmed on the page: at Core `px-qu`'s drawer lists **Text embedding** under _Feeds_; at
Recommended it does not, and `px-embed-cache` shows _Fed by · Per-leg query rewriting_
instead. But at Core that same drawer also lists **LLM query understanding**, a Full-only
component — the drawer is deliberately not template-scoped, and only the hover card is (via
`scoped:true`, whose comment says the drawer "renders these in full"). Level-scoping made
this visible without causing it. Same family as F7: an editorial call for Phase 8, not a
defect this phase introduced.

**F20 · Under Phase 7.1's own rule, the unreached set is now exactly the two held items.**
Sweeping every level for an active component with no inbound `flow | gated | branch | loop |
feeds`, excluding `io` and `control`, returns **`px-expand` and `px-rescore-late`** and
nothing else. That is precisely the `PX_UNREACHED` sketch in 7.1, now true rather than
hoped-for. Two details for whoever writes it:

- The rule must exclude `serves`, as 7.1 already says. With `serves` counted,
  `px-rescore-late` looks reached — `pxd-qdrant-collection` serves it — and H2's whole point
  is that a store answering a component is not the same as something feeding it.
- `px-session` shows up in the raw sweep at Full and is `control:true`, so the marker-based
  exemption covers it. No id needs listing that 7.1 does not already list.

### Exit criteria — Phase 3

- ~~At Core: `px-embed-text` has an inbound edge from `px-qu`.~~ Done, and it carries a
  detail — _"the normalised string, with no per-leg rewriting yet — Core has no rewriter to
  strip constraints out of it"_ — which is the sentence that makes the level-scoping legible
  rather than merely correct.
- ~~At Recommended and Full: `px-embed-cache` has an inbound edge from `px-rewrite`.~~ Done;
  a 309px curve down two rows on the aside side, crossing nothing.
- ~~At Full: `px-bedrock-qu` has an inbound edge from `px-qu` and its drawer shows `Fed by`.~~
  Done. The drawer's section order is `… Notes · In the worked example · Feeds · Fed by ·
Appears in`, and the complementary-not-alternative note renders above it.
- ~~The only remaining request-path components with no data input are the two held items and
  the legitimate I/O and control nodes.~~ Done — see **F20** for the exact sweep and its one
  subtlety.
- ~~Baseline re-emitted, read, pasted.~~ Done, and compared byte-for-byte against a fresh
  emission (**F17**).

---

## Phase 4 — Gate reassignment

Uses the vocabulary installed in Phase 1 — `confidence`/`"classifier uncertain"` and
`config`/`"optional stage"` are both defined and owned already, so this phase is two lines in
`PX_GATES` plus the fixture. ~~**No structural churn — `PX_GATES` changes gate
_kinds_, and edge kinds are part of the asserted inventory, so the baseline still moves.**~~
**Wrong on the second half: the baseline does not move** (**F23**). An edge is `gated` because
its target is gated at all, not because of _which_ kind gates it, so a re-kinding is invisible
to `Model.dependencies`. What it does move is a validator and a legend sentence — see **F21**
and **F24**.

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
  It does, at both widths — see **F25**, which is also where the drawer's silence on
  `conditional` is recorded.

`px-session` already steers `px-personal`, which is what tells the request whether a subject
is known. The two now agree.

### What Phase 4 found

**F21 · The gate-decider validator demanded `steers`, and `confidence` is the first gate whose
decider _feeds_ what it gates.** Assigning the kind threw on the first reload —
`Gate decider does not steer its gated component: px-bedrock-qu`. `PX_GATE_OWNER.confidence`
was installed in Phase 1 and sat inert until this phase used it (F5), so the check had never
been exercised against a decider that is a request-path stage rather than a control node. The
rule it encodes is right — _"a named decider must actually reach what it gates, or the card
claims an authority the relationship model does not grant it"_ — but it was written when every
decider was `px-router` or `routing`, and it mistook the usual form of that authority for the
only one.

**F22 · The obvious fix draws a wire on top of another wire, and the page said so.** Declaring
`px-qu steers px-bedrock-qu` satisfies the old check and was tried first. Both kinds route
`side`, at curve `.4` and `.45`, between two adjacent nodes — and the drawn paths came back
with the same endpoints on the same horizontal line, where the control points make no visible
difference at all:

```
steers  M370.5,652.68 C296.7,652.68 280.3,652.68 206.5,652.68
feeds   M370.5,652.68 C304.9,652.68 272.1,652.68 206.5,652.68
```

Two wires, one visible, the hidden one asserting a relationship the fixture would then carry
for ever. Rejected on the measurement rather than on taste. So the check was widened instead
of the model padded:

```js
const GATE_AUTHORITY_KINDS = Object.freeze(["steers","feeds"]);
```

with the error reworded to _"does not reach its gated component"_. `steers` remains the usual
form — a control node bending a stage it is not otherwise connected to — and `feeds` is how an
in-band decider grants the same authority: `px-qu` does not steer the LLM pass, it hands over
the extraction it was not confident in, on exactly the condition the gate names. A decider
connected by neither still throws. **This is the contract Phase 7 should extend rather than
re-narrow.**

**F23 · The fixture did not move, and that is proved rather than assumed.** `Model.dependencies`
picks `gated` over `flow` from `Model.gate(model,to,tpl)` being truthy — it never reads the
kind or the condition — so re-kinding a gate is invisible to the dependency set, and the
`gated` pairs in the fixture carry no gate label to go stale either. Re-emitted anyway, under
Phase 3's F17 method: `baseline:null`, reload, `emitBaseline("physical")`, then have the page
fetch its own source, slice out the `PHYSICAL_BASELINE` literal and compare the two
character-for-character. **165 pairs, byte-identical, nothing added and nothing removed.** The
round trip is not _required_ for a phase that only re-kinds a gate — but running it is the only
thing that establishes that, and it costs one reload.

**F24 · The physical legend asserted the exact thing this phase falsified.**
`PHYSICAL_SIDEBAR.legendNote` opened _"Retrieval routing owns every request gate"_ — true until
`confidence` was assigned to `px-qu`, false the moment it was, and no validator watches prose.
Rewritten to state the rule instead of the one owner:

> A request gate is owned by whatever produces the fact it turns on — routing for route and
> intent, query understanding for its own confidence — while a deployment gate is a standing
> configuration choice with no runtime owner…

The **logical** `legendNote` carries the same original sentence and stays true: logical
declares `confidence:"understand"` but no logical stage gates on it (F5, and L3 in Phase 6).
The two legends now differ in this one sentence deliberately — recorded in 6.3.

**F25 · The drawer has no `conditional` section; the node chip and the hover card carry it.**
Adding `"conditional"` to `PHYSICAL.hover` works — `px-personal`'s hover card reads
_"optional stage gate — … · Conditional — Runs only where the request carries a known
subject…"_ — and the node renders both chips, `OPTIONAL STAGE · CONDITIONAL`, at 1280px and at
600px. But neither model's `panel` list has a conditional entry, so the drawer shows the gate
and stays silent about the trigger, on the logical side too (`relax`, `crossenc`). Pre-existing
and level-wide, same family as F7 and F19: an editorial call for Phase 8, not a defect this
phase introduced.

### Exit criteria — Phase 4

- ~~`px-bedrock-qu`'s drawer no longer describes a cascade tier passing candidates through.~~
  Done. It reads _"CLASSIFIER UNCERTAIN · per request · ✓ enabled — Runs only when the
  in-process classifier's confidence in its own extraction falls below the floor. Decided by
  Query understanding."_ — agreeing with the purpose line, the design decision, the dial and
  the wire contract instead of contradicting all four.
- ~~`px-personal` shows a config gate _and_ a conditional chip.~~ Done, plus the hover card;
  the drawer is silent on the conditional at both levels (**F25**).
- ~~Gate legend at Full lists **six conditions across four kinds** — route selected · visual
  intent · image query · classifier uncertain · optional pass · optional stage.~~ Done, and
  the usage counts beside them read 3 · 1 · 1 · 1 · 2 · 1, which is the nine gated stages.
  (The plan originally said "four across three", which is the count _before_ this phase: the
  legend is usage-filtered, so the two conditions Phase 1 declared appear only once this phase
  assigns them. See Phase 1 F5.)
- ~~Baseline re-emitted, read, pasted.~~ Re-emitted and compared character-for-character;
  **no paste was needed, because nothing changed** (**F23**).
- Also verified: nine level × plane combinations via `layoutSweep("physical")` — no clipped
  labels, no overlaps, node and wire counts unmoved — and both models still register on a
  fresh load.

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

`configured:true` is already on the six components Phase 1 named, documented beside
`defPhys` with the other structural markers. Phase 2's split did **not** add a seventh:
`px-union` has no dial, so the denominator here is still six (F11). Phase 4 did not touch a
marker either — a gate kind is not a dial — so all three denominators below stand as written.
What is left is the predicate and the count.

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

> **Phase 3 changed the shape of this.** The drawer now already asks the model what the
> _current level_ holds, through `relationsOf` / `reverseRelationsOf` (F14). This section is
> the same question aimed at the other model: `Model.stages(model.crossView.model, tpl)`.
> Follow that pattern rather than reaching into the other model's registries — and note F19
> while you are here, because it is this same section's problem one level along: the drawer
> lists relations to components the current template does not draw, and `crossView` is about
> to grow the vocabulary (`" · not yet at this level"`) that would fix both.

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
| L3  | Leave `personalisation` ungated logically                    | The physical `config` gate is a deployment fact ("do we ship personalisation"), not a retrieval decision. The logical view is right to omit it; Phase 4's `conditional` marker carries the per-request half, and now does — `"Runs only where the request carries a known subject — an anonymous search skips it entirely."` | None — **still owed**: document the asymmetry in `px-personal`'s notes.                  |
| L5  | Leave the logical `legendNote` sentence alone                | Phase 4 rewrote the *physical* legend because assigning `confidence` made "Retrieval routing owns every request gate" false there (F24). Logical declares `confidence:"understand"` and gates nothing on it, so the original sentence is still true at that level. | None — record the asymmetry rather than syncing the two strings.                        |
| L4  | Consider `experiment steers fusion` alongside `fusionpolicy` | Would make the logical view agree with the physical retarget in 6.2 rather than needing a departure to excuse it. **Recommend deferring** — the indirection through the policy component is defensible at the logical level, and a departure is the cheaper record. | Deferred.                                                                               |

Note what Phase 2 did _for free_ here: splitting `px-fuse` **reduced** divergence. The
physical departure shrank from three logical stages to two, and the sufficiency check's
position now matches the logical order exactly. That is the correspondence check paying for
itself. 6.1's `px-fuse → prune` example is unaffected and still needed (F12).

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

**The sweep was run at the end of Phase 3 and returns exactly these two ids** (F20). Write
the invariant against that result rather than re-deriving it, and keep `serves` out of the
kind list — with it counted, `px-rescore-late` looks reached and H2's point is lost.

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

**Do not re-narrow `GATE_AUTHORITY_KINDS`.** Phase 4 widened the gate-decider check from
`steers` to `steers | feeds` on the evidence that the `steers` wire would have been drawn
directly over the `feeds` one (F21, F22). It is the tighter-looking rule that is wrong here,
and a Phase 7 invariant that reinstates it would fail on `px-bedrock-qu`.

~~Also validate `PX_REL_BY_LEVEL`: every key is a real template id, every endpoint exists,
and every endpoint is active at that level.~~ **Done in Phase 3**, in
`validateRelationshipModel`, with a fourth rule the plan did not ask for: a level-scoped
relation may not restate one `PX_REL` already declares, because the merge would then draw the
edge twice and only the fixture would notice. The per-level reverse index is checked for
staleness at every level as well (F15).

### Exit criteria — Phase 7

- Temporarily delete one Phase 3 edge and confirm 7.1 throws naming the component; restore.
  F20's sweep still stands unchanged after Phase 4 — that phase moved no edge (F23).
- Temporarily complete `px-appconfig steers` and confirm 7.3 throws; restore.
- `PX_UNREACHED` contains exactly the two held items.

---

## Phase 8 — Verification and documentation

1. **All three templates, both planes.** For each of Core / Recommended / Full, with data
   stores and write path both off, both on, and each alone: page loads, no console error,
   `PIPELINE_MODELS.physical.validation` clean (`dependencyDiagnostics.validation` is the
   logical model's — see Working rules).
2. **Orphan sweep.** Re-run the data-edge orphan analysis over `PHYSICAL_BASELINE`. Expected
   remaining "no data in" at Full: `px-image` (I/O), the control nodes, and the two
   `PX_UNREACHED` entries. Nothing else.
3. **Correspondence sweep.** Re-run the per-level coverage check: every active logical
   component realised by an active physical one at every level, `q-text` excepted.
4. **Narrow screens.** Below 700px the tool renders textual dependency chips instead of
   wires (`mobileTransition`, `mobileRecovery`, `renderMobileSelection`). Changes to
   relationship display usually need doing twice — check the `writes` kind, the coverage
   note and the new `Fed by` sections all appear there. Phase 3 added five annotated
   relations whose chips carry a payload, and `px-bedrock-qu` gained its first `Fed by` of
   any kind — that is the one to look at first. Already checked at 600px: both node chips
   render (`CLASSIFIER UNCERTAIN`, and `OPTIONAL STAGE · CONDITIONAL`). Note that the hover
   card is `display:none` below 700px, so the conditional trigger has **no** narrow-screen
   home at all until the drawer grows one — see F25 and item 6.
5. **Dark theme.** No new colour tokens are introduced (`writes` reuses `--wire-data`,
   `confidence` reuses `--gate-request`, and the conditional line reuses `--warn` via the
   existing `.hc-condline`) — confirm rather than assume. `confidence` is now actually used,
   so the gate rail and the legend swatch can be looked at rather than reasoned about.
6. **Three editorial calls to accept or fix, all pre-existing and all surfaced by this plan.**
   F7: the relation legend lists every kind a level _may_ draw, so at physical Core `updates`
   has no edge behind it. F19: the drawer lists relations to components the current template
   does not draw, where the hover card scopes them out. F25: the drawer prints a component's
   gate but never its `conditional` trigger, at either level — the one call of the three that
   costs a reader something concrete, because below 700px the hover card is hidden and the
   trigger is then unreachable. Phase 6.1 introduces the copy that would fix F19 if you want
   it fixed; F25 is a line beside the gate note rather than a `panel` entry — `DRAWER_SECTIONS`
   has no `conditional` key, but `setGateNote()` already owns the one place a drawer prints a
   gate for both components and sources, and `HOVER_BLOCKS.conditional` already has the
   sentence; F7 needs a decision, not code.
7. **Documentation.**
   - Update `search-query-pipeline-diagram-tool-architecture.md`: the five sources of truth
     become seven (`PX_WRITES`, `PX_REL_BY_LEVEL`), and the validator wall gains three
     entries. Also record that **a relation can now be level-scoped**, which is a change to
     the layer model rather than to a registry: `model.relations` is no longer what any
     reader consults and `model.reverseRelations` no longer exists — both go through
     `Model.relations(model, tpl)` / `Model.reverseRelations(model, tpl)` (F14, F15). The
     model summary gained `levelRelations` alongside `writeRelations`. **The doc's "all nine edge kinds" is the _logical_ vocabulary and stays at
     nine** — `writes` is `abstraction:["physical"]` (Phase 1 F1), so it is the physical
     count that moves, from ten to eleven. Say which count is which; the current sentence
     does not. The model summary also gained `writeRelations` and `reverseIndexes` went from
     three to four. The region table needs regenerating
     for the physical level regardless — it predates plan 006.
   - `TODO.md`: strike _"Review physical architecture 'Query understanding' and 'LLM query
     understanding' components"_ (Phase 3.1/4.1 answers it — they are **complementary**, and
     the classifier is inside `px-qu`). Note that _"What are the 'Write paths'"_ is partly
     answered by the `writes` kind. Leave _"Add link from Text query to Query expansion?"_
     open and cross-reference the held item below — the physical reading says the answer is
     a loop from the lexical leg, not a link from the query, and after Phase 3 it is the _only_
     missing request-path input left. **Retire line 7** — _"Should
     logical Result sufficiency check come after Candidate pruning and Fusion?"_ — its whole
     premise was the physical grouping "Union, prune and fuse", which Phase 2 dissolved by
     moving the physical to match the logical order rather than the reverse (F13).

---

## Held items — noted, paused for a decision

### H1 · `px-expand` and the pseudo-relevance-feedback loop

> **Phase 3 raised this item's standing.** It is now one of only two request-path components
> with no input, and the other (H2) is a rendering question rather than a missing edge. Every
> other box on the request path acquired a visible input in Phase 3, so `px-expand` no longer
> reads as one omission among several — it reads as the exception. Deciding it is cheap now
> and gets more conspicuous with every phase that does not.

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
