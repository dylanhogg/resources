# Plan 004 — Component sequence and surface fixes

**Target:** `search-query-pipeline-diagram-tool.html` (4675 lines post-Phase 1) — sole file in scope.
**Status:** Phase 1 implemented and verified. Phases 2–4 not started.
**Source:** the four accepted findings from the 25 Aug 2026 sequence review.
**Decisions:** D1 delete `ds-token` · D2 `latererank` stays [optional]/Full · D3 two eval nodes · D4 `expand` aside on the `rewrite` row · D5 new `Evaluation` phase (**confirm before Phase 4**).

Line numbers throughout are as of the file **before any phase landed**, and
Phase 1 has since shifted them by up to +5. Phases 2 and 4 will shift them
again. Re-grep before starting each phase rather than trusting the numbers
written here.

---

## Phase and status summary

| Phase | Change                                                                                            | Fixture? | Size                                     | Risk     | Status      |
| ----- | ------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------- | -------- | ----------- |
| **1** | `dedup` before `diversity`; `expand` off the spine into a left aside feeding `lexical` + `sparse` | yes      | 15 sites, 2 templates                    | low      | **done**    |
| **2** | Remove `Late-interaction retrieval` and `ds-token` entirely                                       | yes      | 20 sites, 9 subsystems                   | **high** | not started |
| **3** | Reword the sufficiency floor as per-leg calibrated floors                                         | no       | 6 sites, prose only                      | lowest   | not started |
| **4** | Add `Experiment assignment` + `Offline evaluation` control nodes                                  | yes      | 2 new components, ~14 sites, 2 templates | medium   | not started |

**Implementation order: 1 → 3 → 2 → 4.** One commit per phase; each must leave
the tool rendering.

1. **Phase 1** first — two small independent edits that exercise the
   fixture-update loop on low-risk changes.
2. **Phase 3** next, out of numerical order: it is prose-only with zero fixture
   risk, so it lands while the tree is still clean and does not have to be
   untangled from a Phase 2 rollback.
3. **Phase 2** third — 20 sites, three throwing traps and one reference
   migration. Run it against an otherwise-quiet tree.
4. **Phase 4** last — additive, and the only phase that structurally touches
   template 2.

Phases 2 and 4 both edit `BASELINE[3]`; separating them avoids a fixture
merge conflict. Phase 3 has no fixture surface, so it can move anywhere in the
order if convenient.

**Blocking question:** D5 (grouping for the two Phase 4 nodes, §4.2) needs an
answer before Phase 4 starts. Phases 2 and 3 are unblocked; **Phase 3 is next.**

---

## 0. The constraint that governs every phase

Two validators run at module load and **throw**, blanking the page:

| Validator                      | Line | What it guards                                                                                                                                                                     |
| ------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validateDependencyBaseline()` | 2954 | Every derived edge must match `DEPENDENCY_BASELINE` (2537) exactly — missing, unexpected and duplicate edges all throw. Serving source/edge counts are asserted separately (2941). |
| `validateRelationshipModel()`  | 2955 | Gate reciprocity, serving-source orphans, relation-detail completeness, routing↔gated-reranker symmetry.                                                                           |

This is a feature, not an obstacle: **there is no such thing as a partially
applied phase here.** Either the fixture is updated in the same commit or the
tool does not render. Every phase below therefore ends with its fixture delta.

Four specific traps, each of which throws:

1. **Gate `altOf` is reciprocal** (`validateRelationshipModel`, ~2495): removing
   `late` without removing `altOf:"late"` from `latererank`'s gate throws
   `Gate alternative is not reciprocal`.
2. **Serving sources may not be orphaned** (~2534): `DATA_SOURCES["ds-token"]`
   with an empty `SERVING_REL` entry throws `Serving source has no consumers`.
   This is what forces D1 — there is no "leave it dangling" option.
3. **Relations out of `assembly` or `behavioural` require a
   `RELATION_DETAILS` entry** with `label`, `payload` _and_ `cadence` (~2521).
   Phase 4's `behavioural → offlineeval` edge cannot be added without one.
4. **Relation endpoints must coexist in at least one template** (~2492).
   Constrains which nodes the Phase 4 additions may wire to.

Flow edges are derived from `row.c` only (`logicalDependencies`, 2658) — a
component in `l:` or `r:` contributes **no spine flow**, only whatever `REL`
declares. That is the mechanism Phase 1.2 relies on.

---

## Phase 1 — Reorder the page-construction tail, and take `expand` off the spine — **DONE**

Two independent edits, no shared surface. Smallest phase; do it first to
exercise the fixture-update loop on a low-risk change.

### 1.1 Deduplication before Diversity

Dedup collapses rows out of a page that Diversity has already capped, leaving
the page short and backfilled from candidates that never saw the diversity cap.
`dedup`'s own rationale (1765) — the near-duplicate pass "only has to be right
for the page being shown" — is circular while Diversity is what decides the
page.

| Site               | Line      | Change                                                                                                                                                          |
| ------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TPL[2].rows`      | 2149–2150 | `{c:["diversity"]}, {c:["dedup"]}` → `{c:["dedup"]}, {c:["diversity"]}`                                                                                         |
| `TPL[3].rows`      | 2179–2180 | same swap                                                                                                                                                       |
| `BASELINE[2].flow` | 2567–2569 | `["freshness","diversity"], ["diversity","dedup"], ["dedup","assembly"]` → `["freshness","dedup"], ["dedup","diversity"], ["diversity","assembly"]`             |
| `BASELINE[3].flow` | 2606–2608 | `["personalisation","diversity"], ["diversity","dedup"], ["dedup","assembly"]` → `["personalisation","dedup"], ["dedup","diversity"], ["diversity","assembly"]` |
| `STEPS[6].stages`  | 2989      | `["business","freshness","diversity","dedup"]` → `["business","freshness","dedup","diversity"]` (cosmetic ordering, but keep it truthful)                       |

**Prose that must follow the reorder** — otherwise the diagram and the drawer
disagree:

- `dedup.decisions[0].why` (1765) currently justifies the split by "where it
  shrinks the most work" vs "the page being shown". The second half is now
  _upstream_ of page construction. Rewrite to: exact-hash collapse at union
  shrinks the expensive stages; near-duplicate collapse here produces the
  distinct set that Diversity then arranges.
- `diversity.contract.in` (1752) — `"ranked candidates"` → `"deduplicated
ranking"`, matching `dedup.contract.out` (1767).
- Consider adding a `diversity` failure row: diversity computed over a list
  containing near-duplicates wastes per-attribute cap slots on the same item.

No `EXAMPLES`, `REFS`, `DIALS` or `CAPS` changes — both keys survive unchanged.

### 1.2 `expand` becomes a left aside feeding two legs (D4)

`expand`'s own decision (1157) sends variants to the **lexical and
learned-sparse legs only** — "dense retrieval already generalises over
vocabulary. Expanding into it mostly adds drift." A centre-spine row asserts
the opposite: that every downstream stage receives the expanded query. The
aside placement follows the `q-image` precedent (2160): sit on the row where
the component is _produced_, wire forward to the consumers.

| Site                | Line      | Change                                                                               |
| ------------------- | --------- | ------------------------------------------------------------------------------------ |
| `TPL[3].rows`       | 2163–2164 | `{c:["rewrite"]}, {c:["expand"]}` → `{l:["expand"], c:["rewrite"]}`                  |
| `REL`               | ~2378     | Add `expand: { feeds:["lexical","sparse"] }`                                         |
| `BASELINE[3].flow`  | 2588–2589 | Delete `["rewrite","expand"], ["expand","prefilter"]`; add `["rewrite","prefilter"]` |
| `BASELINE[3].feeds` | ~2628     | Add `["expand",["lexical","sparse"]]`                                                |

**Edge kind:** plain `feeds`, not `gated`. This matches `q-image → imageimagevec`,
where the gate lives on the _consuming_ node rather than the edge. `expand`
carries no `GATED_EXECUTION` entry today and should not gain one — its
"off by default" status is a configuration stance expressed in its own drawer,
not a per-request route decision.

**No `RELATION_DETAILS` entry is required** (`relationDetail` returns `null`
harmlessly, 2402) — the mandatory-detail rule applies only to relations out of
`assembly` and `behavioural`. Adding one is still worth it for drawer quality:

```js
"expand|feeds|lexical": { label:"query variants", payload:"expanded terms and per-variant weights", cadence:"per request when expansion is enabled" }
```

**Consequences to expect:**

- Template 3 loses a spine row; the query-processing band shortens by one.
  The `+N new vs Core + recommended` pill (3390) recomputes from `tplStages`,
  which walks `l`/`c`/`r` alike (3038) — `expand` still counts as present, so
  the pill is unchanged.
- `expand`'s `caps:["vocab"]` (1158) still registers in Capability coverage;
  `sparse` also carries `vocab`, so the panel is stable.
- Two new curved side wires cross the retrieval fan-out region. Check the
  `feeds` route (`route:"side", curve:.4`, 2451) does not collide with the
  `q-image → imageimagevec` feed on the opposite side. Left cell is free on
  the retrieval row, so the wire has clear vertical space.

### Phase 1 verification — passed

Loaded at 1500x1000 in a browser. Both validators returned rather than threw.

| Check                              | Before          | After           |
| ---------------------------------- | --------------- | --------------- |
| `validation` template 3 edges      | 71 hidden       | 72 hidden       |
| `modelValidation.componentRelations` | 34            | 36              |
| `modelValidation.annotatedRelations` | 4             | 4 (unchanged)   |
| `TPL[3].rows` length               | 22              | 21              |
| template 3 node count              | 37              | 37 (unchanged)  |
| `layoutSweep` overlaps             | 0 in all 6 combinations | 0 in all 6 |
| `layoutSweep` clipped labels       | 2 (template 3)  | 2, the same two |
| edge-label / node collisions       | 0               | 0               |

The two clipped labels — `Image-to-image vector retrieval` and
`Multi-vector / passage retrieval` — are pre-existing at `HEAD`, confirmed by
re-running the sweep against a stashed tree. They are on the retrieval fan-out
row, which Phase 1 does not touch.

Visually confirmed in the Full template: `expand` renders in the left cell beside
`rewrite`, the spine runs `rewrite → prefilter` directly, two feed wires curve
from `expand` into the retrieval group, and the final-ranking tail reads
`Personalisation → Deduplication → Diversity → Results assembly`.

### Phase 1 discoveries

Four things the plan did not anticipate. All are now applied.

**1. The `RELATION_DETAILS` entry had to be dropped — it collided with a node.**
§1.2 proposed an optional detail entry for the `expand` feed. Added, it rendered
an edge label at `position:.56` along the wire, which for a *leftward* side route
lands past the group port and directly on top of the `Lexical & metadata
retrieval` node title — an 82x17px opaque box over 48px of the heading. The four
existing relation labels all sit on rightward wires and are clear, so the
placement rule had never met this case.

Rather than special-case label placement, the entries were removed. Three
reasons, in order of weight:

- `RELATION_DETAILS` today contains **exactly** the four relations the validator
  *requires* to be annotated — those out of `assembly` and `behavioural`. It is,
  in practice, the feedback-plane annotation table. Query-path entries break that
  correspondence for no structural gain.
- `expand.contract.out` already publishes `{ variants[], weights[] }`, which is
  the same payload the detail would have restated, in the place a reader looks
  for it.
- The `q-image → imageimagevec` feed — the precedent §1.2 leans on for the aside
  placement — carries no detail either.

**Follow-up worth its own change:** relation-label placement has no
collision avoidance and no left/right awareness. Any future leftward annotated
relation will hit this. Not fixed here; out of Phase 1's scope.

**2. Two feed wires render as one.** `expand → lexical` and `expand → sparse`
coalesce into a single wire terminating at the retrieval group's left port, with
`data-logical-count="2"`. This is the same treatment `q-image → imageimagevec`
gets, and the specific legs remain visible in the drawer, so it was accepted
rather than worked around. Worth knowing before Phase 4 wires anything into a
grouped target.

**3. `union`'s rationale carried the same circularity as `dedup`'s.** §1.1
flagged `dedup.decisions[0].why`, but `union.decisions[1].why` ended on the same
"only needs to be right for the page you show" claim — which Diversity, not
Deduplication, now decides. Rewritten alongside it.

**4. The legend prose asserted feeds are telemetry.** The Relationships note
said "Feeds carry telemetry; trains is offline". With `expand` feeding query
variants forward, `feeds` now runs in both directions. Reworded to "Feeds carry
data beside the spine — query variants forward, impression telemetry back".

### Phase 1 applied changes

Fifteen sites, five more than the plan's estimate of ten:

| Site                                 | Change                                                     |
| ------------------------------------ | ---------------------------------------------------------- |
| `TPL[2].rows`, `TPL[3].rows`         | `dedup` before `diversity`                                 |
| `TPL[3].rows`                        | `{l:["expand"], c:["rewrite"]}`, one row shorter           |
| `REL`                                | `expand: { feeds:["lexical","sparse"] }`                   |
| `BASELINE[2].flow`                   | three pairs rewritten                                      |
| `BASELINE[3].flow`                   | three pairs rewritten; `expand` pairs replaced by `rewrite → prefilter` |
| `BASELINE[3].feeds`                  | `["expand",["lexical","sparse"]]`                          |
| `STEPS[6].stages`, `STEPS[6].build`  | reordered to match                                         |
| `dedup.decisions[0].why`             | rewritten — no longer circular                             |
| `union.decisions[1].why`             | rewritten — discovery 3                                    |
| `diversity.contract.in`              | `ranked candidates` → `deduplicated ranking`               |
| `diversity.failures`                 | second row: diversifying over near-duplicates              |
| Relationships legend prose           | discovery 4                                                |
| `TEMPLATES` banner comment           | two lines stating the aside invariant, which Phase 4 also relies on |

`EXAMPLES`, `REFS`, `DIALS`, `CAPS`, `REQUIRES` and `GATED_EXECUTION` were not
touched, as predicted. The `+14 new vs Core + recommended` pill is unchanged.

---

## Phase 2 — Remove `Late-interaction retrieval` entirely

The largest phase. `late` is referenced at 20 sites across nine subsystems, and
three of them throw if missed. Delete in the order below; the gate reciprocity
edit (step 2) must land in the same commit as the `def` deletion.

### 2.1 Deletion inventory

| #   | Site                                       | Line(s)   | Action                                                                                                                                                                                                                                                                                                              |
| --- | ------------------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `def({id:"late" …})`                       | 1468–1497 | Delete the whole block                                                                                                                                                                                                                                                                                              |
| 2   | `GATED_EXECUTION.late`                     | 2360      | Delete the entry                                                                                                                                                                                                                                                                                                    |
| 3   | `GATED_EXECUTION.latererank`               | 2362      | Drop the 4th arg: `defGate(3,"route","route selected")` — **reciprocity throws otherwise**                                                                                                                                                                                                                          |
| 4   | `LATE_MODE_OVERLAP` const                  | 1432–1436 | Delete — becomes unreachable once both consumers drop it                                                                                                                                                                                                                                                            |
| 5   | `late.failures` `LATE_MODE_OVERLAP`        | 1490      | Deleted with the block                                                                                                                                                                                                                                                                                              |
| 6   | `latererank.failures` `LATE_MODE_OVERLAP`  | 1618      | Delete the line                                                                                                                                                                                                                                                                                                     |
| 7   | `LEGS`                                     | 2291      | Drop `"late"` — 7 legs → 6                                                                                                                                                                                                                                                                                          |
| 8   | `BASELINE_RETRIEVAL_LEGS`                  | 2509      | Drop `"late"`                                                                                                                                                                                                                                                                                                       |
| 9   | `BASELINE_GATED_FULL_RETRIEVAL_LEGS`       | 2513      | Drop `"late"`                                                                                                                                                                                                                                                                                                       |
| 10  | `DATA_SOURCES["ds-token"]`                 | 2237–2245 | Delete (D1)                                                                                                                                                                                                                                                                                                         |
| 11  | `SERVING_REL["ds-token"]`                  | 2263      | Delete the key                                                                                                                                                                                                                                                                                                      |
| 12  | `REFS.late`                                | 1939–1948 | Delete — **but migrate two entries first, see 2.2**                                                                                                                                                                                                                                                                 |
| 13  | `EXAMPLES.late`                            | 2056      | Delete the key                                                                                                                                                                                                                                                                                                      |
| 14  | `STEPS[9]`                                 | 3001–3003 | Rewrite, see 2.3                                                                                                                                                                                                                                                                                                    |
| 15  | `BASELINE[3].flow` `["late","union"]`      | 2598      | Delete the pair                                                                                                                                                                                                                                                                                                     |
| 16  | `BASELINE[3].steers`                       | 2624      | Recomputed from the shortened `BASELINE_RETRIEVAL_LEGS` — no edit if the array is used, verify                                                                                                                                                                                                                      |
| 17  | `BASELINE[3].serves` `["ds-token","late"]` | 2641      | Delete the pair                                                                                                                                                                                                                                                                                                     |
| 18  | `TPL[3].rows` retrieval row                | 2166      | Drop `"late"` from `c:` — 7 boxes → 6                                                                                                                                                                                                                                                                               |
| 19  | Serving count fixture                      | 2941      | `expectedSourceCounts[3]` 7 → **6**; `expectedServingCounts[3]` 8 → **7**                                                                                                                                                                                                                                           |
| 20  | `RETRIEVAL_REPRESENTATION_OVERLAP`         | 1437–1441 | **Keep** — still cited by `multivec` (1462). Reword: it currently reads "Passage and late-interaction retrieval both enabled", which no longer describes a reachable state. Recast as passage retrieval overlapping the late-interaction _rerank_ tier, or narrow it to passage-vs-document representation overlap. |

`REQUIRES` (2958) has no `late` entry — no change. `PARALLEL_GROUPS.retrieval`
(3443) is membership-derived — no change.

### 2.2 Reference migration — do this before deleting `REFS.late`

`REFS.latererank` (1949–1956) already carries ColBERT and ColBERTv2. Two
entries exist **only** in `REFS.late` and have no other home:

- **PLAID** (2205.09707) — serving efficiency for full-index MaxSim. Genuinely
  retrieval-specific; **drop it** with the component.
- **ColPali** (2407.01449) — late interaction over document _images_. Directly
  relevant to this tool's image-bearing worked example, and nothing else cites
  it. **Migrate to `REFS.latererank`** with a reworded `w:` blurb pointing at
  the rerank placement, or to `REFS.vlmrerank` if it reads better beside the
  vision-language tier.

Losing ColPali silently is the one substantive content regression this phase
can cause.

### 2.3 Prose rewrites — `latererank` becomes the sole MaxSim placement

`latererank` stays **[optional] / Full only** (D2). Its tier, gate kind and
cascade position are unchanged. What changes is that three passages now point
at a component that no longer exists:

| Site                          | Line      | Current                                                                                                                                                                  | Fix                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `latererank.decisions[…].why` | 1604      | "use passage retrieval or full-index late-interaction retrieval instead"                                                                                                 | Drop the second option: "use passage retrieval instead"                                                                                                                                                                                                                                                                                                      |
| `latererank.notes`            | 1615      | "Avoid combining late-interaction reranking with full-index late-interaction retrieval unless…"                                                                          | Delete the note — the combination is now unrepresentable                                                                                                                                                                                                                                                                                                     |
| `latererank.purpose`          | 1599      | "using a bounded shortlist **rather than searching a full token index**"                                                                                                 | The contrast has no referent on the surface. Either keep it as a deliberate statement of what this tier is _not_ (defensible — it explains the design choice), or restate positively. Recommend keeping, reworded so it reads as a design rationale rather than a cross-reference.                                                                           |
| `STEPS[9]`                    | 3001–3003 | title "Passage-level or late-interaction placement", `stages:["multivec","late","latererank"]`, and a `build`/`buys` pair built around choosing between three placements | Retitle to "Passage-level retrieval or late-interaction rerank", `stages:["multivec","latererank"]`, rewrite `build` as a two-way choice (weak candidate recall → passage retrieval; weak multi-facet ordering at high shortlist recall → late-interaction rerank). The `buys` line's "without paying for MaxSim twice by default" is now vacuous — replace. |

**Step count is unchanged**, so the Full-surface divider (4424–4441, keyed to
the first `t===3` step) and the static "Steps 1–7" sentence at line 900 both
stay correct.

**D1 note added to `latererank`.** With `ds-token` gone, nothing on the surface
says where the rerank tier's token vectors come from. Add one note to
`latererank`: stored document token representations, read from the passage or
document store at rerank time — a bounded per-shortlist cost, not a
corpus-scale index. This preserves the honest infrastructure signal that
`ds-token`'s own note carried ("the largest hidden infrastructure commitment
on the Full surface", 2244) at the scale that actually still applies.

### Phase 2 verification

```bash
grep -n '"late"\|late:\|ds-token\|LATE_MODE_OVERLAP\|altOf' search-query-pipeline-diagram-tool.html
```

Expect zero hits for `ds-token` and `LATE_MODE_OVERLAP`, zero `altOf` in
`GATED_EXECUTION`, and no bare `late:` key. `latererank` hits are expected.

In the browser: the Full retrieval fan-out shows **six** legs; the serving
overlay shows **six** sources / seven edges; `window.dependencyDiagnostics
.validation` and `.modelValidation` both resolve; toggling every component off
one at a time in template 3 produces no orphaned edges (the per-component
disable sweep at ~2947 already asserts this).

---

## Phase 3 — Name the sufficiency floor honestly

Smallest phase, no structural change, no fixture change.

`sufficiency` (1237) is defined as "n candidates above a **score floor**", and
sits after `union` and before `prune` (2168). At that point no comparable score
exists: `union`'s own decision text (1506) states that "BM25, cosine and
cross-modal scores are not directly comparable", and the fused score is two
rows away. The floor can only be per-leg and calibrated — which is exactly what
`prune` does (1523–1531, `def:"Top-N per leg plus conservative per-leg floors"`).

The placement is correct and well argued (1241–1245) — keep it. Only the
wording needs to stop implying a single global score.

| Site                           | Line      | Change                                                                                                                                                                       |
| ------------------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sufficiency.sub`              | 1238      | `"enough candidates above the quality floor?"` → `"enough candidates above per-leg calibrated floors?"`                                                                      |
| `sufficiency.decisions[1]`     | 1247–1248 | Options `["n candidates","n candidates above a score floor"]` → `["n candidates","n candidates above per-leg calibrated floors"]`; `def` follows                             |
| `sufficiency.decisions[1].why` | 1249      | Extend: twenty terrible matches are not sufficiency **and** the floor must be per-leg, because no fused score exists yet at this point in the pipeline                       |
| `sufficiency.purpose`          | 1239      | "a check that enough candidates cleared a quality floor" → "…cleared per-leg calibrated floors"                                                                              |
| `sufficiency.dials`            | 1252      | `{n:"Score floor",v:"from the score distribution of judged-relevant docs"}` → make it per-leg: one floor per retrieval leg, from that leg's own judged-relevant distribution |
| `sufficiency.contract`         | 1250–1251 | Verify the `in`/`out` strings don't imply a fused score; adjust if so                                                                                                        |

Add one note to `sufficiency` making the relationship to `prune` explicit: the
same calibrated floors are used here as a gate and there as a cut, which is
precisely why the check must run _before_ pruning (`prune.failures[0]`, 1533,
already names the inverse failure).

**Cross-check `prune`:** its `sub` (1523) says "cap per leg, score floor, cut to
rerank depth". If Phase 3 standardises on "per-leg calibrated floors", use the
same phrase in `prune.sub` so the two nodes visibly share one mechanism.

`sub` strings are rendered on the node face — confirm the longer text does not
overflow the node box at the narrowest supported layout.

### Phase 3 verification

Visual only. Open the drawer for `Result sufficiency check` and confirm no
remaining phrase implies a single comparable score before Fusion.

---

## Phase 4 — Add a minimal evaluation and experimentation plane (D3)

### 4.1 Why two nodes

`assembly.contract.out` (1782) already emits `experiment_ids[]` into the
impression context, and `behavioural`'s notes (1823) already describe using
"experiment context" to interpret events. **No node assigns it.** That is a
live dangling reference, not a hypothetical gap.

Separately, `STEPS[0]` (2965) is titled **"Evaluation harness"** with
`stages:[]` — the build ladder's own first rung has no component. The tool
opens by telling you to build the harness "before anything else" and then never
draws it.

Two nodes close both gaps and keep the request-path/offline split the tool
maintains everywhere else:

- **`experiment` — Experiment assignment** (control, request-time). Assigns the
  variant, stamps `experiment_ids` onto the request, and steers the
  configuration-selecting stages.
- **`offlineeval` — Offline evaluation** (control, offline sink). The labelled
  set, deterministic replay, the rank-1 probe suite, and the debiased
  judgements that `behavioural` produces.

### 4.2 D5 — grouping (**confirm before implementing**)

The two nodes do not fit the existing groups: `experiment` sits physically at
the top of the diagram, `offlineeval` at the bottom, and forcing both into
`"Results"` reproduces exactly the group/position mismatch flagged elsewhere in
the review.

**Recommendation:** add a sixth entry to `PHASES` (3082–3088):

```js
{id:"eval", label:"Evaluation", groups:["Evaluation"], color:"var(--wire-train)"}
```

and give `experiment`, `offlineeval` **and `behavioural`** `group:"Evaluation"`.
Moving `behavioural` out of `"Results"` is what makes the phase coherent — the
event log is the evaluation plane's data source, not a results-assembly
concern — but it changes the "Stages by phase" legend for template 2 as well as
3, which is a visible change to a template this plan otherwise does not touch.

**If that is unwanted:** leave `behavioural` in `"Results"`, put both new nodes
in `"Evaluation"`, and accept a two-member phase. Everything else in Phase 4 is
unaffected either way.

### 4.3 Component definitions

Both are `control:true`, matching `session`, `confidence`, `routing`,
`fusionpolicy`, `degradation` and `behavioural`. Tiers:

| Component     | tier          | intro | Rationale                                                                                                                                                                                                                             |
| ------------- | ------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment`  | `recommended` | 2     | The components list has A/B testing at [recommended]; it must coexist with `routing` (template 2) for its steer edge to validate.                                                                                                     |
| `offlineeval` | `recommended` | 2     | The components list has offline evaluation at [core], but a Core-template node would restructure Core, which is out of scope for this plan. [recommended] puts it in the production target where the build ladder already demands it. |

`tier` and `intro` must stay perfectly correlated (1053) — `recommended`/`2` for
both.

Each `def` needs at minimum `id`, `name`, `group`, `tier`, `control:true`,
`intro`, `sub`, `cx`, `purpose`, `decisions[]`, `contract`. Suggested `cx`: 2
for `experiment`, 3 for `offlineeval` (it is a standing harness, not a service).
Suggested `caps`: `["learning"]` on `offlineeval`, none on `experiment`.

Candidate decision questions, matching the house voice:

- `experiment` — "What is the unit of assignment?" (query / session / user),
  defaulting to user with a note that query-level assignment contaminates
  session metrics.
- `experiment` — "Is the variant logged or inferred?", defaulting to logged,
  because an inferred variant makes every result unattributable.
- `offlineeval` — "Judgements from annotators or from behaviour?", defaulting
  to a labelled set first, debiased behavioural judgements once volume allows.
- `offlineeval` — "Is replay deterministic?", defaulting to yes, tying back to
  `understand`'s warning (1118) that an LLM in the hot path "breaks replayable
  evaluation" — a cross-reference the tool currently makes with nothing to
  point at.

### 4.4 Wiring

```js
experiment:   { steers:["routing","fusionpolicy"], feeds:["assembly"] },
behavioural:  { trains:[…], updates:[…], feeds:["offlineeval"] },   // add feeds
```

Coexistence check (~2492): `experiment`+`routing` coexist in template 2 ✓;
`experiment`+`fusionpolicy` coexist in template 3 ✓; `behavioural`+`offlineeval`
coexist in template 2 ✓.

**Mandatory `RELATION_DETAILS` entry** — `behavioural` is on the enforced list
(~2521), so this edge throws without all three fields:

```js
"behavioural|feeds|offlineeval": {
  label:"debiased judgements",
  payload:"versioned query-document judgements with position-aware features",
  cadence:"offline, on the evaluation cycle"
}
```

`experiment|feeds|assembly` is optional (`experiment` is not on the enforced
list). **Phase 1 found that optional entries are not free** — every detail entry
renders an edge label at `position:.56` along its wire, with no collision
avoidance, and a leftward side route puts that label on top of a node (Phase 1
discoveries, item 1). Add it only after checking where the label lands:

```js
"experiment|feeds|assembly": {
  label:"variant assignment",
  payload:"experiment_ids and assigned variant configuration",
  cadence:"per request"
}
```

The mandatory `behavioural|feeds|offlineeval` label has no such escape — if it
collides, the label-placement follow-up from Phase 1 has to be done first.

**Do not add `experiment` to `GATED_EXECUTION`**, and do not add either node to
`SELECTIVE_RERANK_CASCADE` or `GATED_RERANKERS` — the routing↔gated-reranker
symmetry check (~2501) throws on any reranker-group member that routing does
not steer, and vice versa.

### 4.5 Template placement

Right-hand control cells accept multiple ids (`ids.forEach`, 3417), so no new
rows are needed:

| Template | Row                                   | Line | Change                                  |
| -------- | ------------------------------------- | ---- | --------------------------------------- |
| 2        | `{c:["normalise"]}`                   | 2137 | → `{c:["normalise"], r:["experiment"]}` |
| 2        | `{c:["assembly"], r:["behavioural"]}` | 2151 | → `r:["behavioural","offlineeval"]`     |
| 3        | `{c:["normalise"]}`                   | 2161 | → `{c:["normalise"], r:["experiment"]}` |
| 3        | `{c:["assembly"], r:["behavioural"]}` | 2181 | → `r:["behavioural","offlineeval"]`     |

`experiment` goes on the `normalise` row rather than the `q-text` row because
template 3's `q-text` row already carries `session` on the right (2160), and
because assignment logically precedes any config-dependent stage.

Neither node appears in `row.c`, so **no `flow` edges are created** — the spine
is untouched in both templates. That is the point: the evaluation plane is not
in the request path.

### 4.6 Fixture and build-ladder updates

| Site                  | Change                                                           |
| --------------------- | ---------------------------------------------------------------- |
| `BASELINE[2].steers`  | Add `["experiment","routing"]`                                   |
| `BASELINE[2].feeds`   | Add `["experiment","assembly"]`, `["behavioural","offlineeval"]` |
| `BASELINE[3].steers`  | Add `["experiment",["routing","fusionpolicy"]]`                  |
| `BASELINE[3].feeds`   | Add `["experiment","assembly"]`, `["behavioural","offlineeval"]` |
| Serving counts (2941) | **No change** — neither node has a `serves` relation             |
| `STEPS[0].stages`     | `[]` → `["offlineeval","experiment"]`                            |

Putting the new components on `STEPS[0]` rather than adding a step keeps the
ladder at 14 rungs, so the Full-surface divider (4424–4441) and the static
"Steps 1–7" sentence (line 900) both remain correct. It also finally gives the
tool's own opening instruction a clickable target.

`STEPS[0].build` (2966) already describes the labelled set, deterministic
replay and the probe suite. Extend it with one clause on variant assignment so
the rung covers both new components.

### Phase 4 verification

```bash
grep -n 'experiment\b' search-query-pipeline-diagram-tool.html | grep -v experiment_ids
```

In the browser: `validation` and `modelValidation` both resolve; templates 2
and 3 each show two new dashed control nodes; the `+N new vs Core` pill on
template 3 is unchanged (both nodes are `intro:2`); template 2's pill increases
by 2; clicking `Offline evaluation` opens a drawer whose "Fed by" section names
`Behavioural event log` with the transmission detail; the Stages-by-phase
legend renders the new phase (D5).

---

## 5. Commit boundaries

Order and rationale are in the **Phase and status summary** at the top of this
plan. Two execution notes that belong with the work rather than the summary:

- **Phase 2** — delete top-down in the §2.1 order, and run the browser check
  before writing any of the §2.3 prose rewrites. A throwing validator is much
  easier to locate against a deletion-only diff.
- **Update the Status column** in the summary table as each phase lands, so the
  plan stays readable as a progress record rather than only an intent.

## 6. Out of scope

Recorded so a later reader knows the omissions are deliberate. From the same
review, not addressed here:

- Final-ranking score composition — collapsing `business` / `freshness` /
  `personalisation` into one stage, and the uncapped-boost interaction the
  current order permits. Related to the open "Review business ranking position"
  item in `TODO.md`.
- Query encoding / query-embedding node between `prefilter` and the retrieval
  fan-out.
- Caching (response cache short-circuit, query-embedding cache).
- Generative answer layer, or an explicit scope statement in `TPL[3].desc`
  saying the tool stops at retrieval and ranking.
- `semrerank` removal.
- Recovery loop returning to `routing` rather than `prefilter`.
- Group-tag corrections for `routing`, `degradation`, `sufficiency`, `relax`
  and `zerofallback` (already tracked in `TODO.md`).
