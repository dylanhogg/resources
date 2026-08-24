# Plan 003 — Remove timing, latency, budget and millisecond references

**Target:** `search-query-pipeline-diagram-tool.html` (4738 lines) — sole file in scope.
**Status:** plan only — not implemented.
**Decisions:** D1 delete · D2 plain filter row · D3 keep and rewrite · D4 keep relative cost.

## 1. Scope rule

The tool describes a *logical* query-side search pipeline. Quantified serving
characteristics are claims the diagram cannot support; they will be determined
experimentally later, per service, compute and data conditions.

**Remove — quantified or measured performance claims:**

- Every millisecond figure, range and per-call cost (`lat:[lo,hi]`,
  `~150 ms p95`, `0.5–1.5 ms/doc`, `80–500 ms per call`, `60–80% of the leg's p99`).
- Every percentile: `p95`, `p99`.
- The word **latency** in all its forms (*latency budget, latency-bound,
  latency dial, latency attribution, low latency*).
- The word **budget** wherever it means a time or spend allowance (*headline
  budget, rerank budget, budget-protection, budget-aware, infrastructure
  budget*).
- The literal field label **`timing:`** where it is printed to the reader.

**Keep — qualitative temporal logic intrinsic to the algorithm:**

- *deadline, partial results, times out, slowest leg, concurrently, timely,
  round trip* — the degradation controller and the parallel retrieval group
  are unintelligible without them, and none asserts a measurement.
- Relative-cost adjectives: *cheap first pass, the expensive tier, cost is
  linear in depth, cost scales with document length* (**D4**).
- `cx` — build-and-run complexity index.
- `parallel:` grouping and `conditional:` — properties of the algorithm.
- Candidate *depth* (`k`, top-N, rerank depth) — a logical quantity.

The resulting rule is greppable: **no numbers with time units, no
percentiles, no "latency", no "budget".**

## 2. Deleting `Candidate budget allocation` (D1)

Deleting a component is the largest structural change in this plan. Two
findings make it cleaner than expected, and one is a latent bug.

**Finding 1 — the reference survives on its own.** `REFS.budget` holds the
WAND paper (1927–1930), and `REFS.prune` **already carries the same paper**
(1988). Deleting `REFS.budget` loses no citation.

**Finding 2 — the attach loops are guarded.** `REFS`, `EXAMPLES`, `DIALS` and
`FAILS` are all applied via `Object.keys(X).forEach(k=>{ if(S[k]) … })`
(2063, 2099, 2121, 2129). A leftover `budget:` key would silently no-op rather
than throw — so nothing catches a half-finished deletion. Delete the keys
explicitly; do not rely on the guard.

**Finding 3 — the build-order divider is hard-keyed to step 8.** Line 4511:

```js
return s.n===8 ? '<div class="divider">Steps 1–7 above are Core + recommended …' + html : html;
```

Step 8 *is* `Candidate budget allocation`. Deleting it silently removes the
"Everything below is Full surface territory" divider from the Build order
view. **Renumber steps 9–14 → 8–13**, and re-key the divider to the first
`t===3` step so it cannot rot again.

### Ripple list

| Site | Line(s) | Action |
|---|---|---|
| `def({id:"budget" …})` | 1328–1339 | Delete the whole block |
| `REFS.budget` (WAND) | 1927–1930 | Delete the key — duplicated at `REFS.prune` 1988 |
| `EXAMPLES.budget` ("the budget is 400 ms") | 2078 | Delete the key |
| `FAILS.budget` (2 rows) | 2124–2126 | Delete the key — leaves `FAILS` with only `fusionpolicy` |
| `TPL[3]` row `r:["budget","degradation"]` | 2197 | → `r:["degradation"]` |
| `REL.routing.steers` `LEGS.concat(["budget"])` | 2420 | → `LEGS.concat(GATED_RERANKERS)` |
| `REL.budget` | 2421 | Delete the entry |
| `BASELINE[3].steers` routing row | 2663 | Drop `.concat(["budget"])` |
| `BASELINE[3].steers` budget row | 2664 | Delete the row |
| `STEPS[8]` "Candidate budget allocation" | 3038–3041 | Delete the step, renumber 9–14 → 8–13 |
| Build-order divider `s.n===8` | 4511 | Re-key to the first `t===3` step |
| `PARALLEL_GROUPS.controlPorts.budget:"allocates k"` | 3524 | Delete the port |

### Consequences to expect

- **Capability coverage is unaffected.** `budget` carried `caps:["adaptive"]`,
  but *Per-query adaptivity* is also provided by `session` (1191), `routing`
  (1311) and `fusionpolicy` (1588).
- **Control-plane wires re-space.** `controlPorts` slots are distributed
  evenly along the retrieval-group boundary (`groupControlGeometry`, 3819).
  Dropping from three ports to two (`routing`, `degradation`) moves the
  remaining two wires. Visual only — worth an eyeball on Full surface.
- **`RREL` staleness is checked.** `validateRelationshipModel` throws
  `"RREL is stale"` (2973) if the reverse index disagrees with `REL`, so a
  missed edit at 2420/2421 fails loudly at load. Good.
- **Optional, recommended:** the *logical* idea worth preserving is that
  per-leg candidate depth is part of the route decision. Add one line to
  `routing.notes` — "Per-leg candidate depth is part of the route decision:
  spend depth on the legs the route expects to decide the answer." — rather
  than reintroducing a stage. `routing` already steers every leg.
- Line 1321 `routing.notes` "Learned or **budget-aware** routing" →
  "Learned or depth-aware routing" (needed regardless of D1).

## 3. Inventory of the remaining occurrences

143 matching lines total. Grouped by surface.

### 3.1 Data schema — the `lat` field (38 components)

| Site | Line(s) | Action |
|---|---|---|
| Schema comment `lat: [lo,hi] p95 ms.` | 1028 | Drop the `lat` sentence; keep the `cx` sentence |
| `lat:[a,b]` on every component | 38 sites, 1070–1835 | Delete the field (37 after D1) |
| `budget()` accumulation | 3151–3166 | Delete the lat / parallel / conditional maths |
| `renderBudget()` latency output | 3175–3188 | Delete |
| Drawer chip `p95 <b>…ms</b>` | 4170 | Delete the chip |
| Hover pill + `const latency` | 4408, 4412 | Delete — **`s.lat[0]` is unguarded, so removing the field without this line throws on every hover** |

### 3.2 LHS sidebar panel

| Site | Line(s) | Action |
|---|---|---|
| `<h3>Estimated budget` | 830 | → `Pipeline shape` |
| `Latency, p95` metric (`#latVal`, `#latLo`, `#latHi`, `#latNote`) | 831–835 | Delete |
| `Where it goes` stack + legend | 836–840 | Per **D2**: delete the `#latStack` bar and its `<div class="top">` label; keep `#latLegend`, rename the id to `#phaseLegend` |
| "watch the budget move" | 890 | → "watch the shape change" |
| CSS `.bar i.lo` / `.bar i.hi` | 182–183 | Delete (`.bar i.cx` stays for complexity) |
| CSS `.stack`, `.stack i` | 186–187 | Delete |

`.legendrow` and `.bar` are shared with the Legend and complexity panels — keep.

### 3.3 Composition model and phase filter (D2)

| Site | Line(s) | Action |
|---|---|---|
| `BUDGET MODEL` banner | 3124–3126 | → `COMPOSITION MODEL` |
| `const PHASES` | 3127–3133 | **Keep** — drop the `color` field only if the legend swatches go; keeping the swatch is cheaper and reads better |
| `const MAX_CX` | 3134 | Keep |
| `function budget()` | 3136–3169 | → `composition()`. Return `{cx, models, modelNames, caps, count, off}` plus a `perPhase` **component count** per phase (replaces the latency sum at 3159/3164). The `parallel` / `conditional` branches collapse into a plain count |
| `function renderBudget()` | 3172–3195 | → `renderComposition()`. Delete `#latVal`, `#latLo`, `#latHi`, `#latNote`, `#latStack` and `const scale = 900`. Render the phase row as plain filter buttons: swatch + label + component count, **no percentages, no bar** |
| Phase click wiring | 3205 | Keep, retargeted at `#phaseLegend` |
| Call site `renderBudget()` | 4529 | Rename |

Phase focus itself (`FOCUS.phase`, `applyFocus`, `syncFocusUI`) keeps working
unchanged apart from the id rename and one wording fix:

| Site | Line(s) | Action |
|---|---|---|
| `unified focus: facet \| capability \| latency phase \| gate` | 4032 | Drop "latency" |
| `FOCUS.phase.describe` "components spending time in this phase" | 4046–4047 | → "components in this phase" |
| `syncFocusUI` `#latLegend` selector | 4098–4099 | → `#phaseLegend` |

### 3.4 Component prose

| Component | Line(s) | Change |
|---|---|---|
| `understand` | 1128, 1136 | `why` "costs 100–400 ms and makes every downstream stage non-deterministic" → "is slow and makes every downstream stage non-deterministic". Delete both `Latency (rules)` / `Latency (LLM)` dial rows; fold "Cache-or-don't-ship territory" into the `why` |
| `relax` | 1269, 1281 | `conditionalOn` keeps "one retrieval round trip per pass". `why` "is how you ship a 700 ms p99" → "is how a bounded recovery path becomes an unbounded one" |
| `routing` | 1321 | "budget-aware routing" → "depth-aware routing"; optional depth note per §2 |
| `expand` | 1371, 1372 | Dial note "tune for latency" → "tune for index cost". Failure `s:` "Latency worse than BM25 with no precision gain" → "Costs more than BM25 with no precision gain"; `m:` "per-leg latency and per-leg contribution to top-10" → "per-leg contribution to top-10" |
| `imageimagevec` | 1444 | Keep "route, candidate depth, deadline, calibration and fusion weight" — deadline is qualitative |
| `LATE_MODE_OVERLAP` | 1457 | `m:` "ablate recall, nDCG and latency with each mode alone" → "ablate recall and nDCG with each mode alone" |
| `RETRIEVAL_REPRESENTATION_OVERLAP` | 1462 | `m:` "unique relevant-document contribution per leg; recall gained per millisecond" → keep the first clause only |
| `late` | 1509, 1516 | Note "retrieval quality, latency, and index cost" → "retrieval quality and index cost". Failure `s:` "blow the infrastructure budget" → "blow the index size and memory assumptions" |
| `prune` | 1546 | "Cut the candidate set to what the rerank budget can actually afford" → "Cut the candidate set to the depth the rerank tier is configured for" |
| `fusion` | 1577 | Keep "or only one survives its deadline" |
| `semrerank` | 1603, 1609, 1613, 1614 | `sub` "optional bi-encoder budget-protection tier" → "optional bi-encoder narrowing tier". `why` (budget ×2) → "It exists to narrow what the cross-encoder has to see. If the cross-encoder can already afford the full fused depth, it is a stage that reorders almost nothing — and a stage that reorders nothing is a stage to delete." `def` "Skip it if the cross-encoder can afford the full fused depth" stays. Note "measured budget intervention" → "measured narrowing step". Failure `s:` "Pure added latency" → "Pure added work" |
| `crossenc` | 1650, 1660, 1663, 1670 | `purpose` "the main latency dial in the whole pipeline" → "the main cost dial in the whole pipeline". `why` "Long inputs drive the p99 far above the p95, and most of the discriminating signal is in the first few hundred tokens" → "Long inputs cost far more than short ones, and most of the discriminating signal is in the first few hundred tokens". Delete the `Cost ~0.5–1.5 ms/doc on GPU` dial row. Failure row → `f:` "Cost scales with document length", `s:` "Long documents dominate the rerank stage", `m:` "stage cost by document token length" |
| `vlmrerank` | 1676, 1683, 1687 | `conditionalOn` → "Runs over the top 10–20 candidates only." `why` "At 50–500 ms per call this is the most expensive component on the menu" → "This is the most expensive component on the menu". Failure `s:` "p95 latency doubles for no gain on non-visual queries" → "Cost doubles for no gain on non-visual queries"; `m:` "VLM fire rate by intent class; latency attribution" → "VLM fire rate by intent class; cost attribution" |
| `assembly` | 1811 | Failure `f:` "Hydration is the hidden latency" → "Hydration is the hidden cost"; `s:` "Fast search, slow response" stays; `m:` "per-stage latency attribution" → "per-stage cost attribution" |
| **`degradation`** | 1816–1832 | See §4 |

### 3.5 Degradation controller (D3) — keep, de-quantify

Only the two numeric claims go; the deadline vocabulary stays.

| Site | Line | Change |
|---|---|---|
| `sub:"deadlines, partial results"` | 1817 | Unchanged |
| `purpose` | 1818 | Unchanged |
| `def` / `why` | 1822–1824 | Unchanged — "the slowest nonessential retrieval leg" and "your fast responses are also your least predictable ones" are qualitative |
| `contract` | 1825 | Unchanged |
| Note 1 | 1827 | "Prefer timely partial results to a slow or failed response. ~~The earlier 300 ms and 2 s figures are illustrative only;~~ set actual stage deadlines from product service objectives, workload behaviour, and user expectations." → drop the middle clause and start the second sentence "Set actual stage deadlines from…" |
| Note 2 | 1828 | Unchanged |
| `DIALS.degradation` "Per-leg deadline · 60–80% of the leg's p99" | 2117 | → "Per-leg deadline · derived from the leg's own completion behaviour, never one global value". Keep the "Degradation order" row unchanged |
| `EXAMPLES.degradation` "before missing the deadline" | 2096 | Unchanged |
| `STEPS` step 2 "Instrumentation and degradation deadlines" | 3014–3016 | Unchanged |
| `TPL[2].desc` "operational deadlines" | 2164 | Unchanged |

Note the "300 ms and 2 s" phrase is a dangling back-reference — the figures it
points at were already removed in an earlier pass, so the sentence is stale as
well as quantified.

### 3.6 Templates, examples, build order

| Site | Line(s) | Action |
|---|---|---|
| `TPL[1..3].target` = `~150 / ~400 / ~600 ms p95` | 2146, 2163, 2188 | **Dead field — nothing reads it.** Only `name`, `tagline`, `desc`, `rows`, `extra` are consumed (3083, 3468, 4250, 4593). Free deletion |
| `STEPS[8]` | 3038–3041 | Deleted per D1 |
| `STEPS[9]` trigger "Cross-encoder depth is the latency bottleneck" | 3044 | → "Cross-encoder depth is the binding constraint on the rerank funnel" |
| Build-order divider | 4511 | Re-key per §2 |

### 3.7 References — blurbs only, all four papers stay

| Paper | Line | Blurb change |
|---|---|---|
| WAND, under `budget` | 1929 | Key deleted with the component; the paper survives under `REFS.prune` (1988) with no blurb — optionally give it one there |
| PLAID, under `late` | 1971 | "How to actually serve it at low latency." → "How to actually serve it efficiently." |
| Snippet generation, under `assembly` | 2051 | "a bigger share of the latency budget than most designs assume" → "a bigger share of the work than most designs assume" |
| The Tail at Scale, under `degradation` | 2055 | "Fan-out to N legs means your p99 is the p99 of the slowest leg, every time." → "Fan-out to N legs means the response is governed by the slowest leg, every time." |

### 3.8 Gate model, conditional note, relation details

Required — these are literal `latency` / `budget` / `timing:` strings:

| Site | Line(s) | Action |
|---|---|---|
| `/* Latency, not execution: … */` comment block | 2358–2362 | Delete — it explains maths that no longer exists |
| `CONDITIONAL_BUDGET_NOTE` | 2363 | Delete the constant |
| `conditionalSentence(s)` | 2365–2367 | Collapses to `s.conditionalOn`; inline it at the one call site (4421) or keep the function as a pass-through |
| `GATE_KINDS` doc comment "standing configuration choice about budget or available training data" | 2331 | → "…about corpus shape or available training data" |
| `GATE_KINDS.budget` — the kind name | 2350 | → `config` (matches the existing `--gate-config` colour var) |
| Condition sentence "kept only when budget or training data justifies it" | 2353 | → "kept only when the corpus or available training data justifies it" |
| `defGate(3,"budget","optional pass")` ×2 | 2400, 2403 | → `"config"` |
| Validation comment "A conditional stage leaves the headline budget" | 2832–2833 | → "A conditional stage runs on only some requests, so it owes the reader a reason" |
| Node-card comment "…carries the widest latency range in the diagram, so neither chip may hide the other" | 3410–3413 | Rewrite: gating and cost are still orthogonal, but the latency-range justification is gone |
| `budget()` comment "legs run concurrently: the group costs its slowest member" | 3153 | Deleted with the maths |
| aria label `"; timing: "+edge.detail.timing` | 3764 | **User-visible word "timing"** → `"; cadence: "+edge.detail.cadence` |
| mobile row `' · '+edge.detail.timing` | 3329 | → `edge.detail.cadence` |
| drawer refdetail `detail.timing` | 4229 | → `detail.cadence` |
| `RELATION_DETAILS[*].timing` ×4 | 2430–2440 | Rename the field → `cadence`. Values (`per response telemetry`, `offline model training`, `asynchronous profile update`) are cadences, not durations — unchanged |

Optional hygiene — the gate `timing` field means *per request vs deployment
choice*, never a duration, and is never printed as the word "timing". Renaming
it is only worth doing so your verification grep comes back clean:

| Site | Line(s) | Action |
|---|---|---|
| `GATE_TIMING_LABEL` | 2335, 2827, 4177 | → `GATE_SCOPE_LABEL`; values unchanged |
| `spec.timing` → `spec.decidedAt` | 2338, 2344, 2351, 2383, 2388 | Mechanical |
| `el.dataset.gateTiming` | 3400 | → `gateScope` |
| CSS `[data-gate-timing="deployment"]` ×2 | 209, 350 | → `[data-gate-scope="deployment"]` |
| Gate legend `data-gate-timing=` | 4548 | → `data-gate-scope=` |
| Validation message "Unknown gate timing for kind" | 2827 | → "Unknown gate scope for kind" |

**Skipping this block leaves ~12 `timing` hits in the file**, all internal.

### 3.9 Parallel group

| Site | Line | Action |
|---|---|---|
| `PARALLEL_GROUPS.retrieval.boundary.note` — rendered on the diagram | 3520 | "legs run concurrently · the group costs its slowest member" — **keep**. It is qualitative, it explains why the legs are boxed, and it is the same idea the Tail at Scale reference now carries |

### 3.10 About view

| Site | Line | Action |
|---|---|---|
| "The sidebar updates estimated p95 latency, build-and-run complexity, request-path model count, and capability coverage." | 943 | → "The sidebar updates build-and-run complexity, request-path model count, and capability coverage." |
| "watch the comparative budget and capability effects" | 983 | → "watch the comparative complexity and capability effects" |
| "Capture the chosen stages, boundaries, assumptions, degradation order, latency allocation, and evaluation plan." | 995 | Drop "latency allocation," |
| Guardrail "…capacity model, or guaranteed latency calculator." | 1003 | → "…capacity model, or performance predictor." The following sentence ("The estimates are comparative prompts; benchmark the selected design…") still holds for complexity |

## 4. Implementation order

1. **D1 deletion first** — §2, all 12 ripple sites plus the divider re-key and
   step renumber. Load the page; `validateRelationshipModel` throws on a
   missed `REL`/`RREL` edit, so this self-checks before anything else moves.
2. **Schema pass** — delete the 37 remaining `lat:` fields and the comment at 1028.
3. **Model pass** — `budget()`/`renderBudget()` → `composition()`/
   `renderComposition()` with per-phase counts; update the 4529 call site.
4. **Markup and CSS pass** — sidebar panel (830–840, 890), `#latLegend` →
   `#phaseLegend`, `.bar i.lo/.hi`, `.stack`. This is where the UI simplifies.
5. **Render-site pass** — drawer chip 4170, hover pill 4408/4412 (**required
   or the page throws**), node-card comment 3410, focus wording 4032/4046/4098.
6. **Identifier pass** — `CONDITIONAL_BUDGET_NOTE` removal, gate kind
   `budget` → `config`, `RELATION_DETAILS.timing` → `cadence` and its three
   render sites; then the optional gate-scope rename.
7. **Prose pass** — §3.4, §3.5, §3.6, §3.7, §3.10 in file order.
8. **Dead data** — delete `TPL.target` ×3.
9. **`TODO.md`** — tick off "Remove all mentions of timings and ms". The
   "Remove Conditional pill from hoverover" item sits next to this work
   (hover card 4416) but is a separate change; leave it.

## 5. Verification

Automated — expect zero hits:

```bash
grep -nicE 'latenc|p50|p95|p99|budget|millisecond|[0-9] *ms\b|ms/doc' search-query-pipeline-diagram-tool.html
```

Then, if step 6's optional block was done:

```bash
grep -nc 'timing' search-query-pipeline-diagram-tool.html
```

Straggler sweep — every remaining hit should be a *qualitative* use you chose
to keep (deadline, slowest, timely, concurrently, cost):

```bash
grep -noiE '\bdeadline|\bslowest\b|\btimes out\b|\bslow\b|\bfast\b|\bcost\b|\btiming\b' search-query-pipeline-diagram-tool.html
```

Syntax plus the two load-time validators (`validateDependencyBaseline`,
`validateRelationshipModel`), which throw on a stale `REL`, `RREL`, gate kind
or template row:

```bash
sed -n '/^<script>/,/^<\/script>/p' search-query-pipeline-diagram-tool.html | sed '1d;$d' > /tmp/pipe.js && node --check /tmp/pipe.js
```

Manual UI checks across all three levels and with the data-sources toggle on
(`renderComposition` re-runs on every level switch):

1. **Diagram** — no ms anywhere; `Candidate budget allocation` is gone from
   Full surface; the retrieval group has two control ports (`routing`,
   `degradation`) and its wires land cleanly on the boundary; the group note
   still reads "legs run concurrently · the group costs its slowest member".
2. **Build order** — the "Full surface territory" divider still appears, now
   above step 8 (`Optional semantic pre-rerank`); steps run 0–13 with no gap;
   no step chip points at a deleted component.
3. **LHS panel** — "Pipeline shape" shows stage count, complexity, models and
   capability coverage; no latency metric, no stacked bar, no percentages;
   the phase row is a plain filter with counts and still drives focus
   highlighting; *Per-query adaptivity* is still lit at Full surface.
4. **Hover popup** — pills are group + complexity (+ conditional); no p95
   pill; hovering `relax`, `zerofallback` and `vlmrerank` does not throw and
   shows the trigger sentence with no budget suffix.
5. **RHS drawer** — chips are tier, complexity, model, gate, alternative; no
   ms rows in Dials; relation entries read `label — payload · cadence`;
   `Degradation controller`, `Semantic rerank`, `Cross-encoder rerank`,
   `VLM rerank`, `Candidate pruning` and `Retrieval routing` read cleanly end
   to end; `Candidate pruning` still lists the WAND reference.
