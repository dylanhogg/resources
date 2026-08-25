# Plan 005 — Ordering, tier reconciliation and ladder consistency

**Target:** `search-query-pipeline-diagram-tool.html` (4687 lines at `001e29e`) — plus two
companion docs in Phase 5.
**Status:** **Phase 1 landed** — written 25 Aug 2026, after the post-004 gap review.
Phases 2–5 pending.
**Source:** the gap review over the file as it stands after plan 004 landed all four phases.
**Decisions:** D1 scope is ordering + consistency only · D2 authorization stays prose ·
D3 Business ranking moves last · D4 evaluation stays [recommended], the ladder is what changes.

Unlike plan 004, **nothing here adds or removes a component.** The node inventory is
unchanged at 38 in Full surface, 25 in Core + recommended, 9 in Core. That bounds the
risk sharply: only Phase 1 touches the dependency fixture, and it changes edge
_endpoints_ without changing edge _counts_.

Line numbers cited below are as at `001e29e`. Re-grep before acting on any of them.

---

## Phase and status summary

| Phase | Change                                                                         | Fixture? | Size                       | Risk       | Status  |
| ----- | ------------------------------------------------------------------------------ | -------- | -------------------------- | ---------- | ------- |
| **1** | Move `business` to the end of final ranking, after `diversity`                 | **yes**  | 11 sites, 2 templates      | medium     | **done** |
| **2** | Fix the build ladder and give it a validator                                   | no       | 8 sites + 1 new validator  | low-medium | pending |
| **3** | Justify the seven tier divergences from the source classification              | no       | 7 sites, prose only        | lowest     | pending |
| **4** | Retire `altOf`; fix the `expand`/HyDE contradiction; state the annotation rule | no       | 6 sites + 1 CSS rule       | low        | pending |
| **5** | Regenerate the surface doc, and stop it going stale again                      | no       | 2 docs (+ optional export) | low        | pending |

**Implementation order: 1 → 2 → 3 → 4 → 5.** One commit per phase; each must leave the
tool rendering.

Phases 1–4 are genuinely independent — they touch disjoint sites, and none of them
depends on another's output. The numeric order is therefore just convenience, with two
real constraints:

- **Phase 5 must be last.** It is a snapshot of everything the other four change.
- **Phase 4's `altOf` retirement is fully self-contained** and can be lifted out and
  landed on its own at any point, including before Phase 1, if you want the dead code
  gone sooner. It is the only piece of this plan that deletes rather than corrects.

**One open question**, in §2.1: whether the harness rung keeps its position at the top of
the ladder or moves down and forces a renumber. A recommendation is given; it does not
block starting the phase.

---

## 0. The constraint that governs Phase 1

Two validators run at module load and **throw**, blanking the page:

| Validator                      | Line | What it guards                                                                                                          |
| ------------------------------ | ---- | ----------------------------------------------------------------------------------------------------------------------- |
| `validateDependencyBaseline()` | 2719 | Every derived edge must match `DEPENDENCY_BASELINE` (2535) exactly — missing, unexpected and duplicate edges all throw. |
| `validateRelationshipModel()`  | 2769 | Gate reciprocity, serving-source orphans, relation-detail completeness, routing↔gated-reranker symmetry.                |

There is no partially applied phase: either the fixture moves in the same edit as the
template rows, or the tool does not render.

**What makes Phase 1 safer than plan 004's fixture phases:** moving one component along
a linear chain is a _permutation_, not an insertion. The tail of `BASELINE[2].flow` has
five pairs before and five after; `BASELINE[3].flow` has six and six. No count assertion
moves, `expectedSourceCounts` and `expectedServingCounts` are untouched, and no relation
in `REL` names `business` at all — it is a pure spine component.

Phases 2–5 do not touch `DEPENDENCY_BASELINE`. `STEPS` is **not** covered by either
validator, which is precisely the defect Phase 2 fixes.

---

## Phase 1 — Move Business ranking to the end of final ranking

### 1.1 The problem

`business` currently runs **first** in final ranking:

```
crossenc → business → freshness → dedup → diversity → assembly            (template 2)
ltr      → business → freshness → personalisation → dedup → diversity → assembly   (template 3)
```

Its own `purpose` (1655) says it applies preferences "after relevance ranking", and its
contract (1662) promises `applied_rules[]` is "recorded" — but four or five more re-sorts
run after it. The audit trail therefore describes an ordering the user never sees. If a
promoted cabin is boosted to position 2 and then dropped to position 7 by the diversity
cap, `applied_rules[]` still claims the boost, and the compliance answer the component
exists to give is wrong.

### 1.2 The target order

```
crossenc → freshness → dedup → diversity → business → assembly                     (template 2)
ltr      → freshness → personalisation → dedup → diversity → business → assembly   (template 3)
```

Business ranking becomes the **last thing that touches the ordering** before assembly.
That is what makes `applied_rules[]` honest: it now describes the delivered page.

**What this plan explicitly does not do.** The deeper reading of this defect is that
final ranking should not be a chain of five sequential re-sorts at all — most production
systems compute one blended score and then run post-processing passes. That reframe was
considered and **rejected for this plan** (D3): it changes the diagram's central visual
metaphor, and the ordering fix delivers the audit-trail correctness on its own.
Recorded in §6.

### 1.3 The eleven sites — all applied

| #   | Site                    | Line   | Change                                                                                          |
| --- | ----------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| 1   | `TPL[2].rows`           | 2150   | Move `{c:["business"]}` to sit after `{c:["diversity"]}` (2153)                                 |
| 2   | `TPL[3].rows`           | 2179   | Move `{c:["business"]}` to sit after `{c:["diversity"]}` (2183)                                 |
| 3   | `BASELINE[2].flow` tail | 2563–7 | Rewrite the five pairs                                                                          |
| 4   | `BASELINE[3].flow` tail | 2604–9 | Rewrite the six pairs                                                                           |
| 5   | `STEPS[6].stages`       | 2993   | `["business","freshness","dedup","diversity"]` → `["freshness","dedup","diversity","business"]` |
| 6   | `STEPS[6].build`        | 2994   | Reorder the clause list to match                                                                |
| 7   | `business.contract.in`  | 1662   | `"ranked candidates"` → `"diversified ranking"`                                                 |
| 8   | `business.contract.out` | 1662   | Name the output as the final ranking, not a re-order                                            |
| 9   | `business.purpose`      | 1655   | Sharpen "after relevance ranking" to "last, over the delivered page"                            |
| 10  | `business.notes`        | 1663   | Add the reason the stage runs last                                                              |
| 11  | `business.failures`     | 1664   | Add the diversity-cap interaction row                                                           |

Sites 7–9 mirror what plan 004 Phase 1 did for `diversity.contract.in` when `dedup`
moved ahead of it: the contract seam has to move with the component, or the chain reads
as a sequence of unconnected stages.

### 1.4 Fixture delta

`BASELINE[2].flow`, replacing lines 2563–2567:

```js
(["crossenc", "freshness"],
  ["freshness", "dedup"],
  ["dedup", "diversity"],
  ["diversity", "business"],
  ["business", "assembly"]);
```

`BASELINE[3].flow`, replacing lines 2604–2609:

```js
(["ltr", "freshness"],
  ["freshness", "personalisation"],
  ["personalisation", "dedup"],
  ["dedup", "diversity"],
  ["diversity", "business"],
  ["business", "assembly"]);
```

Five pairs before and after in template 2; six and six in template 3. Hidden edge counts
must be **unchanged** at 34 and 72 — that is the sharpest single check on this phase.

### 1.5 Prose deltas

**`business.contract`** — the input is now the diversified ranking, and the output is the
ranking that assembly hydrates:

```js
  contract:{in:"diversified ranking", out:"final ranking with business_score and applied_rules[] recorded against delivered positions"},
```

**`business.purpose`** — it should say why last:

> Apply soft, auditable preferences the business owns — quality tiers, capped promotions
> and other non-mandatory boosts — as the last stage that changes the order, so the
> recorded rules describe the page the user actually sees.

**`business.notes`** — keep the existing note about hard rules belonging in candidate
filtering (it is the load-bearing half of D2, since authorization has no node of its
own), and add a second note:

> This stage runs last in final ranking for one reason: an audit trail is only worth
> keeping if it describes the delivered order. A boost applied before diversity capping
> or deduplication can be silently undone by them, leaving `applied_rules[]` claiming a
> promotion the page does not show.

**`business.failures`** — add the interaction the move creates. Running last does not
make the stage free; it moves the risk:

```js
{f:"Boosts reintroduce what diversity and dedup removed",s:"Promoted near-duplicates or same-host results climb back into the top-10 after capping",m:"per-attribute distribution of the top-10 before and after business ranking"}
```

**`STEPS[6].build`** — reorder to match, keeping the "after relevance ranking" framing on
the block as a whole:

> Apply intent-aware recency, near-duplicate collapse and per-attribute diversity after
> relevance ranking, then capped soft business preferences last, so the audit trail
> records the delivered order.

### 1.6 Verification — **run, all sites applied**

Measured at 1500x1000 against a same-viewport run of `001e29e` extracted with `git show`,
so every "unchanged" below is a real before/after comparison rather than a recalled number.

| Check                                       | Expected                                                    | Result |
| ------------------------------------------- | ----------------------------------------------------------- | ------ |
| `dependencyDiagnostics.validation`          | passes                                                      | pass — page renders, so neither load-time validator threw |
| `dependencyDiagnostics.modelValidation`     | passes                                                      | pass |
| Hidden edges, templates 2 / 3               | **34 / 72 — unchanged**                                     | 34 / 72 |
| `componentRelations` / `annotatedRelations` | **38 / 5 — unchanged**                                      | 38 / 5 |
| Node counts, templates 1 / 2 / 3            | **9 / 25 / 38 — unchanged**                                 | 9 / 25 / 38 |
| Serving sources / edges                     | **6 / 7 — unchanged**                                       | 6 / 7 |
| Wire counts, six sweep combinations         | unchanged                                                   | 15 / 18 / 35 / 39 / 57 / 64 — identical to baseline |
| `+N new vs …` pills                         | **+16 / +13 — unchanged**                                   | +16 / +13 |
| Build ladder rungs                          | **14 — unchanged in this phase**                            | 14 |
| `Rank & assemble` phase reading             | **6 (t3) / 5 (t2) — unchanged**                             | 6 / 5 |
| Layout sweep, all six combinations          | zero overlaps                                               | zero |
| Clipped sidebar labels                      | unchanged                                                   | same 2 in t3 as baseline (`Image-to-image vector retrieval`, `Multi-vector / passage retrieval`) — pre-existing |
| Edge-label collisions, all three templates  | zero                                                        | **one 4px graze at 1500px — see below** |
| Node-face overflow at 1500px and 375px      | zero                                                        | zero, and no horizontal document overflow at 375px |
| Contract chain across the moved seam        | connected                                                   | `dedup` -> `diversity` "diversified ranking" -> `business` -> `assembly` "final ranking" |
| Visual                                      | `business` sits directly above `assembly` in both templates | confirmed in both |

Every count except the edge-label row came back _unchanged_, which is what a permutation
should produce.

#### The one deviation: a 4px label graze

At 1500px with data sources on, template 3, the `behavioural --trains--> ltr` relation
label (`debiased training set`) overlaps the right border of the `Deduplication` node by
**4px horizontally, 10px vertically**. The 4px falls entirely inside the label's own 6px
left padding, so no text is obscured — but `.elabel` has an opaque background, so it
erases a short section of the node's right border.

**This is the deferred relation-label problem, not a Phase 1 regression.** The evidence:

- The label did not move. `Deduplication` moved up one row into the label's path, because
  `business` vacated the row above it.
- At **1280px the baseline already has seven** label-on-node overlaps and this branch has
  eight. Six are identical in both. The two that differ are the same two labels landing on
  whichever final-ranking node now occupies that row — baseline puts `debiased training
  set` on `Personalisation` and `profile events` on `Deduplication`; this branch puts them
  on `Deduplication` and `Diversity`.

So relation-detail labels already collide with nodes at common widths; the ordering change
shuffled which node each one grazes, and at 1500px specifically it moved one from clear to
touching. Fixing it properly means giving relation labels a collision-avoidance nudge,
which is the item plan 004 deferred and §6 below carries forward. It was **not** fixed
here: Phase 1's premise is that nothing structural changes, and adding a layout algorithm
inside it would break that premise.

The verification row above is left as `zero` on purpose — it records the standard the
phase was measured against, and the standard was missed by 4px.

---

## Phase 2 — Fix the build ladder and give it a validator

### 2.1 `STEPS[0]` claims a precedence the tier system contradicts

Rung 0 (2969–2973) is `t:0`, titled "Evaluation harness", with
`trigger:"Before anything else."` and stages `["offlineeval","experiment"]`. Both of
those components are `tier:"recommended"`, `intro:2` — they do not appear until template 2.

So the ladder says build it before Core; the diagram says it arrives with the recommended
tier. Under **D4** the components keep their tier and the ladder is what changes.

There is a second, quieter symptom. The CSS colours the rung number by tier —
`.step[data-t="1"|"2"|"3"] .num` at 614–616 — and **there is no rule for `data-t="0"`**.
Rung 0 has been rendering with an unstyled badge since it was added. `t:0` is not a tier;
it is a value invented for one rung and then not carried through.

**Recommended fix:** keep the rung where it is, at the top of the ladder, and set `t:2`.
"Step zero — build this before you start tuning" is a real idiom and the argument is
good; what has to go is the claim that it precedes Core, which the tier system flatly
contradicts. Reword:

```js
  {n:0, t:2, title:"Evaluation harness", stages:["offlineeval","experiment"],
   trigger:"Alongside the Core spine, and before the first tuning decision.",
```

`buys` stays as it is — "Without it, all the tradeoffs below are opinions" is still true
and is the reason the rung sits first.

**The open question:** the alternative is to move the harness rung down beside rung 2
(instrumentation, also `t:2`) and renumber the whole ladder. That is tidier — the ladder
would then be strictly non-decreasing in `t` — but it renumbers fourteen rungs, breaks
the `(s.n||"0")` fallback that exists only to render `n:0`, and moves the divider
arithmetic at 4459. It also loses the "measure first" rhetoric, which is one of the
better arguments the file makes. **Recommendation: keep the position, take the reword.**

### 2.2 `fusionpolicy` appears in two rungs

Rung 10 (3007) is "Query-dependent fusion", stages `["fusionpolicy"]`. Rung 13 (3019)
lists `fusionpolicy` again among seven stages. Every other component in the ladder
appears exactly once.

The build view renders `stages` as clickable chips (4445–4446), so `Fusion policy`
appears twice in the ladder and both chips navigate to the same node. The intent was
probably heuristic-policy-then-learned-policy, but the ladder has no mechanism for
"the same component, upgraded" and does not say that anywhere.

**Fix:** remove `fusionpolicy` from rung 13's stages. Rung 10 keeps it. If the
heuristic→learned progression is worth stating, it belongs in rung 13's `build` prose as
a sentence, not as a duplicate chip.

### 2.3 `STEPS[13]` is a grab-bag

```js
  {n:13, t:3, title:"LTR, learned fusion policy, personalisation",
   stages:["ltr","fusionpolicy","personalisation","expand","sparse","session","confidence"],
   build:"Trained ranking over the full feature vector, a trained per-query Fusion policy, and per-user adjustment.",
```

Seven stages, a title naming three, and a `build` describing three. `expand`, `sparse`,
`session` and `confidence` are unmentioned in both — they are four unrelated components
swept into the last rung because the ladder ran out of rungs. A reader clicking the
`Learned sparse retrieval` chip lands on a retrieval leg from a rung about learned ranking.

**Fix:** with `fusionpolicy` removed per §2.2, split the remaining six into three coherent
rungs:

| Rung | Title                                        | Stages                   | Why these belong together                                                              |
| ---- | -------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------- |
| 13   | Learned ranking and personalisation          | `ltr`, `personalisation` | Both are trained, both need behavioural data, both carry a position-bias problem       |
| 14   | Learned-sparse retrieval and query expansion | `sparse`, `expand`       | Both attack vocabulary mismatch, and `expand` feeds `sparse` directly (`REL.expand`)   |
| 15   | Session context and constraint confidence    | `session`, `confidence`  | Both are control-plane context that steers earlier stages rather than ranking anything |

The ladder goes from 14 rungs to 16. Rung 13's existing `trigger` ("Tens of thousands of
labelled or behavioural judgements exist") and `buys` stay with rung 13, which is where
they were always true; rungs 14 and 15 need their own.

The divider at 4459 is keyed to `STEPS.find(s=>s.t===3)`, not to a fixed number, so the
split does not touch it — that was a deliberate defence when the build view was written
and this is the first change to exercise it.

### 2.4 The validator

`STEPS` is the only structural list in the file with no load-time check. That is why
§2.1, §2.2 and §2.3 all survived four plans. Add `validateBuildLadder()`, called
alongside the other two at 2958–2959, asserting:

1. Every id in every `stages` array is a known component.
2. **No component appears in more than one rung** — catches §2.2.
3. Every component except `q-text` appears in exactly one rung. `q-text` is the query
   itself and is deliberately unbuilt; assert it by name so the exemption is visible
   rather than implied.
4. **Each rung's `t` equals the highest `intro` among its stages** — catches §2.1, and
   makes `t:0` unrepresentable.

Rule 4 is the valuable one, and it is worth noting that **it already holds for every
rung except rung 0**. Checked against the current file: rungs 1–13 all satisfy
`t === max(intro of stages)` exactly. The invariant is not being imposed on the data; it
is being written down after the fact, which is the same shape as the `tier`/`intro`
correlation the file already relies on.

This validator throws like the others. Adding it and fixing rung 0 must therefore land in
the same commit.

### 2.5 Verification

| Check                               | Expected                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| All three validators                | pass                                                         |
| Build ladder rungs                  | **14 → 16**                                                  |
| `Fusion policy` chips in the ladder | **2 → 1**                                                    |
| Rung 0 badge                        | styled as recommended (`--rec-bg`), not the unstyled default |
| Divider text                        | still reads "Steps 1–7 above are Core + recommended"         |
| Every chip in rungs 13–15           | navigates to a node in template 3                            |
| Diagram node counts, all templates  | unchanged — `STEPS` does not feed the diagram                |

---

## Phase 3 — Justify the seven tier divergences

### 3.1 The divergences

`1-search-query-pipeline-components-list.md` is the source classification the tool was
built from. Seven entries now disagree with it:

| Component              | Source list                      | Tool                                              |
| ---------------------- | -------------------------------- | ------------------------------------------------- |
| Fusion                 | recommended                      | **core**                                          |
| Business ranking       | core                             | **recommended**                                   |
| Deduplication          | core                             | **recommended**                                   |
| Offline evaluation     | core                             | **recommended**                                   |
| Semantic rerank        | recommended                      | **optional**                                      |
| Image vector retrieval | recommended (one entry)          | **split**: text-to-image rec / image-to-image opt |
| Retrieval routing      | optional, under Query processing | **recommended**, under Candidate retrieval        |

Every one of these is a deliberate improvement made during plans 001–004. None of them is
recorded anywhere. A reader holding both documents just sees two files that disagree.

### 3.2 The fix, in two halves

**Half one — justify each tier in the tool, on its own merits.** The note should never
mention the source list; a reader of the diagram has not seen it. Add one `notes` entry
to each of the seven:

- **Fusion → core.** Two legs with incomparable score scales have no defined ordering
  until something combines their ranks. A hybrid pipeline without an explicit fusion rule
  does not have a relevance ordering, it has two of them.
- **Business ranking → recommended.** Core is the smallest genuinely hybrid retrieval
  path. Business preference is an organisational requirement rather than a retrieval one,
  and a pipeline that promotes nothing is coherent.
- **Deduplication → recommended.** `union` already merges by `doc_id`, so Core cannot emit
  the same document twice. This stage collapses _near_-duplicates — distinct documents
  with near-identical content — which is a page-quality improvement, not a correctness fix.
- **Offline evaluation → recommended.** The tier says when a component appears in the
  request-path diagram, and the harness is not on the request path. When you should
  _build_ it is the build ladder's question, and rung 0 answers it (see §2.1).
- **Semantic rerank → optional.** It earns a place only when cross-encoder depth is the
  binding constraint on the funnel. Recommending it by default adds a stage most pipelines
  measure and then remove.
- **Image vector retrieval → split.** The two legs share an index but not a trigger: one
  fires on visual intent expressed in text, the other requires an uploaded image. Tying
  them to one tier would force image-to-image on pipelines that accept no images.
- **Retrieval routing → recommended, under Candidate retrieval.** It decides which legs
  fire, so it belongs beside the legs rather than in query processing. And once more than
  two legs exist, firing all of them on every query is a cost decision no production
  pipeline actually makes.

**Half two — reconcile the source doc.** Update the tiers in
`1-search-query-pipeline-components-list.md` to match, and add a line under the heading
recording that the tool is the source of truth and the list is the input it was built from.
Do **not** delete the list's original classification wholesale — it is a record of where
the design started.

### 3.3 Verification

Prose only; no counts move. Re-read the seven drawers end to end and confirm the sidebar
tier chips are unchanged. The point of this phase is that the tiers do **not** change —
only their justification is added.

---

## Phase 4 — Retire `altOf`, fix `expand`, state the annotation rule

### 4.1 `altOf` is dead

Plan 004 Phase 2 deleted `Late-interaction retrieval`, which was the only pair the
`altOf` mechanism ever described. It has been carried since as dead machinery. Six sites:

| Site                                            | Line      | Action                                              |
| ----------------------------------------------- | --------- | --------------------------------------------------- |
| `.node .altchip` CSS rule                       | 347–350   | Delete                                              |
| `defGate()` signature and doc comment           | 2329–2337 | Drop the fourth parameter and the two comment lines |
| `validateRelationshipModel()` reciprocity check | 2821–2826 | Delete the whole `if(gate.altOf)` block             |
| Node-face chip render                           | 3352      | Delete the trailing ternary                         |
| Drawer chip render                              | 4123–4124 | Delete the trailing ternary                         |

No `GATED_EXECUTION` entry passes a fourth argument any more, so this is a pure deletion —
about 20 lines plus the CSS rule. Nothing else reads `altOf`.

This is the only destructive edit in plan 005 and it is entirely self-contained. Land it
separately if you prefer.

### 4.2 `expand` advertises HyDE and then routes away from it

`expand.sub` (1159) is `"synonyms, PRF, HyDE, LLM variants"`. Its second decision (1167–1172)
defaults to `"Lexical and learned-sparse legs"` because `"Dense retrieval already
generalises over vocabulary. Expanding into it mostly adds drift."`

HyDE is a dense technique specifically — a hypothetical answer document exists in order to
be embedded. As written, the component names a method in its subtitle that its own default
routing makes unreachable. `REL.expand` confirms it: `feeds:["lexical","sparse"]`, nothing
dense.

The routing default is well argued and should stand. **Fix the contradiction in the
prose**, not the wiring: either drop HyDE from `sub` and mention it in the decision's
`opts` as the dense-only option it is, or keep it in `sub` and add a sentence to the
decision's `why` explaining that HyDE is the one expansion family that must target the
dense legs, which is exactly why it is not the default. The second is more useful — it
turns an inconsistency into the sharpest point on the card.

### 4.3 The annotation rule is implicit

`RELATION_DETAILS` (2383) carries five entries out of 38 component relations. The rule
enforced at 2879–2885 is "every relation out of `assembly` or `behavioural` must have a
complete detail". That rule is real and load-bearing, but it exists only as two hardcoded
string comparisons inside the validator, and nothing states the corresponding negative:
that steering relations are deliberately unannotated because the drawer already carries
the detail.

`experiment` makes the gap visible. It both steers (`routing`, `fusionpolicy`) and feeds
(`assembly`), and its feed is unannotated — correctly, per plan 004's finding that the
label would land in the middle of the retrieval fan-out, but a reader cannot tell that
from the code.

**Fix:** lift the two names into a declared constant and comment the rule at both ends:

```js
/* The feedback plane is the only plane whose wires must declare what they carry:
   these relations move data the reader cannot infer from the component cards.
   Control-plane relations (steers) are deliberately unannotated — the drawer
   already names what each control decides, and a label on every steering wire
   would bury the diagram. `experiment` sits on both planes and is annotated on
   neither: its feed to assembly runs the full height of the diagram, so a label
   would land in the retrieval fan-out. */
const FEEDBACK_PLANE_SOURCES = Object.freeze(["assembly", "behavioural"]);
```

No behaviour changes — `annotatedRelations` stays at 5.

### 4.4 Verification

| Check                               | Expected                                  |
| ----------------------------------- | ----------------------------------------- |
| All validators                      | pass                                      |
| `grep -c altOf`                     | **0**                                     |
| `grep -c altchip`                   | **0**                                     |
| `annotatedRelations`                | **5 — unchanged**                         |
| Gate chips on all eight gated nodes | still render; no node loses its gate chip |
| `expand` drawer                     | reads consistently top to bottom          |

---

## Phase 5 — Regenerate the surface doc, and stop it going stale

### 5.1 How stale it is

`search-query-pipeline-diagram-tool-surface.md` is a flattened outline of template 3. It
carries its own disclaimer — "not the source of truth, a point in time extraction" — and
it has now drifted in five ways:

| Drift                                                                   | Introduced by             |
| ----------------------------------------------------------------------- | ------------------------- |
| Per-stage timings on every line (`0 ms`, `10–40 ms`, …)                 | plan 003 removed them     |
| `Late-interaction retrieval` as a seventh retrieval leg                 | plan 004 phase 2          |
| `Token-vector MaxSim index` as a serving source                         | plan 004 phase 2          |
| `Candidate budget allocation` as a retrieval-group control              | never existed in the tool |
| No `Experiment assignment`, `Offline evaluation`, or `Evaluation` phase | plan 004 phase 4          |
| "quality floor" wording in the sufficiency check                        | plan 004 phase 3          |

The fourth is worth a note of its own. **Candidate budget allocation** is in the source
classification list and in this doc, but has never existed as a component in the tool.
Regenerating the doc silently drops it. That is the right outcome for this plan — D1 puts
new components out of scope — but it should be recorded rather than quietly lost, so §6
carries it.

### 5.2 Regenerate, and consider generating

The doc has gone stale after three consecutive plans. Hand-maintenance is not holding.

**Recommended:** add a `surfaceOutline(templateId)` function to the tool that walks
`TPL[t].rows` and emits the outline, and expose it on `window.dependencyDiagnostics`
alongside the existing five keys. Regenerating the doc then becomes: load the page, call
it, paste. The tool already has every fact the doc contains — group, tier, gate condition,
`sub` line, `REL` edges, serving source — and the outline's structure is a direct read of
the row layout it already renders.

**Fallback:** rewrite it by hand from the post-Phase-4 file, and add a line to the header
naming the commit it was extracted at, so the next reader can tell how stale it is without
diffing.

Either way the doc must be regenerated **after** phases 1–4, since all four change it:
Phase 1 moves business ranking, Phase 2 does not appear in it, Phase 3 changes no tiers,
and Phase 4 removes the "alternative placement of" line that `altOf` used to render.

---

## 5. Commit boundaries

One commit per phase, each rendering:

| Commit | Contents                                                                                    |
| ------ | ------------------------------------------------------------------------------------------- |
| 1      | Phase 1 — templates, fixture, `STEPS[6]`, `business` prose — **applied, not yet committed** |
| 2      | Phase 2 — rung 0, the `fusionpolicy` duplicate, the 13/14/15 split, `validateBuildLadder()` |
| 3      | Phase 3 — seven tier notes, plus the source-list reconciliation                             |
| 4      | Phase 4 — `altOf` deletion, `expand` prose, `FEEDBACK_PLANE_SOURCES`                        |
| 5      | Phase 5 — regenerated surface doc (+ `surfaceOutline()` if taken)                           |

Phase 2's validator and its rung-0 fix cannot be split — the validator throws on the
current data.

---

## 6. Out of scope

Everything in group A of the gap review except where it overlapped, per **D1**. Recorded
here so the reasoning is not lost:

- **Authorization / eligibility filtering as a component.** Per **D2**, it stays prose.
  Three components refer to a filtering stage that has no node — `business.notes`,
  `zerofallback`, `prefilter`. If the tool ever targets enterprise or multi-tenant search,
  this is the first thing to add.
- **Caching.** Nineteen mentions, all `Cache TTL` dials on individual components. No cache
  lookup, no short-circuit, no cache plane. Arguably the largest latency and cost lever in
  a modern pipeline, and currently invisible.
- **Model training loop.** `behavioural.trains` reaches `fusionpolicy` and `ltr` only.
  Nothing trains the embedders or the cross-encoder. Hard-negative mining and relevance
  feedback are both in the source list and both absent.
- **Candidate budget allocation.** In the source list and in the surface doc; never in the
  tool. Phase 5 drops it from the doc. Either it was folded into `routing` + `degradation`
  deliberately — in which case one sentence somewhere should say so — or it went missing.
- **A generative / RAG consumer.** The pipeline ends at Results assembly. Defensible as
  scope, but currently silent rather than stated.
- **Reframing final ranking as a blended score plus post-processing passes.** The root
  cause behind Phase 1; rejected per **D3** because it changes the diagram's central
  metaphor.
- **The `TODO.md` tool features** — URL state params, mermaid export, sidebar ordered by
  `PHASES`, labels on hover. A different kind of plan: tool behaviour, not pipeline content.

### Carried forward from plan 004

- **Relation-label collision avoidance.** _Promoted from "lowest priority" — the premise
  it rested on turned out to be false._ Plan 004 recorded that nothing collides in any
  template. Measuring properly during Phase 1 showed that was only true at 1500px: at
  **1280px, `001e29e` already has seven** relation-label-on-node overlaps across templates
  2 and 3, several of them 40px+ (`impression context` over both `Results assembly` and
  `Behavioural event log`, `conditional legs` over `Retrieval routing`). The check that
  produced "zero" was run at one width.

  Relation-detail labels are placed at a fixed fraction (`position: .56`) along the wire
  and centred with `translate(-50%,-50%)`, with no awareness of what is underneath. As the
  column narrows, the right-hand control-plane wires pass closer to the spine and the
  labels ride onto the nodes. Phase 1 did not create this and did not fix it; it changed
  which node two of the labels graze, and added a 4px graze at 1500px (§1.6).

  A fix wants two things: a real check that sweeps **widths** as well as templates and
  data-source settings, and either a nudge-off-collision pass over `.elabel` after
  placement, or a per-relation `position` override. Worth its own plan.
