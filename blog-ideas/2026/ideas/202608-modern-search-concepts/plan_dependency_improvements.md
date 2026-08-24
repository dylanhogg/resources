# Dependency Diagram Improvements Plan

## Phase 0 — Confirm semantics and freeze invariants

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Added a manually specified logical relationship fixture for Core, Core + recommended, and Full.
- Added a pure logical inventory builder that is independent of SVG layout and group-edge collapsing.
- Added startup validation for hidden and shown source states, duplicate/missing/unexpected relationships, disabled endpoint removal, source deactivation, and expected serving counts.
- Exposed read-only `window.dependencyDiagnostics.inventory(...)` and validation summaries for local diagnostics.
- Verified in the rendered UI that showing sources leaves latency and complexity unchanged in every template.

Verified logical inventories:

| Template           | Sources hidden | Sources shown | Source nodes | `serves` edges |
| ------------------ | -------------: | ------------: | -----------: | -------------: |
| Core               |        9 edges |      12 edges |            3 |              3 |
| Core + recommended |       31 edges |      35 edges |            4 |              4 |
| Full               |       76 edges |      84 edges |            7 |              8 |

Discoveries:

- Logical inventories must be tested separately from SVG path counts. The renderer intentionally collapses group-wide steering relationships, so Full currently draws 58 non-serving paths for 76 logical non-serving relationships.
- A shown data-source card remains present but becomes inactive when all of its consumers are disabled; its serving wires disappear. This is the intended meaning of “deactivates source.”
- The image ANN source correctly remains active with either visual retrieval consumer enabled and becomes inactive only when both are disabled.
- The fixture deliberately duplicates the expected retrieval-leg IDs instead of importing the production `LEGS` constant. This prevents a future production edit from silently changing its own expected baseline.

Before editing, document and preserve these invariants:

- Core remains the irreducible, always-on path.
- Core + recommended remains the production target.
- Full remains a catalogue of gated capabilities, not “run everything.”
- Hiding data sources removes only source nodes and `serves` edges. It must not change pipeline flow, control relationships, enabled components, or budgets.
- Showing data sources adds the following relationships:

  | Template           | Source nodes | `serves` edges |
  | ------------------ | -----------: | -------------: |
  | Core               |            3 |              3 |
  | Core + recommended |            4 |              4 |
  | Full               |            7 |              8 |

- Disabling a component removes its incoming and outgoing relationships and makes a source inactive when none of its consumers remain enabled.

Create an expected-relationship fixture for each template before changing the renderer. This provides a regression baseline for later phases.

**Acceptance criteria:** All existing relationships produce the same endpoint inventory in all three templates, with data sources hidden and shown.

## Phase 1 — Normalize the relationship model

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Added one canonical `RELATION_TYPES` registry for all seven relationship kinds, including public forward/reverse labels, descriptions, colours, dash patterns, line caps, markers, opacity, width, routing mode, accessibility wording, and legend metadata.
- Removed the rendered aliases `ctrl`, `train`, and `feedback`. SVG paths, logical inventories, drawers, diagnostics, and legend metadata now use the canonical names `steers`, `trains`, and `feeds`.
- Rebuilt the visible relationship legend from registry metadata. Its three Phase 1 entries and appearance remain unchanged; completing the legend remains Phase 2 work.
- Separated logical dependencies from render targets. Render edges now target typed nodes, groups, or data sources and retain the logical relationships represented by a collapsed group path.
- Added `data-logical-count` and generated accessibility labels to rendered paths, so a group-level path remains traceable to its represented endpoints.
- Added startup model validation for unknown endpoints, unsupported or duplicate relations, endpoints that never coexist, sources without consumers, incomplete visual or legend definitions, conflicting marker colours, and stale `RREL` or `RSERVING` reverse indexes.

Verification:

- Phase 0 inventories remain unchanged: Core 9/12, Core + recommended 31/35, and Full 76/84 logical edges with sources hidden/shown.
- Rendered path counts remain unchanged: Core 9/12, Core + recommended 27/31, and Full 58/66 with sources hidden/shown.
- Latency and complexity values are unchanged by the refactor or source visibility.
- All six template/source combinations loaded without browser warnings or errors.
- Line colours, dash patterns, caps, opacity, and arrow colours match the pre-refactor treatments.
- Component drawers use the same `Steers`, `Trains`, `Feeds`, `Steered by`, `Trained by`, and `Fed by` terminology as the canonical registry.

Discoveries:

- The largest current group path represents seven logical steering relationships. Keeping `logical[]` on the render edge resolves the previous loss of endpoint information during SVG collapsing.
- The former `feedback` name existed only in rendering; the data model and drawers already called the relationship `feeds`. Canonical naming therefore required no semantic change.
- The generic group render target is ready for a future gated-reranker group, but Phase 1 does not invent that grouping before the Phase 4 semantic decisions are made.
- Legend visibility is now data-driven, so Phase 2 can expose the four missing relationship types without another renderer refactor.

Refactor relationship metadata without changing the visual output yet.

- Use the same public relation names throughout the data model, renderer, drawers, and legend.
- Rename the internal rendered kind `feedback` to `feeds`.
- Centralize each relationship type’s:
  - display name
  - description
  - CSS colour
  - dash pattern
  - arrow marker
  - opacity
  - accessibility label
  - legend visibility

- Separate logical relationships from rendering targets. A dependency can target:
  - an individual node
  - the parallel retrieval group
  - a gated reranker group
  - a data source

- Add validation for:
  - unknown source or target IDs
  - relations whose endpoints do not coexist in any template
  - duplicate relations
  - missing visual or legend definitions
  - serving sources without consumers
  - stale drawer reverse relationships

**Acceptance criteria:** All existing relationships retain the same logical endpoints in all three templates, and drawers and rendered edges use the same terminology.

## Phase 2 — Complete and clarify the legend

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Expanded the relationship legend from three entries to all seven canonical kinds: flow, branch, returns, steers, feeds, trains, and serves.
- Added the missing data-plane/source node swatch and separated the legend into clearly labelled Nodes and Relationships sections.
- Replaced approximate CSS border samples with inline SVG samples generated from `RELATION_TYPES`. Legend strokes, arrowheads, widths, dash arrays, line caps, colours, and opacity now use the same metadata as the rendered connectors.
- Assigned a distinct redundant treatment to every relationship kind. In particular, `feeds` now uses a purple dash-dot line, `serves` a slate dotted line, `trains` a teal long dash-dot line, and `steers` a short grey dash.
- Preserved orange solid recovery lines, while increasing ordinary flow and control-line contrast in both light and dark themes.
- Added concise semantic accessibility phrases to the canonical registry. Rendered paths now read as, for example, “Request-time steering from Retrieval routing to parallel group…” and labelled branches append their outcome.
- Strengthened startup validation so every relationship kind must provide accessibility wording as well as complete visual and legend metadata.

Verification:

- All seven legend samples exactly match a rendered connector of the same kind for stroke token, width, dash pattern, line cap, and opacity.
- Every one of the 66 paths in Full with sources shown has a descriptive `aria-label`; the seven-target collapsed steering path retains all represented target names.
- Phase 0 logical inventories remain unchanged: Core 9/12, Core + recommended 31/35, and Full 76/84 with sources hidden/shown.
- Rendered path counts remain unchanged: Core 9/12, Core + recommended 27/31, and Full 58/66 with sources hidden/shown.
- Latency and complexity remain unchanged across all six template/source combinations.
- Light- and dark-theme browser inspection showed all seven treatments remain distinguishable, with no browser warnings or errors.

Discoveries and decisions:

- A single-column relationship list is clearer than a two-column layout in the 288 px sidebar because each sample remains adjacent to an unbroken label.
- Rendering legend samples from the canonical registry eliminates a subtle maintenance risk: a CSS approximation can drift even when its name remains correct.
- The `serves` key remains visible while data sources are hidden. This explains the existing “Show data sources” control before activation and keeps the legend stable when the source layer is toggled.
- Dark-theme legend text needed the stronger secondary-text token after visual inspection; the line treatments themselves did not require further adjustment.

Add all rendered dependency types to the legend:

- request/candidate flow
- conditional branch
- recovery/retry
- request-time steering
- non-spine data feed or telemetry emission
- offline training
- external serving source

Also add the missing data-plane/source node swatch.

Improve visual redundancy:

- Do not rely on colour alone.
- Give branch, steering, and training patterns visibly different rhythms.
- Make `feeds` and `serves` distinguishable by both pattern and colour.
- Keep the orange recovery treatment.
- Increase ordinary flow contrast slightly.
- Ensure legend samples exactly match rendered stroke width, dash pattern, and opacity.

Add descriptive accessibility text such as “request-time steering from Retrieval routing to the retrieval group.”

**Acceptance criteria:** Every visible node and connector treatment can be decoded solely from the legend.

## Phase 3 — Fix parallel retrieval geometry

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Promoted parallel retrieval to a measured, first-class layout object with explicit bounds, visual rows, active members, fan-out and fan-in lanes, and named controller ports.
- Replaced the predecessor-to-every-leg and every-leg-to-union curves with one semantic fan-out edge and one semantic fan-in edge. Each uses a shared bus plus short branches to the enabled retrieval cards.
- Kept all represented logical endpoints on the two collapsed render edges. Decorative bus scaffolding shares the same hover identity but is hidden from the dependency inventory and accessibility tree.
- Added visible “retrieval fan-out” and “candidate fan-in” captions so the concurrent execution model does not depend on interpreting geometry alone.
- Terminated group-wide steering at labelled ports: Retrieval routing → “gates legs”, Candidate budget allocation → “allocates k”, and Degradation controller → “cuts or skips”.
- Made controller ports choose the top, bottom, left, or right side from the controller's measured position, so the same model works when controls move around the group.
- Reserved a small routing rail around the retrieval cards and recalculated buses from the cards' actual wrapped rows. Showing data sources, changing template, resizing, or disabling legs therefore causes a fresh layout rather than reusing stale coordinates.
- Preserved individual `serves` connectors to their actual retrieval consumers.
- Routed the two recovery returns through separate lanes outside the measured retrieval group, preventing recovery paths from crossing retrieval cards.

Verification:

- Phase 0 logical inventories remain unchanged: Core 9/12, Core + recommended 31/35, and Full 76/84 with sources hidden/shown.
- The clearer collapsed geometry intentionally reduces semantic SVG paths to Core 7/10, Core + recommended 23/27, and Full 46/54 with sources hidden/shown. Summing `data-logical-count` still produces the Phase 0 inventories exactly.
- Latency and complexity remain unchanged across all six template/source combinations.
- Full was verified with a two-column/four-row retrieval wrap when sources are hidden and a one-column/seven-row wrap when sources are shown; both produce two buses with no invalid coordinates.
- Disabling five optional retrieval legs leaves branches only to Lexical and Text vector retrieval while retaining valid fan-in/fan-out and controller geometry.
- Sampled fan-out, fan-in, branch, and recovery geometry has no interior intersections with retrieval cards in either source-visibility state.
- All semantic paths retain descriptive accessibility labels. The decorative bus paths are `aria-hidden`, and controller labels are included in their relationship descriptions.
- Light- and dark-theme browser inspection showed the buses, arrowheads, captions, and controller ports remain readable with no browser errors.

Discoveries and decisions:

- Source visibility changes the retrieval group's wrapping more dramatically than expected, so calculating only a bounding rectangle is insufficient. Grouping cards by measured top coordinate provides stable per-row buses without assuming a column count.
- Disabled cards must remain part of the group's outer bounds even though they receive no active branch. This keeps the routing rails outside every visible card and prevents fan-in curves from cutting through disabled placeholders.
- The old recovery curves clipped the Learned sparse card only in the source-hidden Full layout. Dedicated, slightly offset recovery lanes fixed the collision and keep the two retry meanings visually separable.
- Controller cards can be above, beside, or below the group depending on layout. Choosing a port side from relative geometry is simpler and more robust than template-specific coordinates.
- The bus scaffolding is deliberately decorative rather than a new logical dependency type. The underlying endpoint fixture, drawers, metrics, and data-source semantics therefore remain unchanged.

Replace the current all-to-all flow curves with explicit fan-out/fan-in geometry:

```text
Metadata pre-filter
        │
   retrieval bus
   ├─ lexical
   ├─ text vector
   ├─ text-to-image
   └─ other enabled legs
        │
  Candidate union
```

Implementation details:

- Calculate the retrieval group as a first-class layout object.
- Draw one flow into the group and one flow out, with short branches between the bus and enabled legs.
- Keep individual serving-source edges connected to their actual consumers.
- Terminate group-wide controller dependencies at labelled group ports:
  - Routing: “gates legs”
  - Budget: “allocates k”
  - Degradation: “cuts or skips”

- Make geometry independent of whether nodes wrap into one or multiple columns.
- Recalculate source-column routing when data sources are shown without changing the logical pipeline edges.
- Ensure branch and recovery curves never travel through retrieval cards.

**Acceptance criteria:** Retrieval legs read as concurrent at every supported desktop width and in all three templates.

## Phase 4 — Represent gating and optionality correctly

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Added `gated` as an eighth canonical relationship kind. Its blue dash-dot treatment, marker, description, accessibility wording, and legend sample are distinct from solid flow, conditional decision branches, recovery returns, and grey control steering.
- Added one validated `GATED_EXECUTION` registry for the stages whose request-path entry depends on route, intent, image presence, budget, or policy.
- Applied gated entry to Text-to-image retrieval from Recommended onward and, in Full, to Image-to-image, Learned sparse, Multi-vector/passage, and Late-interaction retrieval.
- Applied gated entry to the optional Full reranking passes: Semantic rerank, Late-interaction rerank, VLM rerank, and Learning-to-Rank. Cross-encoder remains the solid primary pass.
- Added a blue inset rail and a specific gate chip to every gated card. Image-to-image reads “image query”; the visual stages read “visual intent”; route-selected and optional passes say so directly.
- Marked full-index Late-interaction retrieval and Late-interaction rerank as “alternative placement” rather than implying both should run.
- Added a measured “Selective rerank cascade” boundary around the five Full reranking stages, with the explicit rule “route-selected passes · skipped stages pass candidates through”.
- Extended the Phase 3 retrieval bus so deterministic and gated branches share one fan-out scaffold while retaining separate semantic paths and styles.
- Added gate context to connector accessibility labels, component drawers, hover summaries, and the node legend.
- Added startup validation for gated-stage definitions and selective-cascade membership.

Verification:

- Endpoint counts remain unchanged: Core 9/12, Core + recommended 31/35, and Full 76/84 logical relationships with sources hidden/shown.
- Core has zero gated relationships and zero gated cards. Recommended has one gated relationship/card for Text-to-image retrieval. Full has nine gated logical relationships and nine gated cards.
- The mixed deterministic/gated fan-out intentionally produces semantic SVG counts of Core 7/10, Core + recommended 24/28, and Full 47/55 with sources hidden/shown.
- Source visibility changes only `serves` paths; gated and other non-serving paths remain unchanged.
- Disabling all four optional reranking passes creates a direct solid Fusion → Cross-encoder flow. The retrieval gates remain, and no disabled reranker remains on the request path.
- All eight legend samples exactly match their rendered connector type for stroke, width, dash rhythm, line cap, and opacity.
- Gate cards, connectors, chips, cascade boundary, and captions were inspected in light and dark themes with no browser errors or invalid SVG coordinates.

Discoveries and decisions:

- Retrieval routing already declares `rerankers_to_fire[]` in its contract, so the existing content supports request-time reranker gating without inventing a new owner. Phase 4 does not add more steering edges; Phase 5 still owns the narrower routing-to-reranker relationship decision.
- Gating belongs on entry to a stage, while the stage's successful output remains ordinary candidate flow. This avoids turning every connector around an optional stage into a new semantic type.
- A gate pattern alone is insufficient in a dense diagram. The card rail, concise condition chip, group boundary, and bypass note provide redundant explanations without relying on colour.
- Cross-encoder is the stable production reranker in Recommended and the primary pass in Full; preserving its solid entry keeps the production path legible inside the selective catalogue.
- Learning-to-Rank is marked as an optional pass rather than specifically route-selected: its execution may depend on model availability or policy even when Retrieval routing does not own it.

Introduce a clear visual treatment for request-time gated execution. It should be distinct from decision branches and control steering.

Apply it to:

- conditionally fired retrieval legs
- image-to-image retrieval when no image query is present
- optional reranking tiers in Full
- VLM reranking
- any route-dependent late-interaction operation

For the Full reranking surface:

- Avoid one unqualified solid chain that implies every reranker executes.
- Create a “selective rerank cascade” group or gate boundary.
- Preserve ordering constraints where they are real.
- Mark stages that are alternatives or optional passes.
- Keep the Core + recommended Cross-encoder path straightforward and solid.

**Acceptance criteria:**

- Core looks deterministic.
- Core + recommended looks like the production path with bounded conditional recovery.
- Full clearly communicates available gated stages without suggesting they all execute.

## Phase 5 — Correct feedback and learning semantics

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Renamed Behavioural signals to **Behavioural event log** and made its boundary explicit: Results assembly emits position-aware impression context, while client applications emit clicks, saves, enquiries and reformulations directly into the log.
- Kept the existing component count instead of adding a User interactions box. The event-log contract and notes identify the external interaction producer without implying that Results assembly originates user actions.
- Clarified Personalisation as two separate mechanisms in one component: a request-time user-profile lookup on the ranking spine and asynchronous profile updates for future requests.
- Added `updates` as a ninth canonical relationship type. Its magenta irregular dash treatment, arrow, legend sample, forward/reverse drawer terminology, description and accessibility wording distinguish durable profile-state changes from request flow, telemetry feeds and offline model training.
- Reclassified Behavioural event log → Personalisation from `trains` to `updates`. Behavioural event log continues to train Learning-to-Rank and the learned form of Fusion policy offline.
- Added one validated `RELATION_DETAILS` registry for all four feedback/learning relationships. Every entry declares a concise visible label, the transmitted payload and its timing:
  - Results assembly → Behavioural event log: `impression context` · per-response telemetry
  - Behavioural event log → Fusion policy: `debiased training set` · offline model training
  - Behavioural event log → Learning-to-Rank: `debiased training set` · offline model training
  - Behavioural event log → Personalisation: `profile events` · asynchronous profile update

- Rendered those labels as small relation-coloured captions and included the full payload/timing in connector accessibility text and both ends' drawer relationship summaries.
- Resolved reranker ownership in favour of the existing Retrieval routing contract, which already emits `rerankers_to_fire[]`. Routing now steers all four gated rerankers—Semantic rerank, Late-interaction rerank, VLM rerank and Learning-to-Rank—but not the deterministic Cross-encoder.
- Collapsed those four logical steering dependencies into one labelled `selects gated passes` edge terminating at the measured Selective rerank cascade boundary. Its accessibility label retains all four represented endpoints.
- Added startup validation for the gated-reranker set, exact Routing ownership, feedback annotation completeness, annotation-to-edge integrity and the expanded reverse relationship index.

Verification:

- Core remains 9/12 logical relationships with sources hidden/shown; Core + recommended remains 31/35. Full intentionally moves from 76/84 to **79/87** because Routing now owns three additional gated-reranker endpoints.
- Drawn semantic path counts remain Core 7/10, Core + recommended 24/28 and Full 47/55. The four Routing endpoints collapse into the single reranker-gate path, so the clearer ownership does not add visual clutter.
- Source visibility still changes only `serves`: Full adds eight serving relationships and no request, control, feedback, training or update relationship.
- Latency and complexity remain unchanged: Core 34–118 ms / Lean · 15; Core + recommended 97–332 ms / Substantial · 48; Full 172–654 ms / Heavy · 100.
- Full renders all nine legend kinds, and every legend sample matches the corresponding connector's colour token, width, dash pattern, cap and opacity.
- Full renders exactly four feedback captions: one impression-context feed, two offline-training dependencies and one profile update. All semantic paths retain descriptive accessibility labels.
- Disabling Semantic rerank removes that logical endpoint and card connection; the cascade control edge remains with a logical count of three and continues to name the active group.
- The event-log and Personalisation drawers show the same payload, timing and relation terminology as the canvas and legend.
- Wide desktop and source-expanded states were inspected in light and dark themes with no browser warnings, errors or invalid group geometry.

Discoveries and decisions:

- A separate User interactions node would add layout and lifecycle ambiguity without adding a pipeline dependency. The client-side producer is clearer as an explicit external input in the event-log contract; only impression context originates at Results assembly.
- Personalisation is not accurately described as only a trained model. Treating logged events as profile updates preserves the important distinction between asynchronous state mutation and the runtime profile read used by the ranking stage.
- The routing contract is authoritative for gated rerank execution. Pointing Routing only at Late-interaction rerank was inconsistent with both `rerankers_to_fire[]` and the other visibly gated rerankers.
- The cascade boundary is the right render target for shared control, while the logical model must retain the four actual endpoints. This preserves dependency clarity in drawers, diagnostics and accessibility without four more long control curves.
- The Phase 0 source-count invariants remain valid. Phase 5 changes only non-serving Full endpoints, and the new 79/87 baseline is the deliberate semantic successor to the earlier 76/84 fixture.

Implement the decisions made in the clarifying questions below.

Likely relationship shape:

```text
Results assembly ── emits impression context ──▶ Behavioural event log
User interaction ── emits clicks/saves/etc. ───▶ Behavioural event log

Behavioural event log ── trains ──▶ LTR
Behavioural event log ── trains ──▶ learned Fusion policy
Behavioural event log ── updates/learns from ──▶ Personalisation
```

Also resolve Retrieval routing → Late-interaction rerank:

- If routing owns all gated execution, target a reranker gate or group consistently.
- If routing only selects retrieval legs, remove this one specific reranker edge.

**Acceptance criteria:** Every feedback arrow identifies what is transmitted and whether it is request-time data, telemetry, a profile update, or offline model training.

## Phase 6 — Restore dependency clarity on narrow screens

**Status: Complete — 2026-08-23**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Replaced the former arrow-only mobile fallback with a first-class textual dependency view generated from `logicalDependencies(...)`, the same inventory that drives desktop connectors, diagnostics and regression fixtures.
- Added a compact mobile introduction that states the active logical-relationship count, shows only the relationship kinds present in the current template/source state, and explains how to inspect endpoints.
- Added relationship summaries between stacked pipeline rows. They collapse repeated fan-out/fan-in endpoints while retaining separate flow and gated-entry treatments and real decision-outcome labels.
- Replaced the generic recovery arrow with an indented **Recovery decision** block that explicitly preserves both branches and returns:
  - `too few` → Constraint relaxation → returns `relaxed predicate` to Metadata pre-filter
  - `zero` → Zero-result fallback → returns `recovery mode` to Metadata pre-filter

- Added a mobile **Selective rerank cascade** marker with the same route-selected/pass-through rule as the desktop boundary.
- Added dependency chips inside every active control-plane card. Dense endpoint sets are compacted to meaningful groups such as `retrieval legs` and `gated rerankers`; smaller sets retain their component names.
- Added `serves` chips to shown data-source cards and hid the older duplicate source sentence at narrow widths.
- Changed narrow-screen card activation from immediately covering the canvas with the drawer to inserting an inline dependency panel beside the selected card. The panel:
  - highlights every connected card without moving them
  - lists every active incoming and outgoing logical relationship using canonical forward/reverse terminology
  - preserves branch outcomes and Phase 5 payload/timing annotations
  - lets keyboard or pointer users select connected endpoints
  - provides an explicit **Open full component details** action for the existing drawer

- Added responsive-state handling so crossing the 700 px breakpoint rebuilds the appropriate representation. Desktop SVG paths and stage boundaries return above the breakpoint; mobile summaries return at or below it.
- Kept all mobile-only DOM hidden from layout and accessibility above 700 px. Only geometrically impractical SVG curves are removed on narrow screens; the dependency model remains complete.

Verification:

- At 390 px, the mobile relationship totals exactly match the logical fixtures in all six template/source states: Core 9/12, Core + recommended 31/35 and Full 79/87.
- Core shows flow summaries; Recommended adds gated entry, recovery, steering and telemetry; Full adds training and profile updates. Showing sources adds `serves`, giving Full all nine canonical relationship kinds.
- Mobile renders zero semantic SVG paths by design, while the same Full state immediately restores 47 semantic paths representing all 79 logical relationships after resizing to desktop.
- The breakpoint is exact: 700 px shows the textual representation with no curves; 701 px hides it and restores the SVG renderer.
- Core renders seven between-row summaries, Recommended fifteen plus one recovery block, and Full twenty-one plus one recovery block and one cascade marker.
- Full control cards render twelve compact dependency chips. With sources shown, all seven data-source cards expose `serves` chips; the shared image ANN source lists both visual consumers.
- Selecting Retrieval routing exposes two incoming and twelve outgoing active endpoints while its card compacts them to `retrieval legs · gated rerankers · Candidate budget allocation`.
- Selecting the shared image ANN source lists both serving endpoints; dependency-reference navigation moves the inline panel and connected-card highlight to the chosen endpoint.
- Disabling Semantic rerank removes its incident dependencies and changes the Full mobile inventory from 79 to 77 and row summaries from 21 to 20 without leaving a stale selected endpoint.
- Inline selection, full-drawer handoff, close cleanup, source visibility, light/dark themes, mobile-to-desktop-to-mobile resizing and the exact breakpoint were exercised with no browser warnings or errors.

Discoveries and decisions:

- The logical inventory is a better responsive source than the desktop render-edge list. Render edges intentionally collapse groups for geometry; starting from them would discard endpoints precisely where the textual view needs full detail.
- A short transition summary after each model row communicates sequence more reliably than a decorative downward arrow, especially when a row contains parallel or gated alternatives.
- Compact card chips and complete inline selection serve different levels of detail: chips make control ownership scannable, while the selection panel prevents grouping from hiding actual endpoints.
- The drawer remains valuable for contracts and implementation guidance, but it is too disruptive as the first dependency interaction on a phone. Making it an explicit second step keeps the pipeline and its highlighted dependencies visible.
- The desktop rerank boundary cannot be meaningfully measured around a long mobile stack. A marker carrying the same group name and pass-through rule preserves the semantics without pretending that a mobile geometric enclosure is useful.
- Source cards already contained a desktop-oriented `Serves` sentence. Hiding that sentence only on mobile avoids duplication while keeping the new canonical relationship chip visible.
- Relationship-kind chips in the mobile introduction intentionally reflect current state. In particular, `serves` appears only when data sources are shown, unlike the stable explanatory sidebar legend.

The current `≤700px` behavior removes all lines. Replace that with a deliberate responsive representation.

Recommended implementation:

- Keep the stacked cards.
- Add compact relationship summaries between groups.
- Show dependency chips inside control and data-source cards, such as “Steers: retrieval legs.”
- Preserve the recovery branches as an indented decision block.
- Allow node selection to highlight and list incoming and outgoing dependencies.
- Hide only geometrically impractical curves, not the dependency information itself.

**Acceptance criteria:** A narrow-screen user can still identify flow, steering, recovery, serving, and training dependencies.

## Phase 7 — Interaction and accessibility polish

**Status: Not complete — 2026-08-24**

- On hover or keyboard focus, highlight both endpoints and the connecting relation.
- Show a compact relation caption without permanently labelling every line.
- Ensure collapsed group edges highlight all represented targets.
- Extend node drawers to use the same relation names as the legend.
- Add keyboard-accessible dependency summaries.
- Maintain adequate dark- and light-theme contrast.
- Avoid moving source cards or pipeline cards during hover.

**Acceptance criteria:** Every logical relationship is discoverable by mouse and keyboard, and highlighting remains correct for group-collapsed relationships.

## Phase 8 — Verification matrix

**Status: On hold — 2026-08-23**

Test every combination below:

| Dimension   | Cases                                                                    |
| ----------- | ------------------------------------------------------------------------ |
| Template    | Core; Core + recommended; Full                                           |
| Sources     | Hidden; shown                                                            |
| Theme       | Light; dark                                                              |
| Width       | Wide desktop; constrained desktop; narrow/mobile                         |
| State       | Defaults; individual components disabled; whole optional groups disabled |
| Interaction | Hover; keyboard focus; drawer; capability focus; source selection        |

Structural assertions should verify:

- Exact logical endpoint sets per template.
- No `serves` edges while sources are hidden.
- Showing sources does not change non-serving edges.
- The shared image ANN source produces two edges only in Full.
- Disabling one image consumer preserves the shared source for the other.
- Disabling all consumers deactivates the source and removes its wires.
- Group-collapsed steering still exposes all logical targets in drawers and accessibility text.
- No connector path intersects a card except at its endpoint.
- No unexplained connector style exists in the DOM.
- Narrow layouts retain a textual dependency representation.

## Clarifying questions

1. In Full, which rerankers are genuinely ordered stages, and which are alternatives or independently gated passes?
2. Does Retrieval routing own all request-time gating, including rerankers, or only selection of retrieval legs?
3. Should the diagram add a small “User interactions” node, or should Behavioural signals be renamed to a telemetry/event-log component without adding another box?
4. Does Personalisation represent a trained model, a runtime user-profile lookup, ongoing profile updates, or a combination of those?
5. Is replacing most individual retrieval flow curves with a shared fan-out/fan-in bus acceptable, while retaining individual source-serving lines?
6. On narrow screens, should the tool prioritize a compact textual dependency view or preserve the full graphical diagram through horizontal scrolling?
7. Should gated or optional execution become a formal relation type in drawers and the legend, or remain a property of ordinary flow?
8. Should the line legend always show `serves`, or display it only while “Show data sources” is enabled?

## Phase 9 — Type the gate vocabulary

**Status: Complete — 2026-08-24**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Added the validated `GATE_KINDS` registry — `route`, `intent`, `budget` — carrying timing,
  deciding component, colour, and the declared conditions with their explanatory sentences.
  Added `GATE_TIMING_LABEL` as the single source of allowed timings.
- Replaced the free-string `GATED_EXECUTION` entries with `defGate(from,kind,label,note)`
  calls that resolve against the registry at definition time. An entry naming an unknown kind
  or an undeclared condition now throws where it is written.
- Added `--gate-request` and `--gate-config` tokens. `--gate-request` is a lazy alias of
  `--wire-gate`, so a card rail cannot drift from the connector it belongs to; `--gate-config`
  is defined per theme.
- Gave deployment gates a muted, dashed chip and a muted rail, both driven by a `--gate-color`
  custom property set from the registry rather than by per-kind CSS classes.
- Replaced the ungrammatical `Request gated by <label>` and `gated: <label>` phrasing. The
  drawer chip now reads `Gate <label> · per request` or `· deployment choice`, and the full
  registry sentence renders in a new drawer gate note and hover-card line.
- Added the `title` attribute `.gatechip` alone among the node chips lacked.
- Split the gate chip and conditional chip from an either/or into two independent chips.
- Extended startup validation to the registry itself: timings, colours, non-empty conditions,
  and that a named decider is a real component which actually `steers` what it gates.
- Extracted `setGateNote()` so the shared drawer header cannot carry a stale gate note from a
  component onto a data source.

Verification:

- Logical inventories are byte-identical to the pre-change build: Core 9/12,
  Core + recommended 31/35, Full 79/87 with sources hidden/shown.
- Gated card counts per template are unchanged: Core 0, Core + recommended 1, Full 9
  (4 route, 3 intent, 2 budget).
- Every gated card in every template and both themes resolves a `--gate-color` and a chip
  title; none is missing.
- Deployment gates render `#7a8496` light / `#93a0b5` dark with a dashed chip; request gates
  render `#2563b8` light / `#60a5fa` dark solid, confirming the lazy alias tracks the theme.
- All three `conditional` components now render their chip, including VLM rerank, which
  previously lost it to the gate chip.
- Negative tests: an unknown kind and a condition borrowed from another kind both throw at
  definition; a decider that does not steer its gated stage is rejected.
- Drawer, hover card, and node card were inspected in both themes with no console errors.

Discoveries:

- All nine gated stages are steered by Retrieval routing, so `decidedBy` is `routing` for both
  request-time kinds. The route/intent split is therefore about what drives the decision, not
  who owns it. The two budget gates keep `decidedBy: null`, preserving the Phase 4 finding that
  Learning-to-Rank may depend on model availability or policy that routing does not own.
- A kind cannot own a single label: `intent` legitimately carries both "visual intent" and
  "image query". Kinds therefore declare a condition set, and the entry names one of them.
  This keeps the vocabulary validated without collapsing two real conditions into one.
- The 76/84 Full inventory recorded in Phase 0 and repeated through Phase 4 is stale. The
  committed baseline before this phase already measured 79/87 — Phases 5 and 6 added edges
  without restating the figure. Phase 9 changed neither number; later phases should baseline
  against 79/87.
- Connector styling was deliberately left alone. The gate kind is a property of the stage, so
  differentiating the card keeps all eight legend samples matching their rendered connector.
- Naming the decider on the card was held back for Phase 10, which owns surfacing and linking
  it. Phase 9 only stores it and validates it.

Phase 4 introduced four gate labels but modelled them as free strings on `GATED_EXECUTION`.
`route-selected` and `optional pass` therefore render in identical styling with no definition
anywhere, leaving the reader to infer the distinction Phase 4 recorded in its discoveries:
route and intent gates are per-request decisions, while an optional pass is a deployment
choice about budget or available training data.

- Add a validated `GATE_KINDS` registry — `route`, `intent`, `budget` — carrying the public
  label, a full explanatory sentence, the deciding component (or `null`), and its visual
  treatment. Model it on the existing `RELATION_TYPES` registry.
- Give each `GATED_EXECUTION` entry a `kind` and derive its label from the registry.
  Reject unknown or missing kinds at startup, as `RELATION_TYPES` already does.
- Distinguish budget gates visually from route and intent gates. Request-time gates keep the
  gate colour; a deployment-time gate should read as configuration, closer to the existing
  `.altchip` treatment than to `.gatechip`.
- Replace the raw-label phrasing at every render site. "Request gated by _optional pass_" and
  "gated: optional pass" are not grammatical — the labels describe the stage, not the
  condition. Use the registry sentence in the drawer and hover card.
- Add the `title` attribute that `.gatechip` alone among the node chips lacks. `.tierchip`
  and the relation legend already explain themselves on hover.
- Render the gate chip and the conditional chip together rather than as an either/or. VLM
  rerank is both gated and conditional, so the gate currently suppresses the amber warning on
  the widest latency range in the diagram (80–500 ms), while its drawer still says
  "Conditional p95". The two facts are orthogonal: when a stage runs, and what it costs
  when it does.

**Acceptance criteria:** Every gate label resolves through a typed registry entry with a
definition reachable from the canvas. Route-selected and optional-pass stages are
distinguishable without reading the label. Endpoint inventories are unchanged by the phase.

## Phase 10 — Attribute the gating decision

**Status: Complete — 2026-08-24**

Implemented in `search-query-pipeline-diagram-tool.html`:

- The drawer gate chip is now a button that opens its `decidedBy` component, and the gate note
  names that component in prose: "Decided by Retrieval routing."
- Budget gates render the same chip as a plain span with no "Decided by" line, so the
  asymmetry appears twice — once as a missing link, once as a missing sentence.
- Added `chipTag(goto, html)`, which returns a button when there is somewhere to go and a span
  otherwise, so no header chip can present an affordance that does nothing.
- Replaced the free-text `note:"alternative placement"` with `altOf`, naming the paired
  component. Late-interaction retrieval and Late-interaction rerank each name and link the
  other, on the card and in the drawer.
- Added reciprocity validation: an `altOf` target must exist, must not be the stage itself,
  and must name it back.
- Wired `#dChips [data-goto]` clicks. The existing delegation covered only `#dBody`, so header
  refs would have been inert.
- Dropped the uppercasing from `.altchip`. It now carries a proper noun rather than a category
  label, and a shouted component name is harder to read.

Verification:

- Logical inventories unchanged: Core 9/12, Core + recommended 31/35, Full 79/87.
- Gated card counts unchanged: Core 0, Core + recommended 1, Full 9.
- Clicking the gate chip on Learned sparse retrieval opens Retrieval routing. Clicking the
  alternative chip moves between the two late-interaction stages and back.
- Learning-to-Rank and Semantic rerank expose zero header links.
- Negative tests against the reciprocity clause: a self-reference, a one-way claim, a pair
  pointing at a third stage, and an unknown target each throw; the real pair passes.
- Gate chip titles remain present on every gated card in every template and both themes.
- Opening a gated component, then a data source, leaves no stale note or link.
- No console errors; light and dark inspected.

Discoveries:

- The route condition sentence had to be reworded from "Runs only when retrieval routing
  selects it for this query" to "Runs only for queries whose route includes it." Once the note
  appends "Decided by Retrieval routing" uniformly, the original named the owner twice.
- Both request-time kinds link to the same component, because Retrieval routing steers all
  seven of them. The route/intent split remains a distinction about what drives the decision,
  not who owns it. If that shared destination proves confusing in use, the fix is to reconsider
  the kind boundary, not to add a second decider the relationship model does not support.
- Naming the decider in the note and linking it from the chip were kept as one affordance each
  rather than making the note text a second link. Two link targets for one destination in
  adjacent elements read as clutter without adding reach.

"Route-selected" invites the question _selected by what?_ Retrieval routing already declares
`rerankers_to_fire[]` in its contract and steers every gated reranker, but no gate surface
names it. This phase closes plan question 2 in the UI rather than only in the data.

- Make the drawer gate chip activate its `decidedBy` component, so the gate is one hop from
  the component that owns the decision.
- Leave budget gates unlinked. The asymmetry is the lesson: the chip that leads nowhere is
  the one that is not a runtime decision.
- Replace the free-text `alternative placement` note with a component reference, so
  Late-interaction retrieval and Late-interaction rerank each name the other. A pill reading
  "alternative placement" does not say what the alternative is.

**Acceptance criteria:** Every request-time gate names and links its deciding component. The
two late-interaction placements are legible as one either/or choice rather than two unrelated
caveats.

## Phase 11 — Expose the vocabulary in the legend

**Status: Complete — 2026-08-24**

Implemented in `search-query-pipeline-diagram-tool.html`:

- Added `GATE_CONDITIONS` and its `GATE_CONDITION` lookup, flattening the registry into the
  unit the legend and focus panel actually address: the condition, which is what a card shows.
- Added a generated `Gate conditions` legend block, built from that index the way the
  relationship legend is built from `RELATION_TYPES`. Each row carries the literal card chip,
  a count of the stages it gates, and its sentence.
- Removed the single `gated stage` swatch and its now-unused `.gate-key` rule. The generated
  block supersedes it.
- Added `gate` as a fourth kind in the existing unified `FOCUS` registry, so selecting a
  condition highlights every stage it gates and lists them in the focus panel. This reuses the
  facet/capability/phase mechanism rather than adding a parallel one.
- Rows render only for conditions the active template can reach, and the whole block hides in
  Core. A template switch that retires the focused condition clears the focus rather than
  leaving a pressed row with nothing behind it.
- Rewrote the legend note, which previously described gating as "route or intent" and omitted
  deployment gates entirely.
- Casing: restored `.altchip` to uppercase, matching every other pill in the tool, and added
  `.chip .tagval` so a condition name inside a sentence-case drawer chip is cased like the pill
  it refers to. A component name is not a label tag and stays as written.

Verification:

- Logical inventories unchanged: Core 9/12, Core + recommended 31/35, Full 79/87.
- Legend rows per template: Core none and block hidden; Core + recommended one
  (visual intent, 1); Full four — route-selected 4, visual intent 2, image query 1,
  optional pass 2, totalling the nine gated stages.
- Selecting `route-selected` highlights exactly the four route-gated stages; `optional pass`
  highlights Semantic rerank and Learning-to-Rank; re-selecting clears.
- Focusing a Full-only condition and switching to Core + recommended clears the focus, the
  pressed state, and the dimming. A condition that survives the switch keeps its focus and
  re-highlights against the new template.
- Every pill resolves `text-transform: uppercase`: tier, group, gate, conditional, alternative,
  legend condition, boundary title, legend section label.
- Light and dark inspected; pressed rows take the accent border and soft accent background in
  both. No console errors.

Follow-up fix — 2026-08-24:

- The Phase 10 rewording of the `route-selected` sentence removed its subject so the drawer
  would not name Retrieval routing twice, but only the drawer composed the "Decided by"
  suffix. The card tooltip and hover card were left showing a subjectless sentence that also
  leaned on "route" as an undefined term.
- Reworded to "Runs on some queries and not others, depending on what the query needs", which
  needs no prior knowledge of what a route is and still composes with the suffix.
- Extracted `gateSentence(gate)` and used it for the card tooltip, hover card, drawer note, and
  focus panel, so those four surfaces can no longer disagree about attribution. Legend rows
  keep the bare sentence on purpose: the legend note already states that Retrieval routing owns
  every request gate, and repeating it on three of four rows is noise.

Discoveries:

- The block is one row per condition, not per kind. The `intent` kind carries two conditions
  with different sentences, and filtering by condition is strictly more precise than filtering
  by kind while costing one extra row.
- Both selective-cascade captions — the desktop boundary note and the mobile stage marker —
  read "route-selected passes", omitting the two budget passes inside the same boundary. They
  now read "gated passes". This was the same omission the legend note carried.
- The existing `FOCUS` registry absorbed gate filtering in four lines. No new highlight,
  dimming, or panel machinery was needed, which is why this phase touches interaction without
  touching layout.

The node legend carries a single `gated stage` swatch, and its note — "Gated entry runs only
when route or intent activates its stage" — omits budget gates entirely.

- Generate a gate-conditions block from `GATE_KINDS`, as `renderRelationLegend()` is
  generated from `RELATION_TYPES`. One row per kind, with its swatch and sentence.
- Correct the legend note to cover all three kinds.
- Make the rows filter buttons using the existing `aria-pressed` legend styling, so selecting
  a kind highlights every stage it gates. Reading the shape of what routing controls should
  not require hunting pills across the canvas.

**Acceptance criteria:** Every gate kind in the data model appears in the legend with its
definition. Selecting a kind reveals its full set of stages in one action.

Each phase re-runs the Phase 8 verification matrix.
