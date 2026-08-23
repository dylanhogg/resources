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

| Template | Sources hidden | Sources shown | Source nodes | `serves` edges |
|---|---:|---:|---:|---:|
| Core | 9 edges | 12 edges | 3 | 3 |
| Core + recommended | 31 edges | 35 edges | 4 | 4 |
| Full | 76 edges | 84 edges | 7 | 8 |

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

  | Template | Source nodes | `serves` edges |
  |---|---:|---:|
  | Core | 3 | 3 |
  | Core + recommended | 4 | 4 |
  | Full | 7 | 8 |

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

- On hover or keyboard focus, highlight both endpoints and the connecting relation.
- Show a compact relation caption without permanently labelling every line.
- Ensure collapsed group edges highlight all represented targets.
- Extend node drawers to use the same relation names as the legend.
- Add keyboard-accessible dependency summaries.
- Maintain adequate dark- and light-theme contrast.
- Avoid moving source cards or pipeline cards during hover.

**Acceptance criteria:** Every logical relationship is discoverable by mouse and keyboard, and highlighting remains correct for group-collapsed relationships.

## Phase 8 — Verification matrix

Test every combination below:

| Dimension | Cases |
|---|---|
| Template | Core; Core + recommended; Full |
| Sources | Hidden; shown |
| Theme | Light; dark |
| Width | Wide desktop; constrained desktop; narrow/mobile |
| State | Defaults; individual components disabled; whole optional groups disabled |
| Interaction | Hover; keyboard focus; drawer; capability focus; source selection |

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
