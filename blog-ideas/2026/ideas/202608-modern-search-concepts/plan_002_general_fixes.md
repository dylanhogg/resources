# General UI/UX Fixes Plan

Scope: `search-query-pipeline-diagram-tool.html`. This plan covers navigation, orientation,
layout defects, and interaction clarity. It is separate from `plan_001_dependency_improvements.md`,
which owns the relationship model, gate vocabulary, and connector rendering.

Overlap to respect: Phases 2 and 3 below take over the interaction and accessibility work that
`plan_001` Phase 7 currently holds as _Not complete_. When Phase 2 lands, mark `plan_001`
Phase 7 as superseded rather than implementing it twice. Phase 10 below implements the URL-state
item already listed in `TODO.md`.

## Implementation status summary

| Phase | Title                                                 | Status      | Priority | Risk   |
| ----: | ----------------------------------------------------- | ----------- | -------- | ------ |
|     0 | Baseline measurements and regression guard            | Complete    | Blocking | Low    |
|     1 | Fix sticky header, context bar, and sidebar geometry  | Not started | High     | Low    |
|     2 | Make the detail drawer non-modal and keyboard-correct | Not started | High     | Medium |
|     3 | Complete focus mode: dim wires, navigate to hits      | Not started | High     | Low    |
|     4 | Sidebar information architecture                      | Not started | High     | Medium |
|     5 | Orientation: stage rail, search, clickable diff       | Not started | High     | Medium |
|     6 | Reduce competing visual languages                     | Not started | Medium   | Medium |
|     7 | Soften the data-sources layout jump                   | Not started | Medium   | Medium |
|     8 | Drawer depth navigation                               | Not started | Medium   | Low    |
|     9 | Narrow-screen control access                          | Not started | Medium   | Low    |
|    10 | Durable state: URL and local persistence              | Not started | Medium   | Low    |
|    11 | Copy, naming, and cross-view continuity               | Not started | Low      | Low    |
|    12 | Verification matrix                                   | Not started | Ongoing  | Low    |

Phase 0 is complete: `window.dependencyDiagnostics.layout()` and `.layoutSweep()` now reproduce
every number below on demand. Phases 1 to 12 are not started. Phase 0 was blocking because
Phases 1, 4, 5, and 7 all move layout, and without recorded baselines there is no way to tell a
fix from a regression.

Recommended landing order: 0, 1, 3, 2, 4, 5, then the rest by priority. Phase 3 before Phase 2
because Phase 3 is a small, self-contained win that Phase 2's larger interaction change benefits
from being able to lean on.

## Phase 0 — Baseline measurements and regression guard

**Status: Complete — 2026-08-24**

Implemented in the `LAYOUT DIAGNOSTICS` section of `search-query-pipeline-diagram-tool.html`,
which also becomes the single definition site for `window.dependencyDiagnostics` (the earlier
export next to the validators moved here so the object is frozen once, with everything on it).

Two new entry points, both measurement-only:

- `dependencyDiagnostics.layout(scrollY)` — one snapshot of the live page at the given scroll
  offset, restoring the offset afterwards. Returns the active level and data-source setting,
  viewport, document height, sidebar scroll and client heights, counts (nodes, data-source
  nodes, rows, wire paths, legend relationship types, capability chips), the bounding boxes of
  `.hdr` / `.ctxbar` / `.side`, the derived overlaps between those sticky layers, and every
  sidebar label currently clipped by its container.
- `dependencyDiagnostics.layoutSweep(scrollY)` — the same snapshot across all six level and
  data-source combinations at the current window width, then restores the level, data-source
  setting, and focus it borrowed. It calls `drawWires()` synchronously because the render path
  defers wires to the next frame, so a naive sweep would measure stale counts.

Overlaps are reported as pixels of a lower sticky layer hidden behind the one above it, so
Phase 1 has a number to drive to zero rather than a screenshot to argue about.

### Wide desktop — 1600x1000, measured at scroll offset 800

| Level | Sources | Document | Sidebar scroll / visible | Nodes | Source nodes | Rows | Wires |
| ----- | ------- | -------: | -----------------------: | ----: | -----------: | ---: | ----: |
| 1     | off     |   1240px |                1909/947px |     9 |            0 |    8 |    17 |
| 1     | on      |   1395px |                1909/947px |     9 |            3 |    8 |    20 |
| 2     | off     |   2446px |                2500/947px |    23 |            0 |   16 |    34 |
| 2     | on      |   2627px |                2500/947px |    23 |            4 |   16 |    38 |
| 3     | off     |   3437px |                3245/947px |    38 |            0 |   22 |    57 |
| 3     | on      |   3832px |                3245/947px |    38 |            7 |   22 |    65 |

Relationship types in the legend (9) and capability chips (16) are constant across all six.

Sticky geometry is identical in all six combinations: `.hdr` occupies [0, 62], `.ctxbar`
[53, 160], `.side` [53, 1000]. That yields **9px of context bar behind the header** and
**107px of sidebar behind the context bar** — the sidebar heading "Estimated budget" and the
p95 value are unreachable at any scroll offset. This is the defect Phase 1 fixes.

### Constrained desktop — 1100x1000, measured at scroll offset 800

| Level | Sources | Document | Sidebar scroll / visible | Wires |
| ----- | ------- | -------: | -----------------------: | ----: |
| 1     | off     |   2718px |               1441/1441px |    17 |
| 1     | on      |   2873px |               1441/1441px |    20 |
| 2     | off     |   4517px |               1980/1980px |    34 |
| 2     | on      |   4643px |               1980/1980px |    38 |
| 3     | off     |   6236px |               2586/2586px |    57 |
| 3     | on      |   6535px |               2586/2586px |    65 |

Below the 1180px breakpoint the sidebar stops being a scroll container — scroll height equals
client height — and the document nearly doubles: 6535px against 3832px for the same content.
The overlaps grow rather than shrink: 9px and **142px**, because the context bar wraps taller
while `.side` keeps its hardcoded `top:53px`.

### Narrow — 390x844, measured at scroll offset 800

| Level | Sources | Document | Sidebar height | Wires | Sidebar top |
| ----- | ------- | -------: | -------------: | ----: | ----------: |
| 1     | off     |   3583px |         1729px |     0 |      1054px |
| 1     | on      |   3946px |         1729px |     0 |      1417px |
| 2     | off     |   6100px |         2284px |     0 |      3015px |
| 2     | on      |   6609px |         2284px |     0 |      3524px |
| 3     | off     |   8943px |         2950px |     0 |      5193px |
| 3     | on      |   9814px |         2950px |     0 |      6063px |

Wires are zero by design — `drawWires()` bails below 700px and the textual mobile-transition
cards take over. Sticky overlaps are zero because sticky positioning is off at this width.

"Sidebar top" is the distance from the viewport top at scroll offset 800, so at Full surface
with sources shown the level switch, data-source toggle, and component list first become
visible **6863px into a 9814px document**. That is the number Phase 9 has to move.

### Clipped sidebar labels

Only two entries currently overflow, both at Full surface on wide desktop, both in the
component list at a 176px container:

| Label                            | Content width | Container |
| -------------------------------- | ------------: | --------: |
| Image-to-image vector retrieval   |         180px |     176px |
| Multi-vector / passage retrieval  |         178px |     176px |

Neither appears at 1100px, where the sidebar is wider. Phase 4 asserts `clippedLabels` is
empty at every width.

**Acceptance criteria:** met. `dependencyDiagnostics.layoutSweep(800)` at 1600x1000 reproduces
every number in the wide-desktop table, the overlap pair, and the clipped-label list; the same
call at any width produces a comparable table after each later phase.

**Verification:** existing `validation` and `modelValidation` results are unchanged
(9/12, 31/35, 79/87 hidden/shown logical edges), the console is clean, and a sweep followed by
a screenshot confirms the page returns to Core with sources hidden — the sweep leaves no trace.

**Discoveries:**

1. The constrained-desktop layout is worse than the wide one on every axis measured, not just
   narrower. Phase 4 should treat 1100px as the design target rather than an afterthought.
2. The sticky overlap is width-dependent (107px against 142px), which rules out simply
   correcting the `top` constant to a second hardcoded number. Phase 1 needs the measured
   `--hdr-h` / `--ctx-h` approach.
3. Clipped labels are far rarer than expected — two entries, both by under 5px. Phase 4's
   truncation work is small; its real content is the sidebar's 3245px length.

## Phase 1 — Fix sticky header, context bar, and sidebar geometry

**Status: Not started**

`.ctxbar` and `.side` both use a hardcoded `top:53px`. The header does not measure 53px, and the
context bar is not zero-height, so the three sticky layers overlap.

Measured at scroll offset 800, 1600x1000, Full surface:

| Element   | top | bottom |
| --------- | --: | -----: |
| `.hdr`    |   0 |  61.75 |
| `.ctxbar` |  53 | 159.75 |
| `.side`   |  53 |   1000 |

Two consequences:

- The context bar underlaps the header by 8.75px. `.hdr` has `z-index:40` and `.ctxbar` has
  `z-index:35`, so the header paints over the context bar's top padding and hides it.
- The context bar covers the top 106.75px of the sticky sidebar. Once scrolled, the
  "Estimated budget" heading and the p95 latency value — the tool's primary readout — are
  hidden behind it, and the sidebar visibly begins mid-sentence at
  `(+5-20 ms), zero-result fallback (+10-40 ms).`

Work:

- Measure `.hdr` and `.ctxbar` heights on load, on resize, and after any change that reflows
  the context bar, writing them to `--hdr-h` and `--ctx-h` on the root element.
- Stack the layers from those variables: `.ctxbar{top:var(--hdr-h)}`,
  `.side{top:calc(var(--hdr-h) + var(--ctx-h))}`, and
  `.side{max-height:calc(100vh - var(--hdr-h) - var(--ctx-h))}`.
- Apply the same variables to the `@media (max-width:900px)` and `(max-width:700px)` overrides
  that currently reset `top` to `0` or restate `53px`.
- Add `scroll-margin-top` derived from the same variables to node cards, so the Build-order
  `scrollIntoView` in the pipeline view does not land a card under the sticky bars.

**Acceptance criteria:** At every scroll offset and every supported width, no sticky layer
overlaps another, the "Estimated budget" heading and latency value remain visible whenever the
sidebar is pinned, and a node scrolled to from Build order lands fully below the sticky bars.

## Phase 2 — Make the detail drawer non-modal and keyboard-correct

**Status: Not started**

The drawer behaves modally while declaring that it does not. `.scrim.open` sets
`pointer-events:auto` at `z-index:60`, blocking the canvas, and the hover handler early-returns
whenever the drawer is open, so connection highlighting is suppressed too. The element carries
`aria-modal="false"`, and neither `openDrawer` nor `closeDrawer` moves focus into the panel or
restores it on close.

The result is that the tool's core loop — read a component, then read the one it feeds — always
costs close, scroll, click. For a keyboard user the panel is effectively unreachable without
tabbing the remainder of the document.

Work:

- Remove the blocking scrim. Keep the canvas live and interactive while the drawer is open, so
  clicking a second node swaps the drawer contents in place.
- Keep the selected node visibly selected and scrolled into view when the drawer opens, using
  the Phase 1 scroll margins.
- Re-enable hover connection highlighting while the drawer is open; the drawer and the hover
  highlight describe the same relationships and should agree.
- Move focus to the drawer heading on open and restore focus to the originating node on close,
  and keep `Escape` closing the drawer as it does today.
- Reserve canvas width for the drawer rather than covering content, so an open drawer never
  hides the data-source column at Full surface.
- If a modal presentation is still wanted below 700px, gate it on width and set
  `aria-modal="true"` only in that branch.

**Acceptance criteria:** A user can move between components by clicking cards with the drawer
open, keyboard focus enters and leaves the drawer correctly, and no state exists in which
`aria-modal` disagrees with the drawer's actual behaviour. Supersedes `plan_001` Phase 7.

## Phase 3 — Complete focus mode: dim wires, navigate to hits

**Status: Not started**

Focus mode applies `.dimmed` at `opacity:.22` to node cards and data-source cards only. At Full
surface, 65 wire paths stay at full contrast on top of the dimmed nodes, so the highlight
competes with the noise it was meant to remove.

Separately, `scrollIntoView` is currently wired only to Build-order references. Tracing the
`sleeps 6` facet highlights Query understanding, Metadata pre-filter, and Lexical & metadata
retrieval, spread across a 3832px canvas, with no scroll to the first hit and no way to step
between them.

Work:

- Add a focus-mode class to the wires layer and dim every path whose endpoints are not both in
  the hit set, matching the node dim treatment. Paths between two hits stay at full strength.
- For a collapsed group path, treat it as a hit when any represented logical endpoint is a hit,
  so `plan_001` Phase 1's `logical[]` information is honoured rather than lost.
- Scroll the first hit into view when a focus is set, and flash it briefly on arrival.
- Make the existing Focus panel reference rows scroll to and flash their node, instead of only
  opening the drawer. The panel already lists the hits, so the data needed is present.
- Add previous/next hit controls to the Focus panel header, with a position readout, so a
  multi-hit trace is walkable without hunting.
- Keep the current behaviour of leaving the diagram alone when nothing matches.

**Acceptance criteria:** With any facet, capability, phase, or gate condition focused, the only
full-contrast connectors are those between highlighted nodes, and every hit is reachable in one
action without manual scrolling.

## Phase 4 — Sidebar information architecture

**Status: Not started**

The sidebar is 2500-3245px tall inside a 947px scroll container. The budget metrics the user is
meant to watch sit at the top; the component toggles that move them sit roughly 1400px below.
Watching a number change while operating the control that changes it is currently impossible.

The same column also carries a 400px legend block whose explanatory note is eight sentences of
dense prose at 288px wide, and two component names overflow their row: `Image-to-image vector
retrieval` and `Multi-vector / passage retrieval`.

Work:

- Add a compact sticky budget strip pinned to the top of the sidebar's scroll container:
  p95 latency, complexity band, and request-path model count. The full breakdown stays where it
  is; only the three headline numbers follow the scroll.
- Move the legend out of the permanent column into a collapsible panel anchored to the canvas,
  opened from a control near the diagram. It is reference material consulted occasionally, not a
  permanent occupant of a third of the scroll height.
- Split the legend note into short labelled lines rather than one prose block, and let the
  detail sit behind disclosure.
- Fix the two truncating component names: allow two-line wrapping in the list rows, and add a
  title attribute as a fallback.
- Reconsider whether "Capability coverage" and "Components" both need to be fully expanded at
  all times, or whether the group sections in the component list should default to collapsed
  beyond the current template's additions.

**Acceptance criteria:** The headline budget numbers are visible while any component toggle is
being operated, no sidebar row overflows its container at 288px in either theme, and sidebar
scroll height at Full surface with sources shown is materially below the Phase 0 baseline of
3245px.

## Phase 5 — Orientation: stage rail, search, clickable diff

**Status: Not started**

The canvas is 3832px tall with 38 nodes and 65 wires, and offers no overview affordance: no
zoom-to-fit, no minimap, no stage rail, no search. Orientation is the single largest gap in the
tool.

Work:

- Add a sticky stage rail beneath the context bar: Understand, Constrain, Retrieve, Fuse,
  Rerank, Policy, Assemble. It shows the current position while scrolling and jumps on click.
  This also answers the `TODO.md` question about the left-hand vertical group text — a
  horizontal rail does that job without consuming canvas width.
- Add a component search field above the sidebar list. Typing filters the list and highlights
  matching nodes using the existing focus machinery. With 38 components across 8 groups, name
  recall is already unreliable.
- Make the level-difference pill actionable. It is currently a plain `<span id="diffPill">`
  reading, for example, `+ 15 new vs Core + recommended`. Clicking it should focus-highlight
  exactly those newly added nodes — the fastest available answer to "what does this level add",
  reusing focus mode rather than adding machinery.
- Consider a zoom-to-fit control for the canvas, evaluated after the stage rail lands; the rail
  may remove the need.

**Acceptance criteria:** From any scroll position a user can identify which pipeline stage they
are viewing and jump to any other stage in one action, find any component by name without
scrolling the list, and reveal a level's additions in one click.

## Phase 6 — Reduce competing visual languages

**Status: Not started**

A single card currently carries three independent encodings at once: tier as fill colour, group
as a chip, and gate/conditional/alternative chips. Around it, the legend teaches nine
relationship types and sixteen capabilities. `TODO.md` already questions the group tags and the
left-hand vertical group text.

Work:

- Collapse the nine relationship kinds into three families in the default legend — request path
  (flows, gated, branches, returns), control (steers, serves), and offline (feeds, trains,
  updates) — with a disclosure that expands to the full nine. Keep the drawn colours and dash
  patterns from `plan_001` Phase 2 unchanged; this reduces the legend's teaching load, not the
  drawing's precision.
- Remove the group chip from the card face and express group membership through the Phase 5
  stage rail bands instead. A card then reads as name, one-line subtitle, tier, and only its
  exception chips.
- Keep gated, conditional, and alternative chips on the card. Those carry genuine per-card
  information; group membership is positional and does not.
- Re-evaluate the capability list length once search exists; sixteen always-visible chips may
  be better as a searchable list.

**Acceptance criteria:** A card face carries at most one classification chip plus exception
chips, the default legend teaches three relationship families rather than nine, and no
information currently available is unreachable — only relocated behind disclosure.

## Phase 7 — Soften the data-sources layout jump

**Status: Not started**

Toggling "Show data sources" rewrites `.row` from three grid columns to five and applies
`width:calc(100% + 118px); margin-left:-118px` to the canvas. Every card shifts horizontally at
once, and the reading position is lost.

Work:

- Preserve scroll anchoring on the node nearest the viewport centre across the toggle.
- Animate the transition rather than snapping, respecting `prefers-reduced-motion`.
- Evaluate an alternative representation in which a data source appears as an attached badge on
  its consuming node, expanding in place, so the toggle does not restructure the grid at all.
  Decide between the two before implementing; do not build both.
- Whichever representation wins, keep the `plan_001` Phase 0 invariant intact: showing sources
  must not change pipeline flow, control relationships, enabled components, or budgets.

**Acceptance criteria:** Toggling data sources preserves the user's reading position, and the
logical inventories in `plan_001` Phase 0 are unchanged by the new representation.

## Phase 8 — Drawer depth navigation

**Status: Not started**

A single component drawer measures 1754px of scroll across nine sections: What you are choosing,
Stage contract, Dials, Notes, Failure modes, In the worked example, Background reading,
Steered by, Appears in. There is no in-drawer navigation and no way to move to an adjacent
component.

Work:

- Add a sticky section-chip row under the drawer title that scrolls to each section.
- Add previous/next controls stepping through components in pipeline order, so a drawer can be
  read as a sequence rather than a series of lookups. This composes with the Phase 2 change
  that lets the canvas stay live.
- Consider defaulting the lower reference sections to collapsed, with the decision-bearing
  sections open.

**Acceptance criteria:** Any section of an open drawer is reachable in one action, and moving to
the next component in the pipeline does not require returning to the canvas.

## Phase 9 — Narrow-screen control access

**Status: Not started**

At 700px and below, `.stage-area` takes `order:1` and `.side` takes `order:2`. Every control —
budget, capability coverage, legend, and all 38 component toggles — sits below a linear diagram
of roughly 30 cards. The narrow-screen dependency fallback itself works well; the controls are
simply unreachable.

Work:

- Move the budget summary and level controls above the canvas at narrow widths, or present the
  sidebar as a bottom sheet reachable from a persistent control.
- Reduce the vertical cost of the level selector, which currently stacks to three full-width
  rows via `.seg button{flex:1 1 100%}`.
- Confirm the Phase 4 sticky budget strip degrades sensibly at narrow widths rather than
  duplicating the sheet.

**Acceptance criteria:** At 375px width, a user can change level, toggle a component, and read
the resulting budget without scrolling past the diagram.

## Phase 10 — Durable state: URL and local persistence

**Status: Not started**

Already listed in `TODO.md`. There is no `history.replaceState`, no `location.hash` handling,
and no `localStorage`. Theme, level, data-source visibility, per-level component toggles, focus,
and the open drawer all reset on reload, so a composed configuration cannot be recovered or
shared.

Work:

- Encode view, level, data-source visibility, disabled component set, active focus, and open
  drawer into the URL, and restore from it on load.
- Update the URL with `replaceState` on change so the back button is not flooded.
- Persist theme separately in `localStorage`, since it is a preference rather than a document
  state, and keep the current system-preference default when nothing is stored.
- Add a copy-link control near the level selector once the encoding exists.

**Acceptance criteria:** Any composed configuration survives a reload and reproduces exactly
when its URL is opened in a fresh session, and theme choice survives independently of it.

## Phase 11 — Copy, naming, and cross-view continuity

**Status: Not started**

- The header subtitle is four dot-separated clauses: "Progressive pipeline levels - optional
  serving-data layer - hybrid text & image retrieval - logical architecture, not frameworks".
  That is a feature manifest, not orientation, and the manifest already exists in About. Replace
  with one sentence saying what the tool is for.
- "Core + recommended" names a level by arithmetic. If the four-level split in `TODO.md` lands,
  prefer names over formulas: Core, Production, Extended, Full surface.
- Build order does not reflect the current selection. Add a "you are here" marker at the
  selected level; the relationship between the two views is currently explained in prose only.
- Review the Build-order intro paragraph against the level names once they change.

**Acceptance criteria:** A first-time visitor can state what the tool is for from the header
alone, level names are readable as names, and the current level is visible in Build order
without reading prose.

## Phase 12 — Verification matrix

**Status: Not started**

Re-run after each phase, extending the `plan_001` Phase 8 matrix with layout dimensions:

| Dimension   | Cases                                                            |
| ----------- | ---------------------------------------------------------------- |
| Template    | Core; Core + recommended; Full                                   |
| Sources     | Hidden; shown                                                    |
| Theme       | Light; dark                                                      |
| Width       | Wide desktop; constrained desktop; tablet; narrow/mobile         |
| Scroll      | Top; mid-document; bottom                                        |
| State       | Defaults; individual components disabled; whole groups disabled  |
| Interaction | Hover; keyboard focus; drawer open; focus mode; source selection |
| Input       | Mouse; keyboard only; reduced motion                             |

Structural assertions to add:

- No sticky layer overlaps another at any scroll offset or width.
- The budget headline numbers are visible whenever the sidebar is pinned.
- No sidebar row overflows its container.
- In focus mode, every full-contrast connector has both endpoints in the hit set.
- Setting a focus scrolls the first hit into view below the sticky bars.
- Opening the drawer leaves the canvas interactive and moves focus into the panel.
- Closing the drawer restores focus to the originating node.
- Toggling data sources preserves the anchored node's viewport position.
- All `plan_001` Phase 0 logical inventories are unchanged by every phase here.
- A restored URL reproduces the full state it encoded.

**Acceptance criteria:** The matrix passes in full, and the Phase 0 diagnostics table can be
regenerated and compared after every phase.

## Clarifying questions

1. Should the drawer reserve canvas width when open, or overlay the canvas without a blocking
   scrim? Reserving width avoids hiding the data-source column but reflows the diagram.
2. For Phase 6, is removing the group chip from the card face acceptable if group membership is
   still visible through the stage rail, or must it remain readable per card?
3. For Phase 7, which representation is preferred: the current dedicated source column with
   preserved scroll anchoring, or per-node attached source badges that expand in place?
4. Should the four-level split from `TODO.md` land before or after Phase 5, given that the
   diff pill and stage rail both depend on level boundaries?
5. Should focus mode dim the connector layer, or hide non-hit connectors entirely? Dimming
   preserves context; hiding is cleaner at Full surface.
6. Is a zoom-to-fit control still wanted once the stage rail exists, or does the rail make it
   redundant?
7. Should the URL encode the full disabled-component set explicitly, or only the delta from the
   level default, which is shorter but breaks if defaults later change?
