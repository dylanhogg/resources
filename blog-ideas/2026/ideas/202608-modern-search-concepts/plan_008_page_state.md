# Plan 008 — Page state in the URL

Target file: `search-query-pipeline-diagram-tool.html` (single file, zero build).
Companion reading: `search-query-pipeline-diagram-tool-architecture.md` (point-in-time,
25 Aug 2026 — the region table has drifted twice since, but the **layer model, the
validator-wall contract and the recipes are still accurate**). That doc's closing line
"There is no persistence and no URL state (both are open TODOs)" is what this plan retires.

**Status:** Not started. Written on top of `0474e2f` (plan 007 phases 1–5).
Update this line as each phase lands, in the style of plans 006 and 007
(`Phase N done (<sha>, <date>)`).

## What this delivers

Every meaningful thing the reader has chosen is written into the query string as they
choose it, and reading that URL back reproduces the page exactly. Copy the address bar,
paste it into a blog post or a Slack message, and the recipient lands on _your_ diagram —
the physical level at Full surface with the write path on, the LLM reranker switched off
and "Fine visual detail" traced — not on the default Core logical view.

**The invariant to internalise:** the URL is a _projection of `state`_, never a second
source of truth. It is written from `state` and parsed into `state`; no renderer, no
validator and no view ever reads `location`. Anything the URL cannot express is state the
reader can re-reach in one click, and the codec says so out loud rather than silently
dropping it.

### Decisions taken (from review, 26 Aug 2026)

| #   | Question                  | Decision                                                                                                                              |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | How much state to capture | **Fully lossless** — both levels, all three templates' `off` sets. Only non-defaults are ever written, so the common URL stays short. |
| 2   | Open component drawer     | **Included.** `?d=px-vlm` deep-links a component card. Restored without overriding the URL's level.                                   |
| 3   | Collapsed sidebar groups  | **Excluded.** Incidental UI state; lengthens the URL without communicating intent.                                                    |
| 4   | Scroll position           | **Excluded.** Fragile across viewport widths and the narrow-layout breakpoint.                                                        |
| 5   | History behaviour         | **`replaceState` only**, no `popstate` handler. The address bar is always shareable; Back leaves the page exactly as it does today.   |
| 6   | Share affordance          | **Copy-link button** in the header beside the theme toggle. Without it most readers never notice the URL is live.                     |

### Phase summary

| Phase                       | What it does                                                                                                                                                 | On screen         | Depends on | Status      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ---------- | ----------- |
| **1 — The codec**           | New `PAGE STATE IN THE URL` section: vocabulary tables, `serialiseState`, `parseState`, slug helpers, load-time validators, diagnostics handle. No wiring.   | Nothing changes   | —          | Not started |
| **2 — Read on load**        | `applyUrlState()` before the views are built, plus a three-line `<head>` script so a `th=dark` link does not flash light. Sanitisation and canonicalisation. | Links restore     | 1          | Not started |
| **3 — Write on change**     | `syncUrl()` with rAF coalescing, wired to eight call sites. `file:` hash fallback.                                                                           | URL goes live     | 2          | Not started |
| **4 — Drawer deep link**    | The `d` param: restore path that opens a card without moving the level, and model inference from the id.                                                     | Deep links work   | 3          | Not started |
| **5 — Copy-link button**    | `#shareBtn`, clipboard write with an `execCommand` fallback, "Copied" feedback.                                                                              | New header chrome | 3          | Not started |
| **6 — Verification & docs** | Round-trip sweep, both levels × three templates × every plane combination, narrow screens, dark theme, then `TODO.md` and the architecture doc.              | Nothing changes   | 1–5        | Not started |

---

## The state being captured

Read off the live page (`window.dependencyDiagnostics`, 26 Aug 2026). Everything below is
already in `state`; nothing new is being stored.

| Where                      | Field             | Values                                              | Default     | In URL  |
| -------------------------- | ----------------- | --------------------------------------------------- | ----------- | ------- |
| `state`                    | `view`            | `pipe` \| `phys` \| `build` \| `about`              | `pipe`      | ✅      |
| `document.documentElement` | `data-theme`      | `light` \| `dark`                                   | `light`     | ✅      |
| `state.models[m]`          | `tpl`             | `1` \| `2` \| `3`                                   | `1`         | ✅      |
| `state.models[m]`          | `planes`          | logical `{serving}`; physical `{data, consistency}` | all `false` | ✅      |
| `state.models[m]`          | `off[1..3]`       | `Set` of component ids, per template                | empty       | ✅      |
| `state.models[m]`          | `focus`           | `{kind, id}`, kind ∈ `facet cap phase gate runtime` | `null`      | ✅      |
| `state.models[m]`          | `sel`             | open drawer subject: component id or data-source id | `null`      | ✅      |
| `state.models[m]`          | `collapsedGroups` | `Set` of sidebar group names                        | empty       | ❌ (D3) |
| `state.models[m]`          | `prevTpl`         | dead field — written by `setTpl`, read by nothing   | `null`      | ❌      |

`m` ∈ `logical` (38 components, 6 data sources, 9 core / 16 recommended / 13 optional) and
`physical` (39 components, 8 data sources, 12 / 15 / 12). Core components are locked and
cannot be switched off, so an `off` set can only ever hold recommended and optional ids —
29 at most on the logical level, 27 on the physical one, and one to five in practice.

**`state.view` is deliberately global, `tpl` deliberately is not.** Note the existing quirk
in `setView`: crossing between the two pipeline tabs copies the departing model's `tpl` onto
the arriving one. That is why per-model levels in the URL are still worth having and also
why restore must set `state.view` _before_ the bootstrap `setView` call — at bootstrap
`previousView === v`, so the copy never fires, and both levels survive the load intact.

---

## The URL grammar

### Design constraints

1. **Short keys, readable values.** The four things a reader changes constantly — tab,
   level, planes, theme — get names a human can guess. The long tail (`off` lists) is
   allowed to be terse.
2. **Non-defaults only.** A page at defaults has _no_ query string at all, not `?v=pipe&t=1`.
3. **No percent-encoding in the common case.** `,` and `:` are legal unencoded in a query
   component (RFC 3986 sub-delims / pchar) and every browser displays them as typed.
   `URLSearchParams.toString()` would escape both, so **the query string is built by hand**
   and parsed with `URLSearchParams` (which decodes `%2C` and `%3A` transparently, so a
   hand-mangled URL still works).
4. **Repeated keys instead of a group separator.** Per-template `off` sets are emitted as
   `off=2:relax&off=3:vlm`, read with `getAll()`. No `;`, no nested grammar.

### Keys

Bare keys are the **logical** level; `px.` prefixes the **physical** one, matching the
codebase's own naming for that level. `v` and `th` are page-level.

| Key              | Scope | Value grammar                                    | Default (omitted) | Example                  |
| ---------------- | ----- | ------------------------------------------------ | ----------------- | ------------------------ |
| `v`              | page  | `logical` \| `physical` \| `build` \| `about`    | `logical`         | `v=physical`             |
| `th`             | page  | `dark` \| `light`                                | `light`           | `th=dark`                |
| `d`              | page  | one component or data-source id                  | none              | `d=px-vlm`               |
| `t` / `px.t`     | model | `1` \| `2` \| `3`                                | `1`               | `px.t=3`                 |
| `p` / `px.p`     | model | comma list of plane ids                          | none              | `px.p=data,consistency`  |
| `off` / `px.off` | model | `<level>:<id>,<id>` — repeatable, once per level | none              | `px.off=3:px-vlm,px-ltr` |
| `f` / `px.f`     | model | `<kind>:<id>`                                    | none              | `px.f=cap:visual`        |

Physical ids keep their `px-` / `pxd-` prefix. Stripping it would save three characters per
id at the cost of a second naming rule and an ambiguity between `px-vlm` and `pxd-cache`;
the ids as written are copy-pasteable straight against the registry.

Focus kinds are spelled in full — `facet`, `cap`, `phase`, `gate`, `runtime` — because they
are already short and a one-letter code here would be unreadable for no gain. `runtime` is
physical-only (the logical model declares no `RUNTIMES`); parse validation catches
`f=runtime:gpu` on the logical side and drops it.

Gate focus ids are **slugified condition labels** — the labels themselves contain spaces:

| Condition label        | URL slug               |
| ---------------------- | ---------------------- |
| `route selected`       | `route-selected`       |
| `visual intent`        | `visual-intent`        |
| `image query`          | `image-query`          |
| `classifier uncertain` | `classifier-uncertain` |
| `optional pass`        | `optional-pass`        |
| `optional stage`       | `optional-stage`       |

Derived, not declared: `slug = label.replace(/\s+/g,"-")`, reversed by matching against
`model.gateConditions`. A load-time validator asserts the six slugs are unique per model,
so a future condition whose slug collides fails at load rather than silently focusing the
wrong row.

### Worked examples

```
(no query string)                        Logical · Core · light · nothing off
?t=3                                     Logical · Full surface
?t=2&off=2:relax                         Logical · Core + recommended · constraint relaxation off
?v=physical&px.t=3&px.p=data,consistency Physical · Full surface · data stores + write path
?v=physical&px.t=3&px.off=3:px-vlm,px-ltr&px.f=cap:visual&th=dark
?v=build                                 Build order tab
?d=px-vlm                                Physical tab inferred from the id, VLM reranker card open
```

The longest realistic URL — physical, Full surface, both planes, five components off, a
focus and a drawer — is about 130 characters. The pathological one (every optional
component off on both levels at all three templates) is about 900; it is reachable only by
deliberately clicking every toggle, and it still works.

---

## Phase 1 — The codec

**Location.** A new banner section, `PAGE STATE IN THE URL`, immediately after the
`VIEW STATE` banner and the `state` declaration (currently ~line 6463), before
`SHARED PAGE FURNITURE`. It needs `PIPELINE_MODELS`, `state` and `Model.stages` — all of
which exist by that point — and nothing else. Placing it there makes `syncUrl` a hoisted
function declaration visible to the view module below it, with no forward-declaration
dance of the `let revealInModel = ()=>{}` kind.

**One preparatory move.** `VIEW_MODEL` and `MODEL_VIEW` (currently ~line 8732, in
`VIEWS AND PAGE CHROME`) are pure data about which tab draws which level. Move both up into
the `VIEW STATE` section, above `state`. The codec needs them to map `v=physical` onto
`state.view === "phys"`, and they belong with the state they describe. `setView` and
`revealComponent` keep working unchanged.

### What the section contains

```
URL_VIEW          { logical:"pipe", physical:"phys", build:"build", about:"about" }  (derived from MODEL_VIEW)
URL_MODEL_PREFIX  { logical:"", physical:"px." }
URL_FOCUS_KINDS   ["facet","cap","phase","gate","runtime"]
URL_DEFAULTS      { view:"pipe", theme:"light", tpl:1 }

slugifyCondition(label)              → "route selected" → "route-selected"
conditionForSlug(model, slug)        → reverse lookup against model.gateConditions
focusIdValid(model, kind, id)        → per-kind vocabulary check (see below)
serialiseState()                     → string, "" when everything is default
parseState(search)                   → plain object of validated, model-keyed values
```

`serialiseState` walks `state` and emits, in a fixed order so two identical states always
produce byte-identical URLs (which matters: `replaceState` fires on every change and an
unstable key order would make the address bar flicker):

1. `v` — when `state.view !== "pipe"`
2. `th` — when `data-theme !== "light"`
3. logical keys `t`, `p`, `off…`, `f`
4. physical keys `px.t`, `px.p`, `px.off…`, `px.f`
5. `d` — when the active view's slice has a `sel` **and** the drawer is open

Each emitted only when it differs from the default. The whole thing is joined with `&`
behind a single `?`; an empty result means the URL is rewritten to the bare path.

`parseState` is the mirror, and it is **the only place in the file that treats its input as
untrusted**. The validator wall throws at load because internal drift is a bug; a mangled
URL is user input, and blanking the page over one is the wrong answer. So every value is
checked against the live registries and **silently dropped** if it fails:

| Value | Check                                                                                                                                                                                           |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v`   | key of `URL_VIEW`                                                                                                                                                                               |
| `th`  | `dark` or `light`                                                                                                                                                                               |
| `t`   | key of `model.templates`                                                                                                                                                                        |
| `p`   | each id in `model.planes`; a plane with `requires` forces its parent on, exactly as `setPlane` does                                                                                             |
| `off` | level is a template key; each id is in `model.components`, is a stage of that template, and is **not** `tier:"core"` (core is locked — an injected core id would be a state no click can reach) |
| `f`   | kind ∈ `URL_FOCUS_KINDS`; id valid for that kind (below)                                                                                                                                        |
| `d`   | id ∈ some model's `components` or `dataSources`                                                                                                                                                 |

`focusIdValid` per kind: `facet` → `model.facets`, `cap` → `model.capabilities`,
`phase` → `model.phases`, `runtime` → `model.runtimes` (absent on the logical level, so
always false there), `gate` → `conditionForSlug` resolves. This check is not optional
politeness: `FOCUS[kind].describe` does `FACETS.find(...).label` with no guard, so an
unvalidated `f=cap:bogus` would throw a `TypeError` into the sidebar render.

After parsing, `off` sets are run once through the `REQUIRES` cascade (a dependent follows
its parent down) so a hand-edited URL cannot produce a composition the UI itself would never
build.

### Load-time validators, in the spirit of the wall

Cheap, and each throws when the assumption behind the grammar stops holding:

- **Gate slugs are unique** within each model.
- **No id, plane id or template key contains a reserved character** (`& = , : ; ? #`) —
  asserted across both models' components, data sources and planes. A future component
  called `a,b` would break the grammar silently; this makes it break loudly.
- **`URL_FOCUS_KINDS` matches `Object.keys(FOCUS)` exactly** — asserted inside
  `createPipelineView`, where `FOCUS` is in scope. Adding a sixth focus kind and forgetting
  the codec then fails at load instead of quietly dropping the reader's selection.
- **`URL_VIEW` covers every tab** — every `data-view` in the markup has a URL name.

### Diagnostics

Extend `window.dependencyDiagnostics` with a `url` handle, so this is checkable from a
console the way everything else in this file is:

```js
dependencyDiagnostics.url.write(); // current state → query string
dependencyDiagnostics.url.read(search); // query string → parsed object
dependencyDiagnostics.url.roundTrip(); // sweep: see Phase 6
```

**Phase 1 changes nothing on screen.** Nothing calls the codec yet.

---

## Phase 2 — Read on load

### The head script

`<html>` carries `data-theme="light"` as its hard default, so a `th=dark` link would paint
the whole page light and then flip once the main script reaches the end of `<body>`. One
line in `<head>`, immediately before `<style>`:

```html
<script>
  {
    const m = location.search.match(/[?&]th=(dark|light)/);
    if (m) document.documentElement.setAttribute("data-theme", m[1]);
  }
</script>
```

This is the only duplicated parsing in the plan and it is worth a comment saying why:
theme is the one piece of state that must be applied before first paint, and the
alternative — moving 7,700 lines of script into `<head>` with `defer` — is not one.
`applyUrlState` sets the same attribute again, idempotently.

### applyUrlState()

Called once, at the bottom of the new codec section — **before** `createPipelineView` runs,
so each view's first and only render already shows the restored composition. No flash, no
double render, no wasted `drawWires` pass.

```
applyUrlState()
  parsed = parseState(location.search)          // never throws
  state.view = URL_VIEW[parsed.v] ?? "pipe"
  documentElement.data-theme = parsed.th ?? "light"
  for each model:
    slice.tpl    = parsed[m].t ?? 1
    slice.planes = merge(defaults, parsed[m].p)
    slice.off[k] = new Set(parsed[m].off[k])    for each named level
    slice.focus  = parsed[m].f ?? null
  drawer subject held aside for Phase 4
```

The whole body is wrapped in `try/catch`; a throw logs one console warning and leaves
`state` at its defaults. **A bad link must degrade to the default page, never to a blank
one** — the same failure mode the validator wall deliberately produces for internal drift,
and deliberately must not produce here.

### Canonicalisation

At the very end of bootstrap — after `setView(state.view)` — a single `syncUrl()` call
rewrites the address bar from the state that actually took effect. A reader who hand-types
`?t=9&off=3:nonsense&f=cap:bogus` watches it correct itself to `?` (nothing). This is why
Phase 2 lands before Phase 3 is wired: the read path is testable on its own by pasting URLs
and reading `dependencyDiagnostics.url.write()`.

---

## Phase 3 — Write on change

### syncUrl()

```
let urlSyncQueued = false;
function syncUrl(){
  if(urlSyncQueued) return;
  urlSyncQueued = true;
  requestAnimationFrame(()=>{ urlSyncQueued = false; writeUrl(serialiseState()); });
}
```

The rAF coalescing earns its keep three ways: a single click can reach `syncUrl` through
both a handler and `renderAll`; `setOff` cascades over dependents in a loop; and
`layoutSweep()` drives `renderAll` through a dozen synthetic compositions before restoring
the reader's own. Because the sweep is synchronous and restores state before it returns,
the one coalesced write that follows it is already correct — no suspend flag needed.

`writeUrl` compares against `location.search` first and returns early when unchanged, so an
idle re-render costs nothing.

### Call sites

Eight, chosen as _intent_ sites rather than render sites so the URL tracks what the reader
did:

| Site                                         | Covers                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------- |
| `setView()` (page)                           | `v`                                                                       |
| `#themeBtn` handler (page)                   | `th`                                                                      |
| `renderAll()` (per view)                     | `t`, `p`, `off` — every composition change, including the reset button    |
| `setFocus()` (per view)                      | `f`                                                                       |
| `openDrawer()` / `openSourceDrawer()` (view) | `d`                                                                       |
| `clearSelection()` (per view)                | `d` cleared                                                               |
| `closeDrawer()` (page)                       | `d` cleared — belt and braces, it already routes through `clearSelection` |

`renderAll` is the one render site in the list, and it is there because `setTpl`,
`setPlane`, the component toggles and the reset button all funnel through it — seven
handlers collapsed into one call, exactly as the drawer redraw already does two lines above.

### `replaceState`, and the `file:` fallback

Per D5, every write is `history.replaceState(null, "", url)`. No history entries, no
`popstate` handler, Back behaves as it does today. Browsers throttle `replaceState` at
roughly 100 calls per 30 seconds; rAF coalescing on user-driven changes keeps us orders of
magnitude under.

**The `file://` problem.** This tool's stated usage is "open the file directly in a
browser", and a `file:` document has an opaque origin, which can make `replaceState` with a
query string throw `SecurityError`. The deploy target is `https://` (see the `Makefile`) and
local work uses the `python3 -m http.server` config in `.claude/launch.json`, so this is a
degraded mode rather than the main one — but it must not throw an uncaught error into the
console of anyone who double-clicks the file.

Two lines of defence:

```
const urlMode = location.protocol === "file:" ? "hash" : "search";
```

- **`search` mode** — `history.replaceState`, wrapped in `try/catch`. A single failure flips
  the session permanently to hash mode.
- **`hash` mode** — assign `location.hash = "?" + qs` (a hash assignment never navigates and
  never throws). Reading accepts a hash that starts with `?` as if it were a search string.

`parseState` therefore reads `location.search` first and falls back to
`location.hash.startsWith("?") ? location.hash.slice(1) : ""`. The same URLs work in both
modes; only the separator character differs. **This is the one item in the plan I could not
verify in-browser** (the preview pane refuses `file://`); Phase 6 carries a manual check.

---

## Phase 4 — Drawer deep link

`d` is page-level rather than model-scoped, because the drawer is a document singleton —
only one can be open. The owning model is resolved by looking the id up in each model's
`components` and `dataSources` rather than by trusting the `px-` prefix, so a future
renaming cannot break the link.

**Write.** `d` is emitted only when the active view's `slice.sel` is set _and_ the drawer
is actually open. `slice.sel` also carries the narrow-layout selection panel's subject and
survives a tab switch with the drawer still showing another level's card; neither is worth
a second parameter.

**Read.** Restoration happens at the bootstrap tail, after `setView(state.view)`, deferred
one frame (a hidden canvas measures as zero, which is the same reason `revealComponent`
already uses a `setTimeout`):

```
restoreDrawer(id)
  modelId = ownerOf(id)                       // components ∪ dataSources lookup
  if VIEW_MODEL[state.view] !== modelId: drop it and return
  view = VIEWS[modelId]
  if id is a data source: view.openSourceDrawer(id)
  else: view.openDrawer(id); view.revealNode(id)
```

**It must not call `revealComponent`.** That helper is built for cross-tab navigation and
calls `view.setTpl(<level where the component first appears>)` — which would quietly
override the level the URL just restored. Two things that look alike and are not.

**Model inference.** When `v` is absent and `d` names a component the physical model owns,
`v` is inferred as `physical`. This makes a bare `?d=px-vlm` pasted into prose work, which
is exactly how these links get written by hand. An explicit `v` always wins.

**Dropped rather than honoured:** a `d` whose owner is not the active tab's model (e.g.
`?v=build&d=understand`). The drawer would float over the build ladder with no diagram
behind it. `syncUrl` removes the parameter on the next write, so the reader sees the URL
correct itself.

---

## Phase 5 — Copy-link button

Markup, beside the existing theme toggle in `.hdr`:

```html
<div class="hdr-actions">
  <button
    class="icon-btn"
    id="shareBtn"
    title="Copy a link to this view"
    aria-label="Copy a link to this view"
  >
    🔗
  </button>
  <button
    class="icon-btn"
    id="themeBtn"
    title="Toggle theme"
    aria-label="Toggle colour theme"
  >
    ◐
  </button>
</div>
```

`.hdr` is a flexbox with `gap:18px` and `margin-right:auto` on `.brand`, so the two icons
need their own wrapper with a tighter gap: `.hdr-actions{display:flex;gap:8px}`. That is
the entire CSS change.

Behaviour: `navigator.clipboard.writeText(location.href)`, then swap the glyph to a check
for ~1.4s and announce "Link copied" through a visually-hidden `aria-live="polite"` region.
`navigator.clipboard` needs a secure context — `https:`, `localhost` and (in Chrome)
`file:`, but **not** `http://192.168.x.x`, which is a realistic way to view this on a phone
— so a `document.execCommand("copy")` fallback over a temporary off-screen `<textarea>`
sits behind it. If both fail, the button selects the URL text in a prompt-free way and the
title attribute already tells the reader what to copy.

The button is not a new source of truth: it copies `location.href`, which `syncUrl` has
already made correct.

---

## Phase 6 — Verification and documentation

### Round-trip sweep

`dependencyDiagnostics.url.roundTrip()` asserts `parse(write(s))` deep-equals `s` over a
generated sample:

- both models × three templates × every valid plane combination (the existing
  `planeCombinations()` already enumerates these, minus the unreachable "dependent plane on,
  parent off" cases)
- the empty `off` set, a single-id set, and every optional component off at once
- every focus kind × its first and last valid id, plus `null`
- both themes, all four tabs, drawer open on a component and on a data source

This is the check that would have caught a reserved character in an id, an unstable key
order, or a focus kind added without a codec entry.

### By hand

1. **Default page** — the address bar shows a bare path, no `?`.
2. **Every control** — click through level, both plane toggles, five component toggles,
   reset, each focus family, both tabs, theme; confirm the URL tracks each one and that
   reloading reproduces the page.
3. **Losslessness** — customise logical, switch to physical, customise it, copy the URL,
   open in a fresh tab, switch tabs both ways: both compositions survive. Watch for the
   `setView` level-copy quirk on the _second_ tab switch — it is existing behaviour and
   should still apply, but must not fire on load.
4. **Hostile input** — `?t=9`, `?t=abc`, `?off=3:nope`, `?off=1:q-text` (a locked core id),
   `?f=cap:bogus`, `?f=runtime:gpu` on the logical level, `?d=nonsense`, `?px.p=consistency`
   without `data`, and a URL with `%3A`/`%2C` escapes. Every one renders the page and
   canonicalises the URL; none blanks it or throws.
5. **Narrow layout** — restore a link below 700px, then cross the breakpoint both ways. The
   full re-render on crossing must not perturb the URL.
6. **Dark theme** — a `th=dark` link paints dark on first frame with no light flash.
7. **`file://`** — open the file directly, confirm hash mode engages and no `SecurityError`
   reaches the console. This is the one behaviour that could not be verified from the
   preview pane.
8. **Validator wall intact** — the three existing validators still pass; the page is not a
   dead shell.

### Documentation

- `TODO.md` — remove "Enable diagram state to be managed via url query params for sharing
  and bookmarking" from Features.
- `search-query-pipeline-diagram-tool-architecture.md` — the `state` paragraph's closing
  sentence ("There is no persistence and no URL state (both are open TODOs)") is now wrong.
  Replace it with two sentences: the URL is a projection of `state`, and the grammar lives
  in the `PAGE STATE IN THE URL` section. Add a `Recipes` entry: **adding a piece of state
  to the URL** — declare its key, extend `serialiseState`/`parseState`, add its validation,
  add a `syncUrl()` call site, extend the round-trip sample. The doc's own note asks agents
  not to update it; this is a two-line correction of a statement the plan actively falsifies,
  and the recipe is what stops the next change from being guesswork.

---

## Risks and things that will bite

| Risk                                                                                     | Mitigation                                                                                                        |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `replaceState` throws on `file://`                                                       | Protocol check up front plus `try/catch` backstop; hash mode is a full peer, not a stub. Manual check in Phase 6. |
| `FOCUS[kind].describe` throws on an id that no longer exists                             | Validated on parse against the live registries; unknown ids dropped before they reach a render.                   |
| A registry rename introduces a reserved character into an id                             | Load-time validator over every id, plane id and template key.                                                     |
| The codec drifts from `FOCUS` when a sixth focus kind is added                           | `URL_FOCUS_KINDS` asserted against `Object.keys(FOCUS)` inside `createPipelineView`.                              |
| Restore fights `setView`'s level-copy, or `revealComponent` overrides the restored level | Set `state.view` before the bootstrap `setView`; restore the drawer with `openDrawer`, never `revealComponent`.   |
| URL churn during `layoutSweep()`                                                         | rAF coalescing; the sweep restores state synchronously before the single write lands.                             |
| Unstable key order makes the address bar flicker                                         | Fixed emission order, asserted by the round-trip sweep.                                                           |
| Hand-edited URL produces a composition the UI cannot build                               | `REQUIRES` cascade applied after parse; core ids rejected from `off`.                                             |

## What this plan deliberately does not do

- **No `localStorage`.** The architecture doc records "no `fetch`, `import`, `require`,
  `localStorage` or `sessionStorage` calls anywhere" as a property of the file. A URL is a
  thing the reader chose to share; a storage key is a thing that follows them around. The
  theme included — a shared `th=dark` link should not repaint someone's next visit.
- **No `popstate` handling** (D5). Adding it later means adding a re-render path that
  reapplies state to two already-built views; it is a phase of its own, not a flag.
- **No short-link or state-compression scheme.** The pathological URL is ~900 characters and
  reachable only on purpose.
- **No export of the diagram itself** — the mermaid export in `TODO.md` is a separate piece
  of work that shares nothing with this one but the word "sharing".
