# TODO

Note: this list is for humans, not for AI agents.

## Questions

- What are the "Write paths" in Physical architecture diagram? How can this be made more clear to the user?
- Review Query Expansion - what does it feed exactly? Anything missing? Is it really required?
- Is "Build order" logical or physical or both? Is it helpful? How can it be more helpful and clear?

## Minor fixes

- Rename "Pipelines" tab to "Logical architecture"
- Rename "What you are choosing" for clarity
- Apply the formatting of component hover popup to the RHS info panel so that the same information in both is presented in the same way. Hover popup is the preferred UI.
- RHS panel should scroll to top when clicking on a new component, not retain the scroll depth for new click
- Add a little more vertical space around the "Selective Rerank Cascade" box. It's a bit tight above and below.
- Add link from Text query to Query expansion?
- Remove Conditional pill from hoverover since it's redundant.
- Review all steered by, steers info are correct on component popup & RHS panel
- (?) Ordering the sidebar by PHASES
- (?) Only show line labels on highlighting a source or target node for that line.
- (?) Hover over (or click?) a line pops up a tooltip with the line label, source and target node labels, and short description of the relationship.
- (?) Separate html from datasructures
- (?) Split Full view into say Extended and Full so there are 4 levels: Core, Recommended, Extended, Full.

## Features

- Add GA

- What is unhelpful and/or confusing in the whole app (LHS filters, logical & physical diagrams, component hover popup, RHS info panel, Build order, About page)

- WIP: Add a new tab between "Buid order" and "About" that addresses physical implemetations (one on open search, other on qdrant)

- Later: add physical substrate selector (e.g. only OpenSearch)

- wip - needs a redo: Review "What you are choosing" section in RHS popout. List all existing values across all components in a table with added column "clarity" and "validity" (clarity: clear, unclear, confusing; validity: valid, invalid, questionable). Add a column for "suggested change" if clarity or validity is not good.

- Review the group tags (Query processing, Constraint handling, etc)

- Enable diagram state to be managed via url query params for sharing and bookmarking.

- Enable diagram to be exported to mermaid code (or similar) for sharing and editing in other tools.

- Enable components to be moved and re-ordered in the diagram via drag and drop.
