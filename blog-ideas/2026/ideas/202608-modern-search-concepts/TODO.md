# TODO

Note: this list is for humans, not for AI agents.

## Questions

- In the Full logical pipeline, should Personalisation come before Freshness & temporal component (also to be in line with physical pipeline where all the recommended components are grouped into Final ranking passes)
- Describe what the predicate compiler is, and is it necessary?
- Should logical pipeline Result sufficiency check come after Candidate pruning and Fusion? (But stay before expensive reranking). This would match the physical pipeline closer which groups "Union, prune and fuse"
- Should Candidate union + dedup be separated from fusion in physical diagram, such that sufficiency check is in between?
- What are the "Write paths" in Physical architecture diagram? How can this be made more clear to the user?
- Review Query Expansion - what does it feed exactly? Anything missing? Is it really required?
- Is "Build order" logical or physical or both? Is it helpful? How can it be more helpful and clear?

## Minor fixes

`In the search exploration tool search-query-pipeline-diagram-tool.html physical architecture diagram...`

- xMake all components text left aligned (some are centered)
- xUpdate logical pipeline "Candidate union" to "Candidate union, dedupe" to better represent what it does.
- xComponent hover popup should display on left or right side depending on
- xRemove "Build & run complexity" and "Operational burden" in LHS panel (it's obvious from the pipelines)
- xCan Final ranking passes in physical pipeline address Freshness, Near dedupe, Diversity, and Business ranking somehow in the component subtitle?
- xRename "Deduplication" component in logical view to "Near duplicate collapse" to be truer to its function. Update subtitle accordingly to not be the same text.
- xKeep the same template option selected when changing from logical to physical pipeline view
- Embedding could be on GPU or CPU depending on model selcted.
- Apply the formatting of component hover popup to the RHS info panel so that the same information in both is presented in the same way. Hover popup is the preferred UI.
- Add link from Text query to Query expansion?
- Review all steered by, steers info are correct on component popup & RHS panel

- (?) Ordering the sidebar by PHASES
- (?) Only show line labels on highlighting a source or target node for that line.
- (?) Hover over (or click?) a line pops up a tooltip with the line label, source and target node labels, and short description of the relationship.
- (?) Separate html from datasructures
- (?) Split Full view into say Extended and Full so there are 4 levels: Core, Recommended, Extended, Full.

- Ensure this rationale for where the Sufficiency component is within the pipelines is included (hover and RHS info panel): For a architecture diagram, I would consider sufficiency before fusion the simpler and more conventional default if its primary purpose is constraint relaxation and zero/low-result recovery. Put it after fusion only if you explicitly want quality-aware sufficiency. i.e. After union, before fusion -> Do we have enough candidates? After lightweight fusion -> Do we have enough plausible/relevant candidates?

## Features

- Add GA

- Write a Python interface stub file that outlines the physical pipeline Orchestrator

- Enable diagram state to be managed via url query params for sharing and bookmarking.

- Enable diagram to be exported to mermaid code (or similar) for sharing and editing in other tools.

- What is unhelpful and/or confusing in the whole app (LHS filters, logical & physical diagrams, component hover popup, RHS info panel, Build order, About page)

- Turn views of the logical & physical pipelines into blogposts & short videos tracing sample queries through the pipeline

- Later: physical implemetations to add physical substrate selector (e.g. only OpenSearch, onl Qdrant)

- wip - needs a redo: Review "What you are choosing" section in RHS popout. List all existing values across all components in a table with added column "clarity" and "validity" (clarity: clear, unclear, confusing; validity: valid, invalid, questionable). Add a column for "suggested change" if clarity or validity is not good.

- Review the group tags (Query processing, Constraint handling, etc)

- Enable components to be moved and re-ordered in the diagram via drag and drop.
