# TODO

Note: this list is for humans, not for AI agents.

## Minor fixes

- Change the retrieval fan-out and candidate fan-in to be in a dashed "retreival components" box similar to "selective rerank cascade" and reduce the number of fan-in and fan-out lines. Other steering lines can still point to specific components in the dashed box if needed for specificity.
- Only show line labels on highlighting a source or target node for that line.
- Hover over (or click?) a line pops up a tooltip with the line label, source and target node labels, and short description of the relationship.
- Separate html from datasructures?
- Split Full view into say Extended and Full so there are 4 levels: Core, Recommended, Extended, Full.

## Features

- wip - needs a redo: Review "What you are choosing" section in RHS popout. List all existing values across all components in a table with added column "clarity" and "validity" (clarity: clear, unclear, confusing; validity: valid, invalid, questionable). Add a column for "suggested change" if clarity or validity is not good.

- wip - implementing now: Review the query search pipeline exploration tool: search-query-pipeline-diagram-tool.html.html. Would is make sense to add required data source components (e.g. lexical search engine, vector database, etc) to the pipeline diagram to show external dependencies to serve the various retrieval components? (These are assumed to be present in the background, but it might be useful to show them explicitly.)

- Review business ranking position - should it be later?

- Review the group tags (Query processing, Constraint handling, etc)
- Review the vertical text groups display on the LHS - are they needed with group tags as well?

- ? Give notes a type: warning, info, or choice.

- Enable diagram state to be managed via url query params for sharing and bookmarking.

- Enable diagram to be exported to mermaid code (or similar) for sharing and editing in other tools.

- Enable components to be moved and re-ordered in the diagram via drag and drop.

- Enable components to be added and removed from the diagram.
