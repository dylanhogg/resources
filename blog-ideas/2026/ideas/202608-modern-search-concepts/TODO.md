# TODO

Note: this list is for humans, not for AI agents.

## Minor fixes

- Change the retrieval fan-out and candidate fan-in to be in a dashed "retreival components" box similar to "selective rerank cascade" and reduce the number of fan-in and fan-out lines. Other steering lines can still point to specific components in the dashed box if needed for specificity.
- (?) Only show line labels on highlighting a source or target node for that line.
- (?) Hover over (or click?) a line pops up a tooltip with the line label, source and target node labels, and short description of the relationship.
- (?) Separate html from datasructures
- (?) Split Full view into say Extended and Full so there are 4 levels: Core, Recommended, Extended, Full.

## Features

- wip - needs a redo: Review "What you are choosing" section in RHS popout. List all existing values across all components in a table with added column "clarity" and "validity" (clarity: clear, unclear, confusing; validity: valid, invalid, questionable). Add a column for "suggested change" if clarity or validity is not good.

- Review business ranking position - should it be later?

- Review the group tags (Query processing, Constraint handling, etc)

- Enable diagram state to be managed via url query params for sharing and bookmarking.

- Enable diagram to be exported to mermaid code (or similar) for sharing and editing in other tools.

- Enable components to be moved and re-ordered in the diagram via drag and drop.

- Enable components to be added and removed from the diagram.
