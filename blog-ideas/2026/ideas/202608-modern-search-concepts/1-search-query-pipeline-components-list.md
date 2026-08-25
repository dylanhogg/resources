# Query search pipeline components

https://chatgpt.com/c/6a880532-f940-83ec-bddc-d7e8fb97d080

https://claude.ai/chat/c3bc1980-cd77-4d70-89a4-8014c9e5d175

Core/Optional/Recommended classification of components in a modern hybrid search query pipeline.

This list is the **input** the diagram tool was built from, not its output.
`search-query-pipeline-diagram-tool.html` is the source of truth: where the two disagree,
the tool is right and this file is stale. Seven entries below were reclassified or moved
during plans 001–004 and are marked `— was …`; each one is justified in the corresponding
component drawer in the tool, on its own merits. The classification as first written is
preserved verbatim in `1-search-query-pipeline-components-list-original.md`.

- Query inputs
  - Text query [core]
  - Image query [optional]
- Query processing
  - Query understanding [core]
  - Query normalisation [core]
  - Query rewriting [recommended]
  - Query expansion [optional]
  - Session/context-aware search [optional]
- Constraint handling
  - Metadata pre-filter [core]
  - Constraint confidence estimation [optional]
  - Retrieval count / result sufficiency check [recommended]
  - Constraint relaxation policy [recommended]
  - Geographic broadening [optional]
  - Range tolerance widening [optional]
  - Low-confidence constraint removal [optional]
  - Retry retrieval [optional]
  - Zero-result fallback / recovery [recommended]
- Candidate retrieval
  - Retrieval routing / dynamic retrieval [recommended] — was [optional], under Query processing
  - Lexical & metadata retrieval [core]
  - Learned sparse retrieval [optional]
  - Text vector retrieval [core]
  - Text-to-image vector retrieval [recommended] — was one entry, Image vector retrieval [recommended]
  - Image-to-image vector retrieval [optional] — was one entry, Image vector retrieval [recommended]
  - Multi-vector / passage-level retrieval [optional]
  - Late-interaction retrieval [optional]
- Candidate generation
  - Candidate union [core]
  - Candidate pruning [recommended]
  - Candidate budget allocation [optional]
- Fusion
  - Fusion [core] — was [recommended]
  - Reciprocal Rank Fusion [recommended]
  - Weighted score fusion [optional]
  - Query-dependent fusion [optional]
  - Learned fusion [optional]
  - Modality-weighted fusion [optional]
- Reranking
  - Semantic rerank [optional] — was [recommended]
  - Cross-encoder rerank [recommended]
  - VLM rerank [optional]
  - Learning-to-Rank [optional]
- Final ranking
  - Business ranking [recommended] — was [core]
  - Freshness / temporal ranking [recommended]
  - Personalisation [optional]
  - Diversity [recommended]
  - Deduplication [recommended] — was [core]
- Results
  - Results assembly [core]
- Feedback and evaluation
  - Behavioural / implicit relevance signals [recommended]
  - Click feedback [recommended]
  - Save / enquiry feedback [recommended]
  - Query reformulation signals [optional]
  - Hard-negative mining [optional]
  - Offline evaluation [recommended] — was [core]
  - Relevance feedback [optional]
  - Online experimentation / A/B testing [recommended]
