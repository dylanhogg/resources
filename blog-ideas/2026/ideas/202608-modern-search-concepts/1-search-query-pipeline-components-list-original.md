# Query search pipeline components

https://chatgpt.com/c/6a880532-f940-83ec-bddc-d7e8fb97d080

https://claude.ai/chat/c3bc1980-cd77-4d70-89a4-8014c9e5d175

Core/Optional/Recommended classification of components in a modern hybrid search query pipeline.

- Query inputs
  - Text query [core]
  - Image query [optional]
- Query processing
  - Query understanding [core]
  - Query normalisation [core]
  - Query rewriting [recommended]
  - Query expansion [optional]
  - Session/context-aware search [optional]
  - Retrieval routing / dynamic retrieval [optional]
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
  - Lexical & metadata retrieval [core]
  - Learned sparse retrieval [optional]
  - Text vector retrieval [core]
  - Image vector retrieval [recommended]
  - Multi-vector / passage-level retrieval [optional]
  - Late-interaction retrieval [optional]
- Candidate generation
  - Candidate union [core]
  - Candidate pruning [recommended]
  - Candidate budget allocation [optional]
- Fusion
  - Fusion [recommended]
  - Reciprocal Rank Fusion [recommended]
  - Weighted score fusion [optional]
  - Query-dependent fusion [optional]
  - Learned fusion [optional]
  - Modality-weighted fusion [optional]
- Reranking
  - Semantic rerank [recommended]
  - Cross-encoder rerank [recommended]
  - VLM rerank [optional]
  - Learning-to-Rank [optional]
- Final ranking
  - Business ranking [core]
  - Freshness / temporal ranking [recommended]
  - Personalisation [optional]
  - Diversity [recommended]
  - Deduplication [core]
- Results
  - Results assembly [core]
- Feedback and evaluation
  - Behavioural / implicit relevance signals [recommended]
  - Click feedback [recommended]
  - Save / enquiry feedback [recommended]
  - Query reformulation signals [optional]
  - Hard-negative mining [optional]
  - Offline evaluation [core]
  - Relevance feedback [optional]
  - Online experimentation / A/B testing [recommended]
