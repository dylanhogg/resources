# DR training embedding model prompt

Perform [$deep-research](app://connector_openai_deep_research) and conduct a rigorous, source-backed investigation into \*\*how to train, adapt, evaluate and deploy domain-specific embedding models\*\* using millions of proprietary records.

Use \*\*real-estate search\*\* as the primary case study, with lessons drawn from e-commerce, product search, recommendations, biomedical retrieval, legal search and other specialised domains.

The goal is not simply to survey embedding models. It is to determine \*\*whether domain adaptation is worthwhile, which approaches offer the best return on compute and engineering effort, and how to implement them reliably in production\*\*.

Act as a senior ML researcher and engineer specialising in representation learning, embedding models, and large-scale information retrieval.

Develop a **research-backed, practical strategy for training domain-specific embedding models** using millions of proprietary text records (e.g. real-estate listings containing descriptions, attributes, locations and images).

Prioritise retrieval quality, cost efficiency, engineering simplicity, reproducibility and operational scalability.

Primary retrieval tasks:

\| Task | Query | Retrieved documents |\
\| ---------------- | -------------------------------------------- | ---------------------------------------------------------------- |\
\| Text-to-text | Natural-language property search | Listing descriptions and structured text |\
\| Text-to-image | Natural-language visual description | Property photographs |\
\| Image-to-image | Reference photograph | Visually or semantically similar photographs |\
\| Text-to-property | Natural-language search | Entire properties represented by text, metadata and images |

Cover the following:

1. **Base model selection:** Compare leading open-weight architectures (e.g. ModernBERT, Qwen3-Embedding, BGE, E5, GTE, Jina, SigLIP, CLIP etc). Consider encoder vs decoder, bi-encoders, asymmetric query/document embeddings, multilingual support, model size, embedding dimensions and licensing.
2. **Text and multimodal approaches:** Examine text-to-text, text-to-image and image-to-image retrieval. Compare adapting existing multimodal models against training separate specialised encoders, including shared embedding spaces and cross-modal alignment.
3. **Training data preparation:** Leverage millions of unlabelled domain records, structured metadata, images, historical interactions and synthetic queries. Cover weak supervision, positive/negative pair construction, hard-negative mining, deduplication, data quality and leakage prevention.
4. **Training pipeline:** Compare continued pretraining, contrastive learning, supervised fine-tuning, knowledge distillation, LoRA/PEFT and full fine-tuning. Explain objectives, losses, training stages and practical frameworks (e.g. Sentence Transformers, Transformers, PEFT).
5. **Compute and cost:** Propose realistic training configurations for (a) a single NVIDIA L4 24 GB GPU and (b) larger multi-GPU infrastructure. Estimate dataset sizes, training duration, memory requirements, GPU-hours and approximate costs. Identify feasibility limits and efficiency optimisations.
6. **Evaluation:** Design robust offline benchmarks for domain-specific semantic retrieval, including Recall@K, nDCG@K, MRR, hard negatives, out-of-domain generalisation and cross-modal retrieval. Include baseline comparisons, statistical significance, ablations and online evaluation.
7. **Serving and inference:** Compare deployment options, embedding latency, throughput, batching, quantisation, dimensionality reduction, ANN indexing, vector storage and integration with OpenSearch/Qdrant hybrid retrieval. Consider re-embedding millions of records and ongoing updates.
8. **Prior art:** Identify relevant papers, implementations and case studies involving domain-adapted embeddings, particularly real estate/property search, e-commerce, product search, recommendations, biomedical and legal domains. Distinguish peer-reviewed evidence from vendor claims, with links to sources.
9. **Recommended approach:** Propose a staged, cost-conscious implementation roadmap, from strong pretrained baselines through lightweight adaptation to larger-scale training. Include concrete model choices, training recipes, expected benefits, compute budgets, decision gates and criteria for stopping or scaling.

Provide a comparison table covering architecture, size, dimensionality, licence, training requirements, inference characteristics, benchmark evidence and suitability for this use case.\
\*\*Do not assume that the newest or largest model is necessarily the best adaptation starting point.\*\*

Assess whether \*\*text-only adaptation should be pursued independently from multimodal adaptation\*\*, and whether a single shared vector space offers sufficient benefit to justify its complexity.

**Research requirements:**

- Prioritise recent research, reproducible experiments and open-source implementations.
- Distinguish established evidence from hypotheses and estimates.
- Quantify performance, cost and training time wherever credible data exists.
- Discuss when domain adaptation is unlikely to outperform a strong general-purpose embedding model.
- Compare alternatives in concise tables with explicit trade-offs.
- Provide a clear final recommendation optimised for **retrieval quality, engineering simplicity, compute efficiency and production scalability**.

Write for an experienced ML engineering audience. Be technically rigorous, concise and actionable, with links to primary sources rather than generic explanations.
