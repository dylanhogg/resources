I would read approximately 15 of the 34 papers in depth, then selectively read the remainder. Given your ML background, I'd focus on the engineering decisions involved in adapting an existing embedding model rather than learning transformer fundamentals.

The most useful progression is:

1\. Understand the training approaches

Papers 1–3

2\. Generate domain-specific training data

Papers 4–8

3\. Optimise contrastive training and negatives

Papers 9–13

4\. Optimise deployment and deeper adaptation

Papers 14–15

## Part 1 — Core reading sequence

These are the 15 papers I would prioritise, in order.

| #   | Paper                                                                                           | Why read it                                                                                                                                                                                           |
| --- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Qwen3 Embedding (2025)](https://arxiv.org/abs/2506.05176)                                      | Start with a modern embedding training pipeline: foundation models, unsupervised training, supervised contrastive learning, synthetic data and model merging.                                         |
| 2   | [Don't Stop Pretraining (2020)](https://arxiv.org/abs/2004.10964)                               | Understand domain-adaptive pretraining (DAPT) versus task-adaptive pretraining (TAPT), and why more pretraining can help. Its evidence is primarily for downstream NLP tasks, not retrieval directly. |
| 3   | [SciRepEval (2022)](https://arxiv.org/abs/2211.13308)                                           | Learn why evaluating representations on the right downstream tasks matters. Establish evaluation before training.                                                                                     |
| 4   | [TSDAE (2021)](https://arxiv.org/abs/2104.06979)                                                | Understand unsupervised embedding adaptation using only unlabelled domain text, without relevance annotations.                                                                                        |
| 5   | [GPL (2021)](https://arxiv.org/abs/2112.07577)                                                  | One of the most directly relevant papers. Generate synthetic queries, use a cross-encoder for pseudo-labels and adapt a dense retriever without labelled data.                                        |
| 6   | [InPars (2022)](https://arxiv.org/abs/2202.05144)                                               | Explore LLM-generated synthetic query-document training pairs and their effect on retrieval.                                                                                                          |
| 7   | [Promptagator (2022)](https://arxiv.org/abs/2209.11755)                                         | Learn how a handful of representative examples can guide synthetic training-data generation for a specialised retriever.                                                                              |
| 8   | [Adaptation to Financial Filings via LLM Distillation (2025)](https://arxiv.org/abs/2512.08088) | A more advanced, practical pipeline: iterative hard-example retrieval, LLM relevance judgements and retraining a student encoder.                                                                     |
| 9   | [NV-Retriever (2024)](https://arxiv.org/abs/2407.15831)                                         | Deep dive into hard-negative mining, positive-aware filtering and the problem of false negatives.                                                                                                     |
| 10  | [MedCPT (2023)](https://arxiv.org/abs/2307.00589)                                               | Understand learning a domain-specific retrieval model from very large volumes of real user search and click logs.                                                                                     |
| 11  | [Domain-Aware Text Embeddings for C2C Marketplaces (2025)](https://arxiv.org/abs/2512.21021)    | Particularly relevant to property search: noisy listings, asymmetric query/document representations, interaction-derived positives and production evaluation.                                         |
| 12  | [Mine and Refine (2026)](https://arxiv.org/abs/2602.17654)                                      | Move beyond binary relevance. Learn label-aware contrastive training, hard-negative refinement and graded relevance.                                                                                  |
| 13  | [Semantic Retrieval for Product Search in E-Commerce (2026)](https://arxiv.org/abs/2606.01504)  | Study a two-stage dual-encoder training pipeline for product attributes, near-duplicates, substitutes and graded ranking.                                                                             |
| 14  | [Matryoshka Representation Learning (2022)](https://arxiv.org/abs/2205.13147)                   | Learn how to train embeddings that remain useful when truncated, reducing vector storage, memory and retrieval costs.                                                                                 |
| 15  | [Legal Domain Adaptation of Modern BERT Models (2026)](https://arxiv.org/abs/2606.28538)        | A concrete ModernBERT domain-pretraining case study. Compare continued pretraining with training from scratch, and consider whether MLM adaptation is worthwhile.                                     |

## Part 2 — Further reading, ordered by relevance

Read these after the core papers, based on which approaches appear most promising.

| #   | Paper                                                                                                   | Main reason to read                                                                                  |
| --- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 16  | [Climate-ModernBERT (2026)](https://arxiv.org/abs/2609.07798)                                           | Domain corpus composition, continued pretraining and model merging.                                  |
| 17  | [Bi-semantic Chemical Embedder (2026)](https://arxiv.org/abs/2608.03855)                                | Particularly useful two-stage recipe: MLM adaptation followed by Matryoshka contrastive training.    |
| 18  | [Towards Domain Specification of Embedding Models in Medicine (2025)](https://arxiv.org/abs/2507.19407) | GTE adaptation and domain-specific benchmark construction.                                           |
| 19  | [Compass-Embedding v4 (2025)](https://arxiv.org/abs/2601.11565)                                         | False-negative masking, multilingual training, FP8 quantisation and efficient inference.             |
| 20  | [Learn Before Represent (2026)](https://arxiv.org/abs/2601.11124)                                       | An advanced alternative: inject domain knowledge before contrastive representation learning.         |
| 21  | [CardioEmbed (2025)](https://arxiv.org/abs/2511.10930)                                                  | A concrete Qwen3-Embedding-8B domain fine-tuning example using a relatively small curated corpus.    |
| 22  | [Neighborhood Contrastive Learning / SciNCL (2022)](https://arxiv.org/abs/2202.06671)                   | Learn to construct positive and negative samples using graph neighbourhoods.                         |
| 23  | [Contrastive Learning Using Graph Embeddings (2025)](https://arxiv.org/abs/2510.04631)                  | Transfer the graph-derived contrastive learning approach to proprietary industrial data.             |
| 24  | [SAILER (2023)](https://arxiv.org/abs/2304.11370)                                                       | Incorporating structured document fields and domain-specific elements into retrieval training.       |
| 25  | [TermGPT (2025)](https://arxiv.org/abs/2511.09854)                                                      | Fine-grained domain terminology discrimination using token- and sentence-level contrastive learning. |
| 26  | [BioLORD-2023 (2023)](https://arxiv.org/abs/2311.16075)                                                 | Integrating knowledge graphs, LLM-generated data and self-distillation.                              |

## Part 3 — Specialised papers to skim

These are less directly relevant to training a real-estate text retriever, although some contain transferable ideas.

| #   | Paper                                                                               | Useful angle                                                                                        |
| --- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 27  | [BioClinical ModernBERT (2025)](https://arxiv.org/abs/2506.10896)                   | Large-scale continued pretraining with heterogeneous corpora.                                       |
| 28  | [Clinical ModernBERT (2025)](https://arxiv.org/abs/2504.03964)                      | Domain corpus and ontology integration.                                                             |
| 29  | [Patent Language Model Pretraining (2025)](https://arxiv.org/abs/2509.14926)        | Domain-specific pretraining at scale and tokenizer choices.                                         |
| 30  | [Russian ModernBERT Legal Adaptation (2026)](https://arxiv.org/abs/2610.06715)      | Important caution: improved MLM loss does not necessarily translate to downstream task improvement. |
| 31  | [Spatial ModernBERT (2025)](https://arxiv.org/abs/2507.08865)                       | Adding spatial information, but primarily for document extraction rather than retrieval embeddings. |
| 32  | [Structure-Aligned Protein Language Model (2025)](https://arxiv.org/abs/2505.16896) | Aligning representations with a different structural modality.                                      |
| 33  | [DNABERT-2 (2023)](https://arxiv.org/abs/2306.15006)                                | Domain-specific tokenisation and compute-efficient pretraining.                                     |
| 34  | [ProtModernBERT (2026)](https://arxiv.org/abs/2610.19738)                           | Architectural efficiency and pretraining compute trade-offs in a specialised sequence domain.       |

## How I would apply the reading to a property-search embedding project

The sequence above is designed to answer four engineering questions.

| Question                                            | Most useful papers | Experiment to run                                                                                                |
| --------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Can I improve retrieval without domain pretraining? | 1, 5, 6, 7         | Fine-tune an existing embedding model using synthetic query–listing pairs and compare against a frozen baseline. |
| Where should my training signals come from?         | 8, 10, 11          | Compare synthetic pairs, click-derived pairs, structured-attribute matches and LLM-judged relevance.             |
| How do I improve relevance discrimination?          | 9, 12, 13          | Mine hard negatives, filter false negatives and experiment with graded-relevance objectives.                     |
| Is additional pretraining worth the compute?        | 2, 4, 15, 16, 17   | Compare direct contrastive fine-tuning against continued pretraining followed by contrastive fine-tuning.        |

For real estate, I would pay particular attention to false negatives and attribute-sensitive relevance. Two listings might be semantically similar but differ in an essential attribute such as land size, property type, parking or location. Their relevance depends on the query, not simply their textual similarity.

I would therefore prioritise supervised or weakly supervised retrieval adaptation over expensive domain continued pretraining until evaluation demonstrates that the base encoder lacks important domain knowledge.

## The shorter path: only eight papers

If you want the fastest route to implementing a strong first experiment, I would read these in this order:

1. Qwen3 Embedding — modern embedding training recipe.
2. GPL — adapting with unlabelled domain documents.
3. Promptagator — generating task-specific queries.
4. Financial Filings via LLM Distillation — iterative teacher/student training.
5. NV-Retriever — hard-negative selection.
6. C2C Marketplace Embeddings — highly relevant production case study.
7. Mine and Refine — graded relevance and false-negative mitigation.
8. Legal Domain Adaptation of ModernBERT — deciding whether to add continued pretraining.

One limitation of this collection: it is overwhelmingly focused on text embeddings, not multimodal text-to-image or image-to-image retrieval. For the latter, you would want a separate reading track covering CLIP, SigLIP and domain-specific vision-language contrastive fine-tuning.

The central decision I would aim to resolve: Is the quality bottleneck missing domain knowledge in the pretrained model, or missing domain-specific relevance supervision? The latter can often be addressed much more cheaply using synthetic queries, relevance judgements and carefully selected contrastive examples. These papers provide a systematic way to test that distinction rather than assume continued pretraining is required.&#x20;
