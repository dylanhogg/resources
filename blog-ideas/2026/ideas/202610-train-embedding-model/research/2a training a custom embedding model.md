# Training domain-specific embedding models

https://chatgpt.com/c/6ad44fef-b700-83ec-bf97-0f6835759d08

> Search for a list of resources, papers, and examples of training a custom embedding model with data for a specific domain (e.g. ModernBERT, Qwen3-Embedding, etc)

Training domain-specific embedding models: papers, guides and code
I found several useful resources covering fine-tuning open-weight embedding models on custom datasets, including ModernBERT, Qwen3-Embedding, BGE and EmbeddingGemma.
The most relevant approaches are:

- Supervised contrastive fine-tuning: Train on domain-specific query–document pairs, ideally with hard negatives.
- Synthetic data training: Generate queries, relevance labels and negatives from your own documents.
- Unsupervised domain adaptation: Adapt to a specialised corpus when you have few or no relevance labels.
- Knowledge distillation: Use a stronger cross-encoder or LLM to train a smaller, faster embedding model.

1. Best hands-on tutorials and repositories
   These are the resources I'd prioritise for implementing an end-to-end training pipeline.
1. Fine-Tuning ModernBERT on Domain-Specific Data
   Strong match

GitHub notebook · Legal-domain retrieval · ModernBERT
A complete example fine-tuning ModernBERT-embed-base using synthetic legal query–document pairs. Includes baseline evaluation, Matryoshka training, model publishing and post-training evaluation. The author reports improvements exceeding 60% on some retrieval metrics for unseen queries; this is a project-specific result, not a general expectation.
Also provides the training dataset and fine-tuned checkpoint.

GitHub
+1

2. Qwen3-Embedding Official Training Guide
   Official

Qwen · ModelScope SWIFT · 0.6B / 4B / 8B
Direct training instructions for Qwen3-Embedding, including positive pairs, multiple hard negatives, InfoNCE loss, full fine-tuning and support for parameter-efficient methods via SWIFT. Includes data schemas and runnable CLI commands.

GitHub

3. Official ModernBERT Retrieval Training Examples
   ModernBERT · Sentence Transformers · PyLate
   Contains training scripts for single-vector dense retrieval using MS MARCO and GooAQ, plus multi-vector ColBERT-style training using a reranker teacher. Includes BEIR evaluation scripts.

GitHub

4. Sentence Transformers — Training Overview
   General framework · Applicable to many open-weight models
   The central documentation for custom embedding training: datasets, loss functions, trainers, evaluators and query/document prompts. It is a useful foundation for implementing domain-specific retrieval training without relying on model-specific scripts.

Sentence Transformers documentation

5. FlagEmbedding — Fine-Tuning Embedders
   BGE · Contrastive training · Hard-negative mining
   Practical recipes for BGE dense embedding models, including query, pos, neg data, hard-negative mining scripts, multiple training configurations and knowledge distillation. Particularly useful for retrieval-focused training.

GitHub

6. Sentence Transformers — Domain Adaptation
   Unlabelled corpora · GPL · TSDAE · MLM
   Compares approaches for adapting an existing embedding model to specialised text. Explains Generative Pseudo-Labeling (GPL): generating synthetic queries, mining negatives, scoring with a cross-encoder and training a dense retriever.

GitHub

7. Google — Fine-Tune EmbeddingGemma 2
   Google tutorial · Domain-specific triplets · Colab
   An official walkthrough using domain-specific anchor, positive and hard-negative examples. Includes executable notebook options.

Google AI for Developers

2. Additional practical resources
   Resource What it demonstrates
   NVIDIA — Build a Domain-Specific Embedding Model in Under a Day (2026) Complete pipeline: synthetic data generation, hard negatives, contrastive fine-tuning, BEIR evaluation and deployment. Includes a public example dataset and training code. Requires substantial GPU memory for the documented setup. Hugging Face
   Philipp Schmid — Fine-tune Embedding Models for RAG Financial-domain example using 7,000 synthetic pairs from an SEC filing, BGE-base, MultipleNegativesRankingLoss and Matryoshka training. Reports a 7.4% relative nDCG@10 improvement at 768 dimensions. philschmid.de
   Hugging Face — Training and Finetuning Embedding Models Detailed general tutorial covering custom model construction, training data, losses, evaluators and trainer configuration. Hugging Face
   Sentence Transformers — Hard Negative Mining Implementation of mine_hard_negatives() with configurable similarity margins, reranker scoring, sampling and FAISS. Sentence Transformers documentation
   Sentence Transformers — PEFT/LoRA Examples Training embedding models with LoRA adapters, including a retrieval example and comparisons with full fine-tuning. GitHub
   Multi-Vector Embedding Training (2026) ColBERT-style late-interaction embedding training, including dataset, loss and evaluator configuration. An alternative to traditional single-vector embeddings. Hugging Face
   Multimodal Embedding Fine-Tuning (2026) Fine-tuning Qwen3-VL-Embedding-2B on custom text–image pairs for visual document retrieval. Hugging Face
3. Important research papers
   These explain the training strategies used in the repositories above.
   Paper Year Key contribution
   Qwen3 Embedding: Advancing Text Embedding and Reranking 2025 Multi-stage training, synthetic data generation, supervised contrastive fine-tuning and model merging.
   GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval 2022 Generates queries from domain documents, mines negatives and uses cross-encoder pseudo-labels to train a retriever.
   Promptagator: Few-shot Dense Retrieval From 8 Examples 2022 Uses just a few representative examples to generate large task-specific training datasets.
   InPars: Data Augmentation for Information Retrieval using Large Language Models 2022 Generates synthetic query–document training examples from unlabelled corpora.
   NV-Retriever: Improving Text Embedding Models with Effective Hard-Negative Mining 2024 Positive-aware negative mining to improve contrastive training and reduce false-negative contamination.
   Don't Stop Pretraining: Adapt Language Models to Domains and Tasks 2020 Foundational work on domain-adaptive pretraining (DAPT) and task-adaptive pretraining (TAPT).
   Matryoshka Representation Learning 2022 Trains embeddings to retain useful information at multiple truncation dimensions.
   TSDAE: Transformer-based Sequential Denoising Auto-Encoder 2021 Unsupervised sentence-embedding adaptation through denoising, useful where relevance labels are unavailable.

arXiv
+7

For domain-specific retrieval, I would read GPL, Promptagator, NV-Retriever and the Qwen3-Embedding technical report first. Together they cover the main questions of how to construct a domain training set, how to select useful negatives, and how to optimise a pretrained embedding model. 4. Which model should you fine-tune?
For an initial comparison, I'd shortlist these four open-weight checkpoints.
Model Size Why consider it
ModernBERT Embed ~149M Small, efficient, already trained for embeddings; supports query/document prefixes and Matryoshka dimensions.
GTE ModernBERT Base 149M Another strong, retrieval-trained ModernBERT checkpoint with 8,192-token context.
Qwen3-Embedding-0.6B 600M Instruction-aware, multilingual, asymmetric query/document encoding, 1,024-dimensional embeddings.
BGE Base v1.5 109M Lightweight, mature training tooling, extensive examples and good suitability for a baseline.

Hugging Face
+3

Important distinction: answerdotai/ModernBERT-base is a general masked-language-model backbone. nomic-ai/modernbert-embed-base and Alibaba-NLP/gte-modernbert-base have already undergone embedding-specific training. For domain adaptation, I'd normally start with an existing embedding checkpoint rather than learn retrieval behaviour from the original ModernBERT backbone. 5. Suggested training experiment
For a specialised search domain such as real-estate listings, I would start with a small, controlled contrastive-training experiment.
EXAMPLE TRAINING RECORD
Anchor / query
"Quiet apartment near the beach with secure parking"
Positive document

"Two-bedroom apartment, 400 metres from Coogee Beach, secure underground car space, peaceful rear-facing position."
Hard negative

"Modern two-bedroom apartment, 300 metres from Coogee Beach, street parking only, located on a busy road."

This negative is deliberately close to the positive in location and property type but violates important query requirements. It should teach the model task-specific relevance rather than generic textual similarity.

I'd use this progression:

1. Establish zero-shot retrieval baselines using the four models above. Evaluate Recall@k and nDCG@10 against a human-reviewed relevance set.
2. Generate domain-specific query–positive pairs from the corpus, combining real queries with synthetic examples.
3. Mine hard negatives using the baseline retriever and optionally validate them with a cross-encoder to reduce false negatives.
4. Fine-tune with MultipleNegativesRankingLoss or its cached version; use explicit negatives and appropriate query/document prompts.
5. Compare full fine-tuning against LoRA, and potentially Matryoshka training at 256, 512 and 768 dimensions where supported.
6. Evaluate on unseen queries and documents, including a held-out human-labelled test set, and compare quality against embedding latency and index size.
   The Sentence Transformers loss-function guide is particularly useful for choosing the objective. For asymmetric retrieval, maintain the intended query and document encoding conventions throughout training, evaluation and inference.

Sentence Transformers documentation
+1

Recommended reading order
For the most direct path from domain documents to a custom trained embedding model:

1. ModernBERT domain-specific notebook — fastest practical starting point.
2. NVIDIA's 2026 end-to-end training recipe — strongest example of a complete synthetic-data and hard-negative pipeline.
3. Qwen3 official training instructions — for adapting a larger instruction-aware embedding model.
4. GPL paper and NV-Retriever paper — to improve synthetic data quality and negative mining.
   My suggested first experiment: fine-tune nomic-ai/modernbert-embed-base and Qwen3-Embedding-0.6B on the same domain-specific contrastive dataset, keeping the evaluation corpus and relevance judgements fixed. This gives a useful comparison of a small encoder-only architecture against a larger decoder-based embedding model, including the quality–latency trade-off.
