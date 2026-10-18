# Top embedding model architectures

https://chatgpt.com/c/6ad44d71-17b0-83ec-82a3-be8ffde1340a

> What is a list of top embedding model architectures? In particular openweight/opensource models and architectures that can be further trained

As of October 2026, the strongest open-weight embedding models span several distinct architectures. For a project where you want to fine-tune, distil, or continue training an embedding model, I would prioritise these five families:

1. Qwen3-Embedding — strong general-purpose text retrieval, particularly if you want an instruction-aware, decoder-based architecture.
2. BGE-M3 — particularly interesting because a single model supports dense, learned sparse and multi-vector retrieval.
3. ModernBERT / E5 / GTE — efficient encoder-based architectures with mature fine-tuning tooling.
4. ColBERT / ColPali — multi-vector architectures for more expressive retrieval, with higher indexing and serving costs.
5. SigLIP2 / BGE-VL — multimodal architectures for text–image similarity and retrieval.

There are also newer contenders, including NVIDIA Nemotron-3-Embed, Nomic Embed v2 MoE and Jina Embeddings v5.

The important distinction is between the neural network backbone, the retrieval representation, and the training objective. These can be combined in different ways.

## 1. Main embedding architecture families

| Architecture                    | Representation                                  | Strengths                                                             | Limitations                                |
| ------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------ |
| Encoder-only bi-encoder         | One dense vector per input                      | Fast, efficient, straightforward to train                             | Loses token-level detail                   |
| Decoder-based bi-encoder        | One dense vector, often last-token pooled       | Strong semantic understanding; leverages pretrained LLMs              | Typically more expensive                   |
| Sparse neural encoder           | Learned token-weight vector                     | Lexical matching, interpretability, exact terms                       | Large vocabulary-space representation      |
| Late-interaction / multi-vector | Multiple vectors per input, typically per token | Fine-grained matching and better relevance for some tasks             | Larger indices, more complex serving       |
| Unified hybrid encoder          | Dense + sparse + multi-vector                   | Several retrieval signals from one backbone                           | Training and indexing complexity           |
| Multimodal dual-encoder         | Text and images in a shared space               | Cross-modal retrieval; separately indexable image and text embeddings | Requires suitable image–text training data |
| Multimodal late-interaction     | Multiple text/visual vectors                    | Detailed matching of pages, images and visual regions                 | Substantially higher storage and compute   |

For learning or experimenting with embedding training, I would start with encoder-only dense contrastive retrieval, then investigate decoder-derived encoders, multi-vector retrieval and multimodal contrastive learning.

These are not mutually exclusive categories. For example, Matryoshka Representation Learning is a training technique for making truncated dense vectors useful; it is not a separate backbone architecture.

## 2. Best open-weight text embedding models

These are the models I would shortlist for research and further training. They're not ranked by a single benchmark score: architecture, licensing, cost and ease of adaptation all influence the selection.

| Model family                                                                                                       | Architecture / size                   | Why it's interesting                                                    | Licence                     |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ----------------------------------------------------------------------- | --------------------------- |
| [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B?utm_source=chatgpt.com)                         | Decoder-only, 0.6B / 4B / 8B          | Strong general retrieval; instruction-aware; 32K context; Matryoshka    | Apache 2.0                  |
| [Microsoft Harrier OSS](https://huggingface.co/microsoft/harrier-oss-v1-0.6b?utm_source=chatgpt.com)               | Decoder-only, 270M / 0.6B / 27B       | Strong 2026 multilingual results; contrastive learning and distillation | MIT                         |
| [Octen Embedding](https://huggingface.co/Octen/Octen-Embedding-0.6B?utm_source=chatgpt.com)                        | Qwen3-derived, 0.6B / 4B / 8B         | Strong retrieval benchmark results; existing LoRA fine-tunes            | Apache 2.0                  |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com)                                                | XLM-R encoder, 569M                   | Dense, sparse and ColBERT-style outputs; unified training               | MIT                         |
| [ModernBERT Embed](https://huggingface.co/lightonai/modernbert-embed-large?utm_source=chatgpt.com)                 | Bidirectional encoder, \~395M         | Efficient, relatively easy full fine-tuning; modern attention           | Apache 2.0                  |
| [Nomic Embed v2 MoE](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe?utm_source=chatgpt.com)               | Encoder MoE, 475M total / 305M active | Efficient MoE research, fully published training pipeline               | Apache 2.0                  |
| [Snowflake Arctic Embed v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0?utm_source=chatgpt.com) | Encoder, \~305M                       | Efficient multilingual retrieval; practical production candidate        | Apache 2.0                  |
| [E5 Instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct?utm_source=chatgpt.com)               | XLM-R encoder, \~560M                 | Proven contrastive retrieval baseline; query instructions               | MIT                         |
| [SPLADE-v3](https://huggingface.co/naver/splade-v3?utm_source=chatgpt.com)                                         | Sparse BERT encoder                   | Learned lexical expansion for inverted-index retrieval                  | CC BY-NC-SA 4.0             |
| [ColBERTv2](https://github.com/stanford-futuredata/ColBERT?utm_source=chatgpt.com)                                 | BERT-style multi-vector               | Fine-grained token matching; mature research implementation             | MIT code; verify checkpoint |
| [Nemotron-3-Embed](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16?utm_source=chatgpt.com)                  | Bidirectional LLM-derived encoder, 8B | Strong multilingual retrieval with mean pooling                         | OpenMDW 1.1                 |
| [Jina Embeddings v5](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano?utm_source=chatgpt.com)            | EuroBERT-derived, 239M / 677M         | Compact distillation-focused models                                     | CC BY-NC 4.0                |

The Qwen3, BGE-M3, Nomic and ModernBERT specifications come from their model cards and training repositories.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+3

&#x20;Microsoft Harrier and Octen are also worth evaluating against their parent Qwen3 checkpoints rather than assuming the original remains strongest.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+2

### Especially interesting architecturally

BGE-M3 — unified representations

Three trainable retrieval heads share one backbone: dense embeddings, learned sparse weights and token-level multi-vectors. This makes it an unusually useful research platform for comparing retrieval strategies under closely related representations.

Reference: [FlagEmbedding training recipes](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

Nomic Embed v2 — sparse mixture of experts

Eight experts with top-2 routing; approximately 475M total parameters but 305M active. Both its training code and training data are released, making it attractive for architectural experiments, not just conventional fine-tuning.

Reference: [Nomic Contrastors](https://github.com/nomic-ai/contrastors?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

ModernBERT — efficient bidirectional encoding

Uses local/global attention and modern Transformer optimisations. The underlying ModernBERT backbone has 149M and 395M variants; use a retrieval-trained derivative such as ModernBERT Embed, or train your own embedding head and contrastive objective.

Reference: [ModernBERT backbone](https://huggingface.co/answerdotai/ModernBERT-base?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

Nemotron-3-Embed — LLM converted to an encoder

Based on Ministral-3-8B, but using bidirectional attention and average pooling rather than a conventional causal decoding setup. Interesting for investigating whether generative LLM pretraining followed by encoder adaptation offers advantages over traditional BERT-style models.

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

nvidia/Nemotron-3-Embed-8B-BF16 at main

+1

## 3. Open-weight multimodal embedding architectures

For text-to-image and image-to-image retrieval, these deserve their own shortlist.

| Model                                                                                                         | Architecture                                                 | Best application                                               | Licence               |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- | --------------------- |
| [SigLIP2](https://huggingface.co/collections/google/siglip2?utm_source=chatgpt.com)                           | Contrastively trained text/vision dual-encoder               | General image–text retrieval; fine-tuning image–text alignment | Apache 2.0            |
| [BGE-VL](https://huggingface.co/BAAI/BGE-VL-base?utm_source=chatgpt.com)                                      | CLIP-derived dual-encoder                                    | General multimodal retrieval, including composed queries       | MIT                   |
| [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0?utm_source=chatgpt.com)                                | VLM with ColBERT-style multi-vector output                   | Visual-document retrieval and fine-grained image matching      | Apache 2.0            |
| [Jina Embeddings v4](https://huggingface.co/jinaai/jina-embeddings-v4?utm_source=chatgpt.com)                 | Multimodal VLM with single- and multi-vector representations | Visually rich documents and text–image retrieval               | Qwen Research License |
| [Jina Embeddings v5 Omni](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small?utm_source=chatgpt.com) | Multimodal embedding architecture                            | Cross-modal retrieval involving image and other media          | CC BY-NC 4.0          |

SigLIP2 and BGE-VL are particularly suitable if your goal is to retain a conventional ANN index with one vector per document or image. ColQwen2 is more appropriate when the additional fine-grained matching justifies storing many vectors per item.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

a google Collection

+3

## 4. Which architectures are easiest to train further?

| Research or engineering goal                  | Preferred starting point           | Training approach                                              |
| --------------------------------------------- | ---------------------------------- | -------------------------------------------------------------- |
| Domain adaptation with limited compute        | ModernBERT Embed, Snowflake Arctic | Full fine-tuning or LoRA with contrastive loss                 |
| Highest-quality general text embeddings       | Qwen3 / Harrier / Octen            | LoRA with hard negatives and teacher distillation              |
| Experiment with dense versus learned sparse   | BGE-M3                             | Joint or separately weighted retrieval objectives              |
| Research model efficiency                     | Nomic Embed v2 MoE                 | Expert routing, distillation, Matryoshka                       |
| Improve lexical retrieval                     | SPLADE, BGE-M3 sparse head         | Sparse regularisation, hard-negative training                  |
| Improve fine-grained matching                 | ColBERT                            | Multi-vector contrastive or distillation training              |
| Train property-specific text–image similarity | SigLIP2, BGE-VL                    | Image–text contrastive learning with domain-specific negatives |
| Train visual-document retrieval               | ColQwen2                           | LoRA on the VLM and projection layers                          |

For newer training infrastructure, [Hugging Face's August 2026 multi-vector training guide](https://huggingface.co/blog/train-multi-vector-encoder?utm_source=chatgpt.com) and [Sentence Transformers training documentation](https://www.sbert.net/docs/sentence_transformer/training_overview.html?utm_source=chatgpt.com) are useful starting points. Sentence Transformers v6 now supports multi-vector training directly, rather than requiring the original ColPali engine.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+2

## 5. Training objectives worth investigating

The major gains from further training often come from the data and objective rather than changing the backbone.

| Technique                                 | Purpose                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| InfoNCE / Multiple Negatives Ranking Loss | Pull positive query–document pairs together and negatives apart          |
| Hard-negative mining                      | Teach the model nuanced distinctions between plausible matches           |
| Teacher–student distillation              | Transfer the relevance behaviour of a larger embedding model or reranker |
| Matryoshka Representation Learning        | Train useful embeddings at multiple truncated dimensions                 |
| Contrastive multimodal learning           | Align text and image representations                                     |
| Multi-vector MaxSim learning              | Preserve token/patch-level interactions                                  |
| Sparse regularisation                     | Encourage efficient, indexable lexical representations                   |
| Multi-task training                       | Combine retrieval, classification, clustering or similarity objectives   |

For example, contrastive retrieval fine-tuning commonly optimises:

\\[ \mathcal{L}=-\log\frac{\exp(s(q,d^+)/\tau)} {\exp(s(q,d^+)/\tau)+\sum_j\exp(s(q,d_j^-)/\tau)} \\]

Where \\(d^+\\) is the relevant document and \\(d_j^-\\) are negatives. The quality of the negative examples is critical, especially for domain-specific retrieval.

If dimensionality is an important design constraint, Sentence Transformers also supports [MatryoshkaLoss](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html?utm_source=chatgpt.com) combined with contrastive objectives, so models can be explicitly trained for shorter embeddings instead of simply truncated afterwards.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.sbert.net&sz=32)

Sentence Transformers documentation

+1

## 6. My recommended shortlist for a training experiment

Text-only retrieval

5 candidates

Qwen3-Embedding-0.6B

First choice

Strong combination of pretrained retrieval quality, manageable size, available fine-tuning techniques and a permissive licence.

Microsoft Harrier OSS 0.6B

Strong alternative using the same general architecture family, but a different training recipe.

BGE-M3

Best for explicitly testing dense, sparse and multi-vector retrieval within one model family.

ModernBERT Embed

Good efficiency-oriented baseline, particularly if full fine-tuning and CPU inference matter.

Nomic Embed v2 MoE

Particularly useful for investigating training reproducibility, expert routing and embedding compression.

For multimodal retrieval, I would begin with SigLIP2 versus BGE-VL, then add ColQwen2 if fine-grained visual matching looks promising.

A practical evaluation would compare the unchanged checkpoint, a domain-adapted fine-tune and a distilled smaller model. Measure nDCG\@10 and Recall\@K alongside query latency, document-encoding throughput, index size and quality after truncating or quantising the embeddings.

One licensing caveat: open-weight does not automatically mean open-source or unrestricted commercial use. SPLADE-v3 and Jina v5 have non-commercial checkpoint licences; Jina v4, EmbeddingGemma and NVIDIA Nemotron-3-Embed have different custom terms. For commercially deployed, further-trained models, Apache 2.0 and MIT checkpoints are generally the cleaner starting points.

For current performance comparisons, use the [MTEB leaderboard](https://leaderboard.mteb.org/?utm_source=chatgpt.com) filtered by retrieval task, model size and openness rather than an overall average score.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://leaderboard.mteb.org&sz=32)

MTEB Leaderboard

+1
