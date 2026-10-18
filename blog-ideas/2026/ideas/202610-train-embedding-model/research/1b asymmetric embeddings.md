# Asymmetric embeddings

https://chatgpt.com/c/6ad44d71-17b0-83ec-82a3-be8ffde1340a

> Describe asymmetric embeddings in embedding models

Asymmetric embeddings are a particularly important architecture for search, where queries and documents play different roles.

For example:

- Query: "coastal family home with a pool"
- Document: A 500-word property listing describing a four-bedroom house, swimming pool, garden and proximity to the beach.

Rather than treating both inputs identically, an asymmetric embedding model learns to map a short expression of intent to a longer description of relevant content.

There are three main approaches:

| Architecture                 | Query encoder                                     | Document encoder                                   | Example                                  |
| ---------------------------- | ------------------------------------------------- | -------------------------------------------------- | ---------------------------------------- |
| Prompt-conditioned           | Shared weights, query-specific prefix/instruction | Shared weights, different prefix or no instruction | E5, Qwen3, Nomic                         |
| Adapter-asymmetric           | Shared backbone + query-specific adapter          | Shared backbone + document-specific adapter        | Jina Embeddings v3                       |
| Two-tower (separate weights) | Independently trained query tower                 | Independently trained document tower               | DPR, custom Sentence Transformers Router |

The distinction matters: many models advertised as asymmetric are actually shared-weight models with different input conditioning. A true two-tower model offers greater architectural flexibility, including using a smaller model for queries and a larger one for documents.

## 1. Open-weight models supporting asymmetric retrieval

| Model                                                                                                              | Mechanism                     | Query/document differentiation                        | Further training           |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------- | ----------------------------------------------------- | -------------------------- |
| [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B?utm_source=chatgpt.com)                         | Query instruction             | `Instruct: ... Query: ...` versus plain document      | LoRA / full                |
| [E5 / multilingual E5](https://huggingface.co/intfloat/multilingual-e5-large?utm_source=chatgpt.com)               | Task prefixes                 | `query:` versus `passage:`                            | Full / LoRA                |
| [Nomic Embed v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5?utm_source=chatgpt.com)                   | Task prefixes                 | `search_query:` versus `search_document:`             | Full / LoRA                |
| [Snowflake Arctic Embed v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0?utm_source=chatgpt.com) | Query prefix                  | `query:` versus plain document                        | Full / LoRA                |
| [BGE v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5?utm_source=chatgpt.com)                                   | Query instruction             | Query retrieval instruction versus plain document     | Full / LoRA                |
| [Mixedbread Embed Large](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1?utm_source=chatgpt.com)         | Query instruction             | Query retrieval instruction versus plain document     | Full / LoRA                |
| [GTE Qwen2 Instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct?utm_source=chatgpt.com)            | Query instruction             | Task-specific query versus plain document             | LoRA / full                |
| [Jina Embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3?utm_source=chatgpt.com)                      | Different LoRA adapters       | `retrieval.query` versus `retrieval.passage`          | Adapter / full             |
| [DPR](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base?utm_source=chatgpt.com)                  | Separate encoders             | Dedicated question and context checkpoints            | Independent towers         |
| [ColBERT](https://github.com/stanford-futuredata/ColBERT?utm_source=chatgpt.com)                                   | Role-conditioned multi-vector | Query/document markers and different token processing | Contrastive / distillation |

The models' documented query/document conventions are important: E5 and Nomic require the respective prefixes, while BGE v1.5 is more tolerant of omitting its optional query instruction. Qwen3 explicitly recommends a task-specific instruction on the query side.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

intfloat/e5-large-v2 at main

+6

Jina v3 is particularly interesting because its query/document asymmetry is implemented using different learned LoRA adapters on a shared backbone, rather than just text prefixes. Its checkpoint is CC BY-NC 4.0, however, so commercial usage requires additional consideration.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

jinaai/jina-embeddings-v3 at main

+1

## 2. True asymmetric architectures are more interesting for training

A conventional shared encoder has:

\\[ s(q,d)=f\_\theta(q)^\top f\_\theta(d) \\]

A genuine asymmetric dual-encoder instead has:

\\[ \boxed{s(q,d)=f\_{\theta_q}(q)^\top g\_{\theta_d}(d)} \\]

The query and document encoders can have completely different architectures, provided their outputs are trained into a compatible embedding space.

Independent towers, jointly aligned during training.

For high-throughput search, this opens up a compelling architecture: train a relatively expensive document encoder offline, then distil a smaller, low-latency query encoder against its document embedding space.

An especially useful training pattern is to freeze the document encoder and only train the query encoder. This can preserve compatibility with an existing vector index while improving retrieval relevance.

### A concrete, production-oriented example

[OpenSearch Neural Sparse Encoding v3](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill?utm_source=chatgpt.com) implements a genuinely asymmetric sparse retrieval model:

- Query side: tokenizer plus a small static learned weight lookup.
- Document side: DistilBERT-based neural sparse encoder.
- Matching: sparse dot product across the shared vocabulary space.
- Licence: Apache 2.0.

The query path does not require Transformer inference. This is especially relevant for high-QPS retrieval, although it is a different approach from dense ANN embeddings.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

## 3. How I would fine-tune these models

I would compare three experiments against the same retrieval evaluation set.

| Experiment                  | Training                                                                | Main advantage                                         |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------ |
| A. Shared-weight asymmetric | Qwen3 or Nomic, with query/document formatting and contrastive training | Simple implementation and strong baseline              |
| B. Separate adapters        | Shared backbone, independent query and document LoRA adapters           | Greater specialisation without separate full backbones |
| C. Independent towers       | Freeze a strong document encoder; train/distil a small query model      | Low online cost, potential to reuse an existing index  |

For experiment C, the [Sentence Transformers Router](https://sbert.net/docs/package_reference/base/modules.html?utm_source=chatgpt.com) is directly relevant. It explicitly supports different query and document model architectures, including a static query encoder and a Transformer document encoder. Crucially, the documentation warns that independently pretrained encoders require additional training to align their vector spaces.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://sbert.net&sz=32)

Sentence Transformers documentation

+1

For a quick starting baseline, asymmetric Nomic encoding looks like:

```
from sentence_transformers import SentenceTransformermodel = SentenceTransformer(    "nomic-ai/nomic-embed-text-v2-moe",    trust_remote_code=True,)queries = ["coastal family home with a pool"]documents = [    "Four-bedroom house near the beach "    "with a swimming pool and garden."]q = model.encode_query(queries, normalize_embeddings=True)d = model.encode_document(documents, normalize_embeddings=True)scores = q @ d.T
```

These methods apply the configured query and passage prefixes automatically for this model.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

nomic-ai/nomic-embed-text-v2-moe at main

+1

## 4. Recommended research direction

For further training and evaluation, I would prioritise:

- Qwen3-Embedding-0.6B: strongest starting candidate for conventional dense asymmetric fine-tuning.
- Nomic Embed v2 MoE: efficient, permissively licensed, prompt-asymmetric architecture.
- Jina v3: useful reference design for query/document-specific adapters, despite its non-commercial licence.
- Sentence Transformers Router with distinct encoders: best for building a custom, truly asymmetric dense embedding model.
- OpenSearch neural sparse v3: an interesting example of extreme query/document computational asymmetry.

The most interesting research question is whether a small query encoder trained against a frozen, larger document encoder can approach the relevance of a much larger shared embedding model, while preserving index compatibility and reducing online latency.

I would evaluate that directly against a conventional shared Qwen3 model using nDCG\@10, Recall\@100, query-encoding latency and index storage. The key is to distinguish improvements due to asymmetric representation learning from those due to simply having a better underlying model.
