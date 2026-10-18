# Domain-specific embedding models for proprietary real-estate search

https://chatgpt.com/c/6ad45658-2624-83ec-b82a-1fbe454419be

## Executive judgement

For a corpus of millions of proprietary property records, **domain adaptation is worth testing, but large-scale continued pretraining or training an embedding model from scratch is not the place to start**. The strongest evidence points to a much cheaper intervention: start from an already retrieval-trained bi-encoder, create in-domain query–document supervision from interactions and synthetic queries, mine difficult negatives, and fine-tune with a contrastive objective. GPL demonstrated improvements of up to **9.3 nDCG@10 points** over an out-of-domain dense retriever on specialised domains; Promptagator and InPars independently showed that synthetic task-specific queries can materially improve retrieval with little or no manually labelled data. Conversely, GPL found that only half of the unsupervised pretraining techniques it tested actually helped retrieval, with the best continued-pretraining method adding only another **1.4 nDCG@10 on average after GPL**.

The practical implication is that **retrieval adaptation should come before language-model domain adaptation**. A raw language encoder such as ModernBERT is attractive technologically, but a retrieval-trained ModernBERT derivative such as `gte-modernbert-base` is a much better starting point: it keeps the 149M-parameter, 8,192-token encoder architecture while already having an embedding objective and a 768-dimensional output. ModernBERT itself was pretrained on roughly two trillion tokens, making “recreate domain pretraining at comparable scale” economically irrational for a property-search project.

For an English-first Australian real-estate system, I would make **GTE-ModernBERT-base the primary low-cost adaptation candidate**, with **Qwen3-Embedding-0.6B as the stronger, more expensive baseline/challenger**. BGE-M3 and multilingual-E5-large-instruct are important controls when multilingual search matters. Qwen3-Embedding comes from a considerably heavier decoder-model family, but the 0.6B checkpoint is still manageable and was explicitly trained through unsupervised embedding pretraining, supervised fine-tuning and synthetic-data generation; the series spans 0.6B, 4B and 8B and is Apache-2.0 licensed.

**Text-only adaptation should be pursued independently from multimodal adaptation.** Real-estate text retrieval has far more and cleaner supervision—search logs, listing metadata, saves, enquiries, inspection actions and synthetic queries—and a small text encoder is much cheaper to retrain and serve. For visual retrieval, use a separate image–text embedding space initially: **SigLIP2 is my preferred efficient adaptation target**, while **Qwen3-VL-Embedding-2B is an excellent unified multimodal quality ceiling/teacher**. Qwen3-VL-Embedding maps text, images, document images and video into a common space, uses Matryoshka representations, and is released in 2B and 8B sizes; the authors report 77.8 on MMEB-V2 for the 8B model.

I would therefore deploy **two vector spaces, not one universal space**:

| Production space             | Purpose                                                                             | Recommended starting model                                          | Why                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| Text/property semantic space | Natural-language query → listings/properties                                        | GTE-ModernBERT-base; Qwen3-Embedding-0.6B challenger                | Highest supervision density, cheapest adaptation and serving   |
| Visual multimodal space      | Text → photo and photo → photo                                                      | SigLIP2-B initially; Qwen3-VL-Embedding-2B as ceiling               | Naturally supports cross-modal and visual similarity retrieval |
| Property ranking             | Combine text, images, structured constraints, lexical match and behavioural signals | Late fusion/reranker, **not another universal embedding initially** | Avoids forcing heterogeneous relevance signals into one metric |

A single universal text/image/property embedding has an appealing operational story, but it couples model updates, increases query cost and forces text relevance, visual resemblance and property-level utility to share one geometry. Qwen3-VL proves that such a shared space is technically viable; it does **not** prove that it is the optimum production architecture for a specialised text-heavy marketplace.

The central decision rule should be:

> **Adapt only when an in-domain benchmark demonstrates a statistically credible gap. Scale training only when the previous cheaper stage has delivered a meaningful gain.**

MTEB itself found that no embedding method dominated across all embedding tasks, while BEIR showed why out-of-domain retrieval is difficult. Public leaderboard position is therefore useful for choosing candidates, but it is not sufficient evidence that a model will rank Australian property listings correctly.

## Model landscape and retrieval architecture

### Candidate model comparison

The table below deliberately favours models with reproducible implementations and commercially workable licensing rather than simply choosing the largest recent checkpoint.

| Model / family                     | Architecture and size                                       | Embedding / context                                                                                                     | Language support         | Licence                                                                      | Adaptation characteristics                                                                                                                | Inference characteristics                                                                      | Evidence and suitability                                                                                                                            |
| ---------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ModernBERT-base**                | Bidirectional encoder, 149M; raw masked-LM                  | Hidden size 768; 8,192 tokens                                                                                           | Primarily English + code | Apache-2.0                                                                   | Very easy full fine-tuning on 24 GB; but **not itself retrieval-trained**                                                                 | Excellent size/latency class                                                                   | ModernBERT is an efficient modern encoder trained on 2T tokens. Use mainly when you explicitly want to build the retrieval objective yourself.      |
| **GTE-ModernBERT-base**            | ModernBERT bi-encoder, 149M                                 | 768; 8,192 tokens                                                                                                       | English                  | Open checkpoint; verify exact derived-checkpoint terms during release review | **Excellent L4 target**; full FT is simple; supports conventional query/document contrastive training                                     | Likely the best cost/quality candidate here; substantially smaller than 0.6B–4B alternatives   | Official card identifies 149M, 768-dimensional output and 8K context. My default text adaptation target.                                            |
| **Qwen3-Embedding**                | Decoder-foundation-model-derived embeddings; 0.6B / 4B / 8B | 0.6B up to 1,024 dimensions; larger checkpoints higher-dimensional; 32K-class long context; configurable dimensionality | 100+ languages           | Apache-2.0                                                                   | Strong pretrained retrieval starting point. 0.6B: LoRA or careful full FT on L4; 4B/8B: multi-GPU preferred                               | More expensive per token than 149M encoder; large models attractive as teacher/quality ceiling | Multi-stage unsupervised + supervised training and synthetic data; strong multilingual benchmark results. Best text challenger.                     |
| **BGE-M3**                         | XLM-R-style encoder, ≈0.57B                                 | 1,024; up to 8,192 tokens                                                                                               | 100+ languages           | MIT/open model release                                                       | Can jointly support dense, sparse and multi-vector/late-interaction retrieval; valuable if one wants to experiment beyond a single vector | Heavier than GTE-ModernBERT; multi-functionality can also increase engineering complexity      | Peer-reviewed Findings of ACL work; particularly interesting for multilingual/hybrid retrieval.                                                     |
| **multilingual-E5-large-instruct** | XLM-R-family encoder, roughly 0.56B                         | 1,024                                                                                                                   | Multilingual             | MIT                                                                          | Mature contrastive recipe; query instructions provide useful asymmetric behaviour                                                         | Straightforward encoder serving; heavier than 149M ModernBERT                                  | Multilingual E5 used contrastive pretraining over **1B multilingual text pairs** followed by labelled fine-tuning. Very useful robustness baseline. |
| **Jina Embeddings v3**             | 570M encoder with task LoRA                                 | 1,024 default; Matryoshka down to 32; 8,192 tokens                                                                      | Multilingual             | **CC BY-NC 4.0 for the public v3 checkpoint**                                | Architecturally attractive because retrieval-specific LoRA already demonstrates PEFT-style task adaptation                                | Reasonable sub-billion encoder; MRL useful for index compression                               | Technically strong but **not my default for a commercial proprietary property platform without separate licensing**.                                |
| **SigLIP2**                        | Image/text dual encoders; B/L/So400m/g family               | Shared vision–text space; dimension checkpoint-dependent                                                                | Multilingual             | Apache-2.0 checkpoints                                                       | Excellent target for image-text contrastive adaptation. Start with B; freeze/unfreeze towers progressively                                | Image encoding is naturally offline; text-query tower remains relatively economical            | Paper reports gains over the original SigLIP family at every scale in image-text retrieval and representation transfer.                             |
| **Qwen3-VL-Embedding**             | Unified multimodal model, 2B / 8B                           | Matryoshka; up to 32K inputs; one shared multimodal space                                                               | 30+ languages            | Apache-2.0                                                                   | Powerful teacher/quality ceiling; PEFT rather than full FT unless substantial infrastructure exists                                       | Much higher cost than SigLIP2 or small text encoders                                           | Maps text, images, screenshots, video and mixed-modal inputs into one space; paper reports strong MMEB-V2 performance.                              |

**CLIP should remain in the benchmark as a historical zero-shot control, not as my first adaptation target.** The important methodological improvement for constrained training budgets is the SigLIP family’s pairwise sigmoid formulation: unlike CLIP-style globally normalised contrastive loss, it avoids a global softmax and was specifically shown to behave well without extremely large batches. SigLIP2 then extends that recipe with captioning, self-distillation, masked prediction and better data curation.

### Encoder versus decoder-derived embedding models

The main attraction of a ModernBERT/GTE-style encoder is not that an encoder is intrinsically more semantically accurate. It is the **quality-per-FLOP and adaptation simplicity**. ModernBERT was explicitly motivated by the favourable performance/size trade-off of encoder-only architectures for retrieval and classification. Qwen3-Embedding, in contrast, inherits the richer language modelling capabilities of a decoder foundation model and then turns them into embeddings through a specialised multi-stage training pipeline.

For millions of relatively short property descriptions, an 8K or 32K context window is not itself a major advantage. Most useful property records can be represented in a few hundred tokens once structured attributes are canonicalised. The meaningful advantages of Qwen3 are broader semantic/multilingual capability and stronger pretrained retrieval behaviour; the meaningful advantage of GTE-ModernBERT is that you can afford to experiment much more aggressively.

My text-model bake-off would therefore contain at least:

**BM25 → GTE-ModernBERT-base → Qwen3-Embedding-0.6B → BGE-M3 → multilingual-E5-large-instruct**, plus one large teacher such as Qwen3-Embedding/Reranker 4B or 8B.

The large model should initially be treated as a **teacher and upper bound**, not automatically as the production encoder. Qwen's own model family is deliberately released at 0.6B, 4B and 8B precisely because effectiveness and deployment efficiency occupy different operating points.

### Query/document asymmetry

Real-estate queries and listing documents have radically different distributions:

> “quiet character home near good schools with a leafy garden and period details”

is not distributed like:

> “Immaculately renovated Federation residence, 4 bed, 2 bath, north-to-rear…”

That asymmetry should be represented during training. Initially I would retain **shared weights with role-specific prefixes/instructions or adapters**, because this keeps offline document encoding and online query serving simple. E5's instruction/query formulation and Jina's task-specific LoRA are examples of this general strategy.

Do not split into independent query and document backbones until an ablation proves a gain. There is nevertheless strong precedent for doing so when interaction data is abundant: MedCPT trained separate biomedical query and document encoders from **255 million PubMed query/article interactions**, together with a tightly coupled reranker, and reported state-of-the-art zero-shot results on six biomedical IR tasks.

### Multimodal architecture

For the four target tasks, the simplest high-quality architecture is:

| Task            | Production representation                                                                        | Retrieval                                                |
| --------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| Text → text     | Domain-adapted text embedding of canonical listing/property text                                 | ANN + BM25 hybrid                                        |
| Text → image    | SigLIP2/Qwen3-VL text embedding against individual photograph embeddings                         | ANN over image collection                                |
| Image → image   | Same multimodal model's image tower                                                              | ANN over image collection                                |
| Text → property | Union of text candidates and visually relevant candidates; aggregate to `property_id` and rerank | **Late fusion**, rather than a universal property vector |

For image indexing, keep **individual image vectors associated with a property**, rather than averaging every photograph into one property vector. A request such as “kitchen with green marble island”, “arched windows”, “resort-style pool” or “mid-century façade” may correspond to exactly one photograph. Averaging ten bedroom/bathroom/exterior images tends to erase that salient visual feature. Property-level scores can instead use `max`, top-\(m\) mean, or a learned aggregation over image similarities.

For multimodal training, SigLIP2 is the more natural production adaptation target because it is fundamentally an image/text encoder family and is available from small ViT-B through larger variants. Qwen3-VL-Embedding should be tested alongside it because a genuinely unified space could eventually simplify cross-modal retrieval, but its 2B minimum size changes the deployment economics materially.

## Data and adaptation pipeline

### Build supervision from proprietary records before touching model weights

The millions of raw listings are useful, but **millions of records are not the same as millions of good retrieval labels**. Training quality will be dominated by how query relevance and negatives are constructed.

A real-estate corpus should first be normalised into a canonical representation such as:

```text
property_type: house
location: Newtown, NSW
bedrooms: 3
bathrooms: 2
parking: 1
land_area: 240 sqm
features: freestanding; north-facing rear; period details; renovated kitchen;
          timber floors; courtyard; walk-to-train
description: ...
```

The dense representation should primarily learn **soft semantic intent**. Price, exact bedroom count, exact geographical boundaries, availability dates and similar hard requirements should remain structured fields and filters. Asking a vector space to guarantee “under $1.5m”, “exactly three bedrooms” or “within 2 km” is substantially less reliable than using the underlying typed value.

The training hierarchy I would use is:

| Signal                           | Example                                                  |                          Value | Main risk                                                  |
| -------------------------------- | -------------------------------------------------------- | -----------------------------: | ---------------------------------------------------------- |
| Explicit high-intent interaction | Query → enquiry, inspection booking, application/contact |                      Very high | Sparse; affected by inventory/exposure                     |
| Save/shortlist                   | Query → property saved                                   |                           High | User preference beyond textual relevance                   |
| Click + substantial dwell        | Query → clicked listing                                  |                         Medium | Position bias and presentation effects                     |
| Impression without click         | Query → shown but ignored                                | Useful negative only with care | **Strong exposure/position bias**                          |
| Synthetic query                  | Listing → generated plausible search query               |               High if filtered | Distribution mismatch / hallucination                      |
| Metadata-derived pair            | Feature/attribute → listing                              |                         Medium | Can make task artificially easy                            |
| Text ↔ image/caption             | “sunlit period kitchen” ↔ matching image                 |            High for multimodal | Listing text often does not describe that particular photo |

Recommendation-system work is directly relevant here: Google showed that negative sampling choices in two-tower systems interact with the selection bias inherent in implicit feedback, motivating mixtures rather than simply treating every unclicked item as a true negative.

### Synthetic query generation is high ROI

The evidence for query generation is strong enough that it should be a first-class pipeline, not an emergency substitute for labels. GPL generates target-domain queries, retrieves hard candidates and obtains teacher pseudo-labels; Promptagator showed that only a handful of task demonstrations can seed task-specific synthetic training; InPars likewise found benefits from LLM-generated query–document examples.

Do **not** generate generic paraphrases of descriptions. Generate the actual intent distribution you expect from users. For property search, explicitly stratify synthesis into categories such as:

| Intent family          | Example                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| Feature                | “houses with a butler's pantry and large island”                          |
| Architecture/style     | “Federation homes with original fireplaces”                               |
| Lifestyle              | “quiet family home close to parks and primary schools”                    |
| Spatial                | “apartments near the station but away from the main road”                 |
| Visual                 | “bright minimalist kitchen with concrete floors”                          |
| Compound               | “3 bedroom period home in the inner west with parking and a leafy garden” |
| Loose semantic         | “something coastal and relaxed, not ultra-modern”                         |
| Comparative/substitute | “similar to a terrace but with more outdoor space”                        |
| Negative constraint    | “modern apartment without carpet”                                         |
| Colloquial/misspelled  | Real query-log spelling, abbreviations and suburb nicknames               |

A practical first experiment needs nowhere near every property. I would select **200k–500k high-quality, diverse listings and generate roughly 2–5 queries per listing**, producing roughly 0.5–2M synthetic positives. That is enough to test whether adaptation is working before paying to synthesise against the entire corpus.

For proprietary data, the query generator should be self-hosted or run through an approved private inference environment where contractual/data-governance requirements permit the records to be processed. Store the generator model/version, prompt, decoding settings and source listing revision alongside every synthetic example.

### Positive and negative construction

The basic text loss should be InfoNCE / Multiple-Negatives Ranking Loss:

\[
L_q =
-\log
\frac{\exp(s(q,d^+)/\tau)}
{\exp(s(q,d^+)/\tau)+
\sum_i \exp(s(q,d_i^-)/\tau)}
\]

where \(s\) is cosine similarity or dot product and \(\tau\) is the temperature. Sentence Transformers documents MultipleNegativesRankingLoss/InfoNCE as a common objective for high-performing embedding models and offers cached variants specifically to obtain much larger effective negative batches without storing every activation simultaneously.

The negative curriculum matters as much as the positive data:

| Negative source            | Real-estate example                                            | Stage         |
| -------------------------- | -------------------------------------------------------------- | ------------- |
| Random/in-batch            | Unrelated property elsewhere                                   | Warm-up       |
| BM25 hard negative         | Same words, semantically wrong property                        | Early         |
| Dense hard negative        | Current model retrieves it highly                              | Main training |
| Attribute-confusable       | Same suburb, beds, baths and price but lacks requested feature | Main training |
| Geographical hard negative | Similar suburb/name but wrong location                         | Main training |
| Visual hard negative       | Similar-looking room but missing target feature                | Multimodal    |
| Teacher-selected           | High student score, low cross-encoder relevance                | Distillation  |

Naïvely taking the highest-scoring non-positive neighbour is dangerous because many are actually unlabelled positives. NV-Retriever's positive-aware hard-negative mining explicitly addresses this by using the teacher's positive score as an anchor and filtering likely false negatives; its authors reported a 60.9 MTEB Retrieval score and first place on the retrieval leaderboard when released.

For real estate, this issue is especially severe: ten different homes can all genuinely satisfy “three bedroom terrace near cafés with rear lane access”. Treating nine as negatives just because only one was clicked trains the wrong notion of relevance.

### Distillation should precede ever-larger students

A large reranker is considerably cheaper to use **offline as a label generator** than online on every corpus object. For each training query:

1. retrieve perhaps 50–200 candidates with BM25 + current dense retriever;
2. score those candidates with a stronger cross-encoder/reranker;
3. retain positive and informative negative score margins;
4. distil the ranking into the compact production bi-encoder.

Sentence Transformers supports score-based and margin-based distillation objectives such as MarginMSE; GPL itself used a teacher/pseudo-label pattern rather than trusting generated positives and negatives blindly.

For a positive \(d^+\), negative \(d^-\), teacher \(t\) and student \(s\), the useful quantity is the teacher preference:

\[
\Delta_t=s_t(q,d^+)-s_t(q,d^-)
\]

and the student can minimise:

\[
L\_{\mathrm{distill}}
=
\left[
(s_s(q,d^+)-s_s(q,d^-))-\Delta_t
\right]^2
\]

This is one of the highest-return ways to transfer Qwen3-4B/8B-level judgement into a 149M production encoder.

### Continued pretraining and PEFT

I would use the following ordering:

| Method                                    | When I would use it                                                         | Expected ROI          |
| ----------------------------------------- | --------------------------------------------------------------------------- | --------------------- |
| **Contrastive full fine-tuning**          | 100–600M encoder, good pairs available                                      | **Highest**           |
| **LoRA/PEFT contrastive FT**              | 0.6B+ models; multimodal; rapid experiments                                 | **High**              |
| **Teacher distillation + hard negatives** | Initial adaptation works and further accuracy matters                       | **High**              |
| **Continued masked/domain pretraining**   | Very unusual domain language, abbreviations, jargon; retrieval FT plateaus  | Medium/uncertain      |
| **Full FT of 4B–8B**                      | Large measured gain justifies substantial infrastructure                    | Low-to-medium ROI     |
| **Training from scratch**                 | Essentially only when research/IP constraints make existing models unusable | **Very low ROI here** |

GPL's finding is the reason continued pretraining sits low in this ranking: among six pretrained/domain-adaptation approaches it examined, only three helped, whereas task-aligned pseudo-labelled retrieval training provided much larger improvements.

For a 149M GTE-ModernBERT model, **full fine-tuning is simpler than PEFT and should be the default**. For Qwen3-Embedding-0.6B, start with LoRA and only test full fine-tuning if the adapter leaves a meaningful gap. For 2B+ multimodal or 4B/8B text models, PEFT is the sensible starting point.

Large effective batches matter for contrastive retrieval because they increase negative diversity. Gradient Cache was specifically developed to decouple contrastive batch size from encoder activation memory and gives the same large-batch gradient while processing subsets sequentially; Sentence Transformers' cached ranking losses offer a production-friendly implementation of the same general idea.

### Multimodal domain adaptation

For SigLIP2, create three distinct training pools rather than pretending every listing/image association is equally meaningful:

**Image ↔ image.** Near duplicates, alternate exposures/crops of the same photo, validated same-room photographs and carefully selected semantic matches.

**Caption/query ↔ image.** Generate or curate image-specific descriptions such as “double-height living room with exposed brick” rather than pairing every photo with an entire listing description.

**Property text ↔ image.** Use this as a weaker positive because a description of an entire property often contains facts not visible in a particular photograph.

Do not automatically label two random images from one property as strong positives. A bathroom and aerial exterior from the same listing have property identity in common but little visual semantic similarity.

SigLIP2 is well suited to this phase because its training recipe explicitly combines image–text alignment with additional self-supervised and captioning signals.

A conservative adaptation sequence is:

`freeze vision tower → train projection/text adapters → LoRA both towers → selectively unfreeze full towers only if image retrieval remains limiting`.

For visual retrieval at massive scale, Matryoshka-style dimensions are particularly attractive. The original MRL work showed that nested lower-dimensional representations can dramatically reduce storage/retrieval computation without requiring a separate model at each dimension; Qwen3-VL and several modern text models now incorporate this idea directly.

### Deduplication and leakage prevention

Property data creates unusually severe leakage if the split key is merely `listing_id`. The same physical property may be listed repeatedly, sold and relisted, duplicated across portals, reuse the same copy, or retain nearly identical photos.

The evaluation unit should therefore be a **canonical physical-property cluster**. At minimum cluster using internal property identity/address where legally appropriate, listing lineage, description similarity and image hashes. All versions of a property must land in one split.

A robust split is:

```text
train:
    older canonical properties and interactions

validation:
    later properties, grouped by physical property

test:
    still later period; no physical property/relisting/photo duplicate
    may appear in train or validation
```

Maintain an additional **future-time OOD test** containing new suburbs/projects, uncommon property types and changing market vocabulary. This catches the otherwise easy failure mode in which the embedding model appears excellent because it memorised almost identical relistings.

## Compute and training economics

The figures below are **engineering estimates, not published benchmark timings**. Actual throughput can move several-fold with sequence length, number of explicit negatives, FlashAttention implementation, gradient caching, dataloader behaviour and whether both sides of a pair must be encoded. I would benchmark 500–1,000 training steps on the exact stack and then extrapolate before approving a full run.

As a rough cloud-cost sanity check, Hugging Face's current endpoint catalogue includes an L4 deployment at about **US$0.80/hour** and 4×A100 examples around **US$10/hour total**; these are service examples rather than universal GPU market prices. I therefore use round planning rates of approximately **US$1/L4 GPU-hour, US$3/A100 GPU-hour and US$5–10/H100 GPU-hour** below.

### Single NVIDIA L4 with 24 GB

The L4 is enough to do useful research. It is not enough to make arbitrary model choice irrelevant.

| Experiment                  | Suggested workload                  | Practical configuration                                                  |                      Estimated GPU-hours | Approx. compute budget\* | Recommendation                     |
| --------------------------- | ----------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------: | -----------------------: | ---------------------------------- |
| GTE-ModernBERT 149M full FT | 1M pairs, 1–3 epochs, q≤64, doc≤256 | BF16; microbatch 16–32; cached effective batch 512–2,048                 |                               **8–24 h** |                **$8–24** | **Do first**                       |
| GTE-ModernBERT full FT      | 2M pairs                            | Same                                                                     |                              **15–45 h** |               **$15–45** | Easy if first run wins             |
| Qwen3-Embedding-0.6B LoRA   | 1M pairs                            | BF16/quantised base as supported; LoRA r≈16–32; activation checkpointing |                              **20–60 h** |               **$20–60** | Strong challenger                  |
| BGE-M3 / E5-large PEFT      | 1M pairs                            | Short sequences; checkpointing; cached contrastive loss                  |                              **20–60 h** |               **$20–60** | Only if baseline warrants          |
| ModernBERT DAPT             | 100–300M domain tokens              | MLM, packed sequences                                                    |                              **15–40 h** |               **$15–40** | Defer until retrieval SFT plateaus |
| SigLIP2-B PEFT              | ≈1M 224/256px image-text pairs      | Freeze one tower initially; mixed precision                              |                              **20–60 h** |               **$20–60** | Feasible multimodal phase          |
| Qwen3-VL-Embedding-2B       | Multimodal PEFT                     | Technically possible only with aggressive memory measures                | Poor utilisation / operationally awkward |                        — | **Do not make this the L4 plan**   |

\*Planning arithmetic, excluding CPU/storage/data generation and managed-service overhead.

ModernBERT's raw backbone is 149M and was deliberately designed around efficient attention/unpadding, which helps explain why it is such an attractive low-cost adaptation substrate.

For Qwen3-0.6B, ordinary Adam-style full fine-tuning consumes much more memory than the 1.2 GB implied by 600M BF16 weights: gradients, optimiser moments, master weights and activations dominate. LoRA removes most trainable optimiser state and buys much more comfortable microbatch sizes. Full FT may be made to fit with enough optimisation, but on one L4 **engineering simplicity favours LoRA before clever memory gymnastics**.

### Larger infrastructure

The point of multi-GPU infrastructure should be to answer a question the L4 experiment could not, rather than automatically repeat everything at greater scale.

| Workload                                          | Infrastructure      | Estimated total GPU-hours |  Approx. planning cost | Justification required                             |
| ------------------------------------------------- | ------------------- | ------------------------: | ---------------------: | -------------------------------------------------- |
| 149M–0.6B student, 5M pairs + hard negatives      | 4–8× A100/H100      |               **100–300** |           ≈$300–$1,500 | Clear gain from more diverse supervision           |
| Qwen3-Embedding-4B PEFT/full FT                   | 8× 80GB-class GPUs  |             **400–1,200** |        ≈$1,200–$6,000+ | Small model has demonstrable quality ceiling       |
| Qwen3-Embedding-8B adaptation                     | 8+× 80GB-class GPUs |            **800–2,000+** |       ≈$4,000–$20,000+ | Rarely sensible as first production student        |
| Qwen3-VL-Embedding-2B PEFT, 2–5M multimodal pairs | ≈8× H100-class      |               **300–900** |         ≈$1,500–$9,000 | Cross-modal query volume/business impact is proven |
| Qwen3-VL-Embedding-8B                             | 8+ high-memory GPUs |                **1,000+** | Five figures plausible | Treat as research/teacher first                    |

These ranges should not be mistaken for vendor quotations. The important economic comparison is that **hundreds of compact-model adaptation experiments can fit inside the cost and engineering envelope of one poorly designed multi-billion-parameter training programme**.

### Dataset size versus data quality

There is no reason to feed all proprietary records into the first contrastive run. The evidence from Promptagator, GPL and InPars is specifically that task alignment and pseudo-label quality can compensate for limited target-domain manual labels.

A sensible progression is:

| Phase                       | Positive pairs |                Hard negatives | Why                                   |
| --------------------------- | -------------: | ----------------------------: | ------------------------------------- |
| Prototype                   |      100k–250k |                      0–2/pair | Validate pipeline                     |
| First serious model         |         0.5–1M |                      2–8/pair | Establish adaptation value            |
| Strong production candidate |           1–3M | 4–16/pair plus teacher scores | Diminishing returns become measurable |
| Scale test                  |          5–20M |                mined/re-mined | Only after learning-curve evidence    |

Plot retrieval score against \(\log(\text{training examples})\). If the curve has flattened by one million pairs, “we own millions of listings” is not a reason to train on twenty million pairs.

### Training configuration I would actually run

For GTE-ModernBERT:

```text
model:               Alibaba-NLP/gte-modernbert-base
query max length:    64–96 tokens
document max length: 256–384 tokens
precision:           bf16
objective:           InfoNCE / CachedMultipleNegativesRankingLoss
explicit negatives:  2–8 hard negatives where implementation permits
effective batch:     512 → 2,048
learning rate:       search ~1e-5 → 3e-5
epochs:              1–3
validation:          every ~5–10% of an epoch
selection metric:    in-domain nDCG@10, with Recall@50 guardrail
```

Those hyperparameters are starting ranges, not universal optima. The important feature is a sufficiently large effective negative set. Gradient caching was specifically developed to make such large contrastive batches feasible under limited accelerator memory.

A second-stage run should re-mine hard negatives using the first adapted checkpoint, teacher-score them, and mix perhaps **50–70% real interaction supervision with 30–50% synthetic supervision** as an initial search range. The exact mixture must be ablated; there is no literature result establishing an optimum mixture for property search.

For a Matryoshka-capable student, train/evaluate candidate dimensions such as **256, 512 and full dimension**, rather than choosing 1,024 merely because the model emits it. MRL was designed specifically to preserve useful nested representations at smaller dimensions.

## Evaluation and prior art

### A benchmark that can actually decide whether adaptation is worthwhile

The in-domain benchmark should be considered a product asset at least as important as the adapted model.

I would build **roughly 10k–30k judged search queries initially**, eventually larger, sampled across frequency and intent. A smaller but meticulously judged set is preferable to millions of implicit clicks masquerading as ground truth.

For each query, pool results from BM25, the current production ranker, several general embedding models and the adapted candidate. Human judges should assign graded relevance, for example:

| Grade | Property-search interpretation         |
| ----: | -------------------------------------- |
|     3 | Strongly fulfils stated intent         |
|     2 | Good substitute; minor mismatch        |
|     1 | Weakly related                         |
|     0 | Not relevant / violates central intent |

This resembles a useful lesson from Amazon's ESCI product-search benchmark, which released **2.6 million manually labelled query–product judgements** across categories such as Exact, Substitute, Complement and Irrelevant. Product search is structurally close to real estate because multiple inventory items may legitimately satisfy a single query, making binary clicked/not-clicked relevance too crude.

The principal metrics should be:

| Task                                | Primary metric                 | Supporting metrics                    |
| ----------------------------------- | ------------------------------ | ------------------------------------- |
| Text → property                     | **nDCG@10**                    | Recall@20/50/100, MRR@10              |
| Text → listing candidate generation | **Recall@50/100**              | nDCG@10, latency                      |
| Text → image                        | **Recall@5/10**, nDCG@10       | Recall@1, mAP where appropriate       |
| Image → image                       | Recall@K / mAP                 | same-room/style/category slice scores |
| Property candidate generation       | Recall of any grade-2/3 result | coverage and candidate count          |
| Reranked production stack           | nDCG@10                        | conversion-weighted online metrics    |

Recall matters for the first-stage ANN retriever because a reranker cannot recover a good property that never entered the candidate set. nDCG matters once the ordering itself is being judged.

BEIR exists precisely because zero-shot retrieval performance varies materially across domains, while MTEB found no universal embedding winner across its diverse tasks. Those findings argue strongly for a property-specific benchmark rather than selecting from a public aggregate leaderboard.

### Required evaluation slices

The headline average should never be the sole release criterion. At minimum report:

| Slice                        | Why                                                  |
| ---------------------------- | ---------------------------------------------------- |
| Head queries                 | Majority of current traffic                          |
| Tail queries                 | Main potential semantic-search benefit               |
| Feature-heavy queries        | “butler's pantry”, “clerestory windows”              |
| Location-heavy queries       | Tests interaction with structured/geographical logic |
| Compound constraints         | Closest to difficult production intent               |
| Style/lifestyle queries      | Core dense-semantic use case                         |
| Exact entity/address/project | Tests whether BM25 hybrid remains necessary          |
| Misspellings/abbreviations   | Actual search behaviour                              |
| Rentals vs sales             | Different query/inventory distributions              |
| New properties/time period   | Drift                                                |
| New suburbs/projects         | OOD generalisation                                   |
| Multilingual queries         | Only if product requirements justify it              |
| Visual-intent queries        | Determines actual multimodal value                   |

Hard-negative evaluation should deliberately include properties matching almost every visible constraint except one. “Same suburb, same price, same bedroom count, no pool” is a much more diagnostic negative for “family house with a pool” than an unrelated studio 30 km away.

For images, ensure that photographs from the same physical property do not leak across train and test. SQID, an extension of Amazon's shopping-query dataset with images for about **190,000 products**, is a relevant public precedent for evaluating multimodal product search rather than treating image retrieval as an isolated computer-vision problem.

### Statistical testing and release gates

Compute metrics per query and compare models with a **paired bootstrap over queries**, reporting 95% confidence intervals on the difference. For high-stakes model decisions I would use at least several thousand bootstrap resamples and validate with a paired randomisation test.

More importantly, specify a minimum practical effect before looking at results. An example decision rule—not a literature constant—would be:

> Ship adaptation only when the 95% CI excludes zero **and** the lift is at least roughly 2% relative nDCG@10 or 3% relative Recall@50 on the target traffic distribution, with no important slice materially regressing.

This prevents an expensive model/model-index migration to capture statistically detectable but commercially meaningless changes.

### Ablations

A credible experiment should answer not merely “did the final model win?” but _why_:

| Ablation                                   | Question answered                                     |
| ------------------------------------------ | ----------------------------------------------------- |
| Base model only                            | What did adaptation buy?                              |
| + synthetic positives                      | Are generated queries useful?                         |
| + interaction positives                    | Are logs more valuable?                               |
| + BM25 hard negatives                      | Does lexical confusion matter?                        |
| + dense hard negatives                     | Does semantic mining add value?                       |
| + teacher filtering/distillation           | Is pseudo-label quality limiting?                     |
| 256 vs 512 vs full dimensions              | Can index cost be reduced?                            |
| LoRA vs full FT                            | Is PEFT leaving quality behind?                       |
| shared vs separate query/doc adapters      | Is asymmetry valuable?                                |
| text-only vs unified multimodal model      | Does shared-space complexity pay?                     |
| description-only vs description+attributes | Does structured text improve semantic representation? |

### Online evaluation

Offline relevance should gate deployment; online behaviour should confirm it.

Use an A/B test or interleaving experiment and measure not only click-through rate but **query reformulation, short clicks, saves, enquiries, inspection/application actions, abandonment and latency**. Query reformulation is especially informative: semantic retrieval may improve user experience even when raw CTR moves little.

Historical clicks should not simultaneously be treated as unbiased training labels and unquestionable online truth. Recommendation research has repeatedly had to account for implicit-feedback selection effects; the same exposure problem applies to property results.

### What prior art actually tells us

| Domain / work                          | Evidence class                                              | Key result / lesson for property search                                                                                                                                                                                                            |
| -------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GPL**                                | Peer-reviewed domain-adaptation research                    | Synthetic queries + hard negatives + teacher pseudo-labels improved specialised-domain dense retrieval by up to 9.3 nDCG@10; strongest direct evidence that lightweight domain adaptation can pay.                                                 |
| **Promptagator**                       | Research paper                                              | Few demonstrations plus LLM query generation produced task-specific dense retrievers that beat heavily engineered MS-MARCO-trained systems by >1.2 nDCG on average over 11 evaluated sets; reranking added further gains.                          |
| **InPars**                             | Research paper                                              | Synthetic query–document training improved retrieval, including when mixed with supervised data.                                                                                                                                                   |
| **NV-Retriever**                       | Research/preprint subsequently published work               | Positive-aware hard-negative filtering demonstrates that _which_ hard negatives are chosen matters substantially.                                                                                                                                  |
| **MedCPT / biomedical**                | Domain-specific research with large real interaction corpus | 255M PubMed search-log interactions trained specialised query/document retrievers and a reranker; outperformed much larger baselines on biomedical retrieval. Strong evidence that high-quality domain interactions can outweigh sheer model size. |
| **Amazon ESCI / product search**       | Public large-scale benchmark                                | 130k queries and 2.6M graded query-product judgements show why graded substitute relevance is important in catalogue search.                                                                                                                       |
| **SQID / multimodal product search**   | Research dataset                                            | Extends product-search evaluation with images, directly analogous to listing text + property photos.                                                                                                                                               |
| **SAILER / legal retrieval**           | SIGIR legal-search research                                 | Encodes legal document structure and legal elements rather than assuming generic representations are enough; analogous lesson: exploit property structure when forming supervision.                                                                |
| **Caseformer / CaseEncoder**           | Legal-domain research                                       | Constructs domain-specific pretraining/sampling around legally meaningful similarity; reinforces the importance of domain-aware positive/negative construction.                                                                                    |
| **Two-tower recommendation retrieval** | Production research                                         | Mixed negative sampling addresses implicit-feedback selection bias; highly relevant to search clicks and impressions.                                                                                                                              |
| **BGE-M3 / E5 / Qwen3**                | General embedding research                                  | Shows increasingly sophisticated contrastive, distillation and synthetic-data pipelines; useful bases, but public aggregate benchmarks cannot substitute for domain evaluation.                                                                    |

A notable finding from this investigation is that **public, reproducible peer-reviewed prior art specifically for large-scale residential real-estate embedding retrieval is sparse compared with product, biomedical and legal search**. That is a reason to borrow the mature methods from those domains, not a reason to assume a generic embedding will already solve the property problem.

The strongest transferable pattern across these domains is remarkably consistent:

\[
\text{domain signals}
+\text{contrastive bi-encoder}
+\text{good negatives}
+\text{strong reranker/teacher}
\]

beats “take a general language model and perform lots of generic domain pretraining” as the default engineering strategy.

## Serving and production design

### Retrieval should be hybrid by design

The production system should not ask a dense model to solve problems lexical and structured systems already solve extremely well.

For:

> “3 bed Federation home in Marrickville under $2m near the station with original fireplaces and a north-facing garden”

a good request plan is conceptually:

```text
Structured parser:
    bedrooms >= 3
    location = Marrickville / permitted geo region
    price <= 2,000,000

Text semantic query:
    "Federation home near the station with original fireplaces
     and a north-facing garden"

Lexical query:
    original user text

Visual-semantic query:
    optional; use for visually grounded concepts

Candidate generation:
    BM25 top 200
    text dense top 200
    visual property candidates top 100

Union:
    property_id

Final ranking:
    lexical score
    text embedding score
    visual score
    structured-match features
    freshness/inventory/business features
    optional cross-encoder score
```

A simple Reciprocal Rank Fusion baseline is often preferable to prematurely hand-calibrating incomparable BM25 and cosine-score scales:

\[
\mathrm{RRF}(d)=\sum_r \frac{1}{k+\operatorname{rank}\_r(d)}
\]

Once enough judged and interaction data exist, replace or augment this with a learned ranker.

### OpenSearch versus Qdrant

If OpenSearch already holds listing text, filters and lexical indexes, **keeping dense retrieval there is the lowest-operational-complexity option**. The main advantage is fewer consistency boundaries: one listing update can modify text, metadata and vector representation within the same search estate.

Qdrant becomes attractive when the architecture is more vector-centric—especially when keeping several representations per entity, image-level vectors, or independent text/visual indexes is operationally cleaner. Qdrant's documentation explains its HNSW-based vector indexing and filter-aware retrieval architecture.

I would not make the database choice based on synthetic ANN benchmarks alone. Benchmark the actual workload with:

`vector count × dimensions × filter selectivity × update rate × top-K × required ANN recall`.

HNSW itself remains a strong general-purpose ANN method because it constructs a hierarchical proximity graph enabling efficient approximate nearest-neighbour search, but index construction parameters trade memory, build time, recall and query latency.

For an existing OpenSearch deployment, my default would be:

> **OpenSearch BM25 + text vector candidate retrieval initially; separate visual index only if image volume or operational characteristics make it worthwhile.**

That maximises engineering simplicity.

### Dimensionality is a production variable, not a model constant

Raw storage before graph/metadata overhead is simply:

\[
N \times d \times \text{bytes per component}.
\]

For **10 million property vectors**:

| Representation | Raw vector storage |
| -------------- | -----------------: |
| 1,024-d FP32   |           40.96 GB |
| 1,024-d FP16   |           20.48 GB |
| 1,024-d int8   |           10.24 GB |
| 768-d FP16     |           15.36 GB |
| 768-d int8     |            7.68 GB |
| 512-d FP16     |           10.24 GB |
| 512-d int8     |            5.12 GB |
| 256-d FP16     |            5.12 GB |
| 256-d int8     |            2.56 GB |

These figures **exclude HNSW graph structure, document payloads and engine overhead**.

The image collection can dominate. Fifty million photographs at 512 dimensions require **51.2 GB merely for FP16 vectors** or **25.6 GB at int8**, before ANN overhead. That is why MRL/dimensionality experiments are much more economically important for image retrieval than arguing whether a leaderboard's 1,024-dimensional model is fractionally ahead. MRL was explicitly designed to enable such nested accuracy/storage trade-offs without maintaining separate independently trained models.

Do not blindly truncate a conventional embedding. Truncate only a checkpoint trained for Matryoshka representations, or learn/evaluate an explicit projection. Every compressed representation must be evaluated end-to-end against ANN recall, not merely cosine correlation.

### Quantisation

For ANN candidate generation, test in this order:

`FP16 baseline → int8 vectors → lower Matryoshka dimension → more aggressive binary/PQ schemes`.

The first two are usually much easier operationally than aggressive product quantisation. Always separate:

1. **embedding-model degradation** from reducing dimensions;
2. **numeric quantisation degradation**;
3. **ANN approximation degradation**.

Otherwise a poor Recall@50 result gives no information about which compression stage caused it.

### Inference and batching

The 149M ModernBERT/GTE class is particularly attractive because query embedding is cheap enough to sit directly in the online request path. GTE-ModernBERT's official model card lists a 149M model and 768-dimensional output, while ModernBERT itself was designed around efficient attention and unpadding.

For production serving:

| Model type            | Serving pattern                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------ |
| ModernBERT/GTE/E5/BGE | Text-embedding runtime, ONNX/TensorRT/PyTorch or Text Embeddings Inference where supported |
| Qwen3-Embedding       | Pooling/embedding-capable transformer runtime; test implementation parity carefully        |
| SigLIP2               | GPU image batch service + lightweight text-query path                                      |
| Qwen3-VL              | Dedicated GPU service; primarily teacher/quality-tier unless traffic justifies it          |

Do not accept a serving-runtime migration merely because it returns vectors of the right shape. Compare the vectors and retrieval metrics against the reference Hugging Face implementation; pooling, EOS handling, normalisation and instruction formatting can silently alter nearest neighbours.

A useful capacity-planning technique is to avoid pretending to know throughput in advance. Benchmark the real length distribution and use:

\[
T*{\text{backfill}}
=
\frac{N*{\text{records}}}{\text{measured records/sec}}.
\]

For 10M listings, for example, **1,000 records/s implies 2.78 hours**, while **250/s implies 11.1 hours**. These are arithmetic scenarios, not claims that a particular model achieves those rates.

### Re-embedding millions of records safely

Never overwrite the production vector field in place.

Use:

```text
embedding_model_v17
        ↓
new vector/index namespace
        ↓
snapshot backfill
        ↓
dual-write all listing mutations to v16 + v17
        ↓
offline benchmark / ANN-recall / production shadow traffic
        ↓
atomic alias or configuration switch
        ↓
retain v16 for rollback window
```

Every vector should carry at least:

```text
model_name
model_revision
training_data_version
preprocessing_version
embedding_dimension
normalisation_version
created_at
```

A model upgrade that cannot be traced back to its data, tokenizer, pooling code and index configuration is not reproducible.

For image backfills, cache vectors by image-content hash. Property sites repeatedly reuse photography across listing revisions; recomputing identical photographs is wasted GPU time.

### Property-level fusion

The first production text-to-property model should not attempt to concatenate every property modality into a single enormous input and hope one embedding discovers the desired weighting.

Use a property score of the form:

\[
S(p,q)
=
w*tS*{\text{text}}
+w*bS*{\text{BM25}}
+w*v\operatorname{Agg}*{i\in I*p}
S*{\text{visual}}(q,i)
+f\_{\text{structured/ranking}}(p,q)
\]

where \(I_p\) is the property's image set.

The weights can start as validation-tuned constants and eventually become a learned ranking model. This architecture allows text encoders, visual encoders and ranking logic to evolve independently—a substantial operational advantage when millions of vectors must be refreshed after each incompatible embedding update.

## Recommended staged roadmap and decision gates

### Baseline and benchmark

**Compute budget:** roughly 1–3 L4-days plus annotation effort; no training required initially.

Create the leakage-safe property benchmark and run:

`BM25`, `GTE-ModernBERT-base`, `Qwen3-Embedding-0.6B`, `BGE-M3`, `multilingual-E5-large-instruct`.

For visual queries, benchmark legacy CLIP as a control, **SigLIP2-B**, and **Qwen3-VL-Embedding-2B**.

Public benchmark results should only decide which models enter this bake-off. MTEB's core finding—that no embedding method dominates every task—is precisely why the real-estate benchmark must decide the winner.

**Decision gate:** If a general pretrained model plus BM25 already meets relevance targets and qualitative errors are mostly price/location/filter/ranking errors, **stop**. Domain adaptation is unlikely to be the highest-return project.

### Cheap text adaptation

**Recommended target:** GTE-ModernBERT-base.

Build 0.5–1M query–positive pairs from a mixture of high-confidence interactions and synthetic queries; use random/in-batch and BM25 negatives. Full fine-tune for one to three epochs with a cached contrastive loss.

**Budget:** roughly 8–24 L4 GPU-hours per serious run under the assumptions above.

GPL, Promptagator and InPars provide the strongest external support for this phase: target-domain query generation and contrastive retrieval adaptation can give sizeable gains without enormous manually labelled datasets.

**Decision gate:** require a statistically credible and practically useful in-domain improvement. A reasonable engineering threshold is roughly **≥2% relative nDCG@10 or ≥3% relative Recall@50**, without important-slice regressions.

If adaptation does not clear this bar after data-quality inspection, **do not scale the model**. A bigger student rarely fixes poor relevance labels.

### Hard negatives and teacher distillation

Mine difficult candidates from the first adapted model. Score them with a stronger reranker; remove likely false negatives; train again using contrastive plus teacher-margin supervision.

NV-Retriever shows the value of positive-aware negative selection, while GPL provides direct precedent for teacher pseudo-labelling in domain adaptation.

At this stage, compare:

`GTE-ModernBERT full FT` versus `Qwen3-Embedding-0.6B LoRA/full FT`.

Also test 256/512/full-dimensional students where the architecture supports it.

**Decision gate:** scale only if the second-stage model moves the learning curve. If adding another million examples or a substantially larger base changes nDCG by less than the predefined minimum effect, the text representation is probably no longer the bottleneck.

### Multimodal adaptation as an independent workstream

Start with the **pretrained SigLIP2-B** visual space and an image-level property index. Evaluate text-to-image and image-to-image before fine-tuning anything.

Then adapt on perhaps 0.5–2M carefully filtered image-caption/query pairs. Use image-specific generated captions, room/category metadata and real visual search interactions if available. Freeze the vision tower first and progressively relax it.

SigLIP2 is a particularly good fit because the family was designed as multilingual vision-language encoders and its second-generation recipe improves image-text retrieval while remaining available at substantially smaller sizes than a 2B VLM-derived embedding system.

Use Qwen3-VL-Embedding-2B as a **quality ceiling and possible teacher**. The fact that it supports one common space for text, images, document images and videos makes it highly relevant—but it should earn its extra compute by producing materially better property-search outcomes.

**Decision gate:** pursue multimodal production only if visual-intent queries form a meaningful traffic/product slice and adapted visual retrieval beats pretrained SigLIP2 enough to justify the model/index lifecycle.

### Unified multimodal property space only after late fusion

Run a direct experiment:

\[
\text{specialised text + specialised visual + late fusion}
\quad\text{vs}\quad
\text{Qwen3-VL shared space}.
\]

Compare all four requested tasks, latency, storage and refresh cost—not merely text-to-image recall.

I would require the unified approach to achieve approximately:

- text retrieval within ~1–2% of the specialised text model;
- meaningfully superior cross-modal retrieval;
- acceptable online query cost;
- simpler enough operations to offset its larger model/backfill cost.

Those are **recommended product gates, not published constants**.

Unless it clears them, retain the two-space architecture.

### Larger training only when a bottleneck is demonstrated

The escalation sequence should be:

| Order | Intervention                         | Reason                                          |
| ----: | ------------------------------------ | ----------------------------------------------- |
|     1 | Better benchmark                     | Without it no model decision is trustworthy     |
|     2 | Better positives                     | Usually cheaper than a bigger model             |
|     3 | Better hard negatives                | Often very high retrieval ROI                   |
|     4 | Teacher pseudo-labels / distillation | Transfers large-model relevance cheaply         |
|     5 | More data                            | Follow measured learning curve                  |
|     6 | Qwen3-0.6B or similar larger student | Only if 149M capacity is demonstrably limiting  |
|     7 | Continued domain pretraining         | Only if language/domain shift remains visible   |
|     8 | 4B+ full/PEFT training               | Only after evidence of a model-capacity ceiling |
|     9 | Training from scratch                | Almost certainly unjustified                    |

This ordering is supported by domain-adaptation evidence: GPL found large gains from retrieval-specific pseudo-labelled training while generic pretraining additions were inconsistent; MedCPT demonstrates that domain interaction supervision can create exceptional specialised retrievers; and product/legal-search work repeatedly shows that domain-specific notions of relevance and negative construction matter.

### Final production recommendation

For the stated priorities—**retrieval quality, engineering simplicity, compute efficiency, reproducibility and scalability**—the architecture I would approve is:

```text
                           ┌───────────────────────┐
natural-language query ───►│ structured constraint │──► price/geo/bed/etc filters
                           │ extraction             │
                           └───────────────────────┘

            ┌──────────────────────────────────────────────┐
            │                                              │
            ▼                                              ▼
   GTE-ModernBERT                                lexical / BM25 search
 domain-adapted encoder
            │                                              │
            └──────────────► candidate union ◄─────────────┘
                                    ▲
                                    │
                           SigLIP2 visual space
                         (only for visual intents)
                                    │
                                    ▼
                     aggregate images → property_id
                                    │
                                    ▼
                   teacher-trained / learned reranker
                                    │
                                    ▼
                              final results
```

**Text recommendation:** begin from **GTE-ModernBERT-base**, full-fine-tuned on approximately 0.5–2M high-quality domain pairs with cached InfoNCE, then hard-negative mining and teacher distillation. Keep **Qwen3-Embedding-0.6B** as the main challenger and use 4B/8B primarily as teachers until evidence justifies their serving cost. ModernBERT's 149M scale makes repeated experimentation unusually cheap, while Qwen3 supplies a strong modern multilingual/reasoning-oriented quality ceiling.

**Multimodal recommendation:** start with **SigLIP2-B**, keeping image vectors separate from text-property vectors. Index individual photographs and aggregate at the property level. Benchmark **Qwen3-VL-Embedding-2B** as a shared-space alternative/teacher, but do not couple the production text stack to it initially.

**Data recommendation:** invest disproportionate effort in canonical-property deduplication, leakage-proof temporal splits, realistic synthetic queries, graded relevance and positive-aware hard negatives. Evidence from GPL, Promptagator, NV-Retriever, MedCPT and large product-search benchmarks indicates that **supervision construction is more likely to determine the return on adaptation than whether the starting encoder has 149M, 600M or several billion parameters**.

**Serving recommendation:** keep structured filters authoritative, fuse dense search with BM25, use HNSW ANN, aggressively evaluate Matryoshka dimensionality/int8 storage, and perform blue-green versioned re-embedding rather than in-place model migrations. HNSW remains a well-established ANN design, while Matryoshka representation learning specifically provides an accuracy/storage mechanism worth exploiting at tens of millions of property and image vectors.

**Economic recommendation:** do not approve multi-GPU 4B/8B adaptation until the L4-scale pipeline proves that in-domain adaptation produces a meaningful lift. A single L4 can answer the strategically important question—_does domain-specific retrieval supervision help us?_—for tens of dollars of accelerator time under the planning assumptions above. The expensive question—_how large should the final model be?_—only becomes worth answering after that.

The strongest research-backed strategy is therefore **not “train a real-estate embedding model” in one large step**. It is a staged retrieval-learning programme:

\[
\boxed{
\text{strong pretrained baseline}
\rightarrow
\text{domain contrastive FT}
\rightarrow
\text{hard-negative mining}
\rightarrow
\text{teacher distillation}
\rightarrow
\text{separate multimodal adaptation}
\rightarrow
\text{scale only on measured evidence}
}
\]

That sequence captures most of the demonstrated benefits of domain adaptation while minimising the two failure modes that recur in specialised retrieval projects: **spending compute on a larger model when the labels are the problem, and adding architectural complexity before an in-domain benchmark has shown that the complexity buys better search.**
