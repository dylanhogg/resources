# Does 3072 dimensions actually buy you better image search?

_Embedding dimensions for image search: what the published evidence shows, and where it is missing._

## 1. Two common assumptions about embedding size

Most of us bring two assumptions to choosing an embedding size. Both are wrong, and understanding why is useful.

The first assumption: more dimensions means a richer vector. Twice the coordinates should mean twice the room for visual detail, so 3072 dimensions must hold more of a photograph than 768 does.

The second is the cross-model version: Gemini Embedding 2 outputs 3072 dimensions and some other model outputs 768, so Gemini has more capacity to work with.

The first assumption has a measurable cost. Take the example I will return to throughout this post: a real-estate search system over 10 million property photos. At float32, 3072-d vectors take 122.88 GB of raw payload before any index overhead. The same catalogue at 768-d takes 30.72 GB. That is roughly a 92 GB difference from a single architectural choice you probably made in an afternoon.

If the assumption holds, that storage is money well spent. If it does not, you have bought nothing with it.

## 2. The question

> When the _same model_ is served at 3072, 1024 and 256 dimensions, what happens to text→image retrieval quality? And does the published evidence cover the models you are most likely to deploy?

The first question has a good answer, from two independent labs. The second does not: for **Gemini Embedding 2 and Cohere Embed 4 — probably the two models on your shortlist — nobody has published the per-dimension curve for image retrieval.** I could not find one at all. That gap is half of what this post is about, so I want to state it up front rather than save it for the end.

## 3. The one commercial ablation that answers the question: Amazon Nova MME

Amazon Nova Multimodal Embeddings is the only major commercial multimodal model I found with a published per-dimension table for text→image retrieval. The architecture, training and benchmarks are held constant; only the vector length changes. Amazon states that the shorter vectors are prefixes of the full 3072-d representation, trained so the front of the vector carries the most signal. Section 5 explains why that works.

| Nova MME dims | TextCaps | MSCOCO | ViDoRe v2 | float32/vector |
| ------------: | -------: | -----: | --------: | -------------: |
|          3072 |     88.9 |   76.7 |      58.7 |         12 KiB |
|          1024 |     87.9 |   75.6 |      57.7 |          4 KiB |
|           384 |     85.6 |   72.9 |      53.4 |        1.5 KiB |
|           256 |     83.1 |   70.6 |      50.2 |          1 KiB |

Three results stand out.

**3072 → 1024 is cheap.** Removing two-thirds of the coordinates costs 1.0 points on TextCaps and 1.1 on MSCOCO.

**1024 → 384 costs noticeably more.** Against the 3072 baseline: 3.3 points on TextCaps, 3.8 on MSCOCO, 5.3 on ViDoRe.

**256 is not free.** It costs 5.8, 6.1 and 8.5 points respectively, or roughly 6.5%, 8.0% and 14.5% in relative terms.

Which benchmark degrades fastest is worth attention. ViDoRe (Visual Document Retrieval, a benchmark of document screenshots with dense text and fine visual detail) loses 8.5 points at 256 dimensions where TextCaps loses 5.8. This is the first sign that visually rich retrieval has a capacity floor that object-centric captioning benchmarks do not expose. I think that matters for property photos, and section 7 returns to it.

One note on metrics if you plan to reproduce any of this: the image-retrieval columns are average Recall@1/5/10 (R@K is the share of queries where the correct image appears in the top K), and ViDoRe is NDCG@5, a graded ranking score over the top five results. Do not compare these figures against benchmarks that quote Recall@1 alone.

## 4. Open-weight corroboration: Jina CLIP v2

One vendor's ablation of its own model is not a pattern. Jina CLIP v2 serves as a control: the weights are open, text→image Recall@5 is published at six lengths, and you can rerun the evaluation yourself.

| Dims | CLIP Benchmark | Crossmodal-3600 | XTD10 |
| ---: | -------------: | --------------: | ----: |
| 1024 |          79.10 |           81.43 | 84.87 |
|  768 |          79.12 |           82.35 | 84.85 |
|  512 |          78.93 |           82.31 | 84.60 |
|  256 |          78.32 |           81.75 | 84.32 |
|  128 |          75.90 |           78.17 | 81.80 |
|   64 |          70.51 |           72.52 | 77.85 |

**768 dimensions is not worse than 1024 here, and on Crossmodal-3600 it is nearly a point better.** Even 256 stays within about one point of full width on all three tests. Quality drops sharply at 128 (3-4 points) and 64 (7-9 points).

This is the same curve shape as Nova's, at a different scale, from a different lab, on different benchmarks. Two independent sources agreeing is the main argument of this post.

## 5. Why the curve has this shape

Both models are trained with **Matryoshka Representation Learning (MRL)**. The training objective is applied not only to the full vector but to nested prefixes of it, so the first 256 coordinates are optimised to work as an embedding on their own, the first 768 likewise, and so on. That corrects the mental model: **in an MRL-trained model the dimensions are not equally informative independent coordinates.** Truncating is not compressing a flat 3072-d space; it is taking a prefix of a representation designed to survive truncation.

This also explains why the cross-model assumption in section 1 fails. Gemini Embedding 2 at 768 dimensions is still the full Gemini network doing the work, which is not the same as swapping in a smaller 768-d vision-language architecture. **Dimension count is not comparable across model families**, only within one.

One reproducibility trap: taking a prefix changes the vector's norm. If you compare an unnormalised truncation against a normalised full vector, you will manufacture a dimensionality effect that is not there. Gemini's API re-normalises automatically when you request a non-default size. If you truncate by hand, L2-normalise before any cosine comparison.

## 6. The gap: the two models you are probably evaluating

The complication below is, I think, the most useful thing this post has to say.

| Model              | Dimension choices                       | Published multimodal per-dimension ablation? |
| ------------------ | --------------------------------------- | -------------------------------------------- |
| Gemini Embedding 2 | 128-3072; recommended 768 / 1536 / 3072 | **No**                                       |
| Cohere Embed 4     | 256 / 512 / 1024 / 1536                 | **No**                                       |
| Amazon Nova MME    | 256 / 384 / 1024 / 3072                 | **Yes**                                      |
| Jina CLIP v2       | 64 / 128 / 256 / 512 / 768 / 1024       | **Yes**                                      |

**Gemini Embedding 2.** What _is_ published is genuinely good: the model is natively multimodal rather than a text model bolted onto a vision encoder, MRL losses are trained at 768- and 1536-d prefixes, and absolute text→image Recall@1 is strong (mean 80.5; DOCCI 93.4, TextCaps 89.6, MSCOCO 62.9). DOCCI (Descriptions of Connected and Contrasting Images) and TextCaps are detailed-caption datasets rather than short object labels. I expect detailed-caption retrieval to map better onto a query like _"sunlit open-plan kitchen with a waterfall-edge stone island and black pendant lights"_ than COCO's dominant-object framing does, but that is my inference about the application rather than something Google measured.

What is **not** published is any 3072 vs 1536 vs 768 text→image table. The dimension ablation in Google's docs covers **Gemini Embedding 001, a text model, evaluated on MTEB** (1536: 68.17, 768: 67.99, 256: 66.19, 128: 63.31). That is encouraging for MRL in general and consistent with everything above, but it is not evidence about Embedding 2 on complex images, and I would rather say so than let it stand in.

**Cohere Embed 4.** Genuinely multimodal, with clean MRL sizes. Its distinctive capability is fusing image and text into a single vector, which is attractive for listings where room type, suburb or other structured attributes are reliable text that you would otherwise force the pixels to carry. There is again no published per-dimension image-retrieval curve.

One external datapoint exists, and it needs careful handling. Amazon's Nova report benchmarked Embed 4 through the public Bedrock API and scored it **22.9 on MSCOCO against Nova's 76.7**. Three caveats come with that number: Amazon is a direct competitor, the figures are not Cohere self-reported, and the Embed 4 output dimension used is not stated. I read it as a reason to benchmark Cohere yourself, not as evidence about Cohere's dimension curve, which it says nothing about.

## 7. Input resolution probably matters more than dimension

Having spent a thousand words on dimension, I should point at the larger lever, because the published evidence says input resolution is one.

Jina CLIP v2 ablates image resolution separately, and the effect is much larger than anything in sections 3 or 4. On ViDoRe, moving from **224 to 384 pixels lifts average NDCG@5 from 0.256 to 0.454**. 512 pixels adds more; going from 512 to 768 costs 2.25x the image patches for a gain of 0.019, which is why the authors settle on 512.

Compare the two effects directly: cutting a well-trained MRL vector from 1024 to 512 dimensions costs a fraction of a point, while under-resolving the image costs a large share of retrieval quality. Preprocessing can destroy the information before vector width ever becomes the binding constraint.

This matters most for queries where a small local feature _is_ the query:

> "gas cooktop beneath a concealed rangehood" · "herringbone timber flooring" · "frameless shower with niche"

Check what your API does here: Cohere documents that Embed v4 downsamples images above 2,458,624 pixels (1568×1568 square), while Gemini's embedding docs publish no equivalent threshold. One caveat: ViDoRe is document screenshots, not property photography, so do not transfer the specific 512px optimum. The transferable part is the _ordering_ of the two effects.

Neither setting fixes one failure mode: an embedding can correctly identify an image as a kitchen and still rank it top for _"white kitchen, black island"_ when it is a black kitchen with a white island. Attribute binding is exactly what COCO-style benchmarks underexpose.

## 8. The corrected mental model

Pulling the findings together:

1. **Choose the model family first, then the dimension.** Reversing the order risks picking a weaker model because it exposes more coordinates.
2. **Within an MRL model, 3-4x compression plausibly costs about a point.** Nova and Jina agree on this from different labs and at different scales.
3. **The cliff is real, and it sits well below the headline width.** Nova breaks at 256, Jina at 128.
4. **Fine-grained visual retrieval degrades faster than object-centric retrieval.** ViDoRe falls fastest in Nova's table, and I would treat property photography as nearer that end.
5. **Resolution before dimension.** It is the larger measured effect in the published evidence.
6. **Dimension and numeric precision are separate axes.** Cohere exposes int8 and binary output; those compound with width rather than substituting for it, and conflating the two will confuse your results. A real axis, but a different post.

My current view on the operating regions — a hypothesis to test on your own corpus, not a set of guaranteed scores: 3072→1536 is rarely justifiable on quality alone; 1536/1024→768/512 is usually the attractive region; 384/256 is viable when memory-constrained but increasingly lossy on complex imagery; 128/64 is too aggressive without a reranker behind it.

That returns us to the 92 GB:

| Dims | Bytes/vector | Raw vectors @ 10M | vs 3072 |
| ---: | -----------: | ----------------: | ------: |
| 3072 |       12 KiB |         122.88 GB |    100% |
| 1536 |        6 KiB |          61.44 GB |     50% |
| 1024 |        4 KiB |          40.96 GB |     33% |
|  768 |        3 KiB |          30.72 GB |     25% |
|  512 |        2 KiB |          20.48 GB |     17% |
|  384 |      1.5 KiB |          15.36 GB |   12.5% |
|  256 |        1 KiB |          10.24 GB |    8.3% |

Those figures are arithmetic on raw float32 payload only. A real ANN index also carries graph or product-quantisation structures, IDs and metadata, and latency has to be measured in your target vector database: dimension is a first-order effect on distance computation and memory traffic, but traversal, cache behaviour and I/O can dominate.

The conclusion the table supports: **Gemini at 768 instead of 3072 dimensions saves roughly 92 GB of raw vectors on a 10M-photo catalogue, at one of Google's own recommended MRL sizes.** That is why a sub-point quality loss deserves this much attention.

## 9. Limits of this analysis

Four limits I want on the record.

- Every number here comes from a **general benchmark**. None is a property-image benchmark.
- Nova and Jina agreeing is suggestive, but it does not guarantee that **Gemini's or Cohere's** curves have the same shape. They may well not.
- The 768 starting point for Gemini is **extrapolated** from Google's MRL training sizes plus Nova and Jina's behaviour. I could not find it demonstrated anywhere.
- Cliff location is model-specific. You will have to find your own, and a few hundred well-judged domain queries will tell you more than another general benchmark will.

That reverses the question we started with. The useful question was never how many dimensions a model has. It is how far down that model's own curve you can go before your queries degrade.
