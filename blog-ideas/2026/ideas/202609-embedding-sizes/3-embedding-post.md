# Does 3072 dimensions actually buy you better image search?

_Embedding dimensions for image search: the evidence, and the gap where you need it most._

## 1. The intuition that costs 92 GB

Two intuitions most of us bring to picking an embedding size, both wrong in a way that turns out to be useful.

The first: more dimensions means a richer vector. Twice the coordinates, twice the room for visual detail, so 3072 must hold more of a photograph than 768 does.

The second is the cross-model version: Gemini Embedding 2 outputs 3072 dimensions and this other model outputs 768, so Gemini has more headroom.

Here's what the first one costs, using the example I'll keep returning to: a real-estate search system over 10 million property photos. At float32, 3072-d is 122.88 GB of raw vector payload before any index overhead at all. The same catalogue at 768-d is 30.72 GB. Roughly a 92 GB difference from a single architectural choice you probably made in an afternoon.

If the intuition holds, that's money well spent. If it doesn't, you've bought nothing with it.

## 2. The question

> When the _same model_ is served at 3072, 1024 and 256 dimensions, what actually happens to text→image retrieval quality? And does the published evidence cover the models you are most likely to deploy?

The first half has a good answer, from two independent labs. The second half has an uncomfortable one: for **Gemini Embedding 2 and Cohere Embed 4 — probably the two models on your shortlist — nobody has published the per-dimension curve for image retrieval.** I couldn't find one at all. That gap is half of what this post is about, so it's worth naming now rather than saving for the end.

## 3. The one commercial ablation that answers it: Amazon Nova MME

Amazon Nova Multimodal Embeddings is the only major commercial multimodal model I found with a published per-dimension table for text→image retrieval. Same architecture, same training, same benchmarks; the only thing changing is the vector length. Amazon states the shorter vectors are prefixes of the full 3072-d representation, trained so the front of the vector carries the most signal. (More on why that works in §5.)

| Nova MME dims | TextCaps | MSCOCO | ViDoRe v2 | float32/vector |
| ------------: | -------: | -----: | --------: | -------------: |
|          3072 |     88.9 |   76.7 |      58.7 |         12 KiB |
|          1024 |     87.9 |   75.6 |      57.7 |          4 KiB |
|           384 |     85.6 |   72.9 |      53.4 |        1.5 KiB |
|           256 |     83.1 |   70.6 |      50.2 |          1 KiB |

Three things fall out of it.

**3072 → 1024 is cheap.** Two-thirds of the coordinates gone, for 1.0 points on TextCaps and 1.1 on MSCOCO.

**1024 → 384 starts to bite.** Against the 3072 baseline: 3.3 points on TextCaps, 3.8 on MSCOCO, 5.3 on ViDoRe.

**256 is not free.** 5.8, 6.1 and 8.5 points respectively, or roughly 6.5%, 8.0% and 14.5% in relative terms.

The detail worth pausing on is which benchmark falls fastest. ViDoRe (Visual Document Retrieval, a benchmark of document screenshots — dense text and fine visual detail) loses 8.5 points at 256 where TextCaps loses 5.8. That's the first hint that visually rich retrieval has a capacity floor that object-centric captioning benchmarks don't expose. I think that matters for property photos, and §7 picks up the thread.

One metric-hygiene note if you plan to reproduce any of this: the image-retrieval columns are average Recall@1/5/10 (R@K: the share of queries where the correct image appears in the top K), and ViDoRe is NDCG@5, a graded ranking score over the top five. Don't compare these against benchmarks quoting Recall@1 alone.

## 4. Open-weight corroboration: Jina CLIP v2

One vendor's ablation of their own model isn't a pattern. Jina CLIP v2 is the control: open weights, published text→image Recall@5 at six lengths, and you can rerun it yourself.

| Dims | CLIP Benchmark | Crossmodal-3600 | XTD10 |
| ---: | -------------: | --------------: | ----: |
| 1024 |          79.10 |           81.43 | 84.87 |
|  768 |          79.12 |           82.35 | 84.85 |
|  512 |          78.93 |           82.31 | 84.60 |
|  256 |          78.32 |           81.75 | 84.32 |
|  128 |          75.90 |           78.17 | 81.80 |
|   64 |          70.51 |           72.52 | 77.85 |

Stated plainly: **768 is not worse than 1024 here, and on Crossmodal-3600 it is nearly a point better.** Even 256 sits within about one point of full width across all three tests. The break comes at 128 (3-4 points) and 64 (7-9 points).

That is the same shape as Nova's curve at a different scale, from a different lab, on different benchmarks. Two independent sources agreeing is the actual argument of this post.

## 5. Why the curve looks like that

The data has earned the explanation, so here it is.

Both models are trained with **Matryoshka Representation Learning (MRL)**: the training objective is applied not just to the full vector but to nested prefixes of it, so the first 256 coordinates are optimised to work as an embedding on their own, the first 768 likewise, and so on. The correction to the mental model: **in an MRL-trained model the dimensions are not equally informative independent coordinates.** Truncating isn't squeezing a flat 3072-d space down; it's taking a prefix of a representation designed to survive being truncated.

This is also what kills the cross-model intuition from §1. Gemini Embedding 2 at 768 is still the full Gemini network doing the work — it is not the same thing as swapping in a smaller 768-d vision-language architecture. **Dimension count isn't comparable across model families**, only within one.

One reproducibility trap worth knowing: taking a prefix changes the vector's norm. If you compare an unnormalised truncation against a normalised full vector, you will manufacture a dimensionality effect that isn't there. Gemini's API re-normalises automatically when you request a non-default size. If you truncate by hand, L2-normalise before any cosine comparison.

## 6. The gap: the two models you are probably evaluating

Here's the complication, and I think it's the most useful thing this post has to say.

| Model              | Dimension choices                       | Published multimodal per-dimension ablation? |
| ------------------ | --------------------------------------- | -------------------------------------------- |
| Gemini Embedding 2 | 128-3072; recommended 768 / 1536 / 3072 | **No**                                       |
| Cohere Embed 4     | 256 / 512 / 1024 / 1536                 | **No**                                       |
| Amazon Nova MME    | 256 / 384 / 1024 / 3072                 | **Yes**                                      |
| Jina CLIP v2       | 64 / 128 / 256 / 512 / 768 / 1024       | **Yes**                                      |

**Gemini Embedding 2.** What _is_ published is genuinely good: natively multimodal rather than a text model bolted onto a vision encoder, MRL losses trained at 768- and 1536-d prefixes, and strong absolute text→image Recall@1 (mean 80.5; DOCCI 93.4, TextCaps 89.6, MSCOCO 62.9). DOCCI (Descriptions of Connected and Contrasting Images) and TextCaps are detailed-caption datasets rather than short object labels. My read is that detailed-caption retrieval maps better onto a query like _"sunlit open-plan kitchen with a waterfall-edge stone island and black pendant lights"_ than COCO's dominant-object framing does — but that's my inference about the application, not something Google measured.

What is **not** published is any 3072 vs 1536 vs 768 text→image table. The dimension ablation in Google's docs is for **Gemini Embedding 001, a text model, on MTEB** (1536: 68.17, 768: 67.99, 256: 66.19, 128: 63.31). Encouraging for MRL in general and consistent with everything above, but not evidence about Embedding 2 on complex images — and I'd rather say so than let it stand in.

**Cohere Embed 4.** Genuinely multimodal, clean MRL sizes, and the distinctive capability is fusing image and text into a single vector — attractive for listings where room type, suburb or structured attributes are known-good text that you'd otherwise force the pixels to carry. Again: no published per-dimension image-retrieval curve.

There's one external datapoint, and it needs handling carefully. Amazon's Nova report benchmarked Embed 4 through the public Bedrock API and scored it **22.9 on MSCOCO against Nova's 76.7**. Three caveats travel with that number: Amazon is a direct competitor, the figures are not Cohere self-reported, and the Embed 4 output dimension used is not stated. I read it as a reason to benchmark Cohere yourself — not as evidence about Cohere's dimension curve, which it says nothing about.

## 7. The knob that probably matters more: input resolution

Having spent a thousand words on dimension, here's a larger lever, because the published evidence says it is one.

Jina CLIP v2 ablates image resolution separately, and the effect dwarfs anything in §3 or §4. On ViDoRe, moving from **224 to 384 pixels lifts average NDCG@5 from 0.256 to 0.454**. 512 adds more; 512 → 768 costs 2.25x the image patches for +0.019, which is why the authors settle on 512.

Side by side: cutting a well-trained MRL vector from 1024 to 512 costs a fraction of a point, while under-resolving the image costs a large share of retrieval quality. Preprocessing can destroy the information before vector width is ever the binding constraint.

That's most concrete for queries where a small local feature _is_ the query:

> "gas cooktop beneath a concealed rangehood" · "herringbone timber flooring" · "frameless shower with niche"

Worth knowing what your API does here: Cohere documents that Embed v4 downsamples images above 2,458,624 pixels (1568×1568 square), while Gemini's embedding docs publish no equivalent threshold. And the honest caveat — ViDoRe is document screenshots, not property photography, so don't transfer the specific 512px optimum. The _ordering_ of the two effects is the transferable part.

One failure mode neither knob fixes: an embedding can correctly know an image is a kitchen and still rank it top for _"white kitchen, black island"_ when it's a black kitchen with a white island. Attribute binding is exactly what COCO-style benchmarks underexpose.

## 8. The corrected mental model

So, pulling the findings together:

1. **Choose the model family first, then the dimension.** Reversing the order risks picking a weaker model because it exposes more coordinates.
2. **Within an MRL model, 3-4x compression is plausible at about a point.** Nova and Jina agree on this from different labs and different scales.
3. **The cliff is real, and lower than the headline width.** Nova breaks at 256, Jina at 128.
4. **Fine-grained visual retrieval degrades faster than object-centric retrieval.** ViDoRe falls fastest in Nova's table, and I'd treat property photography as nearer that end.
5. **Resolution before dimension.** It's the larger measured effect in the published evidence.
6. **Dimension and numeric precision are separate axes.** Cohere exposes int8 and binary output; those compound with width rather than substituting for it, and conflating the two will confuse your results. A real axis, but a different post.

My current read on the operating regions — a hypothesis to test on your own corpus, not guaranteed scores: 3072→1536 rarely justifiable on quality alone; 1536/1024→768/512 usually the attractive region; 384/256 viable when memory-constrained but increasingly lossy on complex imagery; 128/64 too aggressive without a reranker behind it.

Which brings us back to the 92 GB:

| Dims | Bytes/vector | Raw vectors @ 10M | vs 3072 |
| ---: | -----------: | ----------------: | ------: |
| 3072 |       12 KiB |         122.88 GB |    100% |
| 1536 |        6 KiB |          61.44 GB |     50% |
| 1024 |        4 KiB |          40.96 GB |     33% |
|  768 |        3 KiB |          30.72 GB |     25% |
|  512 |        2 KiB |          20.48 GB |     17% |
|  384 |      1.5 KiB |          15.36 GB |   12.5% |
|  256 |        1 KiB |          10.24 GB |    8.3% |

That's arithmetic on raw float32 payload only. A real ANN index also carries graph or product-quantisation structures, IDs and metadata, and latency has to be measured in your target vector DB: dimension is a first-order effect on distance computation and memory traffic, but traversal, cache behaviour and I/O can dominate.

The sentence the table earns: **Gemini at 768 instead of 3072 saves roughly 92 GB of raw vectors on a 10M-photo catalogue, at one of Google's own recommended MRL sizes.** That's why a sub-point quality loss is worth this much attention.

## 9. Where this breaks down

Four limits I'd want on the record.

- Every number here is from a **general benchmark**. None of them is a property-image benchmark.
- Nova and Jina agreeing is suggestive, not a guarantee that **Gemini's or Cohere's** curves have the same shape. They may well not.
- The 768 starting point for Gemini is **extrapolated** from Google's MRL training sizes plus Nova and Jina's behaviour. It is not demonstrated anywhere I could find.
- Cliff location is model-specific. You'll have to find your own, and a few hundred well-judged domain queries will tell you more than another general benchmark will.

Which is the reversal of where we started. The useful question was never "how many dimensions does this model have". It's "how far down this model's own curve can I go before my queries notice".
