# Outline: embedding dimensions for text-to-image search

**Status:** outline for review. Full post to be written later with the `writing-tone` skill.
**Source:** `1-DR Multimodal Embedding Dimensions for Text-to-Image Vector Search.md`

---

## Brief

|                    |                                                                                                                                                                                                                                                                                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Type**           | Pure evidence survey. What is published, what it supports, what it does not. No eval protocol section, no original benchmark from us.                                                                                                                                                           |
| **Reader**         | MLE choosing a multimodal embedding model and output dimension for text→image vector search. Knows what an embedding and Recall@K are. Does not know that MRL changes the shape of this decision.                                                                                               |
| **Takeaway**       | Dimension is a deploy-time efficiency knob _within_ an MRL-trained model, not a proxy for model quality _across_ models. The first 3-4x cut is usually close to free; the cliff is lower than you think. And for the two models you are most likely evaluating, nobody has published the curve. |
| **Domain framing** | Domain-general findings, real-estate image search as the running concrete example.                                                                                                                                                                                                              |
| **Open weights**   | Jina CLIP v2 only. It is the one open-weight model with a genuine published per-dimension text→image ablation.                                                                                                                                                                                  |
| **Length**         | ~1,400-1,800 words. 9 sections. Four tables (Nova §3, Jina §4, evidence gap §6, storage §8).                                                                                                                                                                                                     |
| **Voice**          | Australian English. Confident where Nova/Jina data is strong, explicit that the Gemini/Cohere recommendation is inference.                                                                                                                                                                      |

**Title candidates** (recognisable topic + unresolved tension):

1. _Does 3072 dimensions actually buy you better image search?_
2. _Embedding dimensions for image search: the evidence, and the gap where you need it most_
3. _3072 vs 768: what the published multimodal ablations actually show_

Leaning (1) as the headline, (2) as the subtitle.

---

## Section-by-section

### 1. Open on the tension (~150 words)

Two intuitions an MLE brings to this decision, both wrong in a useful way:

- **"More dimensions means a richer vector."** Twice the coordinates, twice the room for visual detail.
- **The cross-model version:** "Gemini Embedding 2 is 3072-d and this other model is 768-d, so Gemini has more headroom."

Set the stakes concretely: a 10M-photo real-estate catalogue. At float32, 3072-d is 122.88 GB of raw vectors before any index overhead; 768-d is 30.72 GB. That is a ~92 GB difference on one architectural choice, so the intuition is expensive if it is wrong.

Those two figures only — the full cost curve is Table 4 in §8, once the reader knows what the quality side costs. No definitions here, no MRL yet.

### 2. The central question (~60 words)

> When the _same model_ is served at 3072, 1024 and 256 dimensions, what actually happens to text→image retrieval quality? And does the published evidence cover the models you are most likely to deploy?

Flag up front that the second half of that question has an uncomfortable answer. Creates the pull to keep reading.

### 3. The one commercial ablation that answers it: Amazon Nova MME (~250 words)

**This is the A-plot's first payload.** Nova MME is the only major commercial multimodal model found with a published per-dimension table, architecture and benchmark held constant, only prefix length changing.

**Table 1** (reproduce from the report):

| Nova MME dims | TextCaps | MSCOCO | ViDoRe v2 | float32/vector |
| ------------: | -------: | -----: | --------: | -------------: |
|          3072 |     88.9 |   76.7 |      58.7 |         12 KiB |
|          1024 |     87.9 |   75.6 |      57.7 |          4 KiB |
|           384 |     85.6 |   72.9 |      53.4 |        1.5 KiB |
|           256 |     83.1 |   70.6 |      50.2 |          1 KiB |

Read it in three beats:

- **3072 → 1024 is cheap.** 67% fewer coordinates, 1.0-1.1 points.
- **1024 → 384 starts to bite.** 3.3-5.3 points off the 3072 baseline.
- **256 is not free.** 5.8-8.5 points. Roughly 6.5% / 8.0% / 14.5% relative.

**The detail worth pausing on:** ViDoRe (visual documents, dense fine detail) degrades _fastest_ — 8.5 points at 256 vs 5.8 on TextCaps. First hint that visually rich retrieval has a capacity floor that object-centric captioning benchmarks do not expose. This matters for property photos and is the thread picked up in §7.

**Metric hygiene note** (one line, needed for anyone reproducing): these are average Recall@1/5/10 for image retrieval and NDCG@5 for ViDoRe. Do not mix them with benchmarks quoting Recall@1 alone.

### 4. Open-weight corroboration: Jina CLIP v2 (~200 words)

One vendor's ablation is not a pattern. Jina CLIP v2 is the control: open weights, published text→image Recall@5 at six MRL lengths, reproducible by the reader.

**Table 2:**

| Dims | CLIP Benchmark | Crossmodal-3600 | XTD10 |
| ---: | -------------: | --------------: | ----: |
| 1024 |          79.10 |           81.43 | 84.87 |
|  768 |          79.12 |           82.35 | 84.85 |
|  512 |          78.93 |           82.31 | 84.60 |
|  256 |          78.32 |           81.75 | 84.32 |
|  128 |          75.90 |           78.17 | 81.80 |
|   64 |          70.51 |           72.52 | 77.85 |

The result worth stating plainly: **768 is not worse than 1024, and on Crossmodal-3600 it is nearly a point better.** Even 256 sits within roughly one point of full width. The break is at 128 (3-4 points) and 64 (7-9 points).

Same shape as Nova at a different scale and from a different lab. Two independent labs, consistent curve.

### 5. Why (theory as B-plot, introduced only now) (~150 words)

The data has now created the need for the explanation, so introduce Matryoshka Representation Learning here and not earlier.

The key correction to the reader's mental model: **in an MRL-trained model the dimensions are not equally informative independent coordinates.** Training deliberately front-loads signal into early prefixes, so a truncation is a prefix of a representation designed to survive truncation — not a lossy squeeze of a flat 3072-d space.

This kills the cross-model intuition from §1. Gemini Embedding 2 at 768 is still the full Gemini network; it is not equivalent to swapping in a smaller 768-d vision-language architecture. **Dimension count is not comparable across model families.**

**Reproducibility trap, worth one short paragraph:** taking a prefix changes the vector's norm. Comparing an unnormalised truncation against a normalised full vector will manufacture a dimensionality effect that is not there. Gemini's API re-normalises automatically; if you truncate by hand, L2-normalise before any cosine comparison.

### 6. The gap: the two models you are probably evaluating (~300 words)

The complication, and the most useful thing the post has to say.

**Table 3:**

| Model              | Dimension choices                       | Published multimodal per-dimension ablation? |
| ------------------ | --------------------------------------- | -------------------------------------------- |
| Gemini Embedding 2 | 128-3072; recommended 768 / 1536 / 3072 | **No**                                       |
| Cohere Embed 4     | 256 / 512 / 1024 / 1536                 | **No**                                       |
| Amazon Nova MME    | 256 / 384 / 1024 / 3072                 | **Yes**                                      |
| Jina CLIP v2       | 64 / 128 / 256 / 512 / 768 / 1024       | **Yes**                                      |

**Gemini Embedding 2** — what _is_ published: natively multimodal rather than a text model bolted to a vision encoder; MRL losses trained at 768- and 1536-d prefixes; strong absolute text→image R@1 (mean 80.5; DOCCI 93.4, TextCaps 89.6, MSCOCO 62.9). DOCCI and TextCaps are detailed-caption datasets, which map better onto a property query like _"sunlit open-plan kitchen with a waterfall-edge stone island and black pendant lights"_ than COCO's dominant-object framing does. Label that as an application inference, not a published result.

What is **not** published: any 3072 vs 1536 vs 768 text→image table. The dimension ablation in Google's docs is for **Gemini Embedding 001, a text model, on MTEB** (1536: 68.17, 768: 67.99, 256: 66.19, 128: 63.31). Encouraging for MRL in general. Not evidence about Embedding 2 on complex images. Say so directly.

**Cohere Embed 4** — genuinely multimodal, clean MRL sizes, and the distinctive capability is **fusing image + text into one vector**, which suits listings where room type or structured attributes are known-good text. But again no published per-dimension image-retrieval curve.

Handle the one external datapoint carefully: Amazon's report benchmarked Embed 4 via Bedrock and scored it 22.9 on MSCOCO vs Nova's 76.7. **Three caveats, stated plainly** — Amazon is a direct competitor, the numbers are not Cohere self-reported, and _the output dimension used is not stated_. It is a reason to benchmark Cohere yourself, not evidence about Cohere's dimension curve. Including this caveat is the honest move and makes the rest of the post more trustworthy.

### 7. The knob that probably matters more: input resolution (~200 words)

Deliberate late complication. The post has spent 1,000 words on dimension; now show the reader a bigger lever, because the evidence says so.

Jina CLIP v2 ablates image resolution separately, and the effect dwarfs anything in §3 or §4: on ViDoRe, **224 → 384 pixels moves average NDCG@5 from 0.256 to 0.454**. 512 adds more. 512 → 768 costs 2.25x the image patches for +0.019.

The comparison to land: cutting a well-trained MRL vector from 1024 to 512 costs a fraction of a point. Under-resolving the image costs a _large share of retrieval quality_. Preprocessing can destroy the information before vector width is ever the binding constraint.

Make it concrete with real-estate queries where a small local feature is the whole query:

> "gas cooktop beneath a concealed rangehood" · "herringbone timber flooring" · "frameless shower with niche"

Note Cohere's explicit limit (Embed v4 downsamples above 2,458,624 px, i.e. 1568×1568 square) against Gemini's docs, which publish no equivalent threshold. Caveat honestly: ViDoRe is document screenshots, not property photography, so the specific optimum does not transfer. The ordering of the two effects is the transferable part.

Close with the failure mode that neither knob fixes: an embedding can know an image is a kitchen and still rank it for _"white kitchen, black island"_ when it is a black kitchen with a white island. COCO-style benchmarks underexpose attribute binding.

### 8. Synthesis: the corrected mental model (~200 words)

Restate the model the reader should now hold:

1. **Choose the model family first, then the dimension.** Reversing the order risks picking a weaker model because it exposes more coordinates.
2. **Within an MRL model, 3-4x compression is plausible at ~1 point.** Nova and Jina agree on this from different labs and scales.
3. **The cliff is real and it is lower than the headline width.** Nova at 256 and Jina at 128 both break.
4. **Fine-grained visual retrieval degrades faster than object-centric retrieval.** ViDoRe falls fastest in Nova's table; treat property photography as closer to that end.
5. **Resolution before dimension.** The larger measured effect in the published evidence.
6. **Dimension and numeric precision are separate axes.** Cohere exposes int8 and binary; do not conflate them with width. One or two lines, deliberately not a section — it is a real axis but a different post.

Rough operating regions, labelled as a **hypothesis to test rather than guaranteed scores**: 3072→1536 rarely justifiable on quality alone; 1536/1024→768/512 often the attractive region; 384/256 viable when memory-constrained but increasingly lossy on complex imagery; 128/64 too aggressive without a reranker.

**Table 4** pays off the §1 hook — this is where the reader is actually choosing, so the full cost curve belongs beside the operating regions rather than in the opening:

| Dims | Bytes/vector | Raw vectors @ 10M | vs 3072 |
| ---: | -----------: | ----------------: | ------: |
| 3072 |       12 KiB |         122.88 GB |    100% |
| 1536 |        6 KiB |          61.44 GB |     50% |
| 1024 |        4 KiB |          40.96 GB |     33% |
|  768 |        3 KiB |          30.72 GB |     25% |
|  512 |        2 KiB |          20.48 GB |     17% |
|  384 |      1.5 KiB |          15.36 GB |   12.5% |
|  256 |        1 KiB |          10.24 GB |    8.3% |

State the caveat in one line: raw float32 payload only. A real ANN index also carries graph or PQ structures, IDs and metadata, and query latency has to be measured in the target vector DB — dimension is a first-order effect on distance computation and memory traffic, but traversal, cache behaviour and I/O can dominate.

The sentence this table earns: **Gemini at 768 rather than 3072 saves roughly 92 GB on a 10M-photo catalogue, at one of Google's own recommended MRL sizes** — which is why a sub-point quality loss is worth this much attention.

### 9. Where this breaks down (~100 words)

End on honest limits rather than a tidy bow:

- Every number here is a **general benchmark**. None is a property-image benchmark.
- Nova and Jina agreeing is suggestive, not a guarantee that **Gemini's or Cohere's** curves have the same shape.
- The 768 recommendation for Gemini is **extrapolation** from Google's MRL training sizes plus Nova/Jina behaviour. It is not demonstrated.
- The cliff location is model-specific. Find your own.

Closing line lands the reversal from §1: the useful question is not "how many dimensions does this model have" but "how far down this model's own curve can I go before my queries notice".

---

## Evidence ledger

Tag every claim in the draft as one of these. Keeps the post honest and the skill's evidence check satisfiable.

| Claim                                                                  | Basis                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Nova 3072/1024/384/256 scores                                          | Published, Amazon Nova technical report                            |
| Jina CLIP v2 per-dimension R@5                                         | Published, Jina CLIP v2 paper, open weights                        |
| Jina resolution 224→384→512→768 on ViDoRe                              | Published, same paper                                              |
| Gemini Embedding 2 absolute text→image R@1                             | Published, Google technical paper                                  |
| Gemini Embedding 001 MTEB per-dimension                                | Published, Google docs — **text model, not Embedding 2**           |
| Gemini 128-3072 range, 768/1536/3072 recommended, auto-renormalisation | Published, Gemini API docs                                         |
| Cohere Embed 4 dims, int8/binary, pixel limits, image+text fusion      | Published, Cohere docs                                             |
| Cohere Embed 4 scored 22.9 on MSCOCO                                   | Published but **competitor-run, output dimension unstated**        |
| Bytes/vector and 10M-catalogue storage table                           | Arithmetic; raw payload only, excludes index overhead              |
| DOCCI/TextCaps map better to property queries than COCO                | **Inference**, label as such                                       |
| 768-1024 is the right starting point for real-estate                   | **Inference** from Nova + Jina, not a published real-estate result |

## Resolved decisions

All four review questions are settled. Recorded here so the drafting pass does not reopen them.

1. **Nova leads §3, Jina corroborates in §4.** Evidence strength wins over verifiability, because §4 supplies the verifiability anyway and two independent labs agreeing is the actual argument. The cost accepted: the lead table is a vendor's own, and that vendor scores a competitor at 22.9 in §6.
2. **Keep the Cohere 22.9 number, with the three caveats as drafted** (competitor-run, not self-reported, output dimension unstated). It is the most quotable figure in the source material, so the caveats travel with it every time it appears.
3. **Storage gets its own table (Table 4, §8), not folded into §1 prose.** §1 keeps just the 122.88 GB / 30.72 GB hook.
4. **Quantization stays one or two lines in §8.** Real axis, different post.
