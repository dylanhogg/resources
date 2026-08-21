# Search Query Pipeline Diagrams

https://claude.ai/chat/c3bc1980-cd77-4d70-89a4-8014c9e5d175

# Hybrid Multimodal Query Search Pipelines

Three progressive reference templates for a query-time search pipeline over a corpus where each
document has text fields, textual/numeric metadata, and an associated set of images.

**Assumed offline/indexing state (out of scope here):**

- Lexical + metadata index (BM25 / filterable fields) already built
- Vector index with text embeddings (document/passage level) already built
- Vector index with image embeddings in a shared or bridged space with the text encoder
- Optional: learned sparse (SPLADE-style) and multi-vector (ColBERT-style) indexes

**Legend used in all diagrams:**

| Style  | Meaning                                                    |
| ------ | ---------------------------------------------------------- |
| Blue   | `core` — minimum viable pipeline                           |
| Green  | `recommended` — expected in a competent production system  |
| Purple | `optional` — added for quality, scale, or specific domains |

---

## 1. Core only

The smallest pipeline that is still genuinely hybrid: one lexical leg, one dense leg, a naive
union, and business rules applied on top.

```mermaid
flowchart TD
    QT["Text query"]

    QT --> NORM["Query normalisation<br/>unicode NFKC, case, punctuation,<br/>whitespace, spell-safe tokenisation"]
    NORM --> UND["Query understanding<br/>intent classification,<br/>entity and constraint extraction"]

    UND --> PF["Metadata pre-filter<br/>build filter predicate from<br/>extracted constraints"]

    PF --> LEX["Lexical and metadata retrieval<br/>BM25 over text fields<br/>+ filter predicate"]
    PF --> TVEC["Text vector retrieval<br/>ANN over text embeddings<br/>+ filter predicate"]

    LEX --> UNION["Candidate union<br/>merge by doc id,<br/>normalise scores per leg"]
    TVEC --> UNION

    UNION --> BR["Business ranking<br/>availability, quality tier,<br/>policy boosts and demotions"]
    BR --> DD["Deduplication<br/>collapse near-identical docs,<br/>keep highest ranked"]
    DD --> OUT["Results assembly<br/>hydrate fields, images,<br/>highlights, pagination"]

    classDef core fill:#dbeafe,stroke:#1d4ed8,stroke-width:1px,color:#0b1220

    class QT,NORM,UND,PF,LEX,TVEC,UNION,BR,DD,OUT core
```

**Notes**

- With no fusion stage, `Candidate union` must do the merging itself. The usual minimum is
  min-max or z-score normalisation per leg, then a fixed weighted sum. This is the weakest link
  in the core pipeline and the first thing diagram 2 replaces.
- The pre-filter is applied _inside_ both retrievers, not after them. Post-filtering a top-k ANN
  result set silently collapses recall when the filter is selective.
- No sufficiency check here means an over-constrained query returns zero results with no recovery.

---

## 2. Core + recommended

Adds query rewriting, cross-modal image retrieval, a constraint relaxation loop, principled rank
fusion, a rerank stage, and freshness/diversity in the final ranking.

```mermaid
flowchart TD
    QT["Text query"]

    QT --> NORM["Query normalisation"]
    NORM --> UND["Query understanding<br/>intent, entities,<br/>constraint extraction"]
    UND --> RW["Query rewriting<br/>spelling, synonym canonicalisation,<br/>constraint-stripped retrieval query"]

    RW --> PF["Metadata pre-filter"]

    PF --> LEX["Lexical and metadata retrieval<br/>BM25 + filters"]
    PF --> TVEC["Text vector retrieval<br/>ANN over text embeddings"]
    PF --> IVEC["Image vector retrieval<br/>text-to-image ANN over<br/>document image embeddings"]

    LEX --> SUF{"Result sufficiency check<br/>candidates >= threshold?"}
    TVEC --> SUF
    IVEC --> SUF

    SUF -->|"sufficient"| UNION
    SUF -->|"too few"| RELAX["Constraint relaxation policy<br/>ordered ladder, one step per pass,<br/>max N passes"]
    RELAX -->|"relaxed filter, re-run"| PF
    RELAX -->|"ladder exhausted"| ZERO["Zero-result fallback<br/>drop hard filters, widen to<br/>pure semantic, or 'did you mean'"]
    ZERO --> UNION

    UNION["Candidate union<br/>merge by doc id,<br/>keep per-leg ranks"]
    UNION --> PRUNE["Candidate pruning<br/>cap to top-N per leg,<br/>drop below score floor"]

    PRUNE --> FUSE["Fusion<br/>combine per-leg ranked lists"]
    FUSE --> RRF["Reciprocal Rank Fusion<br/>sum 1 / (k + rank), k ~ 60<br/>rank-based, scale-free"]

    RRF --> SEM["Semantic rerank<br/>take top-K fused, K ~ 50-200"]
    SEM --> CE["Cross-encoder rerank<br/>joint query-document scoring<br/>over text fields"]

    CE --> BR["Business ranking"]
    BR --> FRESH["Freshness and temporal ranking<br/>recency decay or<br/>time-window boost"]
    FRESH --> DIV["Diversity<br/>MMR or per-attribute<br/>result capping"]
    DIV --> DD["Deduplication"]
    DD --> OUT["Results assembly"]

    classDef core fill:#dbeafe,stroke:#1d4ed8,stroke-width:1px,color:#0b1220
    classDef rec fill:#dcfce7,stroke:#15803d,stroke-width:1px,color:#0b1220

    class QT,NORM,UND,PF,LEX,TVEC,UNION,BR,DD,OUT core
    class RW,IVEC,SUF,RELAX,ZERO,PRUNE,FUSE,RRF,SEM,CE,FRESH,DIV rec
```

**Notes**

- `Image vector retrieval` here is cross-modal: the _text_ query is encoded with the image model's
  text tower (CLIP/SigLIP-style) and matched against document image embeddings. No image input
  is required for this leg to be useful.
- The relaxation loop deliberately re-enters at `Metadata pre-filter` rather than at retrieval,
  because relaxation changes the filter predicate. Cap passes (2 is typical) and always tag the
  response so the UI can say "showing broader results".
- RRF is the default fusion choice because it needs no score calibration across legs — lexical
  BM25 scores and cosine similarities are not comparable, and score-based fusion requires
  per-corpus tuning that RRF avoids.
- Rerank depth is the main latency dial. Cross-encoding 100 candidates is roughly 10x the cost of
  fusion; budget for it explicitly.

---

## 3. Core + recommended + optional

Full surface area. Most systems will only ever enable a subset of the purple nodes — treat this as
the menu, not the target.

```mermaid
flowchart TD
    subgraph INPUTS["Query inputs"]
        QT["Text query"]
        QI["Image query<br/>uploaded or reference image"]
    end

    subgraph QPROC["Query processing"]
        NORM["Query normalisation"]
        UND["Query understanding<br/>intent, entities, constraints"]
        RW["Query rewriting"]
        EXP["Query expansion<br/>synonyms, PRF,<br/>LLM-generated variants or HyDE"]
        SESS["Session and context-aware search<br/>prior turns, clicks,<br/>anaphora resolution"]
        ROUTE["Retrieval routing<br/>dynamic retrieval<br/>pick which legs to fire"]
        QIEMB["Query image encoding<br/>image tower embedding"]
    end

    subgraph CONSTR["Constraint handling"]
        CONF["Constraint confidence estimation<br/>per-constraint extraction confidence"]
        PF["Metadata pre-filter<br/>hard vs soft constraint split"]
        SUF{"Result sufficiency check"}
        RELAX["Constraint relaxation policy<br/>ordered ladder"]
        LOWC["Low-confidence constraint removal"]
        RANGE["Range tolerance widening<br/>numeric and date bounds"]
        GEO["Geographic broadening<br/>radius or region expansion"]
        RETRY["Retry retrieval<br/>bounded passes"]
        ZERO["Zero-result fallback<br/>recovery and suggestions"]
    end

    subgraph RETR["Candidate retrieval"]
        BUDGET["Candidate budget allocation<br/>per-leg k under latency budget"]
        LEX["Lexical and metadata retrieval"]
        SPARSE["Learned sparse retrieval<br/>SPLADE-style expansion"]
        TVEC["Text vector retrieval"]
        IVEC["Image vector retrieval<br/>text-to-image and image-to-image"]
        MVEC["Multi-vector / passage-level retrieval<br/>chunk hits rolled up to doc"]
        LATE["Late-interaction retrieval<br/>ColBERT-style MaxSim"]
    end

    subgraph CAND["Candidate generation"]
        UNION["Candidate union"]
        PRUNE["Candidate pruning"]
    end

    subgraph FUSION["Fusion"]
        FUSE["Fusion"]
        QDF{"Query-dependent fusion<br/>select strategy and weights<br/>by intent and query type"}
        RRF["Reciprocal Rank Fusion"]
        WSF["Weighted score fusion<br/>normalised score blend"]
        MODW["Modality-weighted fusion<br/>text vs image leg weighting"]
        LFUSE["Learned fusion<br/>model-predicted leg weights"]
    end

    subgraph RERANK["Reranking"]
        SEM["Semantic rerank"]
        CE["Cross-encoder rerank"]
        VLM["VLM rerank<br/>query vs document images,<br/>top-M only"]
        LTR["Learning-to-Rank<br/>GBDT or neural over<br/>retrieval, behaviour, quality features"]
    end

    subgraph FINAL["Final ranking"]
        BR["Business ranking"]
        FRESH["Freshness and temporal ranking"]
        PERS["Personalisation<br/>user profile and history signals"]
        DIV["Diversity"]
        DD["Deduplication"]
    end

    OUT["Results assembly"]

    QT --> NORM
    NORM --> UND
    SESS --> UND
    SESS --> RW
    UND --> RW
    RW --> EXP
    EXP --> ROUTE
    UND --> CONF
    QI --> QIEMB

    CONF --> PF
    ROUTE --> BUDGET
    PF --> BUDGET

    BUDGET --> LEX
    BUDGET --> SPARSE
    BUDGET --> TVEC
    BUDGET --> IVEC
    BUDGET --> MVEC
    BUDGET --> LATE
    QIEMB --> IVEC

    LEX --> SUF
    SPARSE --> SUF
    TVEC --> SUF
    IVEC --> SUF
    MVEC --> SUF
    LATE --> SUF

    SUF -->|"sufficient"| UNION
    SUF -->|"too few"| RELAX
    RELAX --> LOWC
    RELAX --> RANGE
    RELAX --> GEO
    LOWC --> RETRY
    RANGE --> RETRY
    GEO --> RETRY
    RETRY -->|"re-run with relaxed filter"| PF
    RELAX -->|"ladder exhausted"| ZERO
    ZERO --> UNION

    UNION --> PRUNE
    PRUNE --> FUSE
    FUSE --> QDF
    QDF --> RRF
    QDF --> WSF
    QDF --> MODW
    QDF --> LFUSE
    RRF --> SEM
    WSF --> SEM
    MODW --> SEM
    LFUSE --> SEM

    SEM --> CE
    CE --> VLM
    VLM --> LTR
    LTR --> BR
    BR --> FRESH
    FRESH --> PERS
    PERS --> DIV
    DIV --> DD
    DD --> OUT

    classDef core fill:#dbeafe,stroke:#1d4ed8,stroke-width:1px,color:#0b1220
    classDef rec fill:#dcfce7,stroke:#15803d,stroke-width:1px,color:#0b1220
    classDef opt fill:#ede9fe,stroke:#6d28d9,stroke-width:1px,color:#0b1220

    class QT,NORM,UND,PF,LEX,TVEC,UNION,BR,DD,OUT core
    class RW,IVEC,SUF,RELAX,ZERO,PRUNE,FUSE,RRF,SEM,CE,FRESH,DIV rec
    class QI,QIEMB,EXP,SESS,ROUTE,CONF,LOWC,RANGE,GEO,RETRY,SPARSE,MVEC,LATE,BUDGET,QDF,WSF,MODW,LFUSE,VLM,LTR,PERS opt
```

**Notes**

- `Query-dependent fusion` is drawn as a selector over the other fusion strategies rather than a
  parallel strategy, because in practice it is the policy layer that picks RRF vs weighted vs
  modality-weighted per query class. `Learned fusion` replaces the hand-tuned weights in that
  same slot.
- `Constraint confidence estimation` feeds the pre-filter so constraints can be split into hard
  (high confidence, applied as filters) and soft (low confidence, applied as ranking boosts). This
  is what makes `Low-confidence constraint removal` cheap — the relaxation ladder just promotes
  soft constraints out of the filter.
- Relaxation ladder order matters and is domain-specific. A common ordering is: low-confidence
  constraints first, then numeric/date tolerance, then geography, since geographic broadening
  usually degrades perceived relevance the most.
- `VLM rerank` is expensive enough that it should only see the top 10-20 candidates, and usually
  only when the query has visual intent or an image input is present.
- `Late-interaction retrieval` can equally be positioned as a rerank stage over fused candidates
  rather than a first-stage retriever. Placement depends on whether you maintain a ColBERT-style
  index or only score on demand.
- Deduplication appears last to match the component ordering given, but exact-duplicate collapse
  by content hash is usually cheaper at `Candidate union`; near-duplicate collapse stays here.

---

## Cross-cutting concerns (not stages)

Worth reserving space for when you expand these templates:

- **Timeouts and partial results** — each retrieval leg needs an independent deadline; the pipeline
  should degrade to whichever legs returned rather than failing the request.
- **Caching** — query understanding/rewriting output, query embeddings, and rerank scores all cache
  well at different TTLs.
- **Observability** — per-stage candidate counts and latencies, relaxation-pass counters, and
  per-leg contribution to the final top-10 are the metrics that make regressions debuggable.
- **Offline evaluation hooks** — the fusion and rerank stages are where you want deterministic
  replay against a labelled set.
