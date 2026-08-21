# Search Query Pipeline Diagrams

https://chatgpt.com/c/6a880532-f940-83ec-bddc-d7e8fb97d080

Below are three progressively richer **query-time reference architectures**. Indexing, embedding generation and ingestion are intentionally excluded.

Two assumptions are worth making explicit:

- **Image vector retrieval can be driven by a text query** when text and images share a multimodal embedding space, so it appears in diagram 2 even though `Image query` itself is optional.
- In the full pipeline, many optional components are **alternative strategies**, not steps that must all execute for every query. Retrieval routing decides which paths to activate.

## 1. Core only

This is the minimum viable modern hybrid search pipeline: understand and normalise the query, apply hard metadata constraints, retrieve lexical and semantic candidates in parallel, union them, then apply business ranking and deduplication.

```mermaid
flowchart TD
    subgraph INPUT["Query inputs"]
        TQ["Text query"]
    end

    subgraph PROCESS["Query processing"]
        QU["Query understanding"]
        QN["Query normalisation"]

        QU --> QN
    end

    subgraph CONSTRAINTS["Constraint handling"]
        MPF["Metadata pre-filter"]
    end

    subgraph RETRIEVAL["Candidate retrieval"]
        LEX["Lexical & metadata retrieval"]
        TVR["Text vector retrieval"]
    end

    subgraph CANDIDATES["Candidate generation"]
        CU["Candidate union"]
    end

    subgraph RANKING["Final ranking"]
        BR["Business ranking"]
        DD["Deduplication"]

        BR --> DD
    end

    subgraph RESULTS["Results"]
        RA["Results assembly"]
    end

    TQ --> QU
    QN --> MPF

    MPF --> LEX
    MPF --> TVR

    LEX --> CU
    TVR --> CU

    CU --> BR
    DD --> RA
```

Conceptually:

```text
Query
  → understand / normalise
  → constrain
  → lexical + vector retrieval
  → union
  → business ranking
  → deduplicate
  → results
```

---

## 2. Core + recommended

This is a stronger **production baseline**. It introduces query rewriting, image retrieval, candidate pruning, result-sufficiency checks, constraint recovery, fusion, semantic reranking, freshness and diversity.

```mermaid
flowchart TD
    subgraph INPUT["Query inputs"]
        TQ["Text query"]
    end

    subgraph PROCESS["Query processing"]
        QU["Query understanding"]
        QN["Query normalisation"]
        QR["Query rewriting"]

        QU --> QN --> QR
    end

    subgraph CONSTRAINTS["Constraint handling"]
        MPF["Metadata pre-filter"]
        RSC{"Retrieval count /<br/>result sufficiency check"}
        CRP["Constraint relaxation policy"]
        ZRF["Zero-result fallback / recovery"]
    end

    subgraph RETRIEVAL["Candidate retrieval"]
        LEX["Lexical & metadata retrieval"]
        TVR["Text vector retrieval"]
        IVR["Image vector retrieval"]
    end

    subgraph CANDIDATES["Candidate generation"]
        CU["Candidate union"]
        CP["Candidate pruning"]

        CU --> CP
    end

    subgraph FUSION["Fusion"]
        F["Fusion"]
        RRF["Reciprocal Rank Fusion"]

        F --> RRF
    end

    subgraph RERANK["Reranking"]
        SR["Semantic rerank"]
        CER["Cross-encoder rerank"]

        SR --> CER
    end

    subgraph RANKING["Final ranking"]
        BR["Business ranking"]
        FR["Freshness / temporal ranking"]
        DIV["Diversity"]
        DD["Deduplication"]

        BR --> FR --> DIV --> DD
    end

    subgraph RESULTS["Results"]
        RA["Results assembly"]
    end

    TQ --> QU
    QR --> MPF

    MPF --> LEX
    MPF --> TVR
    MPF --> IVR

    LEX --> CU
    TVR --> CU
    IVR --> CU

    CP --> RSC

    RSC -->|Sufficient| F
    RSC -->|Too few| CRP
    RSC -->|Zero results| ZRF

    CRP --> MPF
    ZRF --> MPF

    RRF --> SR
    CER --> BR
    DD --> RA
```

The important addition here is the **feedback loop**:

```text
retrieval
  → candidate count/sufficiency check
      → sufficient → continue
      → too few → relax constraints → retrieve again
      → zero → fallback/recovery → retrieve again
```

This is probably the most useful of the three as a **default reference architecture** for a production search system.

---

## 3. Core + recommended + optional

This shows the broader search architecture and makes the main decision points explicit: contextual query processing, dynamic retrieval selection, adaptive constraints, multiple retrieval families, configurable fusion and multimodal reranking.

```mermaid
flowchart TD
    subgraph INPUT["Query inputs"]
        TQ["Text query"]
        IQ["Image query"]
    end

    subgraph PROCESS["Query processing"]
        QU["Query understanding"]
        QN["Query normalisation"]
        QR["Query rewriting"]
        QE["Query expansion"]
        SCS["Session/context-aware search"]
        RDR["Retrieval routing / dynamic retrieval"]

        QU --> QN
        QN --> QR
        QR --> QE
        QE --> RDR
        SCS --> QU
        SCS --> RDR
    end

    subgraph CONSTRAINTS["Constraint handling"]
        CCE["Constraint confidence estimation"]
        MPF["Metadata pre-filter"]

        RSC{"Retrieval count /<br/>result sufficiency check"}
        CRP["Constraint relaxation policy"]

        GB["Geographic broadening"]
        RTW["Range tolerance widening"]
        LCR["Low-confidence constraint removal"]

        RR["Retry retrieval"]
        ZRF["Zero-result fallback / recovery"]

        CCE --> MPF
    end

    subgraph GENERATION_PRE["Candidate generation"]
        CBA["Candidate budget allocation"]
    end

    subgraph RETRIEVAL["Candidate retrieval"]
        LEX["Lexical & metadata retrieval"]
        LSR["Learned sparse retrieval"]
        TVR["Text vector retrieval"]
        IVR["Image vector retrieval"]
        MVR["Multi-vector / passage-level retrieval"]
        LIR["Late-interaction retrieval"]
    end

    subgraph GENERATION_POST["Candidate generation"]
        CU["Candidate union"]
        CP["Candidate pruning"]

        CU --> CP
    end

    subgraph FUSION["Fusion"]
        F{"Fusion"}

        RRF["Reciprocal Rank Fusion"]
        WSF["Weighted score fusion"]
        QDF["Query-dependent fusion"]
        LF["Learned fusion"]
        MWF["Modality-weighted fusion"]
    end

    subgraph RERANK["Reranking"]
        SR["Semantic rerank"]
        CER["Cross-encoder rerank"]
        VLM["VLM rerank"]
        LTR["Learning-to-Rank"]
    end

    subgraph RANKING["Final ranking"]
        BR["Business ranking"]
        FR["Freshness / temporal ranking"]
        PER["Personalisation"]
        DIV["Diversity"]
        DD["Deduplication"]

        BR --> FR --> PER --> DIV --> DD
    end

    subgraph RESULTS["Results"]
        RA["Results assembly"]
    end

    %% Query processing
    TQ --> QU
    IQ --> RDR

    QU --> CCE
    MPF --> RDR

    %% Retrieval planning
    RDR --> CBA

    %% Dynamically selected retrieval paths
    CBA --> LEX
    CBA --> LSR
    CBA --> TVR
    CBA --> IVR
    CBA --> MVR
    CBA --> LIR

    %% Image query naturally targets multimodal/image retrieval
    IQ --> IVR

    %% Candidate aggregation
    LEX --> CU
    LSR --> CU
    TVR --> CU
    IVR --> CU
    MVR --> CU
    LIR --> CU

    %% Sufficiency / recovery
    CP --> RSC

    RSC -->|Sufficient| F
    RSC -->|Too few| CRP
    RSC -->|Zero results| ZRF

    CRP --> GB
    CRP --> RTW
    CRP --> LCR

    GB --> RR
    RTW --> RR
    LCR --> RR

    RR --> MPF
    ZRF --> RDR

    %% Fusion strategies
    F --> RRF
    F --> WSF
    F --> QDF
    F --> LF
    F --> MWF

    %% Reranking
    RRF --> SR
    WSF --> SR
    QDF --> SR
    LF --> SR
    MWF --> SR

    SR --> CER
    SR --> VLM

    CER --> LTR
    VLM --> LTR

    %% Optional LTR can be bypassed
    CER --> BR
    VLM --> BR
    LTR --> BR

    %% Final output
    DD --> RA
```

### How I would interpret the three templates

**Core** is the irreducible hybrid retrieval path:

```text
understand → filter → lexical/vector retrieval → union → rank
```

**Core + recommended** is the better general-purpose production template:

```text
understand/rewrite
→ filter
→ lexical + text-vector + text-to-image retrieval
→ union/prune
→ check sufficiency / recover
→ RRF
→ semantic/cross-encoder rerank
→ business/freshness/diversity/dedup
```

**Full** turns this into an adaptive search system:

```text
query + context
→ understand / rewrite / expand
→ estimate constraint confidence
→ dynamically select retrieval strategies and budgets
→ heterogeneous retrieval
→ aggregate
→ assess sufficiency
→ selectively relax and retry
→ choose fusion strategy
→ multimodal / learned reranking
→ product/business ranking
→ results
```

The key architectural distinction is that the third diagram should **not** be interpreted as “execute every box”. `Retrieval routing / dynamic retrieval` is effectively the control plane: a simple text query might use only lexical + text-vector retrieval, while an image-heavy or difficult query might activate image retrieval, late interaction, broader candidate budgets, different fusion weights, and a VLM reranker.
