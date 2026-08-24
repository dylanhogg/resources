# 2-preferred-mental-model

A useful way to think about architecture is as **two independent dimensions**:

1. **Abstraction level** — how far you are from implementation detail.
2. **Viewpoint** — which aspect of the system you are examining.

This avoids mixing terms such as “logical”, “application”, “deployment”, and “runtime”, which answer different kinds of questions.

## 1. Architecture abstraction

```text
Contextual
   WHY does this system exist?
        ↓
Conceptual
   WHAT major capabilities are required?
        ↓
Logical
   HOW should those capabilities work together?
        ↓
Physical
   HOW is this concretely implemented?
```

### Contextual architecture — why

Defines the business problem, actors, goals, scope and external environment.

For a modern property search system:

```text
Property seeker
      ↓
Search experience
      ↓
Relevant property listings

Business goals:
- improve search relevance
- support natural-language queries
- support text and image search
- reduce zero-result searches
- increase engagement / enquiry conversion
```

Typical artefacts:

- system context diagram
- business goals and drivers
- stakeholders
- scope / boundaries
- major external systems

Avoid implementation detail.

---

### Conceptual architecture — what

Defines the major **capabilities** required to satisfy the contextual goals.

```text
Query Input
   ↓
Query Understanding
   ↓
Candidate Retrieval
   ↓
Candidate Ranking
   ↓
Search Results
```

A richer property-search version:

```text
Text / Image Query
        ↓
Query Understanding
        ↓
Constraint Handling
        ↓
Candidate Retrieval
        ↓
Fusion
        ↓
Ranking
        ↓
Result Presentation
```

At this level:

- `Candidate Retrieval` is meaningful.
- `OpenSearch` is too specific.
- `Ranking` is meaningful.
- `cross-encoder running on GPU` is too specific.

The conceptual architecture should remain relatively stable even if the technology stack changes.

---

### Logical architecture — how, abstractly

Decomposes capabilities into responsibilities and interactions without binding them to products or deployment units.

```text
                 ┌────────────────────┐
                 │ Query Understanding│
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Constraint Manager │
                 └─────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
     ┌───────▼────────┐          ┌───────▼────────┐
     │Lexical Retrieval│          │Vector Retrieval│
     └───────┬────────┘          └───────┬────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                       Fusion
                           ↓
                      Reranking
                           ↓
                    Result Selection
```

Logical responsibilities might include:

- query normalisation
- intent/entity extraction
- metadata constraint extraction
- constraint relaxation
- lexical retrieval
- text-vector retrieval
- image-vector retrieval
- candidate union
- fusion
- reranking
- business-rule application
- result sufficiency checks

The important property is that these are **logical components**, not necessarily separate services.

For example:

```text
Logical:
Query Understanding
Constraint Handling
Retrieval Routing
```

All three might later be implemented inside one service.

---

### Physical architecture — how, concretely

Maps logical responsibilities onto specific implementation technologies and infrastructure.

```text
Web / Mobile
     ↓
Search API
     ↓
QUAPI service on GKE
     ↓
Retrieval orchestrator
   ↙            ↘
OpenSearch       Qdrant
BM25 index       HNSW vectors
   ↘            ↙
      RRF fusion
          ↓
GPU reranker service
          ↓
Search API response
```

Physical decisions include:

- OpenSearch vs Elasticsearch
- Qdrant vs OpenSearch vectors
- GKE vs ECS
- Python vs Java
- REST vs gRPC
- managed embedding API vs self-hosted model
- GPU type
- index topology
- replication
- caching
- availability zones

These can change substantially while the logical architecture remains unchanged.

---

# 2. Orthogonal viewpoints

The abstraction hierarchy tells you **how detailed** the architecture is.

Viewpoints tell you **what aspect of the architecture you are looking at**.

A search platform may have all of these views:

```text
Business
Application
Data
Integration
Runtime
Deployment
Technology
Security
Observability
```

Each can be expressed at different abstraction levels.

## Application viewpoint

Focus:

> What software responsibilities exist and how are they organised?

Logical example:

```text
Search API
Query Understanding
Retrieval Orchestrator
Fusion
Reranking
```

Physical example:

```text
search-api
query-understanding-api
ranking-service
```

Useful for:

- service boundaries
- ownership
- application decomposition
- API responsibilities

---

## Data viewpoint

Focus:

> What information exists, where does it come from, and how is it represented?

Conceptually:

```text
Property
Listing
Location
Search Query
Image
Search Result
```

Logically:

```text
Listing
 ├── structured attributes
 ├── description
 ├── text embeddings
 └── image embeddings
```

Physically:

```text
OpenSearch index
- listing text
- metadata
- BM25 fields

Vector index
- listing_embedding
- image_embedding
```

For hybrid search, this viewpoint is particularly important because the same property may exist in several retrieval representations.

---

## Integration viewpoint

Focus:

> How do components communicate?

Example:

```text
Browser
   │ HTTPS
   ▼
Search API
   │ gRPC
   ▼
Query Understanding
   │
   ├── OpenSearch API
   ├── Vector DB API
   └── Ranking API
```

Questions include:

- synchronous vs asynchronous
- REST vs gRPC
- request contracts
- timeout propagation
- retries
- event-driven interactions

---

## Runtime / data-flow viewpoint

Focus:

> What actually happens when a request executes?

This is one of the most useful views for search systems.

```text
"3 bedroom house near the beach under $2m"
                  ↓
          Query normalisation
                  ↓
        Query understanding
                  ↓
      location = coastal area
      beds >= 3
      price <= $2m
                  ↓
        Metadata pre-filter
                  ↓
      ┌───────────┴───────────┐
      ↓                       ↓
   BM25 retrieval       Vector retrieval
     20 ms                  25 ms
      └───────────┬───────────┘
                  ↓
                 RRF
                  ↓
             Reranker
                  ↓
             top 20 results
```

This reveals things static component diagrams do not:

- parallelism
- latency
- retries
- conditional paths
- fallback paths
- fan-out
- candidate counts

For search pipelines, the **runtime view is often more operationally useful than the component view**.

---

## Deployment viewpoint

Focus:

> Where does the software run?

```text
GCP
│
├── GKE
│   ├── search-api pods
│   ├── query-understanding pods
│   └── ranking pods
│
├── OpenSearch cluster
│
├── Vector DB cluster
│
└── GPU inference endpoint
```

Adds concerns such as:

- replicas
- regions
- zones
- autoscaling
- GPU pools
- networking
- failure domains

Deployment architecture is therefore usually a **physical viewpoint**, rather than a separate abstraction level.

---

## Technology viewpoint

Focus:

> What technologies and platforms are used?

For example:

```text
API             → FastAPI
Container       → Docker
Orchestration   → GKE
Lexical search  → OpenSearch
Vector search   → Qdrant
Reranking       → PyTorch / vLLM
Events          → Kafka
Observability   → OpenTelemetry
```

This view is useful for platform and engineering decisions, but it should not replace the logical architecture.

---

## Security viewpoint

Focus:

> Where are trust boundaries and controls?

For the property-search pipeline:

```text
Internet
   ↓
API gateway
   ↓
Authenticated internal services
   ↓
Private search infrastructure
```

It may show:

- authentication
- authorisation
- service identities
- private networks
- secrets
- encryption
- PII boundaries
- audit logging

---

## Observability viewpoint

Focus:

> How do we understand system behaviour?

For search:

```text
Query
  ↓
trace_id
  ↓
Query Understanding
  ↓
Retrieval
  ↓
Fusion
  ↓
Ranking
```

Metrics may include:

- query latency
- retrieval latency
- candidate count
- zero-result rate
- constraint-relaxation rate
- lexical/vector contribution
- reranker latency
- nDCG / Recall / MRR
- model confidence

This is particularly valuable for ML/search systems because **quality and infrastructure behaviour need to be observed together**.

---

# 3. Put the two dimensions together

The useful model is therefore a matrix rather than one hierarchy.

|                   | Conceptual                   | Logical                                 | Physical                       |
| ----------------- | ---------------------------- | --------------------------------------- | ------------------------------ |
| **Application**   | Search capabilities          | Query understanding, retrieval, ranking | Concrete services              |
| **Data**          | Listings, queries, images    | Search documents, embeddings            | OpenSearch/Qdrant schemas      |
| **Integration**   | Systems exchange search data | APIs/events                             | REST/gRPC/Kafka                |
| **Runtime**       | Query → results              | detailed pipeline flow                  | actual calls, timings, retries |
| **Deployment**    | usually minimal              | logical execution boundaries            | GKE, clusters, regions         |
| **Security**      | trust domains                | auth responsibilities                   | IAM, network policies          |
| **Observability** | quality/health goals         | metrics and tracing model               | OTel, dashboards, alerts       |

You normally do **not** need every cell.

The point is that they are available when useful.

---

# 4. Example architecture set for a hybrid property search system

A practical architecture document might contain just five diagrams.

### 1. Context diagram

```text
Property Seeker
      ↓
Search Platform
      ↓
Property Listings
```

Explains **why and scope**.

### 2. Conceptual search architecture

```text
Query
 ↓
Understanding
 ↓
Retrieval
 ↓
Ranking
 ↓
Results
```

Explains **major capabilities**.

### 3. Logical search pipeline

```text
                    Query
                      ↓
             Query Understanding
                      ↓
              Constraint Handling
                      ↓
         ┌────────────┼────────────┐
         ↓            ↓            ↓
      Lexical       Text          Image
     Retrieval      Vector        Vector
                    Search        Search
         └────────────┼────────────┘
                      ↓
                    Fusion
                      ↓
                   Reranking
                      ↓
                    Results
```

Explains **how the search algorithm works**.

### 4. Runtime architecture

```text
request
  ↓
parse query
  ↓
generate embedding
  ↓
parallel retrieval
  ├── lexical
  ├── text vector
  └── image vector
  ↓
candidate union
  ↓
RRF
  ↓
rerank top 100
  ↓
return top 20
```

Explains **how a request executes**.

### 5. Physical/deployment architecture

```text
                 GKE
                  │
          Search API pods
                  │
        Retrieval Service
           ↙           ↘
    OpenSearch       Qdrant
                         │
                  GPU Reranker
```

Explains **what actually runs where**.

---

## The core distinction

The most useful mental shortcut is:

```text
Abstraction = how concrete?

Contextual → Conceptual → Logical → Physical


Viewpoint = what are we examining?

Application / Data / Runtime / Deployment /
Integration / Security / Observability / ...
```

For a modern search system, I would normally prioritise **conceptual architecture, logical pipeline, runtime/data-flow, and physical deployment**. Those four views tend to expose most of the important architectural decisions without producing unnecessary documentation.
