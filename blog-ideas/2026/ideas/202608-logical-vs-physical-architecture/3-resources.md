# 3-resources

There are several strong resources, but they use overlapping terminology differently. The most useful approach is to **take the concepts, not adopt any one taxonomy wholesale**.

## Best resources

| Resource                                                                                                                                                                                 | Key idea                                                                                                                                                                                         | Why it is useful for this mental model                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C4 Model — Simon Brown** [C4 Model](https://c4model.com/?utm_source=chatgpt.com)                                                                                                       | Explicit levels of abstraction: **System → Container → Component → Code**, plus dynamic and deployment diagrams.                                                                                 | Probably the best day-to-day approach for software teams. It directly attacks mixed abstraction levels and ambiguous boxes/arrows. ([C4 model][1])    |
| **Software Systems Architecture — Rozanski & Woods** [Viewpoints and Perspectives](https://www.viewpoints-and-perspectives.info/?utm_source=chatgpt.com)                                 | Separates **viewpoints** such as Functional, Information, Concurrency, Development, Deployment and Operational from cross-cutting **perspectives** such as security, performance and resilience. | This is probably the closest formalisation to the model we've developed. ([Viewpoints and Perspectives][2])                                           |
| **arc42** [arc42 overview](https://arc42.org/overview/?utm_source=chatgpt.com)                                                                                                           | Practical architecture documentation organised around context, building blocks, runtime, deployment, decisions, quality requirements and risks.                                                  | Excellent bridge from architecture theory to something a team can actually maintain. ([arc42][3])                                                     |
| **SEI — Views and Beyond** [Views and Beyond collection](https://www.sei.cmu.edu/library/views-and-beyond-collection/?utm_source=chatgpt.com)                                            | Architecture is a set of relevant **views of system structures**, plus information that applies across views.                                                                                    | Strongest rigorous treatment of why one architecture diagram cannot describe a system adequately. ([Software Engineering Institute][4])               |
| **Kruchten — 4+1 View Model** [Original 4+1 paper](https://users.encs.concordia.ca/~eshihab/teaching/readings/kru95.pdf?utm_source=chatgpt.com)                                          | Logical, Process, Development and Physical views, validated by scenarios.                                                                                                                        | Foundational explanation of why static logical structure, runtime behaviour and physical deployment are different things. ([Concordia ENCS Users][5]) |
| **ISO/IEC/IEEE 42010:2022** [ISO 42010](https://www.iso.org/standard/74393.html?utm_source=chatgpt.com)                                                                                  | Formal vocabulary around architecture descriptions, stakeholders, concerns, viewpoints and model kinds.                                                                                          | Useful when you want precise terminology rather than an opinionated diagramming method. ([ISO][6])                                                    |
| **ArchiMate 101 — The Open Group community** [ArchiMate 101](https://archimate-community.pages.opengroup.org/workgroups/archimate-101/?utm_source=chatgpt.com)                           | Explicit Business, Application and Technology layers plus viewpoints spanning them.                                                                                                              | Useful for enterprise-scale systems where software needs to be related to business capabilities and infrastructure. ([GitLab][7])                     |
| **Michael Nygard — Architecture Decision Records** [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions?utm_source=chatgpt.com) | Record **context → decision → consequences**, not just the resulting architecture.                                                                                                               | Adds the missing “why did we build it this way?” dimension. ([Cognitect.com][8])                                                                      |

### The two I'd read first

For what you're trying to achieve, I'd particularly recommend:

**1. Rozanski & Woods.** Their distinction between **viewpoints** and **perspectives** improves the model we've been discussing.

They essentially say:

```text
Viewpoint
    What structural aspect are we examining?

    Functional
    Information
    Concurrency
    Development
    Deployment
    Operational


Perspective
    What system quality are we checking across those views?

    Performance
    Security
    Availability
    Resilience
    Evolution
    ...
```

A security concern isn't really another architecture _level_. It cuts across your application, data, runtime and deployment views. The same applies to latency, scalability and resilience. ([Viewpoints and Perspectives][9])

That gives us a useful extension from **two dimensions to three**:

```text
1. ABSTRACTION
   How concrete are we?

   Contextual → Conceptual → Logical → Physical


2. VIEWPOINT
   What aspect are we describing?

   Functional / Application
   Data / Information
   Runtime
   Integration
   Deployment
   Development
   ...


3. QUALITY PERSPECTIVE
   What properties must hold across those views?

   Latency
   Scalability
   Reliability
   Security
   Observability
   Evolvability
   Cost
   ...
```

For an ML/search system, I think this is substantially better than trying to invent an “observability architecture”, “performance architecture”, “security architecture”, etc. as peer architecture levels.

---

**2. C4.** C4 is particularly strong on the communication problem. Its documentation explicitly calls out common failures such as mixed abstraction levels, unexplained notation, ambiguous elements and unlabelled relationships. ([C4 model][1])

Its review checklist is surprisingly valuable. Among other things, it asks whether:

- the diagram's **type and scope** are obvious;
- every element's **level of abstraction** is understood;
- every element has a clear responsibility;
- every arrow says what the relationship actually means;
- notation, acronyms and technology choices are unambiguous. ([C4 model][10])

For engineering teams, that is arguably more important than picking the theoretically “correct” architecture taxonomy.

---

## One terminology trap worth understanding

Kruchten's famous **4+1** model uses:

```text
Logical
Process
Development
Physical
```

where **Physical essentially means deployment topology**. ([Concordia ENCS Users][5])

But the conceptual/logical/physical hierarchy we've been discussing uses:

```text
Conceptual
Logical
Physical
```

where **Physical means concrete implementation** more generally.

Those aren't contradictory. They are using the word _physical_ along **different dimensions**.

This is exactly why I would avoid saying:

> “This is the physical architecture.”

Instead say:

> **“This is the physical deployment view.”**

or:

> **“This is the logical runtime view.”**

or:

> **“This is the conceptual data view.”**

That little two-word convention removes a surprising amount of ambiguity.

---

# ML-specific resources

Traditional software architecture literature does not adequately emphasise data, models, training systems, evaluation and feedback loops. A few ML references fill that gap.

### Hidden Technical Debt in Machine Learning Systems

[Google Research paper](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/?utm_source=chatgpt.com)

Still one of the most important ML-system architecture papers. It discusses system-level problems such as dependency entanglement, feedback loops, undeclared consumers, data dependencies and boundary erosion. ([Google Research][11])

Its big architectural lesson is:

```text
          ML system

data ──────────────┐
features ──────────┤
configuration ─────┤
models ────────────┤
serving ───────────┤
monitoring ────────┤
evaluation ────────┤
dependencies ──────┘

       model code
          ↑
     only one part
```

So an ML architecture that merely shows:

```text
API → Model → Response
```

is usually hiding most of the architecturally important system.

---

### Google's MLOps architecture guidance

[MLOps: Continuous delivery and automation pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)

Particularly useful because it explicitly distinguishes:

- experimentation;
- data pipelines;
- training;
- validation;
- model registry;
- deployment;
- online serving;
- monitoring;
- continuous training.

It also stresses that production ML systems are much larger than the actual ML model. ([Google Cloud Documentation][12])

This suggests that an ML system often deserves **different runtime views**:

```text
Online search runtime
Query → Retrieval → Ranking → Results


Offline indexing runtime
Listings → Processing → Embeddings → Search indexes


Training runtime
Training data → Train → Evaluate → Model registry


Evaluation runtime
GT dataset → Candidate system → Metrics → Quality gate
```

Trying to combine those four onto one “architecture diagram” is usually a mistake.

---

### Google's Rules of ML

[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml?utm_source=chatgpt.com)

Less about diagramming, but highly relevant architecturally. It repeatedly emphasises getting the **end-to-end pipeline and infrastructure correct**, keeping the initial ML simple, instrumenting metrics early, testing infrastructure separately from ML, and assigning ownership to data/features. ([Google for Developers][13])

That reinforces an important point:

> The architecture should expose the boundaries that teams need to reason about and test, rather than merely showing where the model sits.

---

# A refined mental model for software + ML

After comparing these approaches, I'd refine our earlier model to this:

```text
                  ARCHITECTURE DESCRIPTION
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ABSTRACTION        VIEWPOINT       PERSPECTIVE
          │                │                │
     how concrete?      what aspect?      what quality?
          │                │                │
     Contextual         Context           Performance
     Conceptual         Functional        Reliability
     Logical            Data              Security
     Physical           Runtime           Scalability
                        Integration       Observability
                        Development       Evolvability
                        Deployment        Cost
```

And one more dimension that diagrams alone don't capture:

```text
DECISIONS

Why was it designed this way?
What alternatives were considered?
What trade-offs were accepted?
```

That's where ADRs fit.

## Applied to hybrid property search

For example:

### Logical functional view

```text
Query Understanding
        ↓
Constraint Handling
        ↓
     Retrieval
    ↙    ↓    ↘
lexical text  image
    ↘    ↓    ↙
       Fusion
         ↓
      Ranking
```

It says **what responsibilities exist**.

### Physical functional view

```text
QU API
  ↓
OpenSearch + Qdrant
  ↓
RRF implementation
  ↓
GPU cross-encoder
```

It says **what implements them**.

### Logical runtime view

```text
query
  ↓
understand
  ↓
parallel retrieval
  ↓
fusion
  ↓
rerank
  ↓
results
```

It says **what happens during one request**.

### Physical deployment view

```text
GKE
├─ query pods
├─ retrieval orchestrator
└─ ranking pods

OpenSearch cluster
Qdrant cluster
GPU endpoint
```

It says **where concrete instances execute**.

### Data view

```text
Listing
├─ metadata
├─ lexical representation
├─ text embeddings
└─ image embeddings
```

It explains a completely different concern without contaminating the runtime diagram.

### Performance perspective

Now apply this **across** those views:

```text
Query understanding       < 20 ms
Retrieval                 < 50 ms
Fusion                    < 10 ms
Reranking                 < 70 ms
──────────────────────────────────
End-to-end p95            < 200 ms
```

That might cause changes to:

- functional decomposition;
- parallelism in the runtime view;
- indexes in the data view;
- GPU allocation in the deployment view.

Performance therefore isn't merely another box or diagram. It is a **constraint that cuts through multiple views**.

---

# Practical rules I would adopt for a software/ML team

The literature above points towards a fairly small set of conventions that could prevent most architecture-document ambiguity:

1. **Name every diagram `<abstraction> <viewpoint> view`.** For example, `Logical Search Runtime View` or `Physical Search Deployment View`.

2. **State the question the diagram answers.**
   Example: “How are candidates generated and ranked for an online query?”

3. **Don't casually mix abstraction levels.**
   `Vector Retrieval` and `Qdrant 1.15 on GKE` should normally not appear as peer boxes.

4. **Separate static structure from runtime behaviour.**
   A component dependency graph is not a request sequence.

5. **Map logical → physical explicitly.**

   ```text
   Logical responsibility       Physical implementation
   ----------------------------------------------------
   Lexical retrieval         →  OpenSearch
   Vector retrieval          →  Qdrant
   Reranking                 →  Qwen reranker on GPU
   ```

   This makes technology substitution much easier to reason about.

6. **Treat data/model artefacts as first-class architectural elements in ML systems.** Models, embeddings, GT datasets, indexes and features shouldn't disappear behind generic “ML” boxes.

7. **Use perspectives for cross-cutting qualities.** Performance, reliability, observability, privacy, cost and security should be tested against relevant views.

8. **Capture important decisions separately in ADRs.** The diagram shows _what is_. An ADR explains _why it became that way_. ([Cognitect.com][8])

9. **Use a glossary.** arc42 explicitly includes one for architecture-specific terminology; this becomes particularly important for words such as _component_, _service_, _pipeline_, _model_, _index_, _retriever_ and _ranker_. ([arc42][3])

10. **Create only views that answer a real question.** Both C4 and the older 4+1 work explicitly allow omitting views that add no value. ([C4 model][14])

The overall goal isn't to produce a comprehensive architecture taxonomy. It's to make it difficult for two engineers to look at the same diagram and form **different mental models of what the boxes and arrows mean**. The combination I'd favour for a modern ML/search team is essentially **C4's disciplined abstraction + Rozanski/Woods' viewpoints and perspectives + arc42's lightweight documentation structure + ADRs + ML-specific data/training/evaluation/runtime views**.

[1]: https://c4model.com/introduction?utm_source=chatgpt.com "Introduction | C4 model"
[2]: https://www.viewpoints-and-perspectives.info/home/viewpoints/?utm_source=chatgpt.com "Software Systems Architecture"
[3]: https://arc42.org/overview "arc42 Template Overview - arc42"
[4]: https://www.sei.cmu.edu/library/views-and-beyond-collection/?utm_source=chatgpt.com "Views and Beyond Collection | CMU Software Engineering Institute"
[5]: https://users.encs.concordia.ca/~eshihab/teaching/readings/kru95.pdf?utm_source=chatgpt.com "Architectural Blueprints—The “4+1” View Model of Software Architecture"
[6]: https://www.iso.org/standard/74393.html?utm_source=chatgpt.com "ISO/IEC/IEEE 42010:2022 - Software, systems and enterprise — Architecture description"
[7]: https://archimate-community.pages.opengroup.org/workgroups/archimate-101/?utm_source=chatgpt.com "ArchiMate 101: A Practical Introduction"
[8]: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions?utm_source=chatgpt.com "Documenting Architecture Decisions"
[9]: https://www.viewpoints-and-perspectives.info/home/perspectives/?utm_source=chatgpt.com "Software Systems Architecture"
[10]: https://c4model.com/diagrams/checklist?utm_source=chatgpt.com "Review checklist | C4 model"
[11]: https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/?utm_source=chatgpt.com "Hidden Technical Debt in Machine Learning Systems"
[12]: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=en&utm_source=chatgpt.com "MLOps: Continuous delivery and automation pipelines in machine learning  |  Cloud Architecture Center  |  Google Cloud Documentation"
[13]: https://developers.google.com/machine-learning/guides/rules-of-ml "Rules of Machine Learning:  |  Google for Developers"
[14]: https://c4model.com/diagrams?utm_source=chatgpt.com "Diagrams | C4 model"
