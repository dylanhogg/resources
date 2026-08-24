# Apply the Veritasium formula to a concrete example

> Apply this formula to a concrete example of a blogpost about components in modern hybrid search query pipelines aimed at a technical audience.

For a technical blog post on **modern hybrid search query pipelines**, the Veritasium formula maps well if you treat the reader’s existing architecture assumptions as the “misconception”.

A weak version starts with:

> Modern search systems combine lexical retrieval, vector retrieval, fusion and reranking.

Accurate, but a senior engineer probably thinks: _yes, I already know this._

A stronger version starts by challenging the simplified mental model:

> **Hybrid search is not BM25 + vectors + reranking.**
>
> That architecture works on a whiteboard. In production, much of the difficult work happens before retrieval even starts: interpreting intent, resolving constraints, deciding which retrievers should run, allocating candidate budgets, relaxing over-constrained queries, and recovering when retrieval produces too little evidence.

That creates the same effect as “most people tie their shoelaces wrong”: the reader knows the topic, but is told their existing model may be incomplete.

## Applying the formula

### 1. Misconception → curiosity

Use a familiar industry simplification as the opening misconception:

> A common diagram for hybrid search looks something like:
>
> `query → BM25 + vector search → fusion → reranker → results`
>
> It isn't wrong. It's just missing most of the interesting parts.

Then expose the problem:

> Consider:
>
> **“Apartment with loft bedroom, walk-in pantry, on at least 10 acres in Victoria.”**
>
> Before either BM25 or a vector index receives this query, the system already has several decisions to make:
>
> - Is “Victoria” a hard geographic constraint?
> - Is “10 acres” metadata filtering or semantic intent?
> - Is “loft bedroom” a structured property attribute or free-text concept?
> - What happens if only two listings satisfy all constraints?
> - Should acreage be relaxed before geography?
> - Should image retrieval participate?
> - How many candidates should each retriever contribute?

Now you've created the knowledge gap:

> The interesting question isn't _how do we combine lexical and vector search?_
>
> It's **what actually sits between a user's query and the retrieval engines?**

That becomes the central question of the post.

---

## 2. Question → explanation

Instead of immediately presenting a giant architecture diagram, introduce each component because a concrete problem demands it.

### A-plot: follow one query through the system

Use the property query as the narrative thread:

> **“Apartment with loft bedroom, walk-in pantry, on at least 10 acres in Victoria.”**

Then progressively discover why each stage exists.

### B-plot: explain the architecture

Alternate the running example with technical sections.

For example:

---

### The query arrives

**A-plot**

The user has supplied one string. But that string contains several different kinds of information:

- property type: apartment
- architectural concept: loft bedroom
- feature: walk-in pantry
- land constraint: ≥10 acres
- geography: Victoria

The system cannot simply embed that string and hope retrieval sorts everything out.

**B-plot: Query understanding**

Introduce:

- query understanding
- entity extraction
- intent classification
- query normalisation
- structured constraint extraction
- confidence estimates

Then return immediately to the query.

---

### What should actually become a filter?

**A-plot**

Suppose the parser produces:

```text
property_type = apartment
state = VIC
land_size >= 10 acres
keywords = ["loft bedroom", "walk-in pantry"]
```

Filtering on `state = VIC` is straightforward.

Filtering on `land_size >= 10 acres` also looks reasonable.

But should `property_type = apartment` be hard?

If the corpus contains rural loft conversions classified as houses rather than apartments, a strict filter might destroy otherwise highly relevant results.

Now there's another question.

**B-plot: Constraint handling**

Introduce:

- metadata pre-filtering
- constraint confidence
- hard vs soft constraints
- retrieval sufficiency checks
- constraint relaxation
- geographic broadening
- range widening
- zero-result recovery

The component exists because the example has demonstrated the need for it.

---

### Which retrievers should run?

**A-plot**

“Walk-in pantry” may work well lexically.

“Loft bedroom” may benefit from semantic retrieval.

If the user attached a photo of the kind of loft they want, image-vector retrieval suddenly becomes relevant too.

Running every retriever with the same candidate count is possible, but wasteful.

**B-plot: Retrieval routing**

Introduce:

- lexical retrieval
- learned sparse retrieval
- dense text retrieval
- image-vector retrieval
- late-interaction retrieval
- multi-vector / passage retrieval
- dynamic retrieval routing
- candidate budget allocation

Again, the reader encounters the problem before receiving the abstraction.

---

## 3. Use A-plot ↔ B-plot throughout

For this subject, the two plots could be:

**A-plot — One difficult property query moving through the pipeline**

This gives the reader continuity and something concrete to reason about.

**B-plot — The general architecture**

This introduces the reusable engineering concepts.

So the article rhythm becomes:

> query problem
> → architecture component
> → query problem
> → architecture component
> → unexpected failure
> → more architecture

Rather than:

> definition
> → definition
> → definition
> → definition
> → 40-box diagram

That distinction matters particularly for technical writing. Dense architecture posts often fail not because the material is too technical, but because they give readers **solutions before establishing the problems those solutions solve**.

---

# A concrete article structure

## Title

Instead of:

> **Components of a Modern Hybrid Search Pipeline**

Use something that challenges the simplified model:

> **Hybrid Search Is More Than BM25 + Vectors**

or:

> **What Actually Happens Before Hybrid Search?**

or:

> **The Missing Pieces in Most Hybrid Search Architecture Diagrams**

For a technical audience, I'd favour:

> **The Missing Pieces in Modern Hybrid Search Pipelines**

Subheading:

> Retrieval gets most of the attention. Query understanding, constraint handling, routing, candidate generation and recovery are often where the architecture becomes interesting.

---

## Opening

> Most hybrid search architecture diagrams look roughly like this:
>
> `query → lexical + vector retrieval → fusion → reranking → results`
>
> It's a useful abstraction. It's also hiding a lot.
>
> Consider a real-estate query:
>
> **“Apartment with loft bedroom, walk-in pantry, on at least 10 acres in Victoria.”**
>
> What should happen before retrieval?
>
> “Victoria” is probably a structured geographic constraint. “10 acres” looks like a numeric filter. “Walk-in pantry” could be metadata or text. “Loft bedroom” may be semantic. And if those constraints return only two properties, should the system return two results, relax something, or retry using another retrieval strategy?
>
> Once you start asking those questions, the familiar `BM25 + vectors + reranker` diagram expands quickly.
>
> My current mental model is that a modern query pipeline is better understood as a sequence of **interpretation, constraint management, retrieval, candidate construction, ranking and recovery**.

This gives the reader a reason to want the architecture.

---

# Then reveal the architecture incrementally

## 1. Query inputs

Start simple:

```text
Text query
Image query
Context / session state
```

Question introduced:

> What information has the user actually given us?

---

## 2. Query processing

```text
Query understanding
Query normalisation
Query rewriting
Query expansion
Session/context-aware search
Retrieval routing
```

Question:

> What did the user mean, and what representation of that intent should downstream systems receive?

Return to the example after explaining it.

---

## 3. Constraint handling

This is a particularly good place for a new tension point.

> Imagine query understanding worked perfectly.
>
> We now have five perfectly extracted constraints.
>
> Retrieval returns **zero listings**.
>
> Is that success?

Then explain:

```text
Metadata pre-filter
Constraint confidence
Result sufficiency check
Constraint relaxation policy
Geographic broadening
Range tolerance widening
Low-confidence constraint removal
Retry retrieval
Zero-result fallback
```

This section becomes much more memorable because the reader encounters the failure first.

---

## 4. Candidate retrieval

Now ask:

> Once we know what to search for, which retrieval mechanisms should actually participate?

Then:

```text
Lexical / BM25
Learned sparse retrieval
Dense text retrieval
Image retrieval
Multi-vector retrieval
Late interaction
```

You can briefly explain what each is good at rather than treating them as interchangeable retrievers.

---

## 5. Candidate generation

Introduce another practical tension:

> Suppose BM25 returns 1,000 documents and dense retrieval returns 1,000.
>
> Do we rerank all 2,000?
>
> What if 600 are duplicates?
>
> What if image retrieval is disproportionately useful for this query?

Then:

```text
Candidate union
Deduplication
Candidate pruning
Candidate budget allocation
```

---

## 6. Fusion

Now the question becomes:

> How do results from incomparable scoring systems become one candidate set?

Explain:

```text
Reciprocal Rank Fusion
Weighted score fusion
Query-dependent fusion
```

This is much stronger than introducing RRF in isolation.

---

## 7. Reranking

Question first:

> Retrieval optimises recall. How do we turn a few hundred plausible candidates into the ten results a user should actually see?

Then introduce:

```text
Cross-encoder reranking
Late-interaction scoring
Business-rule adjustments
Diversity
Personalisation
```

---

## 8. Post-ranking and recovery

Return to the original query.

Maybe the system found:

- two strong exact matches,
- five near matches after acreage relaxation,
- several semantically similar rural loft properties.

Now discuss:

```text
Result sufficiency
Diversification
Constraint annotations
Fallbacks
Logging
Evaluation signals
```

And show the final response strategy.

---

# The final reveal

Only near the end would I show the full architecture:

```text
Query inputs
    ↓
Query understanding / normalisation / rewriting
    ↓
Constraint extraction + confidence
    ↓
Retrieval routing
    ↓
Metadata pre-filter
    ↓
┌───────────┬────────────┬────────────┬──────────────┐
│ Lexical   │ Sparse     │ Dense text │ Image/vector │
└───────────┴────────────┴────────────┴──────────────┘
    ↓
Candidate union + pruning + budget allocation
    ↓
Fusion
    ↓
Reranking
    ↓
Result sufficiency check
    ↓
Constraint relaxation / retry if necessary
    ↓
Post-ranking
    ↓
Results
```

The diagram is now a **payoff**, rather than homework at the top of the article.

---

# The formula translated to technical blogging

For this particular post:

> **Misconception**
> “Hybrid search = BM25 + vector search + reranking.”

↓

> **Contradiction**
> “That leaves out many of the components that determine whether the system works in production.”

↓

> **Question**
> “So what actually happens between a user's query and the retrieval engines?”

↓

> **A-plot**
> Follow one difficult real-world query through the system.

↕

> **B-plot**
> Explain each general architectural component when the example creates a need for it.

↓

> **New complications**
> Zero results, ambiguous constraints, multiple retrievers, candidate explosion, incomparable scores.

↓

> **Resolution**
> Reveal the complete architecture and show how the components fit together.

So the technical-writing version of the Veritasium formula becomes:

> **Challenge the reader's mental model → pose an engineering problem → follow a concrete example → introduce abstractions only when needed → alternate example and theory → assemble the complete model at the end.**

For this subject, I think that would produce a considerably stronger post than starting with a comprehensive component taxonomy, even if the final technical content were almost identical.
