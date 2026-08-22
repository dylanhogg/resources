---
name: technical-blog-writing
description: Write informative, engaging technical blog posts for engineering, ML, data, infrastructure, and architecture audiences using concrete examples, curiosity, evidence, and explicit trade-offs.
---

# Technical Blog Writing

## Purpose

Write technical posts that are useful first and engaging by design. Start from a tension, failure, edge case, or incomplete mental model; create a reason to keep reading; then teach through a concrete example and progressively reveal the broader technical model.

## Core Formula

**Tension → Question → Concrete example ↔ Explanation → Trade-off → Complication → Synthesis**

Create demand for the explanation before giving it.

## Workflow

### 1. Define the reader and takeaway

Identify what the intended reader probably already knows and what they should understand, believe, or be able to do differently after reading.

### 2. Open with a useful tension

Start from one of:

- an incomplete mental model or misconception
- an unexpected result or failure
- an awkward edge case
- two reasonable approaches with different trade-offs

Avoid opening with definitions, taxonomy, or a complete architecture diagram.

### 3. Pose the central question

Turn the tension into a concrete technical question the post will resolve.

### 4. Choose one concrete A-plot

Use a realistic query, API call, failure, dataset, benchmark, architecture decision, or debugging scenario as the narrative spine. Return to it throughout the post.

### 5. Introduce theory as the B-plot

Explain concepts only when the example creates a need for them.

Prefer:

> problem → question → concept → application

rather than:

> definition → definition → example

Use code, diagrams, tables, measurements, and concrete examples when they communicate more efficiently than prose.

### 6. Add complications progressively

Use realistic constraints such as ambiguity, latency, cost, scale, low-result cases, conflicting signals, and failure recovery.

Each complication should teach a trade-off, not merely introduce another component:

> problem → options → trade-off → decision → consequence

Resolve one question while naturally exposing the next useful question.

### 7. Reveal the complete model late

Show the full architecture, taxonomy, framework, or diagram only after the reader understands why the pieces exist.

### 8. End with synthesis

Conclude with:

- the corrected or expanded mental model
- the most important trade-offs
- practical heuristics or recommendations
- where the model breaks down or alternatives make sense

## Evidence

Ground important claims in measurements, concrete examples, primary references, or clearly labelled experience. Distinguish established facts from inference, convention, and opinion.

## Titles

Prefer **recognisable topic + interesting tension**.

Make the subject clear to the intended reader while leaving an unresolved question, contradiction, or implication. Avoid vague clickbait.

## Voice

- Use **Australian English** spelling and grammar.
- Write like a technically competent peer explaining how they currently see the problem.
- Sound human: use specific, context-grounded phrasing and natural sentence variation.
- Be confident where evidence is strong and explicit about uncertainty where it matters.
- Make recommendations when warranted, with the reasoning and trade-offs.
- Use hyphens and em dashes sparingly.

## Avoid

- Salesy or absolute framing such as "obviously", "clearly the best", or "the only way".
- Dismissing alternatives without explaining why.
- False humility that obscures a real recommendation.
- Bravado that hides genuine uncertainty.
- Generic transitions, overly polished symmetry, filler, hype, and throat-clearing preamble.
- Explaining solutions before establishing the problems they solve.
- Dumping a glossary, taxonomy, or complete diagram at the top.
- Contrived clickbait or simplifying engineering trade-offs for narrative neatness.

## Recommended Structure

```text
Title: recognisable topic + tension

Reader's familiar model
↓
Tension / edge case / surprising result
↓
Central question
↓
Concrete example
↕
Technical explanation
↓
Trade-off / complication
↕
Further explanation
↓
Full model / architecture
↓
Practical recommendations + alternatives
↓
Corrected mental model
```

## Quality Check

Before finalising, verify:

- Is the intended reader and takeaway clear?
- Does the opening create a genuine technical tension?
- Is there a clear question pulling the reader through the post?
- Is one concrete example the narrative spine?
- Are concepts introduced because the example requires them?
- Do sections teach trade-offs rather than just list components?
- Are important claims grounded in evidence or clearly labelled judgement?
- Does each section create a natural reason to continue?
- Is the complete model shown after its pieces have been motivated?
- Does the conclusion leave the reader with practical judgement, not just terminology?
