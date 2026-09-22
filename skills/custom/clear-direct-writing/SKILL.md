---
name: clear-direct-writing
description: Rewrite technical writing in clear, direct language while preserving meaning, technical accuracy, detail, and tone. Use when prose is indirect, overly clever, compressed, rhetorical, or harder to understand than necessary.
---

# Clear, Direct Writing

Rewrite the target text in clear, direct language.

## Goals

- Preserve the original meaning, technical accuracy, level of detail, and overall tone.
- Make each point understandable on the first read.
- Prefer clarity over cleverness.
- Keep useful technical nuance.
- Do not make the writing simplistic, generic, or unnecessarily terse.

## Prefer

- Direct subject → verb → object sentence structures.
- Explicit statements over implied conclusions.
- Concrete wording over abstract or metaphorical phrasing.
- Normal prose over fragments used for rhetorical effect.
- Shorter sentences when a sentence contains multiple distinct ideas.
- Clear transitions that state how ideas relate to each other.
- Technical terminology when it improves precision.
- Rewrite headings only when needed to make them more direct and clear; do not make them more clever, catchy, or promotional.
- Voice: Australian English. Use Australian spelling and grammar conventions.

## Avoid

- Indirect, roundabout, or overly compressed phrasing.
- Inverted or unusual sentence structures.
- Clever contrasts that require rereading.
- Metaphors when a literal explanation is clearer.
- Unnecessary rhetorical questions.
- Dramatic setup before a straightforward technical point.
- Sentence fragments used primarily for effect.
- Making the reader infer the main point.
- Artificial sincerity markers such as “the honest take”, “honestly”, or “if I’m being honest”. State the assessment directly instead.
- Empty framing such as “here’s the thing”, “here’s where it gets interesting”, “at its core”, or “the key takeaway is”. Remove the framing and state the point directly.
- Formulaic contrast constructions such as “it’s not X — it’s Y” or “this isn’t just X — it’s Y” when the same idea can be stated directly.
- Meta-commentary such as “let’s unpack this”, “let’s break this down”, or “this raises an important question”. Move directly to the explanation or question.
- Filler emphasis such as “it’s worth noting that”, “importantly”, “crucially”, or “notably” when the sentence is already clear without it.
- Repetition introduced by “in other words” when the following sentence does not materially clarify the previous one.
- Stock phrases such as:
  - “the interesting part is…”
  - “the trick is…”
  - “what falls out of this…”
  - “it turns out…”
  - “this is where things get interesting…”
  - “this is where the story starts to break down…”

State the underlying point directly instead.

## Examples

### Indirect opening

**Avoid:**

> Two intuitions most of us bring to picking an embedding size, both wrong in a way that turns out to be useful.

**Prefer:**

> Most of us have two assumptions about choosing an embedding size, and both turn out to be wrong in ways that are useful.

### Rhetorical setup

**Avoid:**

> The interesting part is not that smaller embeddings lose information. Of course they do. It is where that information disappears.

**Prefer:**

> Smaller embeddings lose information. The important question is which information they lose.

### Clever contrast

**Avoid:**

> On paper, 3072 dimensions should win. In practice, the answer is less tidy.

**Prefer:**

> A 3072-dimensional embedding has more capacity, but that does not always translate into better search quality.

### Formulaic contrast

**Avoid:**

> This isn’t just a storage problem — it’s a search quality problem.

**Prefer:**

> Embedding size affects both storage and search quality.

### Vague transition

**Avoid:**

> This is where the simple story starts to break down.

**Prefer:**

> At this point, the relationship between embedding size and search quality becomes more complicated.

### Unnecessary framing

**Avoid:**

> What falls out of these results is a fairly useful rule of thumb.

**Prefer:**

> These results suggest a useful rule of thumb.

### Artificial sincerity

**Avoid:**

> The honest take is that 3072 dimensions are probably overkill for most production workloads.

**Prefer:**

> For most production workloads, 3072 dimensions are probably unnecessary.

### Figurative language

**Avoid:**

> There is no free lunch here, but there may be a surprisingly cheap one.

**Prefer:**

> Reducing embedding dimensions still involves trade-offs, but the quality loss may be small relative to the storage and latency savings.

### Compressed technical claim

**Avoid:**

> A model can look effectively unchanged at 1536 dimensions while quietly losing performance on the tail.

**Prefer:**

> A model may show little change in aggregate metrics at 1536 dimensions while still performing worse on harder or less common queries.

### Clever conclusion

**Avoid:**

> Bigger vectors buy you capacity. Whether that capacity buys you anything useful is an empirical question.

**Prefer:**

> Larger vectors provide more representational capacity, but you need to test whether that additional capacity improves search quality for your workload.

## Editing Rules

Do not mechanically shorten every sentence. The goal is clarity, not minimum word count. A slightly longer sentence is preferable when it makes the logic more explicit.

Do not remove personality from the writing. Remove phrasing whose main purpose is to sound clever rather than communicate the point.

Do not alter technical claims merely to make the prose simpler. If simplification would change the meaning or remove an important qualification, retain the qualification and rewrite it more clearly.

Do not ban the listed phrases mechanically. Rewrite them when they function as filler, rhetorical decoration, artificial emphasis, or conversational scaffolding rather than carrying useful meaning.

Apply these principles consistently throughout the target text.
