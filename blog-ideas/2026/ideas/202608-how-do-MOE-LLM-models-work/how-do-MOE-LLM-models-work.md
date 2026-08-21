# How do MOE LLM models work?

https://chatgpt.com/c/6a87a55a-5f70-83ec-8866-1548bcca5422

At a high level, how are MOE LLM models trained and how do the experts get activated in inference?

At a high level, a Mixture-of-Experts (MoE) LLM replaces some of the normal dense feed-forward layers with a set of separate feed-forward networks called **experts**, plus a small **router/gating network** that decides which experts should process each token.

The key idea is:

> **Many parameters exist, but only a small subset are used for each token.**

So an MoE model might contain, say, 64 experts in a layer but activate only 2 experts per token. This lets the model have a very large total parameter count without paying the full compute cost on every forward pass.

### 1. What an MoE transformer layer looks like

A conventional transformer block is roughly:

```text
token representations
       │
       ▼
self-attention
       │
       ▼
feed-forward network (MLP)
       │
       ▼
next transformer block
```

An MoE block replaces the single MLP with:

```text
                    ┌── Expert 1 ──┐
                    ├── Expert 2 ──┤
token ──► Router ──►├── Expert 3 ──┤──► weighted combination
                    ├──    ...     ┤
                    └── Expert N ──┘
```

Usually the **attention layers remain dense**. It is primarily the large MLP/FFN component that becomes sparse.

---

## 2. How the experts are trained

The experts generally **aren't manually assigned roles** like:

```text
Expert 1 = Python
Expert 2 = French
Expert 3 = mathematics
```

Instead, the router and experts are trained jointly with normal gradient descent.

For token representation (x), the router computes something like:

[
s = W_r x
]

where (W_r) is a small learned routing matrix.

After a softmax:

[
p_i = \text{softmax}(s)_i
]

you get a score for every expert.

If this is a **top-2 MoE**, the two highest-scoring experts receive the token.

For example:

```text
"The"       -> experts 7, 21
"Python"    -> experts 4, 19
"function"  -> experts 4, 12
"Paris"     -> experts 9, 27
```

The selected experts process the token:

[
y = p_4 E_4(x) + p_{19} E_{19}(x)
]

where (E_i) is expert (i)'s MLP.

Backpropagation then updates:

- the selected experts
- the router
- the rest of the transformer

Over enormous amounts of training data, the experts tend to **differentiate naturally**.

Some may become better at particular:

- languages
- syntactic structures
- programming patterns
- semantic concepts
- reasoning operations
- token positions or representations

But the specialisation is often much messier than human-readable categories.

---

## 3. Why specialisation happens

Imagine two experts initially happen to be slightly better at processing mathematical representations.

The router starts sending them slightly more mathematical tokens.

Because they see more mathematical tokens, they become even better at them.

That reinforces the router's preference.

You can get a feedback loop:

```text
small accidental specialization
          ↓
router sends similar tokens there
          ↓
expert trains more on those representations
          ↓
expert becomes better at them
          ↓
router increasingly prefers it
```

This is one mechanism by which specialisation can emerge.

---

## 4. A major training problem: expert collapse

Without additional constraints, the router may discover that a few experts are initially better and start sending nearly everything to them:

```text
Expert 1: ████████████████████
Expert 2: ██████████████████
Expert 3: ▏
Expert 4: ▏
...
Expert 64: ▏
```

That is bad because:

- most parameters become unused
- popular experts become overloaded
- distributed training becomes inefficient
- unused experts don't learn properly

So MoE training normally includes **load-balancing losses** encouraging the router to distribute tokens reasonably evenly.

Conceptually:

[
L = L_{\text{language model}} +
\lambda L_{\text{load balance}}
]

Modern MoE designs have various more sophisticated ways of handling this.

---

## 5. What happens during inference

The routing mechanism learned during training is simply used during inference.

Suppose a layer has:

```text
64 experts
top-k = 2
```

For every token:

```text
token hidden state
        │
        ▼
      router
        │
     scores 64 experts
        │
        ▼
select top 2
        │
        ├── Expert 17
        └── Expert 42
        │
        ▼
combine outputs
```

Then the resulting representation continues through the transformer.

Importantly, this happens **independently at every MoE layer**.

A token might therefore follow:

```text
MoE layer 1: experts 3 + 17
MoE layer 2: experts 9 + 11
MoE layer 3: experts 2 + 55
MoE layer 4: experts 17 + 31
```

There isn't normally one "expert" chosen for an entire prompt.

Routing happens at the level of **individual token representations at individual layers**.

---

## 6. Different tokens in the same sentence can use different experts

For:

> "Write a Python function to calculate mortgage repayments."

you might conceptually see:

```text
Write       -> experts 12, 30
Python      -> experts  4, 18
function    -> experts  4, 21
calculate   -> experts  7, 32
mortgage    -> experts 15, 44
repayments  -> experts 15, 27
```

And later transformer layers might route them completely differently.

The router is operating on the **contextual hidden representation**, not just the literal token.

So `"bank"` in:

> river bank

can route differently from `"bank"` in:

> mortgage bank

even though the original token is identical.

---

## 7. Why MoE gives so much capacity cheaply

Suppose a dense model has a 10B-parameter FFN component.

An MoE equivalent might have:

```text
8 experts × 10B parameters = 80B parameters
```

but activate only 2:

```text
active compute ≈ 20B parameters
```

So the model might contain approximately:

```text
80B expert parameters
```

while requiring compute closer to:

```text
20B active expert parameters
```

This is why you'll see descriptions such as:

> **200B total parameters, 20B active parameters**

The total parameter count describes model capacity/storage, while the **active parameters** are much more indicative of inference compute.

---

## 8. MoE doesn't make attention sparse by itself

This distinction is useful.

An MoE transformer might conceptually be:

```text
Dense attention
      ↓
Sparse MoE FFN
      ↓
Dense attention
      ↓
Sparse MoE FFN
      ↓
...
```

For every token, attention may still operate over the sequence normally.

The sparsity is mainly:

> **Which FFN parameters process this token?**

rather than:

> **Which previous tokens does this token attend to?**

Those are separate optimisation techniques.

---

## 9. Distributed inference is where things get interesting

For large MoEs, experts can be placed on different GPUs:

```text
GPU 1: Experts 1–8
GPU 2: Experts 9–16
GPU 3: Experts 17–24
GPU 4: Experts 25–32
```

After routing, tokens have to be physically sent to the appropriate GPU:

```text
tokens
  ↓
router
  ↓
all-to-all network communication
  ↓
expert GPUs
  ↓
all-to-all network communication
  ↓
tokens continue
```

This is called **expert parallelism**.

It is one reason MoE inference is not simply "a huge model for the price of a small one." Compute is sparse, but you still have challenges involving:

- storing all expert weights
- GPU memory
- routing overhead
- inter-GPU communication
- load imbalance
- batching
- expert capacity

At scale, communication can become one of the dominant engineering concerns.

---

## 10. The simplest mental model

Think of a dense LLM as having:

```text
one enormous department
that every token passes through
```

An MoE has:

```text
many specialist departments
          +
a receptionist/router
that sends each token to the best 1–2 departments
```

But unlike a human organisation, nobody initially says what the departments specialise in.

**The router and experts jointly discover useful divisions of labour during training.**

And the crucial distinction is:

```text
Total parameters     = model capacity
Active parameters    = approximate compute per token
```

That gap is the main reason MoE architectures such as Switch Transformer, Mixtral, DeepSeek-MoE and related designs are attractive for scaling LLMs.
