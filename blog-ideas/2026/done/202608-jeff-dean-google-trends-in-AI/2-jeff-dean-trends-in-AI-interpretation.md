# Lecture interpretation: Trends in AI by Jeff Dean

https://www.youtube.com/watch?v=UTTeXZrpMR0

https://chatgpt.com/c/6a7a7c0a-244c-83ec-939d-cb4888f2475f

## How Modern AI Got Here

### Notes from Jeff Dean on scaling, TPUs, Transformers, sparse models, inference-time compute, Gemini, and where AI systems may be heading next

I watched a recent talk from Jeff Dean, Google’s Chief Scientist and one of the leads on Gemini. The talk is essentially a guided tour through the last 15 years of modern machine learning, from early large-scale neural networks through to Gemini and current research directions.

What I liked about it is that Dean doesn’t frame recent AI progress as the result of one breakthrough.

His argument is closer to this:

> Modern AI is the result of a whole stack improving together: compute, model architecture, training algorithms, distributed systems, hardware, data, and inference techniques.

That framing feels useful for ML engineers. It is easy to compress the history into “Transformers + scaling”, but that misses quite a lot of the engineering that made today’s models possible.

Below are my notes from the talk, mostly following the order Dean presents them.

---

## Scale mattered, but it wasn’t acting alone

Dean starts with a fairly simple observation: over the last decade or so, increasing the amount of compute, data, and model capacity has produced fairly consistent improvements.

But he is careful not to attribute everything to scaling.

Algorithmic and architectural improvements compound with hardware improvements. His example is that a 20× improvement from scaling combined with a 50× improvement in algorithmic efficiency can produce something closer to a 1,000× overall gain.

I think this is an important distinction.

A lot of discussion around frontier AI treats compute as the primary variable. In practice, the systems that make good use of that compute matter just as much.

You can roughly think about the progress as:

```text
better hardware
      ×
better distributed systems
      ×
better architectures
      ×
better training objectives
      ×
more data
      ×
better inference algorithms
      ↓
much better models
```

None of those factors are independent.

---

## Early neural networks: the ideas were already there

Dean first encountered neural networks as an undergraduate in 1990.

What struck him was that gradient descent and backpropagation created a very general trainable abstraction. The problem at the time was largely compute.

His undergraduate thesis explored what we would now call **data parallelism** and **model parallelism**, using multiple machines to train neural networks.

That thread reappeared years later at Google.

Dean and colleagues built distributed training infrastructure that allowed them to train models around **50–100× larger** than the largest models they were seeing elsewhere. The system used asynchronous model replicas sending gradients to distributed parameter servers.

Mathematically, asynchronous updates were imperfect because workers were often computing gradients against slightly stale parameters.

But it worked.

That is a recurring theme in large-scale ML systems: sometimes an imperfect distributed approximation is much more useful than a theoretically cleaner approach that cannot scale.

---

## Representation learning started revealing interesting behaviour

One of Google's early large-scale experiments trained an unsupervised vision model on **10 million random YouTube frames**.

Without explicit labels, some higher-level neurons became responsive to concepts such as:

- faces
- cats
- human shapes

The interesting part wasn’t really the famous “cat neuron”. It was that useful abstractions emerged from the training objective itself.

Those learned representations could then initialise a supervised vision model and significantly improve performance.

Around the same period, word embeddings showed something similar in language.

Train a simple model to predict nearby words and the geometry of the embedding space begins to encode semantic structure:

```text
cat  ≈ tiger ≈ puma

king - man + woman ≈ queen
```

Not because anyone explicitly programmed those relationships, but because they fall out of the statistics of language.

This was an early sign of a pattern we now take for granted: sufficiently large models trained on sufficiently rich data learn useful representations that support many downstream tasks.

---

## Seq2seq moved neural networks deeper into language

Dean then moves through the sequence-to-sequence work based on recurrent neural networks and LSTMs.

The basic idea was elegant:

```text
input sequence
      ↓
   encoder
      ↓
 latent state
      ↓
   decoder
      ↓
output sequence
```

Machine translation was the obvious application.

An English sentence could be encoded into a hidden representation, which the decoder then used to generate the equivalent French sentence.

This worked surprisingly well, but recurrent models had two important limitations that later became central to the Transformer story.

---

## TPUs came from an inference problem

One of the more interesting engineering stories in the talk is the origin of Google's Tensor Processing Unit, or TPU.

Google had developed a much better neural speech recognition system.

Dean did a back-of-the-envelope calculation for deploying it to roughly one billion users for three minutes per day.

The answer was uncomfortable: Google would need to roughly **double its existing computer fleet** just to run the new speech model.

That changed the economics of the problem.

Instead of asking:

> How do we run neural networks faster on general-purpose hardware?

the question became:

> What hardware would we build if neural networks were the workload?

Two properties made specialised accelerators attractive.

Neural networks tolerate relatively low numerical precision, particularly during inference.

And most of their compute is concentrated around a small collection of linear algebra operations, especially matrix multiplication.

So the chip does not need to be good at everything. It needs to be extremely good at a narrow set of operations.

Dean reports that TPUv1 was roughly:

- **15–30× faster** than contemporary CPUs and GPUs
- **30–80× more energy efficient**

Later generations evolved from individual accelerators into large interconnected ML supercomputers.

There is an important systems lesson here: once ML workloads become large enough, model architecture and hardware architecture start influencing each other.

---

## TensorFlow, PyTorch and JAX made the abstractions reusable

Dean also gives some credit to the software layer.

TensorFlow grew out of Google's internal distributed neural network infrastructure. PyTorch followed from Meta and others, while JAX emerged from Google with a more functional programming style.

Google now uses JAX heavily for Gemini training.

This part of the history is easy to overlook.

The hardware matters, but so do programming abstractions that let researchers express ML computations without having to manually manage thousands of accelerators.

---

## Transformers removed a major sequential bottleneck

Dean then gets to the 2017 Transformer work.

He frames the motivation largely in terms of limitations in recurrent networks such as LSTMs.

An LSTM updates a hidden state one token at a time:

```text
token 1 → state
token 2 → state
token 3 → state
...
token N → state
```

That creates two problems.

First, the computation is inherently sequential. Token 50 cannot be processed until token 49 has updated the state.

Second, information from the entire sequence is being compressed into a relatively small hidden state.

Transformers take a different approach.

Instead of repeatedly compressing history into one vector, they preserve representations of the sequence and use a learned **attention** mechanism to decide which parts matter.

That architecture is much friendlier to parallel hardware.

And in Dean's examples, Transformers achieved equivalent or better quality with substantially less compute than comparable LSTMs.

The Transformer breakthrough was therefore not just “attention works”.

It was also:

> attention maps much better onto the hardware we have.

---

## Self-supervised learning gave models an enormous training signal

The next piece is self-supervised learning.

Language provides its own labels.

Given:

```text
The cat sat on the ___
```

the training data already contains the target token.

No human labelling process is needed.

Dean distinguishes two broad objectives.

**Autoregressive modelling**

```text
previous tokens → predict next token
```

**Masked language modelling**

```text
tokens on both sides → predict missing token
```

Autoregressive prediction is particularly useful for generative models because inference naturally continues from the prefix already generated.

The big advantage is scale.

Once the learning signal comes directly from raw text, the amount of available training data increases dramatically.

---

## Sparse models and Mixture of Experts

Dean then spends some time on sparse models, which I found particularly relevant given how important Mixture of Experts (MoE) architectures have become.

A conventional dense model activates essentially all of its parameters for every token.

A sparse model does something closer to:

```text
                     ┌─ expert A
token → router ──────┼─ expert B
                     ├─ expert C
                     └─ expert D

             activate only a subset
```

The router itself is learned.

Different parts of the model can therefore specialise in different types of inputs, while only a small subset of the total parameters needs to execute for each token.

That gives you two useful properties at once:

- much larger total model capacity
- relatively low compute per token

Dean cites experiments where sparse models reduced training compute by around **8× for the same accuracy**, or alternatively produced better quality at the same compute budget.

My current read is that this is one of the more important long-term scaling ideas.

Dense models couple total parameter count closely to inference cost. Sparse models loosen that relationship.

---

## Pathways: making thousands of accelerators look like one machine

Once models span thousands of chips, distributed systems become part of the ML architecture.

Google's **Pathways** system is designed to hide much of this complexity from the researcher.

Conceptually:

```text
Python / JAX program
        ↓
     Pathways
        ↓
thousands of TPU chips
across pods / networks / sites
```

The researcher writes something that looks like a computation over one giant machine.

Pathways works out how to distribute it across TPU interconnects, datacentre networks and potentially wider network links.

This abstraction becomes even more important because large distributed training jobs do not fail cleanly.

---

## At large scale, hardware becomes probabilistic

One of the less glamorous but more interesting parts of the talk is hardware reliability.

When you run enough chips, some of them occasionally return the wrong answer.

Not necessarily a clean crash.

A bit may flip.

A gradient that should have been something like:

```text
0.2
```

can suddenly become:

```text
1e20
```

If that value is then synchronised across thousands of machines, the entire training run can be corrupted.

Google monitors signals such as gradient norms and automatically performs **deterministic replay** when something looks suspicious.

If the computation produces the same result when repeated, the spike was probably caused by the data.

If the result changes, there may be faulty hardware.

Pathways can then remove the affected hardware and replace it with a hot spare while training continues.

This is a useful reminder that frontier ML has become partly a distributed reliability engineering problem.

---

## Inference-time compute became another scaling dimension

Around 2022, another idea started becoming more important: give the model more compute **after training**.

Dean describes chain-of-thought-style reasoning through this lens.

If a model generates intermediate reasoning tokens before producing an answer, every extra token requires another forward pass through the model.

So longer reasoning effectively means more inference compute.

The interesting shift is that model capability now depends on at least two compute budgets:

```text
training-time compute
        +
inference-time compute
```

That becomes particularly important for tasks such as mathematics, coding and planning.

In the Q&A, Dean mentions generating multiple candidate solutions and having the model evaluate them before deciding which answer to return. That can reduce hallucination rates, but it also multiplies inference cost.

Which leads naturally to the next problem: making inference cheaper.

---

## Distillation: moving capability into smaller models

Dean describes distillation as a way of transferring capability from a larger teacher model into a smaller student model.

Instead of training only against a hard target:

```text
correct token = violin
```

the teacher might provide a probability distribution:

```text
violin    0.75
piano     0.15
trumpet   0.06
...
```

That is a much richer learning signal.

The student learns something about the structure of the teacher's uncertainty, not just whether it produced the single correct answer.

Dean says this is one of the techniques Google uses to transfer capability from **Gemini Pro-scale models into Flash-scale models**.

That matters because a capability only becomes useful at very large scale when its serving cost becomes manageable.

---

## Reinforcement learning increasingly shapes model behaviour

Pre-training gives a model broad capabilities.

Post-training helps decide which of those capabilities actually emerge in useful ways.

Dean talks about several sources of reward:

### Human feedback

People compare two answers and say which they prefer.

### Model feedback

Another model acts as the evaluator or reward model.

### Verifiable rewards

Some domains provide objective correctness signals.

For code:

```text
Does it compile?
Do the tests pass?
```

For mathematics:

```text
Can a theorem prover verify the proof?
```

These signals work particularly well because the reward is relatively difficult to game.

Dean highlights an open research question here: **how do we get reliable reinforcement learning signals in domains where the answer cannot easily be verified?**

I think that remains one of the more interesting questions in model post-training.

---

## Speculative decoding makes autoregressive inference less painful

Transformers fixed much of the sequential bottleneck during training.

Generation is still sequential.

You normally generate:

```text
token 1
  ↓
token 2
  ↓
token 3
  ↓
token 4
```

A large model therefore needs to repeatedly load and process its parameters for individual tokens.

**Speculative decoding** introduces a smaller draft model.

The draft model proposes several future tokens:

```text
small model:
A B C D E F G H
```

The larger model then checks them in parallel.

If the first five match what the large model would have generated, you accept all five at once.

Dean's point is that this can improve inference efficiency without retraining or changing the target model architecture, while preserving the target model's output distribution.

It is a good example of something that sometimes gets missed in model discussions: **decoding algorithms themselves are an important research surface**.

---

# Gemini is where all these pieces meet

The Gemini project started in February 2023.

Dean describes its origin as an attempt to consolidate several Google groups working separately on language and multimodal models into one larger effort.

The goal was explicitly multimodal from the beginning.

Gemini models can work across:

- text
- images
- audio
- video

and increasingly other forms of data such as robotic control signals and LiDAR.

What stood out to me is that Dean does not describe Gemini as one clever architectural trick.

It is closer to a synthesis of the entire talk:

```text
TPUs
+ distributed training
+ JAX
+ Pathways
+ Transformers
+ sparse models / MoE
+ distillation
+ long context
+ supervised fine-tuning
+ reinforcement learning
+ inference-time reasoning
+ speculative decoding
= Gemini
```

That is probably the central idea of the talk.

Modern frontier models are systems, not isolated neural network architectures.

---

## Why Google keeps pushing long context

Dean gives an interesting explanation for why context windows matter.

Knowledge stored in model parameters has effectively been compressed from trillions of training tokens.

It is useful, but fuzzy.

Information placed directly into the context window is much more precise.

If I give the model 900 pages of documentation, it can directly inspect those pages rather than relying on whether the relevant information happened to survive pre-training in the right form.

That makes long context particularly valuable for tasks such as:

- codebase reasoning
- document analysis
- personalised assistants
- multimodal search
- agent workflows

But Dean also argues that context windows alone will not be enough.

More on that shortly.

---

## Pro capability eventually becomes Flash capability

Google maintains different model scales.

A **Pro-scale** model prioritises maximum capability.

A **Flash-scale** model aims for lower latency and cost while retaining as much capability as possible.

Their target is roughly:

```text
Flash(N) ≳ Pro(N-1)
```

In other words, the cheaper model in the next generation should outperform the expensive model from the previous generation.

I think this is a useful way to think about model progress commercially.

Frontier models explore what is possible.

Distillation, architecture work and inference engineering then move those capabilities down the cost curve.

---

## Mathematics is a good illustration of inference-time scaling

Dean uses the International Mathematical Olympiad (IMO) as an example of how quickly model reasoning has improved.

Earlier systems used specialised mathematical tooling, theorem provers and separate geometry models.

More recently, Google used a general-purpose Gemini Pro-scale model with a large inference-time reasoning budget and achieved a gold-medal-level result, solving five of six problems.

The important point isn't really the benchmark itself.

It is the architectural shift:

```text
specialised pipeline
        ↓
general model + more inference compute
```

That pattern may repeat in other domains.

---

# From answering questions to generating software

Dean also shows examples of models creating interactive interfaces.

One example takes technical material and produces an interactive visualisation with controls.

Another takes handwritten recipes, transcribes and translates them, then generates a bilingual website.

His prediction is fairly straightforward: if people can describe software rather than manually implement all of it, we will probably end up with **much more software in the world**.

That seems plausible to me.

The interesting question is what happens to the boundary between:

```text
software product
```

and:

```text
temporary generated interface
```

If a model can generate a useful UI for a task in seconds, not every interface necessarily needs to exist as a permanently maintained application.

---

## Models can also reason in visual space

One example I found interesting is what Dean describes as reasoning in “pixel space”.

Instead of only generating textual intermediate reasoning, a model can generate a sequence of images showing how a physical scenario evolves.

For example:

```text
initial scene
    ↓
ball rolls down ramp
    ↓
ball hits another surface
    ↓
trajectory changes
    ↓
ball lands in bucket B
```

That is conceptually different from treating vision as something that only happens at the input layer.

The model can potentially use image generation as part of its own problem-solving process.

---

# How Google organises something the size of Gemini

The talk briefly moves from technical architecture into organisational architecture.

Gemini is now a project involving more than a thousand contributors.

Dean describes teams spanning:

- pre-training
- post-training
- reinforcement learning
- safety
- vision
- audio
- code
- agents
- internationalisation
- training data
- evaluation
- infrastructure
- serving
- longer-term research

They also maintain thousands of internal request-for-comment documents and use common baselines and leaderboards to evaluate experiments.

The workflow sounds roughly like:

```text
many experiments at tiny scale
        ↓
promising experiments
        ↓
medium-scale validation
        ↓
large-scale validation
        ↓
candidate baseline
        ↓
repeat
```

That leads into what I thought was one of the most useful parts of the talk.

---

# Where Dean thinks things may be heading

## Humans supervising teams of AI agents

Today, most AI interaction still looks like:

```text
one human ↔ one model
```

Dean expects more future work to look like:

```text
              agent
                ↑
agent ← human → agent
                ↓
              agent
```

Potentially dozens or hundreds of agents working on someone's behalf.

The human may provide relatively high-level goals while the agents coordinate the actual work.

That creates a new class of problems.

How do you inspect what 50 agents are doing?

How do you delegate?

How do agents coordinate?

How do you detect that one has misunderstood the task?

How much autonomy should each one have?

This feels less like a pure model problem and more like a combination of distributed systems, human-computer interaction and organisational design.

---

## A million tokens probably isn't enough

Dean thinks million-token contexts are useful.

But he asks a more interesting question:

> What if the information you want the model to reason over is closer to a trillion tokens?

You probably do not want to place all of that directly in the Transformer context.

Instead, the system might look something like:

```text
trillion-token corpus
        ↓
learned retrieval
        ↓
30,000 candidate documents
        ↓
lightweight relevance model
        ↓
100 highly relevant documents
        ↓
LLM context window
```

This sounds less like “RAG is going away because context windows are huge” and more like the opposite.

My current read is that **retrieval and context management become more important as agents get access to larger information spaces**.

The retrieval system itself may simply become more learned, multimodal and tightly integrated with the model.

---

## Inference efficiency may become more important than training efficiency

Training frontier models is expensive.

But training happens occasionally.

Inference happens every time someone uses the model.

If we move towards systems where one agent calls ten other agents, each of which performs multiple reasoning rollouts, inference volumes can increase very quickly.

Dean therefore expects increasing attention on:

- specialised inference hardware
- inference-specific model architectures
- better decoding algorithms
- lower latency
- AI-assisted chip design

He gives a simple product comparison:

```text
100 ms response
vs
5 second response
```

Even when the answers are identical, they are very different user experiences.

Latency is not just an infrastructure metric. It changes what kinds of products are possible.

---

# You don't necessarily need 10,000 GPUs to do useful ML research

The final Q&A contains a point I suspect is worth repeating.

Someone asks what researchers without frontier-scale compute should work on.

Dean's answer is essentially: **experiment at small scale and pay attention to the scaling trend**.

Suppose we have two ideas:

```text
quality
  ^
  |              B
  |            /
  |          /
  |   A ----
  |  /
  +-----------------> scale
```

At very small scale, A might outperform B.

But if B improves much faster as scale increases, B may be the more interesting research direction.

Dean argues that researchers should not necessarily optimise for tiny improvements on current state-of-the-art benchmarks.

A genuinely different idea with promising scaling behaviour may be more valuable even if it loses at small scale.

I like this framing.

It also suggests a slightly different research question:

> Not “does this beat the baseline?”, but “how does this behave as we scale it?”

That is a much more accessible question for teams without frontier-scale infrastructure.

---

# What about hallucinations and AI risk?

In the Q&A, Dean gives a relatively pragmatic view of AI safety.

He suggests some catastrophic-risk concerns may be overstated and believes careful engineering around what AI systems are allowed to do can make them deployable safely.

His nearer-term concerns are more concrete:

- realistic generated misinformation
- fake audio and video
- workforce transitions
- helping people adapt to tasks that are increasingly automated

He also points to broader impacts across employment, education, healthcare, science, media, governance and national security.

Whether you agree with his weighting of the risks or not, I think the distinction is useful.

There are questions about hypothetical future systems.

And there are already concrete deployment problems we know we need to solve.

Both deserve attention.

---

# My main takeaways

If I compress the talk down to the points I found most useful, I get something like this.

### 1. Modern AI is a systems problem

The neural network architecture is only one layer.

Progress has come from co-design across:

```text
models
data
training
distributed systems
hardware
inference
```

### 2. Sparse models are one of the important ways to keep scaling

Mixture of Experts lets model capacity grow faster than per-token inference cost.

That looks increasingly important as model sizes increase.

### 3. Inference-time compute is becoming a first-class scaling dimension

Training a better model is no longer the only option.

You can also give an existing model more compute while solving difficult problems.

### 4. Cheap models inherit yesterday's frontier capabilities

Distillation and inference optimisation matter because eventually the expensive capability needs to become economical.

The Pro → Flash progression is a useful mental model for this.

### 5. Long context does not remove the need for retrieval

If anything, agent systems may need retrieval over increasingly enormous information spaces.

The interesting work may move towards learned retrieval, ranking and context construction.

### 6. Inference efficiency is going to matter a lot

Agentic systems can multiply inference calls very quickly.

Latency, throughput and energy efficiency may become as important as raw benchmark quality.

### 7. Small-scale research still matters

You do not necessarily need frontier compute to discover interesting ideas.

Testing scaling behaviour on smaller systems can still tell you a lot.

---

# Open questions

A few questions the talk left me thinking about.

### How far can sparse architectures go?

If routing gets better, can total model capacity become dramatically larger without proportional inference cost?

And what new failure modes appear as expert specialisation increases?

### How should agent systems share context?

If dozens of agents are working together, should they share one memory system?

Independent memories?

A hierarchical knowledge store?

### What replaces simple RAG?

Retrieval-Augmented Generation is already evolving beyond embedding search plus a vector database.

I suspect the more interesting future systems will combine learned retrieval, ranking, summarisation and model-generated context.

### How much inference compute is enough?

Generating 16 candidate answers and evaluating them may improve reliability.

Generating 160 might improve it further.

At some point the quality curve will flatten while the infrastructure bill keeps increasing.

Understanding that trade-off seems important.

### How do we get good rewards outside verifiable domains?

Maths and code have relatively clean evaluation signals.

Strategy, research, writing and many real-world decisions do not.

Reliable reinforcement learning in those domains still seems much harder.

---

# Closing thought

The part of Dean's talk I found most useful was not any individual model or benchmark.

It was the historical perspective.

The current generation of AI systems did not appear because someone discovered one magic architecture.

They emerged because many layers of the stack improved together:

```text
neural networks
    ↓
distributed training
    ↓
representation learning
    ↓
specialised accelerators
    ↓
Transformers
    ↓
self-supervised learning
    ↓
sparse models
    ↓
large-scale distributed systems
    ↓
distillation + RL
    ↓
inference-time reasoning
    ↓
better decoding
    ↓
multimodal systems like Gemini
```

The next phase looks similarly multi-layered.

My current read is that some of the most important questions will sit around the boundaries between traditional ML research and systems engineering: inference efficiency, retrieval, agent coordination, sparse computation, long-lived memory and hardware/software co-design.

That is probably good news for ML engineers.

There is still a lot more to this field than making the model bigger.
