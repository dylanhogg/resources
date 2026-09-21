# gpt5.6 Sol High

> What about for the tasks of: data analysis, interpreting business requirements, and technical blog writing? How do the 2 set ups compare?

The balance shifts further toward **Opus 5 + Claude Code** for these tasks. Of the three, I would currently choose Claude for **business-requirements interpretation and technical writing**, while **data analysis is closer** because Codex's execution efficiency is valuable.

A useful summary:

| Task                                         | Opus 5 + Claude Code | GPT-5.6 Sol + Codex | Preference        |
| -------------------------------------------- | -------------------: | ------------------: | ----------------- |
| Exploratory data analysis                    |                ★★★★★ |               ★★★★½ | **Opus, slight**  |
| Repetitive / well-defined analysis           |                ★★★★½ |               ★★★★★ | **Codex, slight** |
| Interpreting ambiguous business requirements |                ★★★★★ |                ★★★★ | **Opus**          |
| Turning requirements into technical design   |                ★★★★★ |               ★★★★½ | **Opus**          |
| Technical blog research/synthesis            |                ★★★★★ |               ★★★★½ | **Opus**          |
| Technical prose / voice                      |                ★★★★★ |                ★★★★ | **Opus**          |
| Concise technical documentation              |                ★★★★½ |               ★★★★★ | **Codex, slight** |
| Writing + validating lots of example code    |                ★★★★½ |               ★★★★★ | **Codex, slight** |

There is now reasonably good experimental evidence for two of those conclusions.

## 1. Data analysis: much closer than coding

This is an interesting case because it combines:

1. deciding **what analysis actually answers the question**;
2. manipulating data;
3. writing/running Python/SQL;
4. checking results;
5. interpreting the result.

### Opus advantage: analytical judgement

Artificial Analysis recently introduced **AA-AnalystAgent**, specifically intended to simulate day-to-day Business/Data Analyst work. It uses real spreadsheets and documents across 14 domains and tests tasks such as ratios, trends, P&L modelling, valuation and source reconciliation. Importantly, its authors found that the dominant failure mode was **committing too early to the wrong interpretation**. ([Artificial Analysis][1])

That's almost exactly the kind of failure where I'd expect Opus to have an advantage.

Opus 5 max currently achieves **53.8% pass^5**: it gets 53.8% of these tasks correct on _all five independent attempts_. That makes it one of the strongest models tested, and the benchmark authors specifically note its unusually high consistency. ([Artificial Analysis][1])

Anthropic also reports users seeing Opus 5 improve significantly on data-analysis, financial-modelling and statistical workflows, including checking confounders and validating results through independent methods. Those are vendor-provided testimonials rather than independent benchmarks, but they are directionally consistent with AA-AnalystAgent. ([Anthropic][2])

### Codex advantage: analysis execution

Suppose the task is:

```text
Here are 14 CSVs.

Join them.
Clean these columns.
Compute weekly cohorts.
Run these statistical tests.
Plot the results.
Export tables.
```

This plays directly into Sol + Codex's strongest behaviour:

```text
inspect
  ↓
write Python
  ↓
execute
  ↓
inspect results
  ↓
fix
  ↓
repeat
```

The same efficiency advantage from coding applies here.

If the analytical methodology is already known, I would probably prefer **Codex** because there's less value in Opus spending additional reasoning tokens reconsidering what analysis should be performed.

### So I'd split data analysis into two cases

**Ambiguous analytical question**

> “Engagement declined last quarter. Analyse the data and work out what's happening.”

**→ Opus 5 + Claude Code**

because hypothesis formation, confounders, interpretation and deciding which cuts matter dominate.

**Specified analytical procedure**

> “Calculate 30-day retention by acquisition channel, produce confidence intervals and generate these five charts.”

**→ Sol + Codex**

because execution dominates.

For ML/data-science work there is another useful distinction:

```text
Research-style DS
"What is this dataset telling us?"
             │
             ▼
           OPUS

Production-style DS
"Implement and validate this analysis."
             │
             ▼
           CODEX
```

I would put **EDA closer to Opus** and **data engineering / analysis implementation closer to Codex**.

---

# 2. Interpreting business requirements: Opus more clearly

This is probably the strongest case for Claude of the three.

Consider a realistic requirement:

> “Users should be able to search for properties near good schools, but we shouldn't unnecessarily exclude listings if we're unsure about the school catchment.”

There are a large number of unstated questions:

- What does “near” mean?
- What constitutes “good”?
- Hard filter or ranking signal?
- Where does school data come from?
- What does “unsure” mean?
- How should confidence affect retrieval?
- What should happen with zero results?
- What does the product actually promise to the user?
- Which requirements are product decisions rather than engineering decisions?

The important ability isn't implementation. It's **recovering the latent specification**.

That lines up very strongly with Opus's current strengths.

### AA-Briefcase is particularly relevant

Artificial Analysis' **AA-Briefcase** benchmark tests long-running knowledge-work tasks requiring models to produce things like analysis, spreadsheets, presentations and memos from business material.

Current scores:

- **Opus 5 max: 1721**
- **Opus 5 xhigh: 1693**
- **Sol max: 1504**

Opus currently occupies the top three positions at high/xhigh/max effort. ([Artificial Analysis][3])

This is a considerably bigger difference than many coding benchmarks.

GDPval-AA, another benchmark built around real professional tasks across 44 occupations, also puts the strongest Opus configurations roughly at or above Sol, although the exact ordering is much tighter and varies by effort level. ([Artificial Analysis][4])

### Behaviourally, I see the difference as

**Opus tends toward:**

```text
What is the stakeholder actually trying to accomplish?

What assumptions are hidden here?

Is requirement B inconsistent with A?

There appear to be three plausible interpretations.

This part needs a product decision.

Given everything else in the system,
I think interpretation #2 is most likely.
```

**Sol tends more toward:**

```text
Here is a reasonable interpretation.

Here are the resulting requirements.

Here is an implementation plan.
```

The second is faster.

The first is usually what I want during **requirements discovery**.

### Particularly for requirements → architecture

I'd currently favour this workflow:

```text
business docs
stakeholder notes
existing architecture
        │
        ▼
 Opus / Claude Code
        │
        ├── inferred requirements
        ├── ambiguities
        ├── assumptions
        ├── edge cases
        └── architecture options
                 │
                 ▼
             Sol / Codex
                 │
                 └── implementation
```

This is one of the places where using both models actually makes sense rather than being redundant.

---

# 3. Technical blog writing: Opus is currently the clearer winner

Here we have unusually relevant experimental evidence.

ToneBench recently tested both **through the exact harnesses we're discussing**.

Claude Opus 5 max was run through **Claude Code**.

GPT-5.6 Sol was run through **Codex CLI**.

So unlike many model benchmarks, it is fairly close to your actual comparison.

### Current ToneBench results

**Opus 5 max**

- #1 / 136 models
- Writing score: **90.4**
- Elo: **2360**
- particularly strong in:
  - tone & voice;
  - writing craft;
  - flow;
  - hooks. ([Towards AI][5])

**Sol high**

- #15 / 136
- Writing score: **88.5**
- Elo: **2171**. ([Towards AI][6])

Default configurations show essentially the same relationship:

- Opus default: **89.3**
- Sol default: **87.7**. ([Towards AI][7])

The absolute difference isn't huge, but the **qualitative dimensions are revealing**.

For Sol default:

- tone/voice: 87.3
- craft: 87.7
- substance: **89.5**
- flow: 85.2
- hook: 87.1

For Opus default:

- tone/voice: **90.0**
- craft: **90.3**
- substance: **89.5**
- flow: **89.1**
- hook: **91.0**. ([Towards AI][7])

That is almost exactly how I'd characterise them.

## Sol is not lacking substance

Notice:

> Substance: **89.5 vs 89.5**

The difference is primarily **writing quality**, rather than the amount of useful technical information.

Sol is good at producing:

```text
Here's the problem.

There are four approaches.

1. ...
2. ...
3. ...

The trade-offs are ...

I recommend #2 because ...
```

It's excellent technical communication.

But it can feel a bit more like **excellent generated technical documentation**.

Opus is currently better at producing prose that feels authored:

```text
We initially put the sufficiency check after reranking.

That seems reasonable at first: why decide whether there are
enough results until you've produced the final ranking?

The problem is that by then we've already paid for the expensive
part of the pipeline...
```

That distinction matters for a blog.

---

# Where Claude Code specifically helps technical blogging

Claude Code is particularly useful if your blog content lives in a Git repository.

You can give it:

```text
/blog-ideas
/posts
/experiments
/diagrams
/references
/benchmarks
```

and ask it to:

- read your previous posts;
- infer your established voice;
- inspect code examples;
- execute code to verify examples;
- read papers/references;
- maintain Markdown;
- update Mermaid diagrams;
- cross-reference earlier writing;
- apply a `SKILL.md` writing methodology.

That turns blogging into an agentic task rather than merely:

> “Generate me an article.”

Codex can do all of these things too. The difference is that Claude's strengths in **large-context synthesis + prose composition** align unusually well with the workflow.

---

# One area where I'd prefer Codex for writing

**Technical reference material.**

For example:

> Write concise documentation explaining these 17 search pipeline components, with a definition, when to use each, and one example.

Here I may actually choose Sol.

It tends to be better at:

- staying compact;
- maintaining repetitive structures;
- avoiding unnecessary exposition;
- obeying explicit formatting;
- producing information-dense reference material.

OpenAI's August ChatGPT update explicitly targeted more focused responses and reduced unnecessary formatting/verbosity, although importantly **Codex is still using the July Sol build**, not this August ChatGPT variant. ([OpenAI][8])

So:

**blog/article → Claude**

**docs/reference/API explanation → Codex is arguably preferable**

---

# My ranking for the four tasks we've discussed

Adding software development from the previous answer:

| Task                                 | Winner                 | Strength of preference |
| ------------------------------------ | ---------------------- | ---------------------- |
| Understand unfamiliar codebase       | **Opus / Claude Code** | Moderate               |
| Difficult ambiguous debugging        | **Opus / Claude Code** | Moderate               |
| Well-defined implementation          | **Sol / Codex**        | Moderate               |
| High-volume coding                   | **Sol / Codex**        | Strong                 |
| Open-ended data analysis             | **Opus / Claude Code** | Moderate               |
| Well-specified quantitative analysis | **Sol / Codex**        | Slight                 |
| Interpret business requirements      | **Opus / Claude Code** | **Strong**             |
| Requirements → architecture          | **Opus / Claude Code** | Moderate–strong        |
| Technical blog research              | **Opus / Claude Code** | Moderate               |
| Technical blog writing               | **Opus / Claude Code** | **Strong**             |
| Technical documentation              | **Sol / Codex**        | Slight                 |
| Validate lots of code examples       | **Sol / Codex**        | Moderate               |

If I had access to both and was doing typical **Staff-level ML/software engineering work**, I wouldn't choose one globally. I'd use **Claude as the “think, interpret, critique and communicate” agent and Codex as the “build, execute and verify” agent**.

That division seems better supported by the current evidence than treating one of them as simply the superior frontier model. ([Artificial Analysis][3])

[1]: https://artificialanalysis.ai/evaluations/aa-analyst-agent?utm_source=chatgpt.com "AA-AnalystAgent Benchmark Leaderboard | Artificial Analysis"
[2]: https://www.anthropic.com/news/claude-opus-5?src_trk=em67d16f9bd6e9e5.43146414656243079&utm_source=chatgpt.com "Introducing Claude Opus 5 \\ Anthropic"
[3]: https://artificialanalysis.ai/evaluations/aa-briefcase?models=muse-spark&utm_source=chatgpt.com "AA-Briefcase: Agentic Knowledge Work Benchmark | Artificial Analysis"
[4]: https://artificialanalysis.ai/evaluations/gdpval-aa?utm_source=chatgpt.com "GDPval-AA v2 Leaderboard | Artificial Analysis"
[5]: https://benchmark.towardsai.com/models/claude-opus-5-max.html?utm_source=chatgpt.com "Claude Opus 5 (max effort) Writing Benchmark Score | ToneBench"
[6]: https://benchmark.towardsai.com/models/gpt-5-6-high.html?utm_source=chatgpt.com "GPT-5.6 Sol (high) Writing Benchmark Score | ToneBench"
[7]: https://benchmark.towardsai.com/models/claude-opus-5.html?utm_source=chatgpt.com "Claude Opus 5 (adaptive default) Writing Benchmark Score | ToneBench"
[8]: https://openai.com/index/improving-gpt-5-6-sol-in-chatgpt/?utm_source=chatgpt.com "Improving GPT‑5.6 Sol in ChatGPT—and expanding access to GPT-5.6 Luna for free users | OpenAI"
