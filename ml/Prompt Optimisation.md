# Prompt Optimisation & Supporting Tools

WIP

## Automated Prompt Optimisation

Automated prompt optimisation uses algorithms (evolutionary, feedback-driven, gradient-based, or evaluation-loops) to systematically refine prompts for LLMs, reducing manual engineering and improving task accuracy/performance.

https://www.awesomepython.org/?q=auto-prompting

## agent-lightning (Microsoft)

**Repo:** [github.com/microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)

**Description:** Framework to “train” agents by optimizing prompts (APO) and other components with minimal/near-zero code changes, and works across many agent stacks. ([GitHub][1])

**FAQ**

- **What’s the core idea?** Iterate: run your agent → score outputs → generate improved prompts/components → repeat. ([microsoft.github.io][2])
- **Does it fine-tune model weights?** It supports multiple training-style approaches, including prompt optimization and (optionally) other methods like RL/SFT depending on setup. ([GitHub][1])
- **Do I need a labeled dataset?** Not strictly—any repeatable evaluation/scoring function can drive optimization. ([microsoft.github.io][2])

---

## dspy (StanfordNLP)

**Repo:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)

**Description:** A declarative framework for building modular LM “programs” that can be compiled/optimized into better prompts (and sometimes learned parameters) for tasks like classification, RAG, and agents. ([GitHub][3])

**FAQ**

- **How is it different from “prompting”?** You write structured components/signatures; DSPy handles prompt construction and optimization loops. ([GitHub][3])
- **What do I optimize against?** Whatever metric you can compute from outputs (accuracy/F1, retrieval metrics, task-specific checks, etc.). ([GitHub][3])
- **API LLMs or self-hosted?** Both—DSPy is designed to sit above whatever LM backend you configure. ([DSPy][4])

---

## gepa (gepa-ai)

**Repo:** [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)

**Description:** “Genetic-Pareto” optimization for systems made of text components (prompts/specs/code snippets), using reflective edits guided by execution + evaluation traces against any metric. ([GitHub][5])

**FAQ**

- **What does it optimize?** Any text component in a pipeline—single prompts or multi-part textual systems. ([GitHub][5])
- **Does it need gradients?** No—improvements come from iterative generation/reflection and metric-based selection. ([GitHub][5])
- **What metrics can it use?** Anything you can score automatically (task success, F1, rubric scores, custom validators). ([GitHub][5])

**Resources**

- https://dspy.ai/tutorials/entity_extraction/
- https://dspy.ai/tutorials/gepa_ai_program/
- https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/ - structured information extraction and classification
- https://github.com/gepa-ai/gepa/tree/main/src/gepa/adapters/dspy_full_program_adapter - GEPA evolve entire DSPy programs—including signatures, modules, and control flow

**Videos**

- https://www.youtube.com/watch?v=rrtxyZ4Vnv8 - Matei Zaharia - Reflective Optimization of Agents with GEPA and DSPy

---

## promptfoo

**Repo:** [github.com/promptfoo/promptfoo](https://github.com/promptfoo/promptfoo)

**Description:** Prompt/model evaluation toolkit (plus red teaming) to test prompts, compare providers/models side-by-side, and automate checks in CI/CD. ([GitHub][6])

**FAQ**

- **Is this an optimizer?** It’s primarily an eval + testing harness (with security/red-teaming), often used _around_ optimizers. ([GitHub][6])
- **What’s the main workflow?** Define test cases + assertions, run evals, compare results, and gate changes in CI. ([GitHub][6])
- **Does it help with LLM security?** Yes—includes red teaming / vulnerability scanning guidance and tooling. ([Promptfoo][7])

---

## AdalFlow (SylphAI)

**Repo:** [github.com/SylphAI-Inc/AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)

**Description:** A PyTorch-like SDK for building LLM workflows with “auto-differentiative” prompt optimization (zero-shot + few-shot) so you can train prompts against a dataset/metric. ([GitHub][8])

**FAQ**

- **What gets optimized?** Instructions and/or few-shot examples depending on how you mark tunable prompt parameters. ([GitHub][9])
- **Do I need labels?** Typically you need a reward/score; that can be labels, validators, or an LLM-judge rubric. ([GitHub][9])
- **When is it a good fit?** When you want a training-loop vibe for prompts (datasets, objectives, repeatable optimization). ([GitHub][8])

---

## promptomatix (SalesforceAIResearch)

**Repo:** [github.com/SalesforceAIResearch/promptomatix](https://github.com/SalesforceAIResearch/promptomatix)

**Description:** Framework to automate prompt creation/optimization from natural-language task descriptions; supports multiple optimization paths (including a DSPy-powered compiler per the paper). ([GitHub][10])

**FAQ**

- **What do I provide as input?** A task description (and optionally data/examples); the system generates and refines prompts. ([arXiv][11])
- **Is it tied to DSPy?** It can integrate with DSPy as one route, but also includes lighter-weight optimizer paths. ([arXiv][11])
- **Best use case?** Rapidly getting strong baseline prompts without lots of manual prompt-engineering iterations. ([GitHub][10])

---

## PromptWizard (Microsoft)

**Repo:** [github.com/microsoft/PromptWizard](https://github.com/microsoft/PromptWizard)

**Description:** Discrete prompt optimization that “self-evolves” by generating, critiquing, and refining both instructions and in-context examples via iterative feedback. ([GitHub][12])

**FAQ**

- **What’s being optimized?** Both the instruction prompt and the few-shot examples/ICL set. ([Microsoft][13])
- **What drives improvement?** Feedback loops (often LLM-based critique + task scoring) over iterations. ([GitHub][12])
- **When should I use it?** When “prompt + examples” matter a lot and you want an automated refinement loop. ([GitHub][12])

---

## AutoPrompt (Eladlev)

**Repo:** [github.com/Eladlev/AutoPrompt](https://github.com/Eladlev/AutoPrompt)

**Description:** Prompt optimization pipeline aimed at real-world use: auto-generates prompts tailored to intent and iteratively “calibrates” them using challenging edge cases. ([GitHub][14])

**FAQ**

- **How does it improve prompts?** Generate → evaluate → add hard/edge cases → refine prompt iteratively. ([LinkedIn][15])
- **Do I need a test set?** Strongly recommended—calibration is only as good as the cases you evaluate against. ([LinkedIn][15])
- **Good fit for what tasks?** Moderation/classification-ish pipelines and other repeatable “prompt as policy” tasks. ([GitHub][16])

---

## prompt-ops (Meta Llama)

**Repo:** [github.com/meta-llama/prompt-ops](https://github.com/meta-llama/prompt-ops)

**Description:** Open-source prompt optimization centered on PDO (Prompt Duel Optimizer): a label-free method using dueling bandits + Thompson sampling to pick better prompts via pairwise comparisons. ([GitHub][17])

**FAQ**

- **What does “label-free” mean here?** Optimization can be driven by preference/duel outcomes (often via an LLM judge) rather than ground-truth labels. ([arXiv][18])
- **Why duels instead of scoring everything?** Pairwise comparisons can be more sample-efficient and robust when absolute scoring is noisy. ([arXiv][18])
- **When is it useful?** When you can compare outputs reliably (A vs B) even if absolute metrics/labels are hard. ([arXiv][18])

---

## sammo (Microsoft)

**Repo:** [github.com/microsoft/sammo](https://github.com/microsoft/sammo)

**Description:** Structure-aware multi-objective metaprompt optimization: treats prompts as structured “programs” and searches over transformations (e.g., add/remove/replace components) to improve outcomes. ([GitHub][19])

**FAQ**

- **What’s the key idea?** Optimize prompt _structure_ (not just wording) with transformations over prompt “objects.” ([arXiv][20])
- **What objectives can it handle?** Multiple objectives (quality, cost, latency proxies, etc.) depending on how you define scoring. ([GitHub][19])
- **Where does it shine?** Complex prompts like RAG pipelines where modular structure matters. ([Microsoft][21])

---

## automatic_prompt_engineer (keirp)

**Repo:** [github.com/keirp/automatic_prompt_engineer](https://github.com/keirp/automatic_prompt_engineer)

**Description:** Research code for APE (“Large Language Models Are Human-Level Prompt Engineers”): generates many instruction candidates with an LLM and selects the best via a score function. ([GitHub][22])

**FAQ**

- **Is this production-ready tooling?** It’s primarily research code; you’ll likely adapt ideas rather than drop it into prod. ([GitHub][22])
- **What do I need to run it?** A way to propose candidate instructions (LLM) and a repeatable scoring/eval function. ([arXiv][23])
- **What’s the main output?** A stronger _instruction prompt_ (often for zero-shot or few-shot settings). ([arXiv][23])

---

## prompt-optimizer (vaibkumr)

**Repo:** [github.com/vaibkumr/prompt-optimizer](https://github.com/vaibkumr/prompt-optimizer)

**Description:** Prompt compression tooling: minimizes token complexity to reduce API cost/compute while tracking token reduction and semantic similarity; supports “protected tags” to preserve key parts. ([GitHub][24])

**FAQ**

- **Is it about quality or cost?** Mostly cost/latency via shorter prompts, while trying to preserve meaning. ([GitHub][25])
- **How do protected tags work?** You mark sections that must not be altered/removed during optimization. ([GitHub][24])
- **When should I avoid it?** If small wording changes can break strict formats/contracts and you can’t robustly validate outputs. ([GitHub][24])

[1]: https://github.com/microsoft/agent-lightning?utm_source=chatgpt.com "microsoft/agent-lightning: The absolute trainer to light up AI ..."
[2]: https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/?utm_source=chatgpt.com "APO - Agent-lightning"
[3]: https://github.com/stanfordnlp/dspy?utm_source=chatgpt.com "DSPy: The framework for programming—not prompting— ..."
[4]: https://dspy.ai/?utm_source=chatgpt.com "DSPy"
[5]: https://github.com/gepa-ai/gepa?utm_source=chatgpt.com "GEPA: System Optimization through Reflective Text Evolution"
[6]: https://github.com/promptfoo/promptfoo?utm_source=chatgpt.com "promptfoo/promptfoo"
[7]: https://www.promptfoo.dev/docs/red-team/?utm_source=chatgpt.com "LLM red teaming guide (open source)"
[8]: https://github.com/SylphAI-Inc/AdalFlow?utm_source=chatgpt.com "AdalFlow: The library to build & auto-optimize LLM ..."
[9]: https://github.com/SylphAI-Inc/AdalFlow/blob/main/docs/source/tutorials/index.rst?utm_source=chatgpt.com "AdalFlow/docs/source/tutorials/index.rst at main"
[10]: https://github.com/SalesforceAIResearch/promptomatix?utm_source=chatgpt.com "SalesforceAIResearch/promptomatix: An Automatic Prompt ..."
[11]: https://arxiv.org/pdf/2507.14241?utm_source=chatgpt.com "An Automatic Prompt Optimization Framework for Large ..."
[12]: https://github.com/microsoft/PromptWizard?utm_source=chatgpt.com "microsoft/PromptWizard: Task-Aware Agent-driven Prompt ..."
[13]: https://www.microsoft.com/en-us/research/blog/promptwizard-the-future-of-prompt-optimization-through-feedback-driven-self-evolving-prompts/?utm_source=chatgpt.com "PromptWizard: The future of prompt optimization through ..."
[14]: https://github.com/Eladlev/AutoPrompt?utm_source=chatgpt.com "Eladlev/AutoPrompt: A framework for prompt tuning using ..."
[15]: https://www.linkedin.com/posts/jainmanishk_github-eladlevautoprompt-a-framework-activity-7162594634395844608-H31g?utm_source=chatgpt.com "GitHub - Eladlev/AutoPrompt: A framework for prompt ..."
[16]: https://github.com/Eladlev/AutoPrompt/blob/main/docs/examples.md?utm_source=chatgpt.com "AutoPrompt/docs/examples.md at main"
[17]: https://github.com/meta-llama/prompt-ops?utm_source=chatgpt.com "meta-llama/prompt-ops: An open-source tool for LLM ..."
[18]: https://arxiv.org/html/2510.13907v1?utm_source=chatgpt.com "LLM Prompt Duel Optimizer: Efficient Label-Free ..."
[19]: https://github.com/microsoft/sammo?utm_source=chatgpt.com "microsoft/sammo: A library for prompt engineering and ..."
[20]: https://arxiv.org/html/2404.02319v1?utm_source=chatgpt.com "Prompts As Programs: A Structure-Aware Approach to ..."
[21]: https://www.microsoft.com/en-us/research/blog/sammo-a-general-purpose-framework-for-prompt-optimization/?utm_source=chatgpt.com "SAMMO: A general-purpose framework for prompt ..."
[22]: https://github.com/keirp/automatic_prompt_engineer?utm_source=chatgpt.com "keirp/automatic_prompt_engineer"
[23]: https://arxiv.org/abs/2211.01910?utm_source=chatgpt.com "Large Language Models Are Human-Level Prompt Engineers"
[24]: https://github.com/vaibkumr/prompt-optimizer?utm_source=chatgpt.com "vaibkumr/prompt-optimizer: Minimize LLM token complexity ..."
[25]: https://github.com/vaibkumr/prompt-optimizer/blob/master/docs/index.rst?utm_source=chatgpt.com "prompt-optimizer/docs/index.rst at master"
