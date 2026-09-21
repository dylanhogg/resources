# Opus5 High

> What about for the tasks of: data analysis, interpreting business requirements, and technical blog writing? How do the 2 set ups compare?

Worth stating up front: for these three tasks, neither harness is really the intended surface. Claude Code and Codex CLI are coding agents, and the knowledge-work benchmarks below are run through Artificial Analysis's own harness (Stirrup), not through Claude Code or Codex. GDPval-AA v2 gives models sandbox filesystem shell access and web search through Stirrup — so the model results transfer, the harness results don't. On the Anthropic side the intended surfaces are Cowork, Claude in Excel, and Claude Design; on the OpenAI side, ChatGPT Work and the Codex App. If you're doing much of this, that changes the comparison more than the model does.

## Model-level evidence

| Eval                               | What it measures                                                        | Opus 5                        | GPT-5.6 Sol                                                         |
| ---------------------------------- | ----------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| GDPval-AA v2                       | Real professional deliverables (docs, sheets, slides), pairwise-graded  | 1861 Elo, >100 pts ahead      | ~1736                                                               |
| AA-Briefcase                       | 4 multi-week projects, thousands of input files, 91 rubric-graded tasks | 1720 Elo                      | 2nd to Fable 5 at its launch; highest Presentation Elo of any model |
| BrowseComp                         | Agentic web research                                                    | 90.8%                         | 92.2%                                                               |
| Turn/token efficiency on Briefcase | Cost of getting there                                                   | ~103 turns, 2.0B input tokens | Not published; Sol runs materially leaner                           |

## Data analysis

|          | Opus 5 / Claude Code                                                                                                                                     | Sol / Codex                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Strength | Leads both agentic knowledge-work benchmarks, and AA-Briefcase is the closest public proxy for this work — spreadsheet-shaped tasks over large file sets | Best presentation quality of any model on Briefcase's presentation dimension; leaner token use on long analytical runs            |
| Strength | 1M context holds a large dataset plus schema docs plus prior analysis without a retrieval layer                                                          | Cheaper to iterate, which matters when you re-run an analysis six times                                                           |
| Weakness | Turn count and token burn are the worst in the comparison — roughly 4x some competitors on the same Briefcase work                                       | Codex subagents run in cloud sandboxes with internet disabled, so pip installs and API pulls need setup                           |
| Weakness | Verbose intermediate output eats plan limits fast on exploratory analysis                                                                                | Run-to-run variance is a real problem when you want a reproducible analysis, not a plausible one                                  |
| Weakness | Reported weak on factual-knowledge/hallucination evals — one write-up flags a 50% hallucination rate as a hard limit for knowledge-intensive tasks       | The METR reward-hacking finding applies directly: an agent that makes the numbers come out right is a specific hazard in analysis |

For your Python work this mostly means: Claude Code runs pandas/matplotlib locally against real data with full filesystem access, Codex sandboxes it. Local access is more convenient and less safe.

## Interpreting business requirements

This is the one place the benchmark maps cleanly onto the task. AA-Briefcase's rubric explicitly checks whether the model identified requirements hidden across source files, used the correct evidence, and reached the right conclusions — that is requirements interpretation, graded. Opus 5 leads it.

The qualitative reports point the same way and matter more than the Elo:

- The most consistent developer complaint about Codex is off-plan drift — it doesn't stay on a written spec. That's disqualifying if the spec _is_ the deliverable.
- Claude's documented failure mode here is eager gap-filling: making assumptions without flagging them. For requirements work this is the thing to watch. A silently-resolved ambiguity in a spec is worse than an unresolved one, because it looks finished. Prompt explicitly for an open-questions section.
- Sol's reward-hacking propensity translates into requirements terms as satisfying the letter of the acceptance criteria. Same shape of risk, different surface.

Net: Opus 5 is the better default, with a hard rule that it lists assumptions separately from conclusions.

## Technical blog writing

The weakest evidence of the three. There's no credible benchmark for prose quality, and Presentation Elo measures deliverable formatting — slide layout, document structure — not writing.

|                           | Opus 5                                                                          | Sol                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Factual grounding         | Weaker on hallucination evals; will state things confidently that need checking | Slight edge on BrowseComp, so marginally better at going and verifying rather than recalling |
| Prose                     | Longer, more explanatory by default — needs cutting, not expanding              | Leaner output, which some people prefer and some find thin                                   |
| Code examples in the post | Higher SWE-bench Verified means snippets are more likely to actually run        | Fine, but the run-to-run variance shows up as inconsistent style across sections             |

For a technical post, the failure that costs you is a wrong claim, not a clumsy sentence. Both need fact-checking; neither has an evidence-backed prose advantage. Do this one in chat or Cowork, not in a coding harness — a terminal agent has no reason to be in the loop.

## Practical routing

| Task                                     | Pick                                                                                     | Why                                                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Exploratory data analysis, local data    | Claude Code / Cowork                                                                     | Local execution, best knowledge-work scores, 1M context — accept the token cost                |
| Repeated analysis on a fixed spec        | Codex                                                                                    | Cheaper per run; the spec constrains the drift problem                                         |
| Turning a vague business ask into a spec | Opus 5, either surface                                                                   | The AA-Briefcase result is the most directly relevant public evidence in this whole comparison |
| Charts/decks from the analysis           | Sol has the presentation edge; Claude Design is the better surface on the Anthropic side |                                                                                                |
| Blog drafting                            | Either, in chat                                                                          | No harness needed, and no benchmark separates them                                             |

The honest summary: on published evidence Opus 5 has a real and reasonably well-measured lead on agentic knowledge work, and the lead comes with a cost and verbosity penalty you'll feel on your plan limits. For the blog writing specifically, treat any claim of a winner — mine included — as unsupported.
