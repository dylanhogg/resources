# A recent experience with ChatGPT 5.5 Pro | Gowers's Weblog

This post is important because it gives a concrete, expert-audited example of an LLM moving beyond explanation into plausible original mathematical research, forcing researchers to rethink what kinds of problems, training tasks, and verification workflows remain distinctly human.

https://gowers.wordpress.com/2026/05/08/a-recent-experience-with-chatgpt-5-5-pro/

https://x.com/wtgowers/status/2052830948685676605

https://www.reddit.com/r/mathematics/comments/1taug8h/fields_medalwinning_mathematician_says_gpt55_is/

https://www.reddit.com/r/AIDangers/comments/1tatc0x/fields_medalwinning_mathematician_says_gpt55_is/

https://chatgpt.com/c/6a030834-a440-83ec-bf9b-4c2d38de7521

## Summary

Tim Gowers describes a striking experiment in which ChatGPT 5.5 Pro produced what he regards as **PhD-level additive-combinatorics research**, with almost no mathematical steering from him. The core claim is not merely that the model found a known trick, but that it improved an existing research argument and introduced what Isaac Rajagopal judged to be an original, clever construction. ([Gowers's Weblog][1])

## The maths problem

The setting is additive number theory. Given a finite set of integers (A), one studies the size of its (h)-fold sumset:

[
hA = {a_1 + \cdots + a_h : a_i \in A}.
]

For fixed (h) and (k = |A|), (\mathcal R(h,k)) is the set of all possible values of (|hA|). A related quantity, (N(h,k)), asks: how small an interval ({0,1,\dots,N}) is enough to realize all possible sumset sizes for (k)-element sets? ([Gowers's Weblog][1])

The original known constructions used geometric progressions like:

[
{0,1,m,m^2,\dots,m^{\ell-2}}
]

but these have **exponentially large elements**, leading to exponential bounds in (k). ChatGPT’s key contribution was to replace these with polynomially bounded constructions that preserve the needed additive behaviour. ([Gowers's Weblog][1])

## What ChatGPT did

Gowers first asked ChatGPT about the (h=2) case. It produced a construction improving Nathanson’s exponential-looking diameter bound to a **quadratic upper bound**, which Gowers says is clearly best possible. The idea was to replace powers of 2 with a more efficient Sidon set, since Sidon sets can have quadratic diameter. ([Gowers's Weblog][1])

Then he asked about general (h). ChatGPT first improved Rajagopal’s bound from exponential in (k) to roughly exponential in (k^{1/2+\varepsilon}). After further prompting, it produced a claimed **polynomial bound**. Rajagopal reviewed the result and described it as almost certainly correct, not just line-by-line but at the level of ideas. ([Gowers's Weblog][1])

The resulting bound stated in Rajagopal’s guest section is:

[
N(h,k) \leq O(k^{10h^3})
]

for sufficiently large (k). The lower bound is only known to be on the order of (k^h), so the true asymptotic remains open. ([Gowers's Weblog][1])

## The key technical idea

Rajagopal explains that his own constructions used geometric-series-like blocks whose additive relations were useful but whose elements were exponentially large. ChatGPT constructed substitute sets (G) and (H) that behave like “half a geometric series squeezed into a polynomial interval.” ([Gowers's Weblog][1])

The construction uses **(h^2)-dissociated sets**: sets with no nontrivial low-order additive relations. These allow ChatGPT to create sets that have just the required additive collisions, such as relations of the form (mx = y), while avoiding unwanted extra relations. This preserves the combinatorial behaviour of the geometric progression but with polynomial-sized elements. ([Gowers's Weblog][1])

Rajagopal says this use of (h^2)-dissociated sets to control low-order additive relations “feels quite ingenious” and, as far as he can tell, is original. ([Gowers's Weblog][1])

## AI takeaway

The important point is that this was not just literature retrieval or symbolic grinding. Gowers frames the result as a non-trivial extension of recent human work, and Rajagopal says the polynomial improvement involved an idea he would have been proud to find after a week or two. ([Gowers's Weblog][1])

For AI and research, the post suggests a new threshold: in some areas of maths, especially problem-focused areas like combinatorics, “open problem” no longer necessarily means “safe PhD starter problem.” It may mean “not yet tried seriously by a frontier model.” ([Gowers's Weblog][1])

## Broader implications

Gowers argues that mathematical research training may need to change. Beginner-level open problems that once helped PhD students learn research may increasingly be solvable by LLMs, so the useful human task may shift toward **collaborating with models**, judging correctness, supplying taste, choosing good problems, and knowing when an AI-generated argument is meaningful. ([Gowers's Weblog][1])

He also raises an infrastructure question: what should happen to AI-generated mathematical results? They may be valid and publishable in mathematical substance, but journals and arXiv are not obviously the right venue. Gowers suggests a possible moderated repository where human mathematicians, or ideally proof assistants, certify correctness. ([Gowers's Weblog][1])

## One-sentence version

A frontier LLM appears to have produced a serious additive-combinatorics result by replacing exponential-size geometric-series constructions with polynomial-size dissociated-set constructions, suggesting that some “gentle” open research problems may now be within autonomous LLM reach.

[1]: https://gowers.wordpress.com/2026/05/08/a-recent-experience-with-chatgpt-5-5-pro/ "A recent experience with ChatGPT 5.5 Pro | Gowers's Weblog"

---

## Similar posts

Here are the best recent pieces in the same category: **frontier AI crossing from “math helper” into plausible research collaborator**.

| Article                                                                                                           | Why it’s similar / worth reading                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **“The AI Revolution in Math Has Arrived” — Quanta Magazine, Apr 2026**                                           | Probably the closest broad companion piece. It covers First Proof, where AI systems were tested on unpublished research-level math questions and reportedly solved more than half. Good overview of the inflection point. ([Quanta Magazine][12]) |
| **“As AI keeps improving, mathematicians struggle to foretell their own future” — Scientific American, Mar 2026** | Covers the same “what happens to mathematical research now?” question, especially around First Proof and whether LLMs can prove useful lemmas for working mathematicians. ([Harvard Mathematics Department][2])                                   |
| **“Mathematics in the Library of Babel” — Daniel Litt, Feb 2026**                                                 | A skeptical mathematician updating toward “this is real.” Good for understanding why recent AI-math results feel different from earlier hype. ([Daniel Litt][3])                                                                                  |
| **“How GPT-5 helped mathematician Ernest Ryu solve a 40-year-old open problem” — OpenAI, Nov 2025**               | Vendor-authored, so read critically, but directly relevant: a working mathematician used GPT-5 to accelerate exploration and contribute to a long-standing optimization problem. ([OpenAI][4])                                                    |
| **“The story of Erdős problem #1026” — Terence Tao, Dec 2025**                                                    | A concrete case study of AI, humans, existing literature, and online collaboration combining to solve an Erdős problem. More nuanced than simple “AI solved it” headlines. ([What's new][5])                                                      |
| **“Problem 728 and the use of AI on Erdős problems” — Erdős Problems blog, Jan 2026**                             | Useful for seeing how the mathematical community itself is tracking AI-assisted solutions to open problems, including concern about provenance and verification. ([erdosproblems.com][6])                                                         |
| **“An amateur just solved a 60-year-old math problem—by asking AI” — Scientific American, Apr 2026**              | Similar to the Gowers post in that the interesting part is not just speed, but an AI-assisted route to a method experts thought was genuinely novel/useful. ([Scientific American][7])                                                            |
| **“Claude’s Cycles” — Donald Knuth, 2026 PDF**                                                                    | A primary-source mathematical write-up around Claude contributing to a graph-theoretic construction Knuth had been working on. Worth reading because it is from Knuth rather than a tech-company press release. ([Computer Science][8])           |
| **“Mathematical methods and human thought in the age of AI” — Tanya Klowden & Terence Tao, arXiv, Mar 2026**      | More reflective/philosophical than case-study-based, but important for the research-culture implications: what should remain human-centered as AI becomes embedded in mathematics. ([arXiv][9])                                                   |
| **“AI for Mathematics: Progress, Challenges, and Prospects” — arXiv survey, Jan/May 2026**                        | Best technical survey-style follow-up. Especially useful for the formal proof angle: why Lean/Coq-style verification may become crucial as LLM-generated math scales. ([arXiv][10])                                                               |
| **“AI Will Be Top of Mind at ICM, Math’s Biggest Conference” — Simons Foundation, May 2026**                      | Good signal that this is no longer a niche AI topic; it is becoming a central professional issue for mainstream mathematicians. ([Simons Foundation][11])                                                                                         |

Best reading order: **Quanta → Daniel Litt → Tao’s Erdős #1026 post → Knuth’s Claude’s Cycles → Tao/Klowden paper → AI for Mathematics survey**.

[2]: https://people.math.harvard.edu/~williams/scientificamerican4.pdf "As AI keeps improving, mathematicians struggle to foretell their own future"
[3]: https://www.daniellitt.com/blog/2026/2/20/mathematics-in-the-library-of-babel?utm_source=chatgpt.com "Mathematics in the Library of Babel"
[4]: https://openai.com/index/gpt-5-mathematical-discovery/ "How GPT-5 helped mathematician Ernest Ryu solve a 40-year-old open problem | OpenAI"
[5]: https://terrytao.wordpress.com/2025/12/08/the-story-of-erdos-problem-126/?utm_source=chatgpt.com "The story of Erdős problem #1026 | What's new - Terence Tao"
[6]: https://www.erdosproblems.com/forum/thread/blog%3A2?utm_source=chatgpt.com "Blog - Problem 728 and the use of AI on Erdős problems"
[7]: https://www.scientificamerican.com/article/amateur-armed-with-chatgpt-vibe-maths-a-60-year-old-problem/?utm_source=chatgpt.com "An amateur just solved a 60-year-old math problem— ..."
[8]: https://www-cs-faculty.stanford.edu/~knuth/papers/claude-cycles.pdf?utm_source=chatgpt.com "Claude's Cycles"
[9]: https://arxiv.org/abs/2603.26524 "[2603.26524] Mathematical methods and human thought in the age of AI"
[10]: https://arxiv.org/html/2601.13209v5 "AI for Mathematics: Progress, Challenges, and Prospects"
[11]: https://www.simonsfoundation.org/2026/05/04/ai-will-be-top-of-mind-at-icm-maths-biggest-conference/ "AI Will Be Top of Mind at ICM, Math’s Biggest Conference"
[12]: https://www.quantamagazine.org/the-ai-revolution-in-math-has-arrived-20260413/ "The AI Revolution in Math Has Arrived | Quanta Magazine"
