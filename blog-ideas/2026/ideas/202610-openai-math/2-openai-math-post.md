# Why AI automated maths early, and how mathematicians are reacting

_How mathematicians responded to OpenAI's October 2026 maths release, and what their responses suggest about the order in which AI automates different kinds of work._

## 1\. Why AI automated maths early

In the same week that OpenAI published hundreds of AI-generated maths papers, Peyman Milanfar posted a theory about which fields AI will automate first. His post opens with this line:

> "Speed of automation is in inverse proportion to the entropy of the subject domain."
> — [Peyman Milanfar](https://x.com/docmilanfar/status/2107678058857873520)

Milanfar argues that AI performs well at maths and coding because both fields are structured and orderly. He then ranks other fields by how disordered they are. The hard sciences come next, slowed by noisy measurements and laboratory work. Financial markets, medicine and law are progressively harder, and culture, art and leadership are the least exposed.

I think Milanfar's idea explains much of why maths was automated before other research fields. Maths gives a model an unusually clear feedback signal. A proof is either valid or invalid, and Lean, an interactive theorem prover, can check a formal proof automatically. This makes maths close to an ideal domain for reinforcement learning with verifiable rewards, where a model is trained on tasks whose answers a program can check.

The mathematicians' reactions show where the idea needs refining. AI can now produce proofs at scale, but producing proofs is only one part of mathematical research. Researchers also decide which questions matter, develop an understanding of why results are true, train students and build careers. These tasks are much less structured, and most of the mathematicians' concerns are about them.

## 2\. What OpenAI released

In early October, OpenAI published a large set of AI-generated manuscripts in its [openai/math repository](https://github.com/openai/math).

| Measure                              | Reported figure                       |
| ------------------------------------ | ------------------------------------- |
| Manuscripts                          | 719                                   |
| Families of related results          | 372                                   |
| Problems posed to the model          | About 4,000                           |
| Average reasoning compute per result | About 3 hours of ChatGPT Pro thinking |
| Top-line results formalised in Lean  | About 42%                             |
| Model                                | Unreleased internal OpenAI model      |

The results cover number theory, group theory, geometry, theoretical computer science and other areas. OpenAI says it started evaluating its models on open research problems because the models had saturated its existing maths benchmarks. OpenAI also says that results without a Lean formalisation may contain errors, and that it will record corrections as new versions in the repository.

The release therefore contains results at different stages of validation, and OpenAI published all of them at once. Normal mathematical review handles results one paper at a time.

## 3\. What mathematicians said

Shortly after the release, the blog Proofs and Prompts collected [more than 100 reactions](https://proofsandprompts.com/2026/10/08/100-reactions-to-100-solutions/) from mathematicians. Few contributors are simply for or against AI, and many are both impressed by the results and worried about their effects. The contributors chose to respond and wrote soon after the release, so the collection is not a representative survey.

**Some results are mathematically sophisticated.** Even contributors who strongly criticise the release accept this. Roman Sauer found that a construction from ergodic theory, which he has used in his own research, appears in OpenAI's solutions to two unrelated problems. Tobias Osborne recognised the individual steps in the proofs but was surprised by the way the model combined them. Ben Green reports that the additive combinatorics results cover about three quarters of the aims of his ERC Advanced Grant, a major European research grant.

**A correct proof does not give mathematicians understanding.** This is the most common argument. Peter Scholze writes that the goal of mathematics is human understanding. Terence Tao notes that the AI-generated papers have no human authors who can answer questions, give talks or develop the surrounding theory. Kimball Martin points out that a Lean check shows a proof is correct but not that it is good. Lean confirms that a formal proof establishes a stated theorem. It does not confirm that the manuscript explains the proof clearly, or that the result is worth knowing.

**The release creates work for mathematicians who did not ask for it.** Contributors describe manuscripts that are hard to read, longer than necessary and missing attribution to earlier work. Kevin Buzzard notes that some papers have already been retracted. Enrico Fatighenti says he rejects poorly written work from students, and asks why AI-generated papers should meet a lower standard. One released result used an Artin group with 116 generators. Giulio Tiozzo used an AI tool to look for a smaller example, and it produced a new one with 12 generators in about ten minutes. He concludes that more careful review before release would have produced much better papers. Tristan Humbert, whose PhD research centres on Katok's entropy conjecture, describes OpenAI's claimed proof of that conjecture as largely unreadable.

**PhD students and postdocs face the greatest career risk.** Open problems provide the topics for PhD theses, grant proposals and postdoctoral applications. When a lab publishes a claimed solution, a junior researcher can lose a project they planned to work on for years. Humbert is reconsidering his postdoc applications. Tasmin Chu has to rewrite a grant proposal that is due the following week. Established mathematicians can move to new problems more easily than junior researchers, who need recognised results within a few years to secure their next position.

**The goals of AI labs and academic mathematicians conflict.** OpenAI used these open problems, at least in part, to evaluate an internal model. Several contributors argue that open problems are a shared resource that mathematicians use to organise research programmes and long-term collaborations, and that treating them as benchmark questions ignores that role. The Advisory Group on Mathematics and Artificial Intelligence (AGMAI) has asked AI companies to stop evaluating proprietary models on open problems. Johannes Schmitt notes that OpenAI followed some of AGMAI's advice, such as releasing results promptly and including partial Lean proofs. However, OpenAI still considers evaluation on open problems important. Alain Valette asks why mathematicians should give free expert feedback to a company he describes as secretive.

**Some contributors expect mathematics to benefit.** Buzzard is excited about the future, although he acknowledges that the field is currently in disorder. Ryan Alweiss welcomes what he calls an age of "proof abundance". Danny Calegari argues that mathematics also involves building frameworks that explain how results fit together, such as the Langlands program, and says the release has not taken that away from him.

## 4\. My view: entropy is a property of tasks, not whole fields

Milanfar applies his idea to whole fields. The reactions suggest that it is more accurate when applied to individual tasks within a field. The table below is my rough breakdown of mathematical research, and the entropy ratings are my own judgement.

| Task                                            | My entropy rating | Status after this release          |
| ----------------------------------------------- | ----------------- | ---------------------------------- |
| Checking a formal proof                         | Very low          | Already automated by Lean          |
| Finding a proof of a well-posed conjecture      | Low               | Substantially automated            |
| Writing a clear, well-attributed exposition     | Medium            | Weak in the released papers        |
| Deciding which questions are worth asking       | High              | Still done by people               |
| Building shared intuition and training students | High              | Still done by people, under strain |

The release automated most of the work in the first two rows of the table. Almost all of the mathematicians' concerns relate to the last three rows. Those tasks have no automatic verifier, so AI will probably automate them more slowly. The release also made those tasks harder, because mathematicians now have hundreds of new results to read, explain and connect to existing work.

This points to an effect that Milanfar's ranking does not describe. When AI quickly automates the low-entropy tasks in a field, it increases the workload on the high-entropy tasks in the same field. I expect the hard sciences to show a similar pattern. Their verifier is a laboratory experiment rather than Lean, so automation there will probably be slower and less certain.

## 5\. What this means for people building models

- **Validation is now the main bottleneck.** Models can generate candidate proofs cheaply, but formal checking, simplification and expert understanding are still expensive. As a result, more proofs do not increase usable mathematical knowledge at the same rate. Tiozzo's example suggests that models could do some of the simplification before results are released.
- **Evaluating a model on open problems changes the field being measured.** A benchmark that leaks into training data mainly loses its usefulness. Publishing results on open research problems changes the research field itself and affects the careers of the people working in it.
- **The success rate needs context.** About 4,000 problems produced 372 families of results. This is impressive, but it does not show that the model can reliably solve any open problem. The rate depends on how OpenAI selected the problems, how much compute it used, how many attempts failed and how thoroughly the results were verified. Some headline results are also partial. Álvaro Lozano-Robledo notes that the quasi-Riemann result is a major advance but not a proof of the Riemann Hypothesis.
- **Novelty needs a careful definition.** Combining techniques from distant areas of mathematics counts as real discovery. The reactions do not settle whether these models create new mathematical abstractions or are very good at recombining existing ones.
- **A complete workflow would be more useful than a collection of proofs.** Mathematicians would benefit more from a system that combines proof generation, formal verification, literature search, simplification and explanatory writing, with expert feedback at each stage.

## 6\. Open questions

- Does Milanfar's idea work better at the level of tasks, as I have argued, or will AI automate the high-entropy parts of maths next?
- Who should pay to verify and explain AI-generated results: the lab that produced them, research funders or the mathematical community?
- How should PhD programmes and funding bodies adapt when a lab can publish a claimed solution to an open problem without warning?
- Will the hard sciences follow the same pattern when the verifier is a laboratory experiment instead of a proof checker?

I would like to hear from mathematicians who have worked through some of the papers, and from ML practitioners who disagree with the task-level framing.
