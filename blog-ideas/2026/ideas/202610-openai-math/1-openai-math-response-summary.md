# Mathematicians react to OpenAI’s October 2026 mathematics release

Summary of [100+ reactions to 100+ solutions](https://proofsandprompts.com/2026/10/08/100-reactions-to-100-solutions/), published 8 October 2026 and updated 12 October.

## Executive summary

The responses reveal a mathematical community experiencing a profound shift in what AI can achieve, combined with serious uncertainty about the future of mathematical research.

The central tension is that AI appears increasingly capable of producing mathematical discoveries faster than humans can understand, verify, or build upon them.

The reactions are far from uniformly negative. Many mathematicians are genuinely impressed by the technical achievements and excited about the possibility of accelerating research. Others describe grief, anger and disorientation as problems they have spent years pursuing are apparently resolved overnight.

Crucially, much of the criticism is not directed at the mathematical capabilities themselves. It concerns how the results were generated, released and communicated, and the consequences for the research community.

Several broad conclusions emerge:

- The capability advance is substantial. Researchers recognise potentially important results, including solutions involving unexpected combinations of mathematical techniques.
- Correctness is not the same as understanding. A machine-generated proof may establish a theorem without providing the conceptual insight mathematicians normally seek.
- The research bottleneck is shifting. Generating candidate proofs is becoming easier; verification, explanation, synthesis and identifying worthwhile questions are increasingly important.
- The disruption is uneven. Early-career researchers, particularly PhD students and postdocs, face greater risks than established mathematicians.
- There is a conflict between commercial AI development and academic norms. Researchers question access, transparency, attribution, publication quality and the use of their research as model-development benchmarks.
- Mathematics is not finished. Even contributors who regard the results as extraordinary argue that mathematical research extends well beyond solving existing conjectures.

The overall sentiment is best characterised as technical admiration combined with deep concern about how mathematical research will be organised in the future.

## 1. What OpenAI actually released

On 6 October 2026, OpenAI published a large collection of AI-generated mathematical manuscripts in its [openai/math GitHub repository](https://github.com/openai/math).

| Measure                                  | Reported figure                         |
| ---------------------------------------- | --------------------------------------- |
| Manuscripts released                     | 719                                     |
| Families of related mathematical results | 372                                     |
| Problems evaluated                       | Approximately 4,000                     |
| Average reasoning compute per result     | 3 hours of ChatGPT Pro thinking compute |
| Top-line results formalised in Lean      | Approximately 42%                       |
| Model                                    | Unreleased internal OpenAI model        |

The release spans multiple mathematical disciplines, with claims involving longstanding conjectures in areas such as number theory, group theory, geometry and theoretical computer science.

OpenAI acknowledges that not every result has been formally verified and that some unformalised arguments may contain errors. The repository includes proof artefacts and selected reasoning summaries, but not everything needed to reproduce the full experimental process.

For an ML audience, the important distinction is that the collection is neither simply a benchmark score nor a uniformly verified set of theorems. It contains research outputs at different stages of validation.

The scale is itself a significant part of the story. Mathematics has traditionally progressed through a relatively slow process of discovery, proof development, peer review and community understanding. This release places hundreds of claimed advances into that process simultaneously.

## 2. The main themes in the responses

### A. The mathematical capability is genuinely surprising

Even researchers who strongly oppose OpenAI's approach acknowledge the scale and apparent sophistication of the results.

An important observation is that the models are not merely reproducing standard textbook techniques. Some proofs appear to combine ideas from distant areas of mathematics in ways that experts find unexpected.

Examples from the responses:

Roman Sauer describes seeing a specialised construction from ergodic theory, previously used in his own research, appear in solutions to two very different problems. He regards the combination as deeply creative.

Tobias Osborne observes that individual steps in the proofs are often recognisable. The surprising part is their composition: the model combines familiar techniques in ways a human researcher might dismiss as implausible.

Ben Green describes particularly surprising advances involving Dirichlet L-functions and additive combinatorics, noting that several results overlap with the aims of a recently awarded research grant.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

+2

These observations raise an interesting ML question: is the model discovering fundamentally new mathematical concepts, or has it become exceptionally effective at combining and extending existing ones?

The responses do not establish a definitive answer. Several mathematicians suggest that the current achievements may still lie largely within the conceptual space created by existing human research.

Either way, the practical capability is significant.

### B. Proof generation and mathematical understanding are different things

This is perhaps the strongest recurring intellectual argument.

Traditionally, producing a proof involves developing intuition, investigating unsuccessful approaches, discovering useful abstractions and gradually building an explanation.

The final theorem is only one product of that process.

AI changes this relationship. A system can potentially produce a valid proof without any human having developed the underlying understanding.

Formal verification helps, but only addresses part of the problem.

Lean, the interactive theorem prover used in the release, can check that a formal proof establishes a specified theorem from its stated assumptions. This offers a much stronger correctness guarantee than checking informal prose with another LLM.

However, Lean verification does not establish that the informal manuscript accurately explains the formal proof, that the result is mathematically interesting or that the proof provides useful intuition.

Several contributors argue that mathematical knowledge requires more than a correct proposition. It requires humans to understand why it is true, what techniques matter and how the result connects to other ideas.

Peter Scholze emphasises that mathematics remains a pursuit of human understanding. Terence Tao similarly notes that AI-generated results lack the human authors who would normally answer questions, give seminars and help develop the surrounding theory.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

+2

The underlying concern is that mathematics might become a collection of verified facts without enough shared conceptual understanding to make those facts useful.

### C. The release creates a substantial verification and interpretation burden

A repeated criticism concerns the quality and quantity of the published material.

Contributors describe manuscripts containing unclear explanations, unnecessarily long arguments, missing attribution, minor errors and occasionally withdrawn results.

The criticism is not that every proof is incorrect. It is that releasing hundreds of imperfect manuscripts transfers considerable work to mathematicians who did not choose to undertake it.

Enrico Fatighenti asks why AI-generated papers should be exempt from the presentation standards expected of students and academic researchers.

Giulio Tiozzo gives a concrete example: after examining a construction involving an Artin group with 116 generators, he used an AI model to obtain a much simpler example with 12 generators. He takes this as evidence that more careful review could have substantially improved the release.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

+1

There is a clear asymmetry here. An AI lab can generate results at machine speed, while human mathematicians must invest considerable time to verify, explain, simplify and contextualise them.

Some researchers welcome this work because the results may contain important insights. Others argue that mathematicians should not provide unpaid validation for commercially developed AI systems.

OpenAI has committed to funding workshops and other programmes to support understanding of the results, but this does not fully resolve the concerns over responsibility and publication standards.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

### D. Early-career researchers face the greatest disruption

This is one of the most consistent concerns across otherwise very different viewpoints.

Academic mathematics relies heavily on research problems that remain open for years or decades. They provide the basis for PhD theses, research grants, postdoctoral projects and career development.

The sudden publication of a claimed solution can undermine this structure before the result has even been fully assessed.

Tristan Humbert describes how OpenAI's claimed solution to Katok's entropy conjecture affected research central to his PhD and forced him to reconsider his postdoctoral applications.

Tasmin Chu describes having to rewrite a grant proposal after a problem that had motivated years of interest appeared in the release.

Both accounts illustrate a deeper problem: researchers lose not only the opportunity to establish a result first, but also the opportunity to develop mathematical intuition through years of working towards it.

Senior mathematicians often have established reputations and greater freedom to change direction. Junior researchers are more dependent on measurable achievements within a limited period.

The responses repeatedly call for academic institutions to reconsider how they train, fund, hire and evaluate mathematicians.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

+2

### E. The incentives of AI labs and academia are not well aligned

OpenAI says these results arose from evaluating internal models on open research problems after existing mathematical benchmarks had become insufficiently challenging.

That distinction matters. The researchers' mathematical questions were being used, at least partly, as evaluation tasks for improving an AI system.

The mathematical results were a product of that process, but advancing mathematical understanding was not necessarily the primary objective.

Several contributors argue that open problems are a shared scientific resource, not merely a collection of benchmark questions. They help organise research programmes, motivate collaboration and provide long-term goals.

There are also concerns over equitable access. The most capable models remain proprietary, creating a situation in which researchers with access to frontier compute may have a substantial advantage over conventional academic groups.

The independent Advisory Group on Mathematics and Artificial Intelligence (AGMAI) has explicitly declined to endorse the process through which OpenAI obtained its results. It argues that mathematicians must retain the ability to choose their own research questions, develop independent approaches and access powerful research tools fairly.

OpenAI has engaged with the group, but maintains that evaluating internal models on mathematical research remains important. This is a substantive disagreement, not simply a problem of how papers are formatted.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://agmai.org&sz=32)

agmai.org

+2

### F. Mathematics may need to redefine what it rewards

A more forward-looking set of responses considers what mathematical research should value when theorem production becomes much cheaper.

Danny Calegari argues that mathematics is also about developing frameworks, concepts and compelling ways of understanding a subject. A significant contribution might be a new theory or an explanatory framework rather than a particular solved problem.

Kevin Buzzard is strongly optimistic that mathematical truth and understanding will eventually emerge from the current disorder, and that AI-generated explanations will improve.

Ryan Alweiss likewise welcomes an era of abundant proofs, arguing that collaboration between humans and AI could enable substantially more ambitious mathematics.

Others believe that the shift could improve aspects of academic culture by reducing the emphasis on being first to prove a theorem and increasing recognition for explanation, collaboration and education.

The concern is whether these activities will remain professionally sustainable if the strongest incentives and most substantial resources continue to favour automated proof generation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

+2

## 3. Where the contributors differ

The positions are not simply pro-AI versus anti-AI. Many researchers hold both positive and negative views simultaneously.

| Perspective                         | Core argument                                                                                                 | Representative contributors                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Technically optimistic              | The results are extraordinary and could accelerate mathematical discovery                                     | Kevin Buzzard, Ryan Alweiss, Andreas Thom         |
| Impressed but deeply concerned      | Valuable advances are arriving in a way that damages existing research practices                              | Terence Tao, Martin Bridson, Bryna Kra            |
| Strongly critical of the release    | The volume, presentation and commercial motivations impose unacceptable costs                                 | Henry Wilton, Enrico Fatighenti, Alexandre Martin |
| Focused on human understanding      | Mathematics must remain about concepts, intuition, exploration and communication                              | Peter Scholze, Danny Calegari, Marco Moraschini   |
| Concerned about future capabilities | The mathematical results may signal broader changes to science and society, including possible security risks | Grigori Avramidi, Peter Scholze                   |

These are illustrative positions rather than formal groups. The collection is a set of voluntary, immediate reactions, not a representative survey of mathematicians.

A particularly important distinction is that many critics accept the technical achievements while rejecting the process by which they were produced and distributed.

## 4. What this means from an ML perspective

There are several useful interpretations for people working with frontier models.

Capability evaluation is changing. Open research problems now appear increasingly useful for measuring the upper end of model capability, but this creates an unusual benchmark problem: evaluating a model can materially alter the research domain being evaluated.

Validation is becoming a separate bottleneck. A substantial gap can exist between candidate generation, formal correctness and domain-expert understanding. Simply generating more outputs does not necessarily increase usable scientific knowledge at the same rate.

Novelty needs a more careful definition. Combining established ideas in previously unexpected ways can be an important discovery. However, the extent to which these models create new mathematical abstractions, rather than recombining existing ones, remains an important open question.

Success rates need context. Approximately 4,000 problems were attempted, producing 372 families of results. This is impressive, but does not imply the model can reliably solve arbitrary open problems. The choice of evaluation problems, available compute, unsuccessful attempts and strength of verification all matter.

The workflow matters as much as the model. A more useful scientific system might integrate candidate generation, formal verification, literature review, simplification, explanatory writing and expert feedback, rather than treating a completed proof as the sole deliverable.

It is also important to avoid overstating the release. Several very difficult problems received partial or related advances, but not full solutions to their original formulations. For example, the reported quasi-Riemann result is not a proof of the Riemann Hypothesis itself.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proofsandprompts.com&sz=32)

Proofs and Prompts

## 5. Overall assessment

The collection captures a field confronting a possible transition from scarcity of mathematical proofs to an abundance of candidate proofs.

That could lead to a substantial acceleration in scientific understanding. It could also undermine the human institutions and intellectual practices that made the achievements possible.

The contributors broadly agree that mathematical research is changing, but disagree about whether the current trajectory will produce a healthier discipline.

The most constructive direction emerging from the responses is not to reject AI or to hand mathematical discovery entirely to machines. It is to develop research practices in which powerful AI systems contribute to discoveries while humans continue to formulate questions, build intuition, understand results and shape the direction of the field.

The central question is no longer just whether AI can produce important mathematical proofs. It is whether the resulting capability can be integrated into science in a way that increases human understanding rather than simply generating results faster than anyone can absorb them.
