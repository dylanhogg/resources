Create a concise technical introduction to Lean 4 as both a proof assistant and a functional programming language.

Audience: senior software engineers and machine learning engineers. Assume strong Python knowledge and some familiarity with Scala or ML-family functional programming. Do not cover installation, editor setup, package management, or project scaffolding.

Primary sources:

- [https://github.com/leanprover/lean4](https://github.com/leanprover/lean4)
- [https://lean-lang.org/](https://lean-lang.org/)
- [https://leanprover-community.github.io/](https://leanprover-community.github.io/)

Goal: fast-track the reader into writing small mathematical proofs and functional programs in Lean.

Write in high-density technical prose. Be direct, precise, and example-driven. Avoid motivational filler.

Cover:

1. Mental model

- What Lean is: dependent type theory, theorem prover, functional language, compiler.
- How propositions-as-types and proofs-as-programs work.
- Difference between computation, type checking, elaboration, tactics, and kernel verification.
- Why the small trusted kernel matters.

2. Lean as a functional language

- Core syntax for definitions, functions, algebraic data types, pattern matching, recursion, namespaces, modules.
- Compare briefly with Python and Scala where useful.
- Explain `def`, `inductive`, `structure`, `class`, `instance`, `match`, `let`, lambdas, implicit arguments, and type inference.
- Show concise examples using `Nat`, lists, options, recursive functions, and simple data models.

3. Lean as a proof assistant

- Explain theorem statements, hypotheses, goals, contexts, and proof terms.
- Introduce `example`, `theorem`, `by`, `rfl`, `rw`, `simp`, `exact`, `apply`, `intro`, `cases`, `induction`, and `calc`.
- Show minimal proofs of equality, implication, conjunction, universal quantification, and induction over natural numbers or lists.
- Explain tactic mode versus term-style proofs.

4. Type system essentials

- Universes, dependent functions, dependent pairs only as far as needed.
- Propositions versus data: `Prop`, `Type`, proof irrelevance.
- Explain coercions, implicit parameters, named arguments, and notation.
- Show how error messages relate to failed unification or missing instances.

5. Mathematical workflow

- How to read a goal state.
- How to search or discover lemmas conceptually.
- How to decompose a proof into smaller lemmas.
- How `simp` and rewriting form the core proof automation workflow.
- How Mathlib changes practical Lean development.

6. Functional programming workflow

- Writing total functions.
- Structural recursion and termination checking.
- Separating executable code from specifications.
- Using tests/examples versus proving properties.

7. Worked path

- Provide a compact sequence of examples that progresses from:
  a. defining functions,
  b. proving simple equalities,
  c. proving a property by rewriting,
  d. proving a property by induction,
  e. defining a small custom datatype,
  f. proving a property about it.

8. Practical heuristics

- Common beginner failure modes for experienced engineers.
- How to think when Lean rejects a program or proof.
- When to use tactics, when to write helper lemmas, and when to simplify definitions.
- What to learn next after this introduction.

Output format:

- Use section headings.
- Keep explanations terse but complete.
- Prefer small Lean code blocks over long prose.
- Include comments inside code only when they clarify non-obvious Lean behavior.
- Avoid long historical background.
- Avoid installation instructions.
- Avoid generic theorem-proving hype.
- End with a compact cheat sheet of core syntax and tactics.
