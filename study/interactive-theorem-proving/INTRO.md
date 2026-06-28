# Technical Introduction to Lean 4 for Senior Engineers and ML Engineers

## Executive Summary

Lean 4 is both a dependently typed functional programming language and an interactive theorem prover. It lets engineers write executable programs, precise mathematical specifications, and machine-checked proofs in one language. The fastest useful mental model is: **types are specifications, terms are programs, and proofs are values checked by the kernel**.

Lean is not mainly a test framework or symbolic algebra tool. It is a small trusted proof kernel plus a large elaboration, automation, and library ecosystem. The kernel checks whether a proof term inhabits a proposition-as-type. Everything else exists to help humans construct those terms.

## Why Engineers Should Care

For software engineers, Lean teaches how to encode invariants in types rather than comments, tests, or runtime checks. For ML engineers, Lean is increasingly relevant as a target language for formal reasoning, proof generation, theorem-proving benchmarks, verified algorithms, and structured reasoning evaluation.

The practical goal is not to “prove everything.” The practical goal is to learn how to state claims precisely, decompose them into lemmas, use libraries effectively, and build small verified artifacts.

## Core Mental Model

Lean uses dependent type theory. A type can depend on a value.

```lean
def double (n : Nat) : Nat := n + n
```

Here `double` is a program. A theorem is also a declaration:

```lean
theorem zero_add_nat (n : Nat) : 0 + n = n := by
  rfl
```

The proposition `0 + n = n` is a type. The proof after `:=` is a term of that type. `by` starts tactic mode, where commands transform proof goals until Lean can construct the proof term.

## Syntax and Concepts to Learn First

Learn declarations first: `def`, `theorem`, `example`, `structure`, `inductive`, `namespace`, `variable`.

Learn core types: `Nat`, `Int`, `Bool`, `List α`, `Option α`, `Prod`, `Sum`, functions `α → β`, propositions `Prop`.

Lean’s function syntax is close to Scala/Haskell:

```lean
def mapOption (f : α → β) : Option α → Option β
  | none => none
  | some x => some (f x)
```

Pattern matching, recursion, algebraic data types, typeclasses, and higher-order functions are central. Unlike Python, Lean wants total definitions by default: recursive functions must structurally terminate unless explicitly justified.

## Proof Basics

A proof usually starts by introducing assumptions, simplifying definitions, splitting cases, applying known theorems, and invoking automation.

Important tactics:

```lean
intro h      -- introduce variable or hypothesis
exact h      -- solve goal with exact proof
apply h      -- reduce goal using implication/function
rw [h]       -- rewrite using equality
simp         -- simplify using rewrite database
cases h      -- split inductive value/proof
induction n  -- induction on n
constructor  -- split conjunction / build structures
omega        -- solve Presburger arithmetic goals
ring         -- solve algebraic ring equalities
```

Example:

```lean
example (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  constructor
  · exact h.right
  · exact h.left
```

Read this as a proof script. The goal is `q ∧ p`. `constructor` splits it into two subgoals: prove `q`, then prove `p`.

## Functional Programming in Lean

Lean programs are pure by default. Side effects are represented through monads such as `IO`, similar in spirit to effect systems in Scala but enforced more directly.

```lean
def greet (name : String) : IO Unit := do
  IO.println s!"hello, {name}"
```

Use Lean as a functional language by learning:

- algebraic data types via `inductive`
- records via `structure`
- pattern matching
- recursion and termination
- typeclasses
- monadic `do`
- generic functions over type parameters
- proofs attached to data structures

Example of a custom ADT:

```lean
inductive Tree (α : Type) where
  | leaf : α → Tree α
  | node : Tree α → Tree α → Tree α

def size : Tree α → Nat
  | Tree.leaf _ => 1
  | Tree.node l r => size l + size r
```

## Mathlib and Reuse

Mathlib is the standard library for serious Lean proof work. Do not reprove basic algebra, order theory, lists, sets, topology, or number theory from scratch. Search Mathlib first, then write small glue lemmas.

The basic workflow is:

1. State the theorem.
2. Inspect the goal.
3. Search for existing lemmas.
4. Rewrite/simplify.
5. Split cases or induct.
6. Use automation where appropriate.
7. Factor repeated proof fragments into named lemmas.

## Engineering Workflow for Proofs

Lean proof development is interactive. The central loop is not “write complete proof then compile.” It is:

```text
state theorem → inspect goal → apply tactic → inspect new goals → repeat
```

Good Lean code has small theorem statements, short helper lemmas, explicit names, and minimal cleverness. Proof golfing is bad engineering. Prefer maintainable proof scripts over dense automation when onboarding engineers.

## How to Fast-Track Learning

Recommended sequence:

1. Read enough Lean syntax to write functions over `Nat`, `List`, `Option`, and custom `inductive` types.
2. Learn propositions-as-types: implication, conjunction, disjunction, negation, equality, quantifiers.
3. Prove small propositional examples using `intro`, `exact`, `apply`, `constructor`, `cases`.
4. Prove list and natural-number properties using `simp`, `rw`, and `induction`.
5. Learn Mathlib search and reuse.
6. Write small verified programs: parser invariants, sorting properties, bounded data structures, simple algebraic laws.
7. Only then learn advanced dependent types, custom tactics, metaprogramming, or category-theoretic abstractions.

## Python/Scala Translation Guide

Python intuition often fails because Lean does not treat types as optional metadata. Types are executable constraints on what can be constructed.

Scala intuition transfers better: algebraic data types, pattern matching, immutability, parametric polymorphism, and typeclasses all map well. The key additional step is dependent types: values can appear in types, so invariants can be represented statically.

Example concept shift:

```text
Python: assert len(xs) > 0 at runtime
Scala: encode Option/NonEmptyList
Lean: encode the invariant in the type and prove it is preserved
```

## Common Failure Modes

New Lean users usually fail by trying to write large proofs top-down, ignoring Mathlib, fighting simplification, or treating tactics as magic. Better practice is to make definitions reducible, theorem statements minimal, and proof states small.

Another common mistake is overusing dependent types too early. Start with ordinary functional programs and ordinary theorems. Add dependent structure only when it removes real ambiguity or invalid states.

## Minimal Outcome Target

After the introduction, an engineer should be able to:

- read basic Lean declarations
- write recursive functional programs
- define inductive data types
- state simple mathematical properties
- prove propositional and equality-based theorems
- use `simp`, `rw`, `cases`, and `induction`
- search and reuse Mathlib lemmas
- understand why a proof is kernel-checked rather than merely tested

The target is not mathematical maturity in one session. The target is operational literacy: enough Lean to write small programs, state precise claims, and complete simple machine-checked proofs.

[1]: https://lean-lang.org/ "Lean Programming Language"
