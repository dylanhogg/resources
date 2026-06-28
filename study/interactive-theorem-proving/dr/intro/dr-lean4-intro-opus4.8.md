# Lean 4: A Technical Introduction for Engineers

https://claude.ai/chat/39757f52-f983-42e2-bc0c-486193e853b8

---

Lean 4 is one language with two faces: a strict, eagerly-evaluated functional language that compiles to native code, and a proof assistant for dependent type theory. The same `def` mechanism that defines a sorting function also defines a proof. This guide fast-tracks you to writing both. It assumes Python fluency and ML-family/Scala familiarity.

---

## 1. Mental model

Lean is four things at once, layered:

- **A dependent type theory** (the Calculus of Inductive Constructions). Types may depend on values: `Vector α n` is the type of length-`n` vectors, where `n` is a _value_.
- **A functional language** with inductive types, pattern matching, type classes, and a native compiler.
- **A theorem prover**: because types can encode propositions, a term of the right type _is_ a proof.
- **A small trusted kernel** that re-checks every proof term.

### Propositions as types, proofs as programs

The Curry–Howard correspondence is not an analogy in Lean — it is the implementation.

| Logic             | Type theory                            |
| ----------------- | -------------------------------------- |
| proposition `P`   | a type `P : Prop`                      |
| `P` is true       | `P` is inhabited (some `p : P` exists) |
| proof of `P`      | a term `p : P`                         |
| `P → Q` (implies) | function type `P → Q`                  |
| `P ∧ Q`           | product/structure `And P Q`            |
| `∀ x, P x`        | dependent function `(x : α) → P x`     |
| `∃ x, P x`        | dependent pair `Σ`/`Exists`            |

Proving `P → P` is writing the identity function. Proving `P ∧ Q → Q ∧ P` is writing a function that swaps a pair. This is literal:

```lean
theorem imp_self (P : Prop) : P → P := fun h => h          -- proof = identity fn
theorem and_swap (P Q : Prop) : P ∧ Q → Q ∧ P :=
  fun h => ⟨h.2, h.1⟩                                       -- proof = swap pair
```

### Five distinct phases

Engineers conflate these at their peril:

- **Elaboration**: Lean turns surface syntax (with implicit args, coercions, notation, type-class holes) into fully-explicit core terms. Most "magic" and most confusing errors happen here.
- **Type checking**: verifying a fully-elaborated term has the claimed type.
- **Computation / reduction**: definitional unfolding (`2 + 2` reduces to `4`). Used during type checking to decide equality.
- **Tactics**: a metaprogram that _builds_ a proof term. `by ...` runs tactics; the output is an ordinary term. Tactics are a convenience, not a separate logic.
- **Kernel verification**: the final term is re-checked by a tiny kernel that knows nothing about tactics, `simp`, or Mathlib.

### Why the small kernel matters

`simp`, `omega`, `decide`, and the entirety of Mathlib are _untrusted_. They emit proof terms the kernel independently re-checks. A bug in a tactic produces a term the kernel rejects — it cannot produce a false theorem that the kernel accepts. Your trust bottoms out in a few thousand lines, not the millions above it.

---

## 2. Lean as a functional language

Strict evaluation, immutable by default, total functions, exhaustive pattern matching.

### Definitions and functions

```lean
def double (n : Nat) : Nat := n * 2
def add (x y : Nat) : Nat := x + y           -- curried: Nat → Nat → Nat

#eval double 21                              -- 42   (#eval runs the compiler)
#check add                                   -- add : Nat → Nat → Nat
```

`#eval` executes; `#check` reports a type without running. There is no `return`; the body _is_ the value.

### Lambdas, `let`, `match`

```lean
def compose (f g : Nat → Nat) : Nat → Nat := fun x => f (g x)

def classify (n : Nat) : String :=
  let parity := if n % 2 == 0 then "even" else "odd"
  match n with
  | 0 => "zero"
  | _ => parity
```

`match` is exhaustive and the compiler rejects missing cases — closer to Scala's `match` than Python's `match`, with no fall-through.

### Algebraic data types: `inductive`

```lean
inductive Color where
  | red | green | blue

inductive Tree (α : Type) where               -- generic, like Scala's sealed trait + cases
  | leaf
  | node (left : Tree α) (value : α) (right : Tree α)
```

`Nat`, `List`, `Option`, `Bool` are all just `inductive` types in the library, not primitives:

```lean
-- conceptually:
-- inductive Nat where | zero | succ (n : Nat)
-- inductive List (α : Type) where | nil | cons (head : α) (tail : List α)
-- inductive Option (α : Type) where | none | some (val : α)
```

`5` is sugar for `succ (succ (succ (succ (succ zero))))`; `[1,2,3]` for `1 :: 2 :: 3 :: []`.

### Recursion and pattern matching

```lean
def length : List α → Nat
  | []      => 0
  | _ :: xs => 1 + length xs                  -- structural recursion on the tail

def sum : List Nat → Nat
  | []      => 0
  | x :: xs => x + sum xs

def lookup? : List (Nat × String) → Nat → Option String
  | [],            _ => none
  | (k, v) :: rest, n => if k == n then some v else lookup? rest n
```

The `?` in `lookup?` is a naming convention for `Option`-returning functions, not syntax.

### `structure` — product types / records

```lean
structure Point where
  x : Float
  y : Float

def origin : Point := { x := 0, y := 0 }
def Point.norm (p : Point) : Float := Float.sqrt (p.x^2 + p.y^2)

#eval origin.norm                             -- dot notation resolves to Point.norm
#eval { origin with x := 3 }.x                -- 3.0   (functional record update)
```

A `structure` is a single-constructor `inductive` with named projections. `p.norm` works because `p : Point` and `Point.norm` exists — namespace-driven dot dispatch.

### `class` / `instance` — type classes

Like Scala's `given`/`using` or Rust traits; the mechanism behind `+`, `==`, etc.

```lean
class Describe (α : Type) where
  describe : α → String

instance : Describe Bool where
  describe b := if b then "yes" else "no"

instance : Describe Nat where
  describe n := s!"the number {n}"            -- s!"..." is string interpolation

def announce [Describe α] (x : α) : String :=  -- [..] = instance-implicit argument
  s!"got {Describe.describe x}"

#eval announce true                            -- "got yes"
```

`[Describe α]` is an **instance-implicit**: Lean synthesizes it by instance search, not from the call site.

### Namespaces and modules

```lean
namespace Geometry
  def area (w h : Float) : Float := w * h
end Geometry

#eval Geometry.area 3 4
open Geometry in
#eval area 3 4                                 -- `open` brings names into scope
```

Each file is a module; `import Foo.Bar` pulls in `Foo/Bar.lean`. No separate interface files (unlike OCaml/SML).

### Implicit arguments and inference

```lean
def identity {α : Type} (x : α) : α := x       -- {α} is implicit, inferred from x
#eval identity 5                               -- α := Nat, inferred
#eval @identity Nat 5                          -- @ makes all implicits explicit
```

Three binder kinds: `(x : α)` explicit, `{x : α}` implicit (inferred by unification), `[x : α]` instance-implicit (inferred by type-class search).

---

## 3. Lean as a proof assistant

### Anatomy of a proof obligation

A **goal** is a target type to inhabit, under a **context** of hypotheses. Lean displays it as `hypotheses ⊢ goal`:

```
P Q : Prop
h : P ∧ Q
⊢ Q ∧ P
```

`theorem` names a proof; `example` is an anonymous theorem (a typechecked assertion you don't reuse). Both demand a term of the stated type.

### Term style vs tactic style

Two ways to supply the proof term:

```lean
-- term style: write the proof term directly
theorem and_swap₁ (h : P ∧ Q) : Q ∧ P := ⟨h.2, h.1⟩

-- tactic style: `by` runs a metaprogram that builds the same term
theorem and_swap₂ (h : P ∧ Q) : Q ∧ P := by
  constructor       -- split goal Q ∧ P into two goals: ⊢ Q and ⊢ P
  · exact h.2        -- · focuses one goal
  · exact h.1
```

They produce identical kernel terms. Term style is crisp for small structural proofs; tactic style scales to large goals where you manipulate state incrementally. Most real proofs are tactic style.

### The essential tactics

- **`rfl`** — closes `a = b` when both sides reduce to the same normal form _definitionally_.
- **`exact e`** — the goal is solved by exactly the term `e`.
- **`apply f`** — unify goal with `f`'s conclusion; spawn subgoals for `f`'s arguments. Backward reasoning.
- **`intro h`** — for goal `A → B` (or `∀ x, ...`), move `A` into the context as `h`, leaving `⊢ B`.
- **`rw [h]`** — rewrite occurrences of `h`'s LHS with its RHS in the goal (`rw [← h]` for the reverse). Auto-closes if the result is `rfl`.
- **`simp`** — repeatedly rewrite with the `@[simp]` lemma set plus anything you pass: `simp [foo, bar]`.
- **`cases h`** — case-split an inductive hypothesis/value into one goal per constructor.
- **`induction n`** — like `cases` but supplies an induction hypothesis in the recursive cases.
- **`calc`** — chained equational/relational reasoning, readable like a paper proof.

### Minimal proofs across logical forms

```lean
-- equality by computation
example : 2 + 2 = 4 := rfl

-- implication: introduce the antecedent
example (P : Prop) : P → P := by intro h; exact h

-- conjunction: build / destructure
example (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := ⟨hp, hq⟩
example (P Q : Prop) (h : P ∧ Q) : P := h.1

-- disjunction: provide one side / case-split
example (P Q : Prop) (hp : P) : P ∨ Q := Or.inl hp
example (P Q : Prop) (h : P ∨ Q) : Q ∨ P := by
  cases h with
  | inl hp => exact Or.inr hp
  | inr hq => exact Or.inl hq

-- universal quantification: it's just a (dependent) function
example : ∀ n : Nat, n + 0 = n := fun n => rfl

-- rewriting with a hypothesis
example (a b : Nat) (h : a = b) : a + 1 = b + 1 := by rw [h]

-- calc for a readable chain
example (a b c : Nat) (h₁ : a = b) (h₂ : b = c) : a = c :=
  calc a = b := h₁
    _    = c := h₂
```

### Induction over `Nat`

`Nat.add` recurses on its _second_ argument, so `n + 0 = n` holds by `rfl`, but `0 + n = n` does not — it needs induction:

```lean
theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl                          -- base: 0 + 0 = 0
  | succ k ih => rw [Nat.add_succ, ih]   -- step: 0 + (k+1) = (0+k)+1 = k+1
```

This asymmetry — one direction free, the other requiring work — is the single most common surprise for newcomers. _Which_ argument a recursive definition pattern-matches on determines what's definitionally true.

---

## 4. Type system essentials

### Universes and the `Prop`/`Type` split

Every type lives in a universe. `Nat : Type`, `Type : Type 1`, `Type 1 : Type 2`, … (stratified to avoid Girard's paradox). `Prop` is the universe of propositions: `Prop : Type`, and any `P : Prop` is a proposition whose inhabitants are proofs.

```lean
#check Nat        -- Type
#check Type       -- Type 1
#check (2 = 2)    -- Prop
#check Prop       -- Type
```

The split is semantic, not cosmetic:

- **`Type`** classifies **data** — things you compute with and extract from. `Nat`, `List α`, `Point`.
- **`Prop`** classifies **propositions** — things you prove. `2 = 2`, `n ≤ m`, `Sorted xs`.

**Proof irrelevance**: any two proofs of the same `p : Prop` are _definitionally equal_. The kernel treats them as interchangeable. This is why proofs carry no runtime cost and are erased during compilation — and why you cannot pattern-match on a `Prop` to recover _which_ proof you had. Data in `Type` is not proof-irrelevant.

### Dependent functions and pairs

A dependent function's return _type_ depends on the argument _value_:

```lean
def replicate (n : Nat) (x : α) : List α :=    -- ordinary
  match n with | 0 => [] | k+1 => x :: replicate k x

-- dependent: the result type mentions n
-- (Vector lives in Mathlib/Batteries; shown for shape)
-- def vreplicate (n : Nat) (x : α) : Vector α n := ...
```

A **dependent pair** (`Σ`, or `Exists` in `Prop`) packages a value with a proof about it. `∃ x, P x` is exactly such a pair where the second component is a proof:

```lean
example : ∃ n : Nat, n > 3 := ⟨4, by decide⟩   -- witness 4, then prove 4 > 3
```

### Coercions, named/implicit args, notation

```lean
-- coercion: Nat ↪ Int inserted automatically where an Int is expected
example (n : Nat) : Int := n                    -- ↑n inserted silently

-- named arguments, like Python keywords
def rect (width height : Float) : Float := width * height
#eval rect (height := 2) (width := 5)

-- notation: define your own infix
infixl:65 " ⊕ " => Nat.add
#eval 3 ⊕ 4                                      -- 7
```

`:65` is precedence. Most operators (`+`, `*`, `=`, `≤`) are notation resolving through type classes, so they work on any type with the right instance.

### Reading error messages

The two errors you will hit constantly:

```lean
-- 1. unification failure
example : Nat := "hello"
-- type mismatch: expected Nat, got String
-- Lean tried to make `String` unify with `Nat` and couldn't.

-- 2. missing instance
-- #eval (fun x => x) = (fun x => x)
-- failed to synthesize  Decidable (... = ...)
-- No `Decidable`/`BEq` instance exists for function equality.
```

Read errors as: _"during elaboration I needed type `X` here, your term gives `Y`, and unification of `X` with `Y` failed"_, or _"I needed an instance `C τ` and instance search found none"_. The fix is almost always at the elaboration boundary, not the kernel.

---

## 5. Mathematical workflow

### Reading a goal state

After each tactic, re-read context-above, goal-below:

```
case succ
k : Nat
ih : 0 + k = k          -- the induction hypothesis you were handed
⊢ 0 + (k + 1) = k + 1   -- what remains
```

The `case` tag tells you which constructor branch you're in. The horizontal line separates _what you have_ from _what you owe_.

### Discovering lemmas

You rarely invent lemmas; you find existing ones. Conceptually:

- **`exact?`** — searches for a single lemma closing the current goal.
- **`apply?`** — suggests lemmas whose conclusion unifies with the goal.
- **`rw?`** — suggests rewrites applicable to the goal.
- **`simp?`** — runs `simp` and prints the exact lemma set it used (so you can make it reproducible).
- **Naming convention** — Mathlib names _describe the statement_: `zero_add`, `add_comm`, `mul_one`, `List.length_append`, `Nat.succ_le_succ`. Guessing the name from the shape of your goal usually works.

### Decomposition into lemmas

A proof that won't yield directly is usually several lemmas wearing a trenchcoat. Extract the sub-facts:

```lean
-- instead of one tangled induction, prove the helper first
theorem add_comm_helper (n m : Nat) : n + (m + 1) = (n + m) + 1 := rfl
-- then use it where the big proof needs that step
```

Small named lemmas are reusable, individually debuggable, and `simp`-taggable.

### `simp` and rewriting as the core engine

The dominant workflow is: state the goal, then drive it to `rfl` by rewriting. `rw` for surgical single steps you control; `simp` for "normalize using everything known." `simp [h₁, h₂]` augments the default set with your hypotheses. The skill is curating _which_ lemmas `simp` may use — too few and it stalls, too many and it loops or rewrites the wrong way.

### What Mathlib changes

Core Lean ships minimal mathematics. **Mathlib** is the community library: a unified algebraic hierarchy (`Monoid`, `Group`, `Ring`, `Field`, topological spaces, measure theory, …), tens of thousands of lemmas, and heavy tactics (`ring`, `linarith`, `polyrith`, `field_simp`, `positivity`). Practical Lean _is_ Lean-plus-Mathlib: you spend most effort locating the right existing lemma and stitching it together, not proving from first principles. Without Mathlib you re-derive arithmetic; with it you `exact?` your way through standard goals.

---

## 6. Functional programming workflow

### Total functions

Every function must be **total**: defined on all inputs, terminating on all inputs. There is no `null`, no uncaught exception in pure code, no infinite loop in a `def` the termination checker accepts. Partiality is made explicit in the type:

```lean
def safeDiv (a b : Nat) : Option Nat :=
  if b == 0 then none else some (a / b)
```

### Structural recursion and termination

Recursion on a strictly smaller piece of an inductive value is accepted automatically:

```lean
def reverse : List α → List α
  | []      => []
  | x :: xs => reverse xs ++ [x]      -- xs is structurally smaller than x :: xs ✓
```

Non-structural recursion needs a justification. Either supply a decreasing measure or use `termination_by` / `decreasing_by`:

```lean
def gcd : Nat → Nat → Nat
  | 0, y => y
  | x+1, y => gcd (y % (x+1)) (x+1)
  termination_by x _ => x             -- prove the first argument decreases
```

If you genuinely need possibly-nonterminating code, `partial def` opts out of the checker — but such functions are opaque to proofs.

### Separating executable code from specifications

A Lean idiom: write the efficient program, write a clear (possibly slow) specification, prove they agree.

```lean
def fastReverse (xs : List α) : List α :=     -- tail-recursive, efficient
  let rec go acc | [] => acc | x :: t => go (x :: acc) t
  go [] xs

-- `reverse` (from above) is the obvious spec; one then proves:
-- theorem fastReverse_eq (xs : List α) : fastReverse xs = reverse xs := ...
```

Now you ship `fastReverse` and trust it because it's proven equal to the readable `reverse`.

### Tests/examples vs proofs

Two levels of assurance, both useful:

```lean
-- a test: checked once, at this input, at compile time
example : reverse [1, 2, 3] = [3, 2, 1] := rfl
#guard reverse [1, 2, 3] = [3, 2, 1]          -- #guard fails the build if false

-- a proof: holds for ALL inputs
theorem reverse_reverse (xs : List α) : reverse (reverse xs) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [reverse, ih]
```

Use `#guard`/`example`-by-`rfl` for cheap spot checks during development; promote to a `theorem` when you need the universal guarantee.

---

## 7. Worked path

A single progression, each step building on the last.

**(a) Define functions.**

```lean
namespace Demo

def sum : List Nat → Nat
  | []      => 0
  | x :: xs => x + sum xs

def double (n : Nat) : Nat := n * 2
```

**(b) Prove a simple equality** (closes by computation):

```lean
example : sum [1, 2, 3] = 6 := rfl
example : double 21 = 42 := rfl
```

**(c) Prove a property by rewriting** (use a hypothesis to transform the goal):

```lean
example (n : Nat) (h : n = 3) : double n = 6 := by
  rw [h]                       -- goal becomes double 3 = 6, then rfl auto-closes
```

**(d) Prove a property by induction** (the `sum` of a doubled list is double the sum):

```lean
def doubleAll : List Nat → List Nat
  | []      => []
  | x :: xs => double x :: doubleAll xs

theorem sum_doubleAll (xs : List Nat) : sum (doubleAll xs) = double (sum xs) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [doubleAll, sum, double, ih]
    ring                        -- 2*x + 2*(sum xs) = 2*(x + sum xs); ring is Mathlib
    -- without Mathlib: `omega` also closes this linear-ish Nat goal
```

**(e) Define a small custom datatype.**

```lean
inductive Tree (α : Type) where
  | leaf
  | node (l : Tree α) (v : α) (r : Tree α)

def Tree.mirror : Tree α → Tree α
  | .leaf       => .leaf
  | .node l v r => .node r.mirror v l.mirror

def Tree.size : Tree α → Nat
  | .leaf       => 0
  | .node l _ r => l.size + 1 + r.size
```

**(f) Prove a property about it** — mirroring is an involution, and preserves size:

```lean
theorem Tree.mirror_mirror (t : Tree α) : t.mirror.mirror = t := by
  induction t with
  | leaf => rfl
  | node l v r ihl ihr =>
    simp [Tree.mirror, ihl, ihr]   -- two induction hypotheses, one per subtree

theorem Tree.size_mirror (t : Tree α) : t.mirror.size = t.size := by
  induction t with
  | leaf => rfl
  | node l v r ihl ihr =>
    simp only [Tree.mirror, Tree.size, ihl, ihr]
    omega                          -- commute the Nat additions

end Demo
```

Note the recursive `node` case supplies _one induction hypothesis per recursive field_ (`ihl`, `ihr`) — directly mirroring the constructor's structure.

---

## 8. Practical heuristics

### Beginner failure modes for experienced engineers

- **Expecting `rfl` to do real math.** `rfl` only sees _definitional_ equality. `0 + n = n` is not `rfl` even though it's "obviously" true — it needs induction. Check which argument your definition recurses on.
- **Reaching for `simp` first, always.** `simp` is a sledgehammer; when it fails it fails opaquely. Prefer `rw` with named lemmas while learning, and use `simp?` to see what `simp` actually did.
- **Confusing `Prop` and `Bool`.** `Decidable`-ness, `decide`, `==` (Bool) vs `=` (Prop) trip people up. `if` needs a `Decidable` proposition; `==` needs `BEq`.
- **Fighting implicit arguments.** When unification "can't figure out `α`," add a type ascription `(x : T)` or go explicit with `@`.
- **Over-engineering data structures** before the proofs are in. Definitions chosen for runtime elegance can be miserable to reason about; sometimes the provable definition and the fast definition should be _different functions you prove equal_ (§6).
- **Treating tactic state as imperative mutation.** Each tactic rewrites a goal; there are no side effects. If a tactic "did nothing," your model of the goal is wrong — re-read the state.

### When Lean rejects a program or proof

1. **Read the error at face value**, starting from "expected X, got Y." The mismatch type pair localizes the bug.
2. **Reduce to the smallest failing `example`.** Strip context until it's minimal; the cause usually becomes obvious.
3. **Inspect the goal with `#check`**, place `sorry` to confirm the _rest_ type-checks, and bisect.
4. **For a stuck proof**, ask whether the statement is even true at the base/edge cases. Often the proof is fine and the _lemma is false as stated_.

`sorry` is your friend: it's an admitted hole that typechecks. Stub a goal with `sorry`, confirm the surrounding structure is right, then fill it in. Lean warns on every `sorry`, so none ship silently.

### Tactics vs helper lemmas vs simpler definitions

- **Use tactics** for the local, mechanical grind: introduce, case-split, rewrite, normalize.
- **Write a helper lemma** when (a) a fact recurs, (b) a goal is too big to hold in your head, or (c) you want it as a `@[simp]` rewrite rule for later automation.
- **Simplify the definition** when proofs about it are persistently painful. A definition that recurses on the "wrong" argument, or bundles too much, makes every downstream proof harder. Changing the definition is often cheaper than out-proving a bad one.

### What to learn next

- **Mathlib tactics**: `ring`, `linarith`, `omega`, `field_simp`, `positivity`, `gcongr` — they collapse routine goals.
- **The Mathlib algebraic hierarchy** and naming conventions, so `exact?`/`apply?` become reliable.
- **`Decidable` and `DecidableEq`**: the bridge between `Prop` proofs and `Bool` computation.
- **Well-founded recursion** (`termination_by`, `decreasing_by`) for non-structural functions.
- **Metaprogramming / custom tactics and `macro`s** once you're writing repetitive proofs.
- Canonical next reads: _Theorem Proving in Lean 4_, _Functional Programming in Lean_, and _Mathematics in Lean_.

---

## Cheat sheet

### Declarations

```lean
def f (x : T) : U := ...          -- function / value
theorem name (h : P) : Q := ...   -- named proof
example : P := ...                -- anonymous, typechecked assertion
inductive T where | c₁ | c₂ ...   -- sum type / data
structure S where field : T       -- product / record
class C (α : Type) where m : ...  -- type class
instance : C T where m := ...     -- instance
abbrev N := T                     -- transparent alias
```

### Binders

```lean
(x : T)    -- explicit
{x : T}    -- implicit (inferred by unification)
[x : T]    -- instance-implicit (inferred by type-class search)
@f         -- make all implicits explicit
(x := v)   -- named argument
```

### Terms

```lean
fun x => e            -- lambda
let x := e; body      -- local binding
match e with | p => r -- pattern match
⟨a, b⟩                 -- anonymous constructor (structures, And, Exists, Σ)
h.1  h.2  h.left      -- projections
x :: xs   []   [a,b]  -- list cons / nil / literal
s!"text {x}"          -- string interpolation
```

### Core tactics

```lean
rfl                 -- definitional equality
exact e             -- close goal with term e
apply f             -- backward-chain through f, spawn subgoals
intro h             -- move ∀/→ antecedent into context
rw [h]              -- rewrite L→R (rw [← h] for R→L); auto-rfl
simp [lemmas]       -- normalize via simp-set + extras
simp only [lemmas]  -- normalize via ONLY the listed lemmas
cases h with ...    -- case-split an inductive
induction n with    -- induction + induction hypothesis
  | zero => ...
  | succ k ih => ...
constructor         -- apply the goal type's constructor
calc a = b := h₁    -- chained equational reasoning
  _   = c := h₂
omega               -- linear arithmetic over Nat/Int
ring / linarith     -- (Mathlib) algebraic / linear-order goals
decide              -- close a Decidable proposition by computation
sorry               -- admit (typechecks, warns)
```

### Search & inspect

```lean
#check e            -- show the type of e
#eval e             -- compile and run e
#guard e            -- build fails unless e reduces to true
exact?  apply?  rw?  simp?   -- lemma / rewrite discovery
```

### Naming intuition (Mathlib)

`zero_add`, `add_zero`, `add_comm`, `mul_one`, `List.length_append`, `Nat.succ_le_succ` — names spell out the statement; guess from the goal's shape.
