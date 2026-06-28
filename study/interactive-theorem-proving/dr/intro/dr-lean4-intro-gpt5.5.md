# Lean 4 for Proofs and Programs

## https://chatgpt.com/c/6a3f86a1-df80-83ec-ad98-db00750f34af

--

## Mental model

Lean 4 is simultaneously an interactive theorem prover, a dependently typed functional programming language, and a compiler toolchain. The official reference describes Lean as an interactive theorem prover based on dependent type theory; the Lean 4 system paper describes it as an extensible theorem prover and efficient functional programming language; the repository itself is explicitly “the repository for Lean 4” and links the theorem-proving tutorial, functional-programming book, and language reference as first-class documentation. citeturn6view0turn22view1turn5view0turn0search0

The core idea is propositions-as-types and proofs-as-programs. A proposition is a term of type `Prop`; a proof of proposition `p` is just a term `t : p`. Implication is a function type, so a proof of `p → q → p` is literally a function that takes a proof of `p`, a proof of `q`, and returns the first one. Lean’s proof language is not a separate logic bolted onto programming syntax; it is the same dependent type theory used for ordinary definitions. citeturn8view0turn19view0

```lean
example (p q : Prop) : p → q → p :=
  fun hp hq => hp
```

To work effectively, separate five layers in your head. **Computation** is reduction or evaluation of terms. **Type checking** verifies that a term inhabits a type, using definitional equality. **Elaboration** turns surface syntax into core terms, filling in omitted arguments, expanding macros, inserting coercions, and running type-class search. **Tactics** are proof-state programs that construct proof terms. **Kernel verification** is the final trusted check that the elaborated proof term obeys the core rules of the type theory. Lean’s reference is explicit that elaboration transforms user-facing syntax into a simpler core theory, tactics construct proof terms behind the scenes, and the kernel checks those terms. citeturn22view0turn6view1turn19view0

The small trusted kernel matters because most of the pleasant tooling is _not_ trusted. Macros, elaborators, tactics, the equation compiler, and recursive-definition machinery can all be buggy; the design goal is that bugs there should cause rejected proofs or confusing diagnostics, not accepted false theorems. The reference and system paper both emphasise a minimal kernel and independent validation paths, and the theorem-proving text notes that pattern matching and recursive proofs are compiled down to primitive recursors outside the trusted code base and then checked by the kernel. citeturn6view0turn19view0turn10search5turn10search6

## Lean as a functional language

Lean programs are built from `def`, `inductive`, `structure`, `class`, `instance`, `match`, `let`, lambdas, and namespaces. Definition-like commands include `def`, `example`, `theorem`, and `opaque`; they elaborate a term against a signature, and—except for `example`, which is discarded—save the resulting core expression in the environment. Inductive types are Lean’s primary mechanism for introducing data; structures are a special case of inductive types with exactly one constructor. citeturn9view4turn2search7turn13view0

```lean
def inc (n : Nat) : Nat := n + 1

def head? {α : Type} : List α → Option α
  | []      => none
  | x :: _  => some x

def sum : List Nat → Nat
  | []      => 0
  | x :: xs => x + sum xs

def mapOption {α β : Type} (f : α → β) : Option α → Option β
  | none   => none
  | some x => some (f x)

def compose {α β γ : Type} (f : β → γ) (g : α → β) : α → γ :=
  fun x =>
    let y := g x
    f y
```

If you know Python, the largest shift is that Lean is expression-oriented, immutable by default, and total by default. If you know Scala or an ML-family language, Lean’s `inductive` declarations feel closer to algebraic data types or `enum`/sealed-sum encodings than to OO classes. Pattern matching is the standard way to consume inductive data, and recursive functions are expected to be structurally recursive unless you provide a termination argument. Lean also enforces exhaustiveness of `match`, which rules out silent fall-through partial functions. citeturn13view0turn13view1turn21view0

```lean
inductive Tree (α : Type) where
  | leaf
  | node (left : Tree α) (value : α) (right : Tree α)

structure User where
  name    : String
  retries : Nat

def defaultUser : User :=
  { name := "alice", retries := 3 }

class Size (α : Type) where
  size : α → Nat

instance : Size User where
  size u := u.retries

def retryBudget {α : Type} [Size α] (x : α) : Nat :=
  Size.size x
```

`structure` gives you product types with generated projections; `class` participates in type-class inference; `instance` declarations are syntactically almost the same as definitions. In practice, `class` plus `instance` is Lean’s ad-hoc polymorphism mechanism, roughly analogous to Scala type classes or `given`/implicit search, not Python duck typing. The reference explains that type classes are collections of overloaded operations, instance declarations are definition-like, and instance synthesis fills square-bracket parameters. citeturn6view4turn6view5turn9view1turn9view2turn18search5

```lean
def id' {α : Type} (x : α) : α := x

def three  : Nat    := id' 3
def greet  : String := id' (α := String) "lean"
```

Implicit arguments are written with `{...}`; instance-implicit arguments are written with `[...]`; named arguments use `x := ...` syntax at call sites. The elaborator creates metavariables for omitted implicit arguments and schedules instance-implicit arguments for type-class synthesis. This means you should read a signature like `def f {α} [C α] (x : α) := ...` as “`α` will be inferred; evidence for `C α` will be synthesised”. citeturn6view7turn6view8

Lean organises APIs with namespaces and source-file imports. A source file is the smallest compilation unit; imports use dotted module names derived from paths; importing a file does **not** automatically open its namespaces. Namespaces are hierarchical and are the primary organisation mechanism for APIs. citeturn24view0turn24view1turn6view12

```lean
import MyLib.Data.Util

namespace Demo

def value : Nat := 42

end Demo
```

## Lean as a proof assistant

A theorem statement is a type to inhabit. The local context contains variables and hypotheses already available; the current goal is the target type still to be constructed. In tactic mode, Lean shows a proof state: an ordered sequence of goals, each consisting of a context plus a target. `example` checks a theorem or term and discards it; `theorem` stores it in the environment for later use. citeturn19view0turn9view4

```lean
example : 2 + 2 = 4 := by
  rfl

example (p q : Prop) : p → q → p := by
  intro hp hq
  exact hp

example (p q : Prop) : p ∧ q → q ∧ p := by
  intro hpq
  cases hpq with
  | intro hp hq =>
      exact And.intro hq hp

example : ∀ n : Nat, n = n := by
  intro n
  rfl

example (x y z : Nat) (h₁ : x = y) (h₂ : y = z) : x = z := by
  calc
    x = y := h₁
    _ = z := h₂
```

The core tactics you need first are small and compositional. `rfl` closes reflexive goals. `exact e` closes a goal if `e` already has the target type. `apply e` matches the goal against the conclusion of `e` and creates subgoals for its premises. `intro` moves a binder or implication premise from the goal into the context. `cases` performs case analysis on an inductive hypothesis. `induction` applies an induction principle and gives you induction hypotheses. `rw` rewrites using equalities. `simp` simplifies using `[simp]` lemmas, supplied rules, and optionally hypotheses. `calc` is stepwise transitive reasoning. citeturn25view0turn20view0turn25view4turn25view1turn25view2turn25view3

Tactic mode and term mode are interchangeable views of the same result. Term mode writes the proof term directly, often with `fun`, `match`, `have`, and `show`; tactic mode is imperative proof-state manipulation that _constructs_ that term. The reference is explicit that tactics are a special-purpose proof language and that each goal corresponds to an incomplete portion of a proof term. For short structural proofs, term mode is often clearer; for case splits and induction, tactic mode is usually faster to write and easier to debug. citeturn19view0turn8view1

```lean
example (p q : Prop) : p → q → p :=
  fun hp hq => hp

theorem zero_add' (n : Nat) : 0 + n = n := by
  induction n with
  | zero =>
      rfl
  | succ n ih =>
      rw [Nat.add_succ, ih]
```

## Type system essentials

Lean has a hierarchy of universes. `Prop` is `Sort 0`; data lives in `Type u`, notation for `Sort (u + 1)`. The important operational distinction is not merely “logic versus data”, but also that propositions are proof-irrelevant and run-time irrelevant: any two proofs of the same proposition are interchangeable, and proofs are erased from compiled code. Also, all function types in the core language are dependent; ordinary arrows are just the special case where the result type does not mention the argument. citeturn9view3turn6view2turn6view3turn8view0

That is why the same syntax works for programs and proofs. `Nat → Nat` is an ordinary function type; `(n : Nat) → Fin (n + 1)` is a dependent function type whose codomain depends on the input value. Dependent pairs (`Sigma`, written `Σ`) package a value together with indexed data; subtypes package a value together with a proof of a predicate, and the proof component is erased at run time. You do not need dependent pairs immediately, but you should recognise them when a value determines the type of accompanying data. citeturn23search0turn23search1

Elaboration handles many conveniences that are _not_ part of the kernel’s core theory: implicit parameters, named arguments, type-class synthesis, notation, and coercions. Coercions are inserted when the elaborator has constructed a term of one type in a context expecting another and can synthesise a suitable `CoeT` chain. Notation is likewise surface syntax translated during elaboration. This is why “what Lean parses” and “what the kernel checks” are related but distinct questions. citeturn6view6turn6view7turn6view8turn3search2turn22view0

When Lean rejects code or a proof, the failure is usually in one of four buckets. A **type mismatch/application type mismatch** usually means unification could not make inferred and expected types definitionally equal. **Don’t know how to synthesise implicit argument** means ordinary implicit inference lacked enough information. **Failed to synthesise instance** means instance search for a square-bracket parameter failed. **Failed to infer structural recursion / termination** means the recursive-definition checker could not justify the recursion pattern. These are different failure modes and need different fixes. citeturn26search3turn18search5turn18search1turn21view0turn21view2

## Working effectively

For proofs, read the goal state literally. Everything above `⊢` is in scope; everything after `⊢` is the type you still need to build. From there, the default workflow is: introduce binders with `intro`; normalise with `simp`; rewrite with `rw`; split cases with `cases`; use `induction` when the object was defined inductively; and use `exact` or `apply` when you already know the relevant lemma or constructor. Lean also ships search-oriented helpers such as `exact?`, `apply?`, and `rw?`, and the community recommends API docs and Loogle for declaration search. citeturn19view0turn20view0turn11search2turn17view1

In practice, `simp` plus rewriting is the centre of day-to-day proof automation. `simp` uses lemmas tagged `[simp]` and optional local rules; `rw` applies specific equalities in a controlled order. Most beginner proofs become much shorter once the right helper lemma is stated, marked for simplification where appropriate, or passed explicitly to `simp [lemma₁, lemma₂]`. This is the standard Lean style because it scales from tiny toy proofs to library developments. citeturn20view0turn6view10turn15search9

Mathlib changes practical Lean development from “prove everything from first principles” to “assemble the right existing abstractions and lemmas”. The community site describes Mathlib as a unified, community-driven library of formalised mathematics that also contains definitions useful for programming, and the documentation stack includes searchable API docs for Mathlib, `Std`, `Batteries`, and even core Lean/compiler modules. Real Lean work therefore depends as much on library navigation as on raw tactic knowledge. citeturn17view0turn17view1turn17view2

For functional programming, keep executable code in `Type`, specifications in `Prop`, and do not mix them prematurely. Lean’s default is total code: `match` must be exhaustive, and recursive definitions must be justified by structural or other accepted termination arguments. For lightweight feedback, `#eval` compiles and evaluates expressions; `example` declarations are good for tiny executable fragments or proof obligations that you want checked but not named. Then prove semantic properties separately with `theorem`. citeturn13view1turn21view0turn21view1turn26search3turn6view2

The most common failure mode for experienced engineers is assuming Lean will “obviously” perform a semantic transformation that is not definitional equality. If two terms only become equal _after_ a theorem is applied, you need `rw`, `simp`, `change`, or a helper lemma; the kernel will not guess that for you. The next most common mistake is hiding information from elaboration: omitted types, an unsolved implicit parameter, or a missing instance looks like a logic failure but is often just an elaboration failure. When in doubt, make arguments explicit, state one helper lemma, and rerun. citeturn18search4turn20view0turn18search5turn18search1

After this introduction, the most useful next texts are the official _Theorem Proving in Lean 4_ for proof language and foundations, _Functional Programming in Lean_ for programming idioms, and _Mathematics in Lean_ once you start using Mathlib seriously. Keep the language reference and API docs open while you work; for non-trivial developments, they are not optional. citeturn17view1turn5view0turn17view0

## Worked path

The shortest path from “I can read Lean” to “I can write small programs and proofs” is to alternate between a definition and a property of that definition. The sequence below goes from plain functions, to reflexive computation proofs, to rewriting from a hypothesis, to induction on naturals, to a custom datatype and an induction proof over it. The tactics used are the standard ones introduced above: `rfl`, `rw`, `induction`, `simp`, and `exact`. citeturn25view0turn20view0turn25view2turn25view1

```lean
def double (n : Nat) : Nat := n + n

def headOr {α : Type} (fallback : α) : List α → α
  | []      => fallback
  | x :: _  => x

example : double 3 = 6 := by
  rfl

example (a b c : Nat) (h : a = b) : a + c = b + c := by
  rw [h]

theorem zero_add' (n : Nat) : 0 + n = n := by
  induction n with
  | zero =>
      rfl
  | succ n ih =>
      rw [Nat.add_succ, ih]

inductive Tree (α : Type) where
  | leaf
  | node (left : Tree α) (value : α) (right : Tree α)

def mirror {α : Type} : Tree α → Tree α
  | .leaf => .leaf
  | .node l x r => .node (mirror r) x (mirror l)

theorem mirror_mirror {α : Type} (t : Tree α) : mirror (mirror t) = t := by
  induction t with
  | leaf =>
      rfl
  | node l x r ihL ihR =>
      simp [mirror, ihL, ihR]
```

Read what is going on operationally. `double 3 = 6` is closed by computation, so `rfl` works. The second theorem works because `rw [h]` rewrites the goal using a hypothesis in the local context. `zero_add'` is not definitionally true by reduction on the second argument of addition, so it needs induction. `mirror_mirror` is the canonical custom-datatype pattern: define a structurally recursive function, then prove its involutive property by induction with one induction hypothesis per recursive field. citeturn6view10turn20view0turn25view2

## Cheat sheet

Use this as the minimal recall set while writing your first files. The syntax items below correspond directly to Lean’s definition, function, inductive-type, namespace, implicit-argument, and function-application machinery. citeturn9view4turn13view0turn24view1turn6view7turn6view8

- `def f (x : α) : β := term` — define a function or constant. citeturn9view4
- `inductive T where | c1 ... | c2 ...` — define a sum type or recursive datatype. citeturn10search10turn13view0
- `structure S where field₁ : A ...` — define a record/product type with projections. citeturn6view4
- `class C (α : Type) where ...` / `instance : C T where ...` — define and populate a type class. citeturn6view5turn9view1
- `match x with | ... => ...` — case analysis and data destruction. citeturn13view0
- `fun x => body` — lambda abstraction; proofs of implications and `∀` often look like this. citeturn8view1turn18search5
- `{α : Type}` — implicit argument; `[C α]` — instance-implicit argument. citeturn6view7turn6view8
- `(α := Nat)` — pass a named implicit argument explicitly. citeturn6view8
- `namespace N ... end N` — qualify related declarations under `N`. citeturn24view1
- `import A.B.C` — import a module by dotted name. citeturn24view0

For proofs, the following tactics are the smallest useful set. Their intended use is standardised in the reference manual and theorem-proving tutorial. citeturn19view0turn20view0turn25view0turn25view1turn25view2turn25view3turn25view4

- `rfl` — close a reflexive goal, often after computation. citeturn25view0
- `intro x` — move one binder or premise into context. citeturn25view4
- `exact t` — finish the goal with a term already of the right type. citeturn20view0
- `apply f` — reduce the goal to the premises of `f`. citeturn20view0
- `rw [h, ← lemma]` — rewrite using equalities. citeturn20view0
- `simp [defs, lemmas]` — simplify using `[simp]` rules and supplied facts. citeturn20view0
- `cases h` — split by constructors of an inductive hypothesis. citeturn25view1
- `induction n with | ...` — prove by induction, receiving induction hypotheses. citeturn25view2
- `calc ...` — write chained equational or transitive reasoning. citeturn25view3
- `exact?`, `apply?`, `rw?` — interactive search aids; use the suggestion, do not leave them in the final proof. citeturn11search2

When diagnostics appear, parse them mechanically. “Application type mismatch” suggests failed unification; “don’t know how to synthesise implicit argument” suggests insufficient type information; “failed to synthesise instance” means missing type-class evidence; “failed to infer structural recursion” means Lean could not justify termination from the syntax of the recursive calls. citeturn26search3turn18search5turn18search1turn21view0
