# **Lean 4 for Systems and Verification Engineers: A Technical Guide**

https://gemini.google.com/app/f5fea363d37c331b

https://docs.google.com/document/d/1wDNs8f_RWc0KP_U8Yp9QRO5SW9cBCkpM8H3hbWd2KPI/edit?tab=t.0

---

Lean 4 is a self-hosted, dependently typed functional programming language and interactive theorem prover1. Unlike traditional general-purpose functional languages or pure proof checkers, Lean 4 is designed as a unified system where executable algorithms, logical specifications, and mathematical proofs co-exist and compile to optimized machine code via intermediate C1. The execution runtime utilizes reference counting with a destructive update mechanism called Functional-but-in-Place (FBIP) to run pure functional structures at native speed3. Soundness is guaranteed by a compact trusted kernel1.

## **Theoretical Foundation and Compilation Architecture**

The power of Lean 4 lies in its mathematical rigor and its highly performance-optimized compiler. The compilation and verification pipelines run in parallel to decouple logical soundness from optimization and usability1.

### **Compilation and Verification Pipelines**

Lean 4 processes source code through distinct structural phases, distinguishing between proof checking and executable generation:

1. **Elaboration:** User-facing syntax containing implicit arguments, coercions, and macro structures is translated into fully explicit, type-checked terms of the Calculus of Inductive Constructions (CIC)1. This translation solves higher-order unification and typeclass synthesis problems1.
2. **Kernel Verification:** The compiled proof terms are passed to a minimal, trusted logical kernel1. The kernel verifies type-correctness and term equivalence, maintaining absolute logical consistency1.
3. **Compilation:** Computationally relevant definitions are lowered to an intermediate representation, optimized, and compiled to C1. The C code is then compiled to machine code using standard compilers such as GCC or Clang1.

### **Architectural Division of Labor**

The system is constructed with strict separation between verified logical soundness and untrusted proof generation:

- **The Trusted Kernel:** Implemented in approximately 6,000 lines of C++ code, the kernel is the sole arbiter of logical truth1. It implements type-checking rules for universe polymorphism, inductive types, and proof irrelevance1. The kernel does not understand tactics, editor actions, or code compilation1.
- **The Elaborator:** Written in Lean 4, this component translates intuitive human declarations into explicit kernel terms1. It is highly complex and untrusted; any bug in the elaborator that produces structurally invalid logic will be rejected immediately by the trusted kernel during type-checking1.
- **The Tactic Engine:** A metaprogramming framework that constructs proof terms1. Tactics function like imperative code generators for proofs; they manipulate an incomplete proof state consisting of local contexts and target goals, generating raw lambda expressions that are verified by the kernel1.
- **The Compiler and Runtime:** Lower-level code generation that operates completely outside the logical kernel1. It leverages the FBIP paradigm to perform destructive in-place updates on uniquely referenced heap objects, reducing memory allocation and copy overheads3.

### **The Propositions-as-Types Paradigm**

Lean 4 is mathematically grounded in the Curry-Howard isomorphism, which maps mathematical logic to type theory1. Under this paradigm, a proposition is represented as a type, and a proof of that proposition is represented as a program that inhabits that type1. Proving a theorem is identical to writing a program that type-checks against the specified logical type2.

Lean  
\-- A logical implication P → Q is literally a function type.  
\-- A proof of the implication is a function that maps a proof of P to a proof of Q.  
theorem imply_transitivity {P Q R : Prop} (hpq : P → Q) (hqr : Q → R) : P → R :=  
 fun hp \=\> hqr (hpq hp)

During compilation, terms whose types live in Prop (propositions) are erased, leaving only the computationally relevant Type parameters for runtime execution7.

## **Lean as a Functional Programming Language**

Lean 4 supports algebraic data types, pattern matching, structures, recursive functions, type classes, and a robust module system4.

### **Syntax and Core Declarations**

In Lean, variables, algorithms, and computational parameters are defined using the def keyword, whereas types are introduced using inductive and structure13.

| Programming Concept      | Python                              | Scala                       | Lean 4                             |
| :----------------------- | :---------------------------------- | :-------------------------- | :--------------------------------- |
| **Variable Binding**     | x \= value                          | val x \= value              | let x := value \[cite: 17, 18\]    |
| **Constant Definition**  | Module-level variables              | val / object                | def \[cite: 13, 19\]               |
| **Tagged Unions (ADTs)** | Union\[T1, T2\] / Class hierarchies | sealed trait / Case classes | inductive \[cite: 14, 15\]         |
| **Product Records**      | dataclass / NamedTuple              | case class                  | structure \[cite: 14, 15\]         |
| **Ad-hoc Polymorphism**  | Protocol / Abstract classes         | Implicits / Typeclasses     | class and instance \[cite: 4, 16\] |
| **Logical Invariants**   | Runtime assert / Unit tests         | Types / Design by Contract  | Prop theorems verified by kernel2  |

### **Structural Representation and Type Classes**

Structures are product types with named fields that automatically generate projection functions14. Type classes allow for extensible ad-hoc polymorphism4.

Lean  
\-- Defining namespaces to avoid global identifier collision  
namespace SystemsModel

\-- A custom inductive option representation  
inductive CustomOption (α : Type) where  
 | none : CustomOption α  
 | some : α → CustomOption α

\-- A structure representing system state  
structure NodeState where  
 nodeId : Nat  
 isActive : Bool  
 payload : String

\-- Type class definition for converting types to JSON strings  
class ToJson (α : Type) where  
 toJson : α → String

\-- Instantiating the typeclass for NodeState  
instance : ToJson NodeState where  
 toJson state :=  
 s\!"\\{ \\"id\\": {state.nodeId}, \\"active\\": {state.isActive} \\}"

end SystemsModel

The standard library includes Nat (which compiles to optimized bignums via arbitrary-precision arithmetic libraries but logically behaves as inductive Peano numbers), List α, and Option α15.

## **Lean as an Interactive Theorem Prover**

Theorem proving in Lean occurs primarily inside tactic-mode blocks, denoted by the by keyword, where commands iteratively manipulate the logical environment10.

### **Proof State Mechanics**

In tactic mode, the compiler provides interactive feedback on the **Proof State** at the cursor position9. The state is divided into context assumptions and target goals10:

α : Type  
xs : List α  
h_empty : xs \= \[\]  
\--------------------  
⊢ xs.reverse \= \[\]

The context (above the turnstile ⊢) defines the current local assumptions10. The goal (below the turnstile ⊢) represents the type that must be inhabited to complete the proof9.

### **Tactic Mode versus Term-Style Proofs**

- **Term-Style:** Proofs are constructed as explicit functional lambda terms11. This style is concise for simple logical implications but rapidly becomes unreadable for complex derivations26.
- **Tactic Mode:** Proofs are constructed step-by-step using a sequence of goal-directed commands9. This mode provides real-time compiler feedback10.

### **Core Proof Tactics**

- intro: Binds universal quantifiers or moves the antecedent of an implication into the context10.
- exact: Solves the current goal if the provided term matches the target type exactly10.
- apply: Applies a function or implication backwards, creating subgoals for any missing parameters10.
- rw: Performs syntactic find-and-replace using an equality identity10.
- simp: Automates rewriting using a global simplifier database and local hypotheses10.
- rfl: Proves reflexive equalities (such as a \= a) by simplifying both sides computationally10.
- cases: Splits an inductive type into its constructor variants to perform case analysis24.
- induction: Generates proof obligations for the constructors of inductive types, providing an induction hypothesis for recursive cases10.
- calc: Groups transitive equality or inequality rewrites into structured, readable chains17.

Lean  
\-- A minimal proof illustrating multiple tactics  
theorem equality_and_implication (a b : Nat) (h : a \= b) : a \+ 0 \= b := by  
 rw \[Nat.add_zero\] \-- Rewrites "a \+ 0" to "a" in the goal  
 exact h \-- Goal is now "a \= b", which matches hypothesis h

## **Type System Essentials**

Lean's type system is based on a cumulative hierarchy of universes, dependent function types (![][image1]\-types), and dependent pair types (![][image2]\-types)1.

### **Universes and Proof Irrelevance**

To prevent logical contradictions, Lean uses a predicative hierarchy of universes19. The base level contains two distinct universes:

- Prop (Sort 0): The universe of mathematical propositions1.
- Type (Sort 1): The universe of computational data structures and standard types19.

The fundamental difference between these two universes is **Proof Irrelevance**1. If a type P has type Prop, any two elements p1 : P and p2 : P are definitionally equal within the kernel1. This enables aggressive proof erasure during compilation7.

### **Dependent Functions and Dependent Pairs**

A dependent function type (denoted (x : α) → β x or as a ![][image1]\-type) represents a function where the output _type_ depends on the input *value*6. A dependent pair (denoted (x : α) × β x or as a ![][image2]\-type) represents a structure where the type of the second element depends on the value of the first10.

Lean  
\-- A dependent function where the output type "Fin n" depends on the input value "n"  
def getFirstElementOfSize (n : Nat) (h : n \> 0\) : Fin n :=  
 ⟨0, h⟩

### **Extensible Coercions and Implicit Parameters**

Lean uses type classes to manage implicit conversions (coercions)32. Coercions are resolved automatically and shown in proof states via the up-arrow operator ↑33.

Lean  
\-- Coerces a boolean to a Prop (decidable proposition)  
instance : Coe Bool Prop where  
 coe b := b \= true

Implicit parameters are declared using curly braces {} and are solved using type unification13. Instance implicit parameters, declared using square brackets \[\], are synthesized via typeclass resolution4. Arguments can also be passed explicitly by name using the (param := value) syntax21.

### **Unification and Debugging Diagnostics**

Unification errors occur when Lean's elaborator cannot verify that an inferred type matches the expected type18. Missing typeclass instances generate errors such as failed to synthesize instance.  
These resolution failures can be debugged in-line by tracing typeclass synthesis:

Lean  
set_option trace.Meta.synthInstance true

## **Mathematical Verification Workflow**

Prover development involves translating mathematical concepts into precise code, interacting with the goal state, and utilizing automation tools1.

### **Navigating the Goal State**

When writing proofs, the developer places the cursor inside tactic blocks to inspect the goal state10. Hypotheses are manipulated using forward reasoning (generating new assertions from existing assumptions using tactics like have) or backward reasoning (modifying the goal using apply and refine)10.

### **Lemma Discovery and Proof Decomposition**

Large proofs are decomposed into smaller helper lemmas using the have keyword17. To discover existing lemmas in Lean's standard library and the mathematical library (**Mathlib**), developers use:

- **Loogle:** A query tool that searches by type signatures and patterns38.
- **exact?:** A tactic that searches the environment for a single matching lemma to solve the goal.

### **Automated Simplification and Mathlib Integration**

The simp tactic repeatedly applies rewrite rules annotated with the @\[simp\] attribute10. While rw performs strict syntactic matching39, simp normalizes terms to a canonical form10.  
Mathlib provides a highly unified algebraic hierarchy where physical dependencies are tightly integrated32. During proof resolution, Mathlib relies on automatically synthesized coercion paths and typeclass derivations32. Over 74% of dependency edges in Mathlib are synthesized implicitly, meaning proof success depends on correct typeclass resolution32.

## **Functional Programming Workflow and Totality**

Lean 4 establishes a mathematically consistent environment by enforcing **Totality** by default2.

### **The Totality and Termination Constraint**

If non-terminating (infinite) loops were allowed in the logic space, they could represent proofs of false statements, rendering the logic unsound40. Thus, every standard function must be proven to terminate on all inputs2.

### **Structural versus Well-Founded Recursion**

- **Structural Recursion:** The recursive step operates on a strict subcomponent of an inductive argument40. The compiler automatically translates this into type recursors, proving termination implicitly40.
- **Well-Founded Recursion:** For non-structural algorithms (such as binary search or gcd), the developer must provide an explicit measure using the termination_by clause41. Lean then generates proof obligations to show that this measure strictly decreases with each recursive call41.

Lean  
\-- Euclid's GCD requires well-founded recursion because arguments are not structurally smaller  
def gcd (n k : Nat) : Nat :=  
 if h : n \= 0 then  
 k  
 else if k \= 0 then  
 n  
 else if n \> k then  
 gcd (n \- k) k  
 else  
 gcd n (k \- n)  
termination_by n \+ k \-- The sum of both inputs serves as the decreasing measure \[cite: 42\]

To bypass totality constraints in non-logical contexts (such as network servers or external systems bindings), functions can be marked partial or unsafe41.

### **Executable Implementations versus Logic Specifications**

Functional verification in Lean follows a code-first approach:

1. **Implementation:** Write a pure functional program focused on performance2.
2. **Specification:** State theorems that define the logical behavior of the program2.
3. **Verification:** Prove that the program satisfies those theorems2.

The correctness of functional logic is verified through assertions verified at compile-time by the kernel, or evaluated interactively using \#eval13.

## **End-to-End Verification Worked Path**

This section presents a complete, compile-valid path that defines a custom datatype, implements helper functions, states properties, and verifies them using rewriting and induction.

Lean  
\-- 1\. Defining a custom algebraic datatype representing a linked list  
inductive CustomList (α : Type) where  
 | nil : CustomList α  
 | cons : α → CustomList α → CustomList α

\-- 2\. Implementing a structural recursive function to compute length  
def customLength {α : Type} (xs : CustomList α) : Nat :=  
 match xs with  
 | CustomList.nil \=\> 0  
 | CustomList.cons \_ ys \=\> customLength ys \+ 1

\-- 3\. Implementing list concatenation  
def customConcat {α : Type} (xs ys : CustomList α) : CustomList α :=  
 match xs with  
 | CustomList.nil \=\> ys  
 | CustomList.cons z zs \=\> CustomList.cons z (customConcat zs ys)

\-- 4\. Proving a simple equality definitionally using reflexivity  
theorem customLength_nil {α : Type} : customLength (CustomList.nil : CustomList α) \= 0 :=  
 rfl

\-- 5\. Proving a property of concatenation by definitional reduction  
theorem customConcat_nil_left {α : Type} (ys : CustomList α) :  
 customConcat CustomList.nil ys \= ys :=  
 rfl

\-- 6\. Proving a property by structural induction  
\-- We prove that appending nil to the right of a list leaves the list unchanged.  
theorem customConcat_nil_right {α : Type} (xs : CustomList α) :  
 customConcat xs CustomList.nil \= xs := by  
 induction xs with  
 | nil \=\>  
 \-- Base case: customConcat nil nil reduces definitionally to nil  
 rfl  
 | cons z zs ih \=\>  
 \-- Inductive step: goal is "customConcat (cons z zs) nil \= cons z zs"  
 \-- This reduces definitionally to "cons z (customConcat zs nil) \= cons z zs"  
 simp only \[customConcat\]  
 \-- The induction hypothesis is "ih : customConcat zs nil \= zs"  
 rw \[ih\]

## **Practical Systems Engineering Heuristics**

When applying Lean 4 to systems engineering, developers often encounter common conceptual roadblocks.

### **Resolving Common Beginner Failure Modes**

- **Syntactic vs. Definitional Equality in Rewrites:** The rw tactic operates via strict syntactic matching39. If a goal is computationally identical but syntactically different (e.g., trying to rewrite a \+ 0 using a lemma for 0 \+ a), rw will fail39.
  - _Heuristic:_ Use simp or dsimp to normalize the expressions to their common logical forms before rewriting, or use change to manually cast the goal to a definitionally equal syntax.
- **Alias Transparency and Typeclass Failures:** If a custom type alias is defined using def, Lean's typeclass engine will often fail to resolve basic traits (such as SizeOf or DecidableEq)35.
  - _Heuristic:_ Use abbrev or the @\[reducible\] attribute to define aliases35. This signals the elaborator to transparently unfold the definition during unification and typeclass resolution35.
- **Well-Founded Loop Resolution:** When a recursive function fails to compile because Lean cannot prove termination, the compiler generates complex logical helper goals containing invImage or sizeOf35.
  - _Heuristic:_ Use all_goals simp_wf inside a decreasing_by block to automatically simplify the generated relational obligations42.

### **Architectural Guidelines for Proof Engineering**

- **When to use Tactics:** Use tactics for inductive proof branches, non-trivial equalities, and complex search goals where automated strategies like omega or aesop can close goals10.
- **When to use Terms:** Use direct proof terms for simple logical mappings, function compositions, and computational helper lemmas17.
- **Proof Decomposition:** If a tactic proof becomes slow to compile or is deeply nested, extract intermediate steps into standalone private theorem helpers to optimize compilation time and maintain readability46.

### **Recommended Paths for Continued Mastery**

To transition from basic verification to industrial-scale proof engineering, developers should study:

1. _Functional Programming in Lean_ (FPIL): Focuses on monadic side effects, I/O, and programming with dependent types38.
2. _Theorem Proving in Lean 4_ (TPIL4): Focuses on the logical foundations of dependent type theory and natural deduction38.
3. _Mathematics in Lean_ (MIL): Introduces the design of Mathlib and advanced tactic usage38.

## **Syntax and Tactic Reference Cheat Sheet**

### **Core Syntax Keywords**

| Keyword        | Operational Purpose                                           | Compiled Runtime Execution                         |
| :------------- | :------------------------------------------------------------ | :------------------------------------------------- |
| def            | Declares a computable function, type alias, or constant13.    | Lowers to an executable C function1.               |
| theorem        | Declares a logical proposition and its verified proof term10. | Completely erased during compilation7.             |
| example        | Introduces an anonymous theorem to check a proof in-line17.   | Erased; does not add a symbol to the namespace.    |
| inductive      | Declares a custom sum type or algebraic data type14.          | Lowers to a tagged structure14.                    |
| structure      | Declares a product type with projection fields14.             | Compiled as a record structure14.                  |
| class          | Declares a typeclass to support ad-hoc polymorphism16.        | Translated to implicit dictionary passing16.       |
| instance       | Registers an implementation for a specified typeclass16.      | Added to the global synthesis lookup table4.       |
| match ... with | Destructures and pattern-matches expressions14.               | Lowers to switch statements with FBIP3.            |
| let            | Binds a local variable with a defined value17.                | Retained in local execution scope.                 |
| by             | Enters tactic-mode to write imperative proof scripts13.       | Interpreted by elaborator; erased on compilation1. |

### **Tactical Guide**

| Tactic    | Syntactic Format                      | Operational Effect on Proof State                                  |
| :-------- | :------------------------------------ | :----------------------------------------------------------------- |
| intro     | intro h \[cite: 10\]                  | Moves a target binder or implication premise into context as h10.  |
| exact     | exact h \[cite: 10\]                  | Closes the main goal if term h matches the goal type10.            |
| apply     | apply h \[cite: 10\]                  | Applies implication h backwards, updating the target goal10.       |
| rw        | rw \[h\] / rw \[←h\] \[cite: 10, 25\] | Syntactically replaces LHS with RHS using equality lemma h10.      |
| simp      | simp \[h1, h2\] \[cite: 10, 29\]      | Simplifies the goal using simplification rewrites10.               |
| rfl       | rfl \[cite: 13, 17\]                  | Solves equality if both sides are computationally equivalent13.    |
| cases     | cases x \[cite: 22, 28\]              | Performs case analysis, splitting the goal per constructor24.      |
| induction | induction x \[cite: 10, 29\]          | Generates proof branches for induction over variable x10.          |
| calc      | calc ... \[cite: 17, 25\]             | Groups sequential equality transformations into readable chains17. |

#### **Works cited**

1. Lean (proof assistant) \- Grokipedia, [https://grokipedia.com/page/Lean\_(proof_assistant)](<https://grokipedia.com/page/Lean_(proof_assistant)>)
2. BRIDGE: Building Representations In Domain Guided Program Synthesis \- arXiv, [https://arxiv.org/html/2511.21104v2](https://arxiv.org/html/2511.21104v2)
3. The Lean 4 Theorem Prover and Programming Language \- Microsoft Research, [https://www.microsoft.com/en-us/research/publication/the-lean-4-theorem-prover-and-programming-language/](https://www.microsoft.com/en-us/research/publication/the-lean-4-theorem-prover-and-programming-language/)
4. The Lean 4 Theorem Prover and Programming Language \- ResearchGate, [https://www.researchgate.net/publication/353088178_The_Lean_4_Theorem_Prover_and_Programming_Language](https://www.researchgate.net/publication/353088178_The_Lean_4_Theorem_Prover_and_Programming_Language)
5. functional programming | Chalmers Security & Privacy Lab, [https://www.cse.chalmers.se/research/group/security/tag/functional-programming/](https://www.cse.chalmers.se/research/group/security/tag/functional-programming/)
6. The Lean Theorem Prover (system description) \- andrew.cmu.ed, [https://www.andrew.cmu.edu/user/avigad/Papers/lean_system.pdf](https://www.andrew.cmu.edu/user/avigad/Papers/lean_system.pdf)
7. “Why not just use Lean?” | Hacker News, [https://news.ycombinator.com/item?id=47922079](https://news.ycombinator.com/item?id=47922079)
8. The Lean FRO Year 1 Roadmap \- Lean Programming Language, [https://lean-lang.org/fro/roadmap/y1/](https://lean-lang.org/fro/roadmap/y1/)
9. Tactic Proofs \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Tactic-Proofs/](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/)
10. The hitchhiker's guide to reading Lean 4 theorems \- LambdaClass Blog, [https://blog.lambdaclass.com/the-hitchhikers-guide-to-reading-lean-4-theorems/](https://blog.lambdaclass.com/the-hitchhikers-guide-to-reading-lean-4-theorems/)
11. Propositions and Proofs \- Lean Programming Language, [https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/](https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/)
12. 11\. Axioms and Computation — Theorem Proving in Lean 3 (outdated) 3.23.0 documentation, [https://leanprover.github.io/theorem_proving_in_lean/axioms_and_computation.html](https://leanprover.github.io/theorem_proving_in_lean/axioms_and_computation.html)
13. The Lean 4 Theorem Prover and Programming Language (System Description), [https://lean-lang.org/papers/lean4.pdf](https://lean-lang.org/papers/lean4.pdf)
14. Datatypes and Patterns \- Lean Programming Language, [https://lean-lang.org/functional_programming_in_lean/Getting-to-Know-Lean/Datatypes-and-Patterns/](https://lean-lang.org/functional_programming_in_lean/Getting-to-Know-Lean/Datatypes-and-Patterns/)
15. 4.4. Inductive Types \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/The-Type-System/Inductive-Types/](https://lean-lang.org/doc/reference/latest/The-Type-System/Inductive-Types/)
16. 10\. Type Classes \- Lean Programming Language, [https://lean-lang.org/theorem_proving_in_lean4/Type-Classes/](https://lean-lang.org/theorem_proving_in_lean4/Type-Classes/)
17. tactic \- When is the lean 4 "by" required? \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/4029/when-is-the-lean-4-by-required](https://proofassistants.stackexchange.com/questions/4029/when-is-the-lean-4-by-required)
18. Coercion Insertion \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Coercions/Coercion-Insertion/](https://lean-lang.org/doc/reference/latest/Coercions/Coercion-Insertion/)
19. 2\. Dependent Type Theory \- Lean Programming Language, [https://lean-lang.org/theorem_proving_in_lean4/Dependent-Type-Theory/](https://lean-lang.org/theorem_proving_in_lean4/Dependent-Type-Theory/)
20. Learning Lean 4 as a programming language 4 – Proofs \- Unreasonable Effectiveness, [https://unreasonableeffectiveness.com/learning-lean-4-as-a-programming-language-4-proofs/](https://unreasonableeffectiveness.com/learning-lean-4-as-a-programming-language-4-proofs/)
21. Lecture 19: Inductive types & proofs, [https://course.ccs.neu.edu/cs2800sp23/l19.html](https://course.ccs.neu.edu/cs2800sp23/l19.html)
22. lean4/doc/examples/tc.lean at master \- GitHub, [https://github.com/leanprover/lean4/blob/master/doc/examples/tc.lean](https://github.com/leanprover/lean4/blob/master/doc/examples/tc.lean)
23. 20.1. Natural Numbers \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Basic-Types/Natural-Numbers/](https://lean-lang.org/doc/reference/latest/Basic-Types/Natural-Numbers/)
24. Natural numbers \- Universitat de València, [https://www.uv.es/coslloen/Lean4/Leancap06.html](https://www.uv.es/coslloen/Lean4/Leancap06.html)
25. 2\. Basics — Mathematics in Lean v4.19.0 documentation, [https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html](https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html)
26. Topic: Term mode vs tactic mode \- Zulip Chat Archive, [https://leanprover-community.github.io/archive/stream/187764-Lean-for-teaching/topic/Term.20mode.20vs.20tactic.20mode.html](https://leanprover-community.github.io/archive/stream/187764-Lean-for-teaching/topic/Term.20mode.20vs.20tactic.20mode.html)
27. Lean 4: Formal Theorems \- Emergent Mind, [https://www.emergentmind.com/topics/formal-lean-4-theorems](https://www.emergentmind.com/topics/formal-lean-4-theorems)
28. Arrays and Termination \- Functional Programming in Lean, [https://leanprover.github.io/functional_programming_in_lean/programs-proofs/arrays-termination.html](https://leanprover.github.io/functional_programming_in_lean/programs-proofs/arrays-termination.html)
29. Interlude: Tactics, Induction, and Proofs \- Lean Programming Language, [https://lean-lang.org/functional_programming_in_lean/Interlude\_\_\_-Tactics\_\_\_-Induction\_\_\_-and-Proofs/](https://lean-lang.org/functional_programming_in_lean/Interlude___-Tactics___-Induction___-and-Proofs/)
30. Don't trust, verify: guarantees for the Lean proof assistant \- MFoCS Seminar Presentation, [https://cs.ru.nl/\~freek/courses/mfocs-2024/slides/rutger.pdf](https://cs.ru.nl/~freek/courses/mfocs-2024/slides/rutger.pdf)
31. Learn Lean 4 in Y Minutes, [https://learnxinyminutes.com/lean4/](https://learnxinyminutes.com/lean4/)
32. The Network Structure of Mathlib \- arXiv, [https://arxiv.org/html/2604.24797v1](https://arxiv.org/html/2604.24797v1)
33. 11.2. Coercing Between Types \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Coercions/Coercing-Between-Types/](https://lean-lang.org/doc/reference/latest/Coercions/Coercing-Between-Types/)
34. Coercions \- Lean Programming Language, [https://lean-lang.org/functional_programming_in_lean/Overloading-and-Type-Classes/Coercions/](https://lean-lang.org/functional_programming_in_lean/Overloading-and-Type-Classes/Coercions/)
35. lean \- Termination for Wrapped Fin in Lean4 \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/3840/termination-for-wrapped-fin-in-lean4](https://proofassistants.stackexchange.com/questions/3840/termination-for-wrapped-fin-in-lean4)
36. invalid apply tactic, failed to unify \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/2210/invalid-apply-tactic-failed-to-unify](https://proofassistants.stackexchange.com/questions/2210/invalid-apply-tactic-failed-to-unify)
37. Lean 4 Tutorial \- Brandon Rozek, [https://brandonrozek.com/blog/lean4-tutorial/](https://brandonrozek.com/blog/lean4-tutorial/)
38. Learn — Lean Lang, [https://lean-lang.org/learn/](https://lean-lang.org/learn/)
39. Difference between \`Nat.add\` and \`+\` in Lean \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/5278/difference-between-nat-add-and-in-lean](https://proofassistants.stackexchange.com/questions/5278/difference-between-nat-add-and-in-lean)
40. Termination Checking: Comparing Structural Recursion and Sized Types by Examples \- McGill School Of Computer Science, [https://www.cs.mcgill.ca/\~dthibo1/papers/termination.pdf](https://www.cs.mcgill.ca/~dthibo1/papers/termination.pdf)
41. 7.6. Recursive Definitions \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Definitions/Recursive-Definitions/](https://lean-lang.org/doc/reference/latest/Definitions/Recursive-Definitions/)
42. Lean 4.6.0 \- Lean Programming Language, [https://lean-lang.org/blog/2024-2-29-lean-460](https://lean-lang.org/blog/2024-2-29-lean-460)
43. Using \`decreasing_by\` to prove termination in Lean 4 \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/2712/using-decreasing-by-to-prove-termination-in-lean-4](https://proofassistants.stackexchange.com/questions/2712/using-decreasing-by-to-prove-termination-in-lean-4)
44. Interacting with Lean \- Lean Programming Language, [https://lean-lang.org/doc/reference/latest/Interacting-with-Lean/](https://lean-lang.org/doc/reference/latest/Interacting-with-Lean/)
45. Termination for Wrapped \`Fin n\` in Lean4 \- Stack Overflow, [https://stackoverflow.com/questions/78183967/termination-for-wrapped-fin-n-in-lean4](https://stackoverflow.com/questions/78183967/termination-for-wrapped-fin-n-in-lean4)
46. How can I do induction on a field in Lean 4? \- Proof Assistants Stack Exchange, [https://proofassistants.stackexchange.com/questions/5127/how-can-i-do-induction-on-a-field-in-lean-4](https://proofassistants.stackexchange.com/questions/5127/how-can-i-do-induction-on-a-field-in-lean-4)
47. Functional Programming in Lean, [https://lean-lang.org/functional_programming_in_lean/](https://lean-lang.org/functional_programming_in_lean/)
48. Introduction \- Lean Programming Language, [https://lean-lang.org/theorem_proving_in_lean4/Introduction/](https://lean-lang.org/theorem_proving_in_lean4/Introduction/)
49. Datatypes, Patterns and Recursion \- Functional Programming in Lean, [https://leanprover.github.io/functional_programming_in_lean/getting-to-know/datatypes-and-patterns.html](https://leanprover.github.io/functional_programming_in_lean/getting-to-know/datatypes-and-patterns.html)
50. Lean for Scala programmers \- Part 4 \- Typista.org, [https://typista.org/lean-for-scala-programmers-4/](https://typista.org/lean-for-scala-programmers-4/)

[image1]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAAAZElEQVR4Xu3LsQ2AIBSE4RvA/TcwRqMM4Bba21nbUQuGo/AKfLGFL6F5PwfUycd3/3gvK4+9Bnraqcdswff40GNW83jQQKbxqIFM40kDmcazBjKNnQYqjjekD7uGqENql4amKQpcZkCikuwjdwAAAABJRU5ErkJggg==
[image2]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAArklEQVR4XmNgGLngPwkYA0gxICT90ORgwI0BIs+ILgECUxnwmA4Fb4HYEl0QBr4zQDRboUtAQSoQl6MLIgN8tjMBcQq6IDJwYoBo/oguQSz4xAAxwBBdghgA0ozXebiAABAvRheEAlkgnowuCAOgOMQVWCDQD8SR6IIwQMifIHkVdEE1qEQcEMcDcQIQJzJA/F0BxBuh8lhd9RKIvwDxTyD+C8T/GBCK0fEoGGEAAFniNAwurcD0AAAAAElFTkSuQmCC
