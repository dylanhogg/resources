# Useful Prompts/Skills

## Planning / Implementation

1. Make plan
```
Make a single PR phase plan to implement <IMPLEMENTATION_TASK>. Ensure all existing functionallity remains the same. Utilise <THINGS> where possible. Ensure solution is elegant, performant and cross platform supported.

Write the plan to plans/<PLAN_FILENAME>.md
```

2. Implement plan
```
Implement <PLAN_DESC> plan at plans/<PLAN_FILENAME>.md. Ensure code is clean, minimal, and modular.
Once feature complete, test to ensure it works as expected according to the plan.
After testing, review the implementation and identify places that can be refactored for clean code, readability, and modularity. Then implement those identified refactorings to improve the quality of the codebase, while keeping it simple.
```

3. Refactor implementation
```
Review the implementation, and identify places that can be refactored for clean code, readability, reducing duplication, and modularity. Then implement those identified refactorings to improve the quality of the codebase, while keeping it simple.
```

## Refactoring Prompts

### Arg, fix this mess:

```
Make files smaller, remove unused function everywhere, unduplicate logic and re-evaluate random helpers. Add structure.
```

- https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-simplifier/agents/code-simplifier.md
