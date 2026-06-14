---
name: lean-python
description: Use when making Python code changes, refactors, tests, CLI updates, packaging changes, dependency work, or Python code reviews in repos that value simple, clean, minimal code. Bias toward the smallest correct implementation, existing project patterns, current third-party docs, uv/ruff/pyright/pytest validation, and avoiding unnecessary abstractions or documentation churn.
---

# Lean Python

## Workflow

1. Read local guidance first: `AGENTS.md`, `pyproject.toml`, `Makefile`, and nearby code.
2. Define the smallest behavior change that satisfies the request.
3. Reuse existing structure, helpers, dependencies, and style.
4. Add no abstraction unless it removes real duplication or risk.
5. Add no dependency unless stdlib or existing deps are a poor fit.
6. Check current docs before using unfamiliar or fast-moving third-party APIs.
7. Prefer clear functions with typed parameters and returns.
8. Prefer functional flow over classes unless state or interfaces justify a class.
9. Keep docstrings and comments sparse; use them only for non-obvious intent.
10. Update tests with the narrowest coverage that proves behavior.

## Python Defaults

- Use `uv` for environment and dependency work.
- Use `pytest` for tests.
- Use `ruff` for format/lint.
- Use `pyright` for types.
- Use `typer`, `loguru`, `pydantic`, and `python-dotenv` when they fit the task.
- Prefer `pathlib`, `dataclasses`, `functools`, `collections`, `contextlib`, and `asyncio` from stdlib before adding packages.
- Use `assert` for internal assumptions; raise meaningful exceptions for user-facing failures.
- Keep secrets in `.env`; never log secret values.

## Avoid

- Broad rewrites.
- Premature plugin, service, factory, registry, or class hierarchies.
- New `__init__.py` files unless required.
- README or docs edits unless explicitly requested.
- Tiny wrapper functions that only rename another call.
- Mock-heavy tests when a direct unit or integration test is simpler.
- Silent fallbacks that hide bad input or configuration.

## Validation

Choose the smallest useful checks, usually in this order:

```bash
uv run pytest
uv run ruff format . --check
uv run ruff check .
uv run pyright
```

Run targeted tests first when the change is narrow. Run full checks before handoff when the change touches shared behavior, packaging, CLI entrypoints, or public interfaces.

## Handoff

Report:
- what changed
- what checks ran
- any skipped checks and why
- any risk that remains
