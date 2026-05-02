<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0
Modified principles: N/A (initial adoption)
Added sections:
  - Core Principles (I–VII)
  - Git Discipline
  - Code Quality Gates
  - Governance
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ compatible (uses dynamic Constitution Check)
  - .specify/templates/spec-template.md — ✅ compatible (no principle-specific references)
  - .specify/templates/tasks-template.md — ✅ compatible (generic task structure)
Follow-up TODOs: None
-->

# CAMEL Constitution

## Core Principles

### I. Clean Architecture

All code MUST follow a layered architecture with strict dependency direction
(outer layers depend on inner layers, never the reverse):

- **Domain** → pure business logic, entities, value objects, domain services
- **Application** → use cases, orchestration, ports (abstract interfaces)
- **Infrastructure** → adapters, repositories, CLI, external services, frameworks

Rules:
- Domain layer MUST NOT import from application or infrastructure
- Application layer MUST NOT import from infrastructure
- Infrastructure implements interfaces defined in the application layer
- Each layer lives in its own package under `src/camel/`

### II. Domain-Driven Design

The codebase MUST model the problem domain explicitly using DDD tactical patterns:

- **Entities**: objects with identity and lifecycle (e.g., `Evaluation`, `Experiment`)
- **Value Objects**: immutable, equality by value (e.g., `Score`, `ModelConfig`)
- **Aggregates**: consistency boundaries with a single root entity
- **Repositories**: abstract persistence behind domain-defined interfaces
- **Domain Services**: stateless operations that don't belong to a single entity
- **Ubiquitous Language**: code naming MUST mirror the domain language used by
  stakeholders — no abbreviations that obscure intent

### III. The Zen of Python

All design decisions MUST align with the Zen of Python (PEP 20). Key mandates:

- Explicit is better than implicit — no magic, no hidden side effects
- Simple is better than complex — prefer straightforward solutions
- Flat is better than nested — limit indentation depth to 3 levels
- Readability counts — code is read far more than it is written
- There should be one obvious way to do it
- If the implementation is hard to explain, it's a bad idea
- Namespaces are one honking great idea — use them
- Always use `uv` commands
- Never hardcode dependencies. use `uv` to resolve them
- NEVER hardcode paths, urls, ports, and specific strings

### IV. DRY & Clean Code

Code MUST NOT repeat itself. Every piece of knowledge has a single,
authoritative representation:

- Extract shared logic into well-named functions or base classes
- Meaningful names: variables, functions, classes MUST reveal intent
- Functions MUST do one thing (Single Responsibility Principle)
- Keep functions short — ideally under 20 lines
- No dead code, no commented-out code in main branches
- Prefer composition over inheritance

### V. Factory Pattern & Dependency Inversion

Object creation MUST be decoupled from usage through the Factory pattern:

- Use Factory functions or classes to instantiate complex objects
- Consumers depend on abstractions (protocols/ABCs), not concrete implementations
- Factories live in the infrastructure layer and return domain/application types
- Configuration and wiring happen at the composition root (entry point)
- This enables testability — swap real implementations with test doubles
  via factory configuration

### VI. Test-First Development

All features MUST be accompanied by tests. The Red-Green-Refactor cycle is
the default workflow:

- Write a failing test that defines the expected behavior
- Implement the minimum code to make the test pass
- Refactor while keeping tests green
- Unit tests for domain logic, integration tests for adapters
- `pytest` as the test framework; `pytest-cov` for coverage enforcement
- Type safety enforced via `mypy --strict`

### VII. Observability & Simplicity

Production code MUST be observable and no more complex than necessary:

- Structured logging with context (experiment ID, model, step)
- MLflow tracking for all evaluation runs
- YAGNI — do not build features until they are needed
- Start with the simplest solution that could work; refactor when evidence
  demands it
- Complexity MUST be justified in commit messages or PR descriptions

## Git Discipline

All contributors MUST follow these git practices without exception:

- **NEVER** use `git add .` — stage files explicitly by path
- Every commit MUST use the project commit template
  (`[type](domain): short description` with Problem/Solution/Impact/Notes)
- Commit messages MUST be atomic — one logical change per commit
- Branch naming: `<type>/<short-description>` (e.g., `feat/llm-judge-scorer`)
  Note: speckit feature tracking uses `NNN-<feature-name>` branches internally;
  rename to `<type>/<description>` before merging to main
- Conventional types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`
- Rebase over merge for linear history on feature branches
- Interactive rebase to squash WIP commits before merging to main
- Use `pre-commit` before committing
- Close related `issues` in GitHub if completed

## Code Quality Gates

All code MUST pass these automated checks before merge:

- **black**: code formatting (line-length 100)
- **isort**: import ordering (black-compatible profile)
- **autoflake**: remove unused imports and variables
- **mypy**: strict type checking with no untyped definitions
- **vulture**: dead code detection (min confidence 80%)
- **pytest**: all tests passing with coverage threshold met
- **pre-commit**: all hooks MUST pass before any commit is accepted

Manual review checklist:
- Architecture layer boundaries respected
- Domain language consistency verified
- No DRY violations introduced
- Factory pattern used for complex object creation

## Governance

This constitution is the supreme authority for project development practices.
All code, reviews, and architectural decisions MUST comply.

Amendment procedure:
1. Propose change via a dedicated commit with type `docs(constitution)`
2. Document rationale in the commit's Problem/Solution/Impact sections
3. Version bump follows SemVer (MAJOR: breaking principle change,
   MINOR: new principle/section, PATCH: clarification/typo)
4. All active contributors MUST acknowledge amendments

Compliance:
- Every PR review MUST verify adherence to these principles
- Violations MUST be flagged and resolved before merge
- Exceptions require explicit justification documented in the PR description

**Version**: 1.0.0 | **Ratified**: 2025-05-01 | **Last Amended**: 2025-05-01
