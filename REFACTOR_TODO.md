# Refactor TODO

Tracked refactor of `zero-shot` into a typed **core** with pluggable
**interfaces**. Work top-to-bottom; each task is reviewed by a fresh agent
before it is marked done.

Decisions locked in with the owner:

- Layout: `zero_shot/core/` + `zero_shot/interfaces/{cli,server}/`; the server
  interface owns `static/`.
- Typing: pydantic for the domain models, the HTTP API contract, and `Config`.
- Commands: keep `zero-shot` (CLI) and `zero-shot-serve` (server); one console
  script per interface.

## Tasks

- [x] **Task 4 — regression tests**
  - `tests/core/` (models, config, classifier, scorer, image_utils) and
    `tests/interfaces/` (schemas, CLI, server), mirroring the package layout.
  - `FakeScorer` fixture keeps the suite deterministic and offline; pytest added
    to the dev group and configured in `pyproject.toml`.
  - `AGENTS.md` tests section updated.
  - Review agent findings: no blockers. Fixes applied: (major) autouse fixture
    clears ambient `ZERO_SHOT_*` env vars so config tests are isolated; (major)
    added coverage for the 413 request/upload limits; (minor) image calibration
    routing contract (`chat_template`/`add_leading_space`), `resolve_precision`
    `"none"`/`None` aliases, GPU-name mismatch, `_render` list/null, unscored and
    all-unscored options, and `temperature <= 0`; seeded the noise image test and
    removed a duplicated PNG helper.
  - Verified: `pytest` 87 passed in ~1 s, offline and with `ZERO_SHOT_*` set.

- [x] **Task 1 — core/interface split**
  - `zero_shot/core/{classifier,scorer,config,image_utils}.py`
  - `zero_shot/interfaces/cli/` (console script `zero-shot`)
  - `zero_shot/interfaces/server/` (console script `zero-shot-serve`, `static/`)
  - Update imports, `pyproject.toml` scripts + packaging, `Dockerfile`,
    `AGENTS.md`, `README.md`, `.opencode` skill references.
  - Review agent findings: (1 blocker) `from . import app as app_module` was
    shadowed by the FastAPI instance exported in `server/__init__.py`, breaking
    `--config`/`--cpu-low`; fixed by importing `app`/`configure` directly and
    removing the shadowing re-exports. (2 minor) reset `_RESOLVED_DEVICE` in
    `configure()`; documented `cpu_low_config_path()` wheel-install fallback.
  - Verified: `zero-shot --help`, `zero-shot-serve --help`, `--config` and
    `--cpu-low` start with the right config, real CPU classification run.

- [x] **Task 2 — pydantic typing**
  - Domain: question specs (`choice`/`noul`/`score`), `TokenScore`,
    `OptionScore`, `Classification`.
  - API: request + response + health + usage + error models.
  - Config: pydantic model with validators (temperature > 0, quantize enum,
    device enum).
  - Preserve the exact JSON wire shape the web UI consumes.
  - Review agent findings: no blockers/majors. Fixes applied: (minor) wrap
    `load_config` in the CLI's error handler so a bad `device` is a friendly
    `error:` not a traceback; (minor) `NoulQuestion.criteria` is now `Any`
    (was `dict|None`) matching the old ignore-anything behavior; (minor)
    `ErrorResponse` is now used by an `_error()` helper and documented via
    `responses=` in OpenAPI; (nit) `classify_one` validates unconditionally so a
    bad spec raises `ValueError`, not `AttributeError`.
  - Verified: `to_dict` output byte-identical to HEAD (reviewer), OpenAPI
    documents `ClassifyResponse`/`ErrorResponse`, real CPU CLI + HTTP runs,
    400/422 error bodies unchanged.

- [x] **Task 3 — ruff + CI**
  - `[tool.ruff]` in `pyproject.toml`, ruff as a dev dependency.
  - `.github/workflows/lint.yml` running `ruff check` on push/PR to `main`.
  - Fix all findings so the tree lints clean.
  - Review agent findings: no blockers/majors. Fixes applied: (minor) pin the
    documented command to `uvx ruff@0.16.8 check` to match CI/uv.lock; (minor)
    enable the `BLE` rule so the existing `# noqa: BLE001` comments are
    load-bearing; (nit) add `.ruff_cache/` to `.gitignore`. Format gate
    (`ruff format --check`) intentionally not added — out of scope (lint only).
  - Verified: `uvx ruff@0.16.8 check` passes on the whole tree; imports, help,
    compileall smoke tests pass; workflow YAML valid and version-consistent.

## Verification log

Recorded per task once its review agent signs off.
