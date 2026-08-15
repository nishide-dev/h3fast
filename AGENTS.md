# H3Fast Project Guide

## Project overview

H3Fast is a research and runtime project for making local MiniMax H3-Base inference faster and more efficient while preserving reproducibility, safe fallbacks, and measurable quality. The intended MVP is a public Bring Your Own Weights (BYOW) runtime: users supply a legally obtained local H3 snapshot, and H3Fast provides validation, manifests, diagnostics, benchmarks, and measured optimizations without redistributing the official weights.

The authoritative product and distribution specification is `docs/spec.md`. Read the relevant sections before changing architecture, packaging, model conversion, serving, optimization, benchmarking, or release behavior.

The repository was initialized from `ml-research-template`. The current MNIST, PyTorch Lightning, Hydra, logger, and training files are disposable template scaffolding, not H3Fast architecture. Do not extend or preserve them merely for compatibility. Remove them when replacing the template with the first H3Fast implementation.

## Current phase and priorities

The project is in Phase 1A implementation and follows this order:

1. Resolve the remaining Phase 0 blockers in `docs/spec.md`: license and territory applicability and the quality reference set. The base revision is immutable and one FL2VA/T2VA E2E smoke has completed on 2×RTX 6000 Ada, but this is not yet a support tier or performance baseline.
2. Extend the fixed BF16 harness from one smoke run to stage-level profiling and the protocol's measured-run set.
3. Keep the smallest single-package BYOW validation and benchmark path reproducible and CPU-import-safe.
4. Use the profile to select and implement one independently measurable optimization.
5. Publish only after legal, quality, reproducibility, and supply-chain gates pass.

Avoid speculative support for multiple GPU vendors, task families, quantization schemes, servers, or package distributions before an actual need and test environment exist.

## Source of truth

- `docs/spec.md` defines product scope, phases, compliance requirements, artifact formats, benchmark policy, and release gates.
- Schemas and tests define machine-checkable behavior once implemented.
- Code and documentation must not silently diverge from the specification. Update the specification in the same change when intentionally changing a documented contract.
- Examples containing placeholders such as `<tested-version>` are not executable defaults.
- Do not present an unverified assumption, external benchmark, or vendor claim as an H3Fast result.

## Language and communication conventions

Use English for names that must work consistently in developer tooling:

- Commit subject lines
- Issue titles
- Pull request titles
- Branch names
- Source-code identifiers, CLI names, schema fields, filenames, and test names

Use Japanese for explanatory content intended for this project team:

- Commit bodies
- Issue descriptions and comments
- Pull request descriptions and review comments
- Design rationale, implementation notes, test reports, and incident notes

Code comments and docstrings should be English. Project documentation may be Japanese unless it is part of an external English-facing API or distribution artifact. Preserve established terminology such as BYOW, AdaLN, FL2VA, Ref2VA, and SGLang rather than inventing inconsistent translations.

## Commit messages

Use Conventional Commits for the English subject line:

```text
<type>(<optional-scope>): <English imperative summary>

<Japanese body explaining why, what changed, and how it was verified>
```

Allowed types:

- `feat`: User-visible capability
- `fix`: Defect correction
- `perf`: Measured performance improvement
- `refactor`: Behavior-preserving code restructuring
- `test`: Test-only change
- `docs`: Documentation-only change
- `build`: Build or dependency change
- `ci`: CI/CD change
- `chore`: Maintenance not covered above
- `revert`: Revert of an earlier change

Rules:

- Write the subject in English, imperative mood, without a trailing period.
- Keep the subject concise, preferably no more than 72 characters.
- Use a meaningful scope such as `manifest`, `benchmark`, `sglang`, `kernel`, or `ci` when helpful.
- Write the body in Japanese. Explain motivation and trade-offs instead of repeating the diff.
- Include verification results in the Japanese body for non-trivial changes.
- Use `!` and a `BREAKING CHANGE:` footer for incompatible public API, CLI, schema, or artifact changes.
- Do not use `perf` unless the change includes a reproducible benchmark against the fixed baseline.

Examples:

```text
docs(spec): clarify BYOW license boundaries

BYOWがH3の利用権や地域制限を代替しないことを明記した。
公式コードを取り込む場合の成果物分類もリリースゲートへ追加した。
```

```text
perf(kernel): fuse AdaLN residual operations

参照実装との数値比較を追加し、H100上で対象stageの中央値を測定した。
測定条件と結果はbenchmark bundleへ記録した。
```

Agents must not create commits, amend history, push branches, or open issues/PRs unless the user explicitly requests that action.

## Branch, issue, and pull request conventions

Use short English kebab-case branch names, preferably prefixed by the change type:

```text
feat/manifest-validator
fix/adaln-schedule-check
perf/sparse-attention-kernel
docs/release-policy
```

Issue titles must be English. Issue bodies must be Japanese and should contain:

- 背景・目的
- 対象範囲と対象外
- 完了条件または再現手順
- 技術的・法的な制約
- 必要な検証環境

Pull request titles must be English and should follow the same Conventional Commit form as commit subjects. Pull request bodies must be Japanese and should contain:

- 目的と背景
- 主な変更点
- 仕様・API・schemaへの影響
- テストおよびbenchmark結果
- GPU、driver、依存versionなどの検証環境
- 既知の制約、fallback、rollback方法
- 関連Issueまたは設計判断へのリンク

Draft pull requests must clearly state what remains incomplete. Performance pull requests must include raw or linked benchmark artifacts and quality-regression results, not only a headline speedup.

## Repository and packaging policy

- Start with one `h3fast` Python distribution under `src/h3fast`.
- Keep manifest, backend, kernel, diagnostics, CLI, and benchmark responsibilities separated by modules.
- Do not create `h3fast-core`, `h3fast-kernels`, or `h3fast-server` distributions until a split condition in `docs/spec.md` is met.
- Add a GPU-specific project under `targets/` only after that target is actually tested and requires a conflicting dependency graph.
- Keep runtime dependencies in `[project.dependencies]` and development-only tools in dependency groups.
- Commit `uv.lock` once real H3Fast dependencies are established. Update affected lockfiles in the same change as dependency metadata.
- Pin Python, uv, PyTorch, Triton, SGLang, CUDA/ROCm, and driver requirements wherever reproducibility depends on them.
- Public wheels must install and import in a clean environment without workspace source overrides.

## Python and runtime code

- Target Python 3.12 until Phase 0 selects a different supported range.
- Use type hints for public and internal interfaces. Avoid untyped dictionaries for stable manifests and API contracts.
- Keep CPU import paths functional. Optional GPU libraries must be imported lazily.
- Do not initialize a GPU, compile Triton kernels, run autotuning, download models, or contact external services at module import time.
- Keep SGLang-specific behavior behind an adapter and pin the supported SGLang version or commit.
- Prefer explicit capability checks and typed errors over implicit fallback.
- Fail closed for model revision, checksum, schedule, schema, or compatibility mismatches.
- Record every fallback and effective optimization in diagnostics and reproducibility metadata.
- Never silently change task family, model ID, schedule, step count, precision profile, or quality profile.

## Optimization and benchmark policy

Every optimization must include:

- A clear hypothesis tied to profiling evidence
- A PyTorch or otherwise trustworthy reference implementation
- Unit or kernel correctness tests
- Boundary, dtype, shape, non-contiguous, NaN/Inf, and unsupported-capability cases as applicable
- A safe fallback or an explicit unsupported error
- An isolated A/B measurement against the pinned baseline
- E2E latency, relevant stage latency, peak memory, and quality-regression results
- Environment and artifact metadata sufficient to reproduce the result

Change one optimization dimension at a time when establishing causality. Separate cold-start, compilation, warmup, and steady-state measurements. Do not call a path lossless or exact until its numerical tolerance and quality gate are defined and passed.

## Testing and quality checks

Run the smallest relevant checks during development and the complete affected suite before handoff. Until the template is replaced, the available commands are provisional:

```bash
uv sync
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check src/
```

When the H3Fast skeleton is implemented, the expected minimum gates are:

- Unit tests and schema validation
- Ruff formatting and lint
- Static type checks
- CPU-only package import
- Clean-wheel install and import
- Reference fallback tests
- GPU kernel correctness for affected backends
- Short E2E audio-video generation smoke test for affected Tier 1 targets
- Performance and quality regression checks for optimization changes

Do not keep template training tests or dependencies merely to make obsolete CI pass. Replace CI in the same change that removes the template implementation.

## Model, data, and artifact safety

- Never commit model weights, adapters, AdaLN caches, generated media, benchmark media with unclear rights, secrets, access tokens, customer inputs, or local model-cache paths.
- Use Safetensors for distributed model artifacts; do not distribute pickle-based model files.
- Do not automatically upload conversion outputs or benchmark artifacts.
- Do not automatically download H3 weights in the initial BYOW path. Operate on an explicitly supplied local snapshot.
- Record immutable base revisions and per-file SHA-256 digests.
- Treat incomplete conversions as invalid and write outputs atomically where practical.
- Keep large generated outputs and compiler/model caches outside Git.

## License, territory, and security

MiniMax H3 has non-standard license, territory, downstream-terms, commercial-display, safety, and Output-use restrictions. BYOW does not remove those restrictions.

- Do not copy MiniMax H3 code, configuration, weights, documentation, or other Materials into H3Fast without recording provenance and completing the artifact classification required by `docs/spec.md`.
- Do not publish H3 Works, Model Derivatives, containers that include them, or services using them before the applicable release gate passes.
- Do not claim that an IP check, Hugging Face gate, checkbox, or CLI warning guarantees legal compliance.
- Never include weights, tokens, user inputs, or generated outputs in OCI image layers.
- Public media APIs must reject unsafe local-file access and defend against SSRF, oversized media, decoder attacks, and private-network redirects.
- Do not log prompt or media contents by default.
- Public dissemination features must clearly disclose AI-generated content; C2PA is supplementary provenance, not a substitute for visible disclosure.

## Documentation expectations

- Document public APIs, CLI options, schemas, compatibility rules, and fallback behavior in the same change as implementation.
- Include exact commands only after verifying them against the pinned environment.
- Label aspirational examples and untested hardware explicitly.
- Keep external source links close to time-sensitive technical or license claims.
- Record significant architectural decisions and rejected alternatives in `docs/` when they are not already captured by the specification.

## Definition of done for a change

A change is complete when:

- It stays within the current Phase and stated scope.
- Relevant specification and documentation are updated.
- Appropriate tests and static checks pass.
- Performance claims include reproducible evidence.
- New failure modes have explicit errors or tested fallbacks.
- Dependency, artifact, license, security, and compatibility impacts are recorded.
- No secrets, weights, caches, generated media, or unrelated template artifacts are introduced.
- The handoff reports what changed, how it was verified, and any remaining limitations in Japanese.
