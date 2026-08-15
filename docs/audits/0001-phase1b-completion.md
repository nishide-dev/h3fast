# Phase 1B completion audit

- Date: 2026-08-16 (Asia/Tokyo)
- Related: Issue #9
- Scope: repository, packaging, schemas, documentation, CI, and release readiness
- Outcome: Internal Phase 1B path is consistent; public release remains blocked

## Audited implementation

追跡対象全fileを確認し、現在の実装を次に限定した。

- 単一`h3fast` Python distribution、Python 3.12、単一`uv.lock`
- local snapshot inspection、manifest/checksum verification、CPU-safe diagnostics
- 固定SGLang/SIF向けpreflight、guarded launch、T2VA benchmark client、measured suite
- 20層BF16 baseline、40層既定candidate、明示的20層rollback
- 固定1 caseのexact decoded artifact reference/report
- CPU unit/schema checks、Ruff、`ty`、wheel/sdist、clean-wheel import CI

`convert`、H3Fast独自kernel、quantization、cache、public serving、OCI distribution、GPU CI、PyPI release、派生重みは実装済み範囲に含めない。

## Findings and corrections

| Finding | Risk | Correction |
|---|---|---|
| `AGENTS.md`がPhase 1Aかつtemplate置換前の状態を示していた | 次の作業順序と不要依存の判断を誤る | Phase 1A/最初のPhase 1B実測完了、release blocker優先へ更新 |
| specの将来CLI/OCI例が現行commandと区別されていなかった | 未実装commandや未公開packageを利用者が実行する | 現在の実装表を追加し、`convert`、`serve`、PyPI、OCI、target例を未実装と明記 |
| specのrepository treeが先行分割済みの構成だった | 不要なkernel/package/test階層を早期作成する | 現在の単一distribution構成へ合わせ、workspace構成は将来条件付きのまま維持 |
| specのPR CI必須項目が実際のCIを超えていた | release gate済みと誤認する | 現行CPU CIとPublic Runtime release前の追加gateを分離 |
| artifact registerがbaseline後、candidate前の状態だった | provenanceと公開可否の判断を誤る | 40層candidate、exact gate、snapshot/runtime実測、未完reviewを反映 |
| 未解決事項が「正式実装開始前」のままだった | Phase 1B完了後の次作業が不明確 | release/Phase 2前blockerへ修正し、owner/deadline未割当と優先順を明記 |
| `serve-guarded`でlifecycle記録を省略できた | 実効resident設定を再現metadataへ残さない経路が生じる | `--lifecycle-report`を必須化し、ready eventにもruntime settingを記録 |
| 一部schemaだけがmeta-schema検証対象だった | 未使用schemaの破損をCIが見逃す | 全`schemas/*.schema.json`をDraft 2020-12として検証するtestを追加 |
| `.gitignore`にtemplate固有commentが残っていた | 現行repository状態を誤認する | 汎用commentへ更新 |

## Template and artifact audit

Git追跡対象にMNIST、PyTorch Lightning、Hydra、logger、training config、notebook、model weight、generated media、runtime cache、benchmark output、token、secretは存在しない。`ml-research-template`由来の実装依存も`pyproject.toml`と`uv.lock`に残っていない。

`models/`、`outputs/`、`benchmark-results/`、`runtime-cache/`、model/media拡張子は引き続き`.gitignore`対象である。commit済みquality referenceはdigest、media metadata、制約だけを含み、生成物またはlocal pathを含まない。

## Release readiness

次は完了している。

- immutable H3/SGLang/runtime identityを持つlocal measured path
- Phase 1A baselineと最初のPhase 1B A/B
- candidate measured 3/3のsingle-case exact gate
- clean wheelのCPU install/import
- source、weight、runtime、Outputの初期artifact分類

次が未完了のため、Draft解除、Public Runtime公開、Support Tier付与、Phase 2開始は不可とする。

1. Applicable Territory、H3 license、H3/SGLang code boundaryの承認とowner/deadline
2. 公開権を確認した10-case smoke / 50-case regression setと知覚・audio・semantic A/V gate
3. clean machineでの20層baselineと40層candidate再現
4. dependency/license scan、secret scan、release notice、必要時のSBOM/vulnerability gate
5. schema owner、release承認者、初期support targetの正式決定

次の実装順序は上記1から4とし、新しい最適化やdistribution分割を先行させない。

## Verification

この監査変更では追加GPU runを行わない。既存のPR #4、#6、#8で保存したlocal evidenceを参照し、CPU/package checks、全schema meta-validation、local Markdown link検査、tracked artifact検査を再実行する。最終結果はPull Request本文へ記録する。
