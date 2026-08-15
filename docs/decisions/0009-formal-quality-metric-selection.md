# ADR 0009: Formal quality metric candidate selection

- **Status:** Candidate selection recorded; implementation and approval pending
- **Date:** 2026-08-16
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16), [ADR 0008](0008-formal-quality-metric-plan.md)

## Context

ADR 0008は6つのmetric familyと承認条件を固定したが、特定の外部metric、model、checkpointまたは実装は選択していない。候補調査を文書だけで管理すると、moving revisionへの置換、source code licenseとcheckpoint termsの混同、未検証entrypointの`planned`扱いが起こり得る。

一方、現時点では固定H3 runtimeにmetric依存を導入しておらず、formal 60-case baselineの自己変動も未測定である。したがって、候補を調査した事実と、実行可能・承認済みという状態を分離する必要がある。

## Decision

[`formal-quality-metric-selection.json`](../../benchmarks/quality/formal-quality-metric-selection.json)をcandidate assessmentとし、[`quality-metric-selection.schema.json`](../../schemas/quality-metric-selection.schema.json)で構造を固定する。これはformal readinessのsource of truthではなく、採用前の候補、固定revision、license scope、入力、score方向、採用条件、blockerを管理する補助artifactである。

候補は次の通りとする。

| Family | Candidate | Disposition | 主な未解決事項 |
|---|---|---|---|
| Prompt adherence | [SigLIP2 base patch16 256](https://huggingface.co/google/siglip2-base-patch16-256) | candidate | H3Fast adapter、固定runtime互換性、自己変動 |
| Perceptual video | [LPIPS 0.1.4](https://github.com/richzhang/PerceptualSimilarity) | candidate | frame alignment、weight来歴、固定runtime互換性 |
| Temporal consistency | H3Fast adjacent-frame LPIPS trajectory | candidate | algorithm契約、boundary test、自己変動 |
| Audio quality | [ViSQOL](https://github.com/google/visqol) | blocked | Bazel/native build、Python 3.12、48 kHz media契約 |
| A/V sync | [Synchformer](https://github.com/v-iashin/Synchformer) | blocked | checkpoint license/digest、旧dependency graph |
| Human pairwise | H3Fast blind randomized ballot | candidate | ballot schema、offline runner、review policy |

外部componentは調査時点の40文字commitへ固定する。version欄だけでimmutable identityとせず、source URL、revision、license、licenseが何を対象とするかを別々に記録する。特にSynchformerのMIT表記はsource codeだけを対象とし、pretrained checkpointの権利を推測しない。

次の条件をすべて満たすまで、[`formal-quality-metric-plan.json`](../../benchmarks/quality/formal-quality-metric-plan.json)の該当familyを`planned`へ変更しない。

- H3Fast内に実在するtyped entrypointとreference/correctness testがある
- Python 3.12、固定PyTorch/H3 runtime、offline loading、GPU memoryを実機確認する
- model/checkpoint/bundled weightを含む全artifactのimmutable identityとlicense scopeを確認する
- formal 60 caseについてbaseline自己変動を測定し、欠損をfailとして扱う
- ownerがmethod、dependency pin、入力contract、score方向をreviewする

budgetとapproval evidenceは自己変動測定後に別PRでreviewし、6 familyの一つの合格で別familyの失敗を相殺しない。

## Consequences

候補調査を再現でき、次に実装するadapterの境界と採用条件が明確になる。現在のcandidate assessmentは実行可能性、metric妥当性、checkpoint配布権または品質同等性を承認しない。

最初の実装対象は、native buildや未確定checkpointを必要としないproject-owned human ballot contractとする。その後、固定runtimeで依存を隔離してsmokeできる外部metricを1 familyずつ追加する。
