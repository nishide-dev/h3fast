# ADR 0009: Formal quality metric candidate selection

- **Status:** Candidate selection recorded; human-pairwise runner/pilot completed (2026-08-16); metric approval pending
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

最初の実装対象として、native buildや未確定checkpointを必要としないproject-owned human ballot/key schema、blind assignment commitment、欠損時fail、aggregate scorerを追加した。seed、ballotとassignmentはprivate fileとして分離し、prompt、media、pathまたはper-case decisionをaggregate出力へ含めない。

2026-08-16にoffline A/B media presentation runner(`stage-human-pairwise`)とselection記録CLI(`record-human-pairwise`)を追加し、private media manifest契約でmediaのdigest検証とblind staging(相対参照のみの`index.html`を含む)を固定した。synthetic mediaによる60-case pilotで、assignment keyを参照しない知覚代理判定がground truthと全件一致し、blind割当・復号・集計をend-to-endで検証した([experiment 0006](../experiments/0006-human-pairwise-pilot.md))。

review policyは[ADR 0010](0010-human-pairwise-review-policy.md)でsingle-reviewer運用として承認した。ただしGPU実出力によるformal実測とimmutable implementation evidenceは未完了である。このためhuman-pairwiseはcandidateのままとし、formal metric planを`planned`へ変更しない。

2026-08-16に最初の外部metric adapterとしてperceptual-video (LPIPS) entrypoint `score-perceptual-video`を追加した。依存は`quality-metrics` dependency groupへ隔離し(lpips 0.1.4、torch 2.11.0 CPU wheel、torchvision 0.26.0)、packageのCPU importとwheelのruntime依存ゼロを維持した。AlexNet backbone checkpointは自動downloadせず、SHA-256 `7be5be79…cdee02`へ固定したlocal fileだけを受け付ける。frame数・解像度・frame rateの完全一致をfail-closedで要求し、torch thread数1の単一process決定性、非有限値のfail、decode失敗の帰属、構築中の予期しないcheckpoint追加拒否を含めてcorrectness testで固定した。cross-machine再現性はformal実測時に確認する。CPU wheel採用は評価の決定性とCI費用のためであり、GPU実行が必要になった場合はconflicting dependency graph条件で`targets/`分離を検討する。formal media contractと固定H3 runtimeでの実機確認、baseline自己変動、backbone checkpointのlicense scope確認が残るため、perceptual-videoもcandidateのままとする。

同日、temporal-consistencyのproject-owned契約を`adjacent-frame-lpips-trajectory-v1`として実装した。各動画の隣接frame間LPIPS列をtrajectoryとし、index対応stepの絶対差のmean/maxをlower-is-betterで比較する。scene cutは除外せず(保持されたcutは相殺、消失は検出)、alignment契約と依存・backbone契約はperceptual-videoと共有し、2 frame未満を拒否する。static scene、flicker、cut除去、決定性、decode失敗帰属、非有限値をcorrectness testで固定した。残blockerはperceptual-videoと同一である。

同日、prompt-adherence adapter(契約`siglip2-base-patch16-256-cosine-v1`)を追加した。transformers 5.14.1をquality-metrics groupへpinし、pinned snapshot(7 fileのSHA-256 manifest、自動downloadなし)のoffline load、prompt本文のprivate file供給とformal case `prompt_sha256`へのbytes一致検証、最大16 frameの固定一様サンプリング、cosine類似度のmean/min集計を実装した。pinned snapshotはmodel_type `siglip`(FixRes変種)でありAutoModel経由でloadしてarchitectureを検証する。pinned実snapshotでのoffline loadと意味的方向性はlocal CPUで確認済みである。残blockerは固定H3 runtime/formal media contractでの確認とbaseline自己変動である。
