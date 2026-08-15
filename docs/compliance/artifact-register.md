# Artifact register

最終更新: 2026-08-16

この表は技術的な来歴記録であり、法的助言ではありません。公開前に適切なreviewerの承認を記録してください。

| 成果物 | 出所 | H3との関係 | License | 配布範囲 | 状態 |
|---|---|---|---|---|---|
| H3Fast source | 独自実装 | H3専用tooling。MiniMax Materialsのsource fileは未収録 | Apache-2.0 | Public候補 | engineering boundary inventory済み、H3 Works非該当の法務承認待ち |
| `h3fast` wheel/sdist | H3Fast sourceからbuild | H3重み・cache・MiniMax source fileを含まない | Apache-2.0 | Public候補 | clean install済み、法務・release gate待ち |
| CPU CI artifacts | H3Fast sourceとtest fixture | 合成fixtureのみ。H3 Materialsを扱わない | Apache-2.0 | CI内部 | 利用可 |
| Initial Runtime release gate | H3Fast独自metadata | owner、期限、evidenceと全required checkを記録。H3 Materialsなし | Apache-2.0 | Public候補 | `blocked`。legal/release/schema owner未指名、公開不可 |
| Initial Runtime territory inventory | H3Fast独自metadata | 開発・GPU・CI・storage・配布・実行・Output利用のregion evidenceを記録。H3 Materials/Output本体なし | Apache-2.0 | Public候補 | `incomplete`。legal owner、物理所在地、利用地域未確認 |
| Formal quality-set record | H3Fast独自metadata | 10/50件のcase identity、coverage、rights/metric approval契約。prompt本文、reference media、生成物なし | Apache-2.0 | Public候補 | `incomplete`。case registry、rights review、quality owner、metrics未登録 |
| SGLang adapter metadata | 独自実装 | SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`との互換性とmedia probeを記録。コードcopyなし | Apache-2.0 | Public候補 | 20層baselineと40層candidateをlocal実測済み。単一case exact gate合格、support未付与 |
| SGLang source checkout | sgl-project/sglang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e` | H3 pipelineを含む外部runtime source。repositoryには未収録 | Apache-2.0ほか依存license | ローカル実験限定・再配布なし | 固定checkoutでE2E実機検証済み。依存license review未完了 |
| SGLang CUDA 12.9 SIF | `lmsysorg/sglang` amd64 manifest `sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f` | H3重みを含まない外部runtime image | image内licenseに従う | ローカル実験限定・再配布なし | 固定digestのSIFでE2E実機検証済み。再配布・SBOM・license review未完了 |
| MiniMax H3 FL2VA snapshot | `MiniMaxAI/MiniMax-H3` commit `42ed227ee7df40d41602854ae760620d6eb651fe` | MiniMax H3 Works | MiniMax H3 Community License | 配布禁止 | 84 files / `144051241571` bytesをlocal検証済み。権利・地域review未完了 |
| H3 derivative weights | 未作成 | Model Derivative候補 | MiniMax H3 Community License | 配布禁止 | 対象外 |
| AdaLN cache / LoRA / quantized weights | 未作成 | Model Derivative候補 | 未確定 | 配布禁止 | 対象外 |
| Benchmark outputs | 固定H3 baselineおよび40層candidateで作成 | H3 Output | H3 license・AUP review必須 | 非公開・Git管理禁止 | 各protocolでwarmup 1回・測定3回済み。candidate measured 3成果物はexact gate合格。配布review待ち |
| Exact quality reference metadata | local Benchmark outputsから生成したdigest・media metadata | H3 Output自体を含まず、Outputのhashとstream metadataを記録 | H3 license・AUP review必須 | Public候補 | 単一caseのplacement-only gateで実測検証済み。一般品質・配布review待ち |
| Benchmark protocols | H3Fast独自metadata | 固定revision、runtime、case、quality methodを記録。prompt本文1件を含む | Apache-2.0候補。promptの公開権reviewが必要 | Public候補 | 20層baselineと40層既定protocolをschema検証済み。formal set未完成 |

## Initial boundary decision

- H3Fast repositoryへMiniMax H3のコード、設定、重み、documentationをcopyしない。
- 初期CIはH3をdownload・import・実行しない。
- SGLangはruntime dependencyにせず、version検出とadapter契約だけを実装する。
- H3を用いるGPU CI、benchmark、変換は、Applicable Territoryと対象環境のreview完了後に別途追加する。
