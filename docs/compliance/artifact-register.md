# Artifact register

最終更新: 2026-08-15

この表は技術的な来歴記録であり、法的助言ではありません。公開前に適切なreviewerの承認を記録してください。

| 成果物 | 出所 | H3との関係 | License | 配布範囲 | 状態 |
|---|---|---|---|---|---|
| H3Fast source | 独自実装 | H3専用tooling。MiniMax Materialsは未収録 | Apache-2.0 | Public候補 | 法務レビュー待ち |
| `h3fast` wheel/sdist | H3Fast sourceからbuild | H3重み・cacheを含まない | Apache-2.0 | Public候補 | 法務・release gate待ち |
| CPU CI artifacts | H3Fast sourceとtest fixture | 合成fixtureのみ。H3 Materialsを扱わない | Apache-2.0 | CI内部 | 利用可 |
| SGLang adapter metadata | 独自実装 | SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`との互換性とmedia probeを記録。コードcopyなし | Apache-2.0 | Public候補 | 単一H3 E2E smoke済み。品質・support未検証 |
| SGLang source checkout | sgl-project/sglang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e` | H3 pipelineを含む外部runtime source。repositoryには未収録 | Apache-2.0ほか依存license | ローカル実験限定・再配布なし | runtime-cacheで検証中 |
| SGLang CUDA 12.9 SIF | `lmsysorg/sglang` amd64 manifest `sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f` | H3重みを含まない外部runtime image | image内licenseに従う | ローカル実験限定・再配布なし | 取得・実機検証中 |
| MiniMax H3 FL2VA snapshot | `MiniMaxAI/MiniMax-H3` commit `42ed227ee7df40d41602854ae760620d6eb651fe` | MiniMax H3 Works | MiniMax H3 Community License | 配布禁止 | ローカル取得・検証中 |
| H3 derivative weights | 未作成 | Model Derivative候補 | MiniMax H3 Community License | 配布禁止 | 対象外 |
| AdaLN cache / LoRA / quantized weights | 未作成 | Model Derivative候補 | 未確定 | 配布禁止 | 対象外 |
| Benchmark outputs | 固定H3 E2E smokeで作成 | H3 Output | H3 license・AUP review必須 | 非公開・Git管理禁止 | 2×RTX 6000 Ada smoke検証済み。品質・配布review待ち |

## Initial boundary decision

- H3Fast repositoryへMiniMax H3のコード、設定、重み、documentationをcopyしない。
- 初期CIはH3をdownload・import・実行しない。
- SGLangはruntime dependencyにせず、version検出とadapter契約だけを実装する。
- H3を用いるGPU CI、benchmark、変換は、Applicable Territoryと対象環境のreview完了後に別途追加する。
