# Artifact register

最終更新: 2026-08-15

この表は技術的な来歴記録であり、法的助言ではありません。公開前に適切なreviewerの承認を記録してください。

| 成果物 | 出所 | H3との関係 | License | 配布範囲 | 状態 |
|---|---|---|---|---|---|
| H3Fast source | 独自実装 | H3専用tooling。MiniMax Materialsは未収録 | Apache-2.0 | Public候補 | 法務レビュー待ち |
| `h3fast` wheel/sdist | H3Fast sourceからbuild | H3重み・cacheを含まない | Apache-2.0 | Public候補 | 法務・release gate待ち |
| CPU CI artifacts | H3Fast sourceとtest fixture | 合成fixtureのみ。H3 Materialsを扱わない | Apache-2.0 | CI内部 | 利用可 |
| SGLang adapter metadata | 独自実装 | `sglang==0.5.15.post1`との互換候補を記録。コードcopyなし | Apache-2.0 | Public候補 | H3 E2E未検証 |
| H3 weights / derivative weights | 未作成 | MiniMax H3 Works / Model Derivative候補 | MiniMax H3 Community License | 配布禁止 | 対象外 |
| AdaLN cache / LoRA / quantized weights | 未作成 | Model Derivative候補 | 未確定 | 配布禁止 | 対象外 |
| Benchmark outputs | 未作成 | H3 Outputを含み得る | H3 license・AUP review必須 | 非公開 | 4×RTX 6000 AdaでのH3 E2E・license review待ち |

## Initial boundary decision

- H3Fast repositoryへMiniMax H3のコード、設定、重み、documentationをcopyしない。
- 初期CIはH3をdownload・import・実行しない。
- SGLangはruntime dependencyにせず、version検出とadapter契約だけを実装する。
- H3を用いるGPU CI、benchmark、変換は、Applicable Territoryと対象環境のreview完了後に別途追加する。
