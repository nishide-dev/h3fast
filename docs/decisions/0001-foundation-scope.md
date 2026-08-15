# Foundation scope

- Status: Accepted
- Date: 2026-08-15

## Context

repositoryはMNIST/PyTorch Lightning用templateで初期化されており、H3Fastの仕様と一致していません。実行環境では4基のNVIDIA RTX 6000 Ada Generation 48GBを確認できましたが、H3のbase revision、license review、品質referenceは未決定であり、H3 E2Eは未検証です。

## Decision

- templateのtraining code、Hydra config、Lightning/PyTorch依存を削除する。
- 初期実装は単一の`h3fast` distributionとする。
- Phase 1AのCPU-only基盤としてsnapshot、manifest、doctor、benchmark protocolの検証を実装する。
- SGLang adapterの参照候補を`0.5.15.post1`へ固定する。ただしH3 E2E検証完了まではsupport済みと表示しない。
- 最初のbaseline候補をFL2VA checkpointによるT2VA、768p short edge、16:9、5秒、4×RTX 6000 Ada Generation 48GBとする。
- H3 MaterialsをrepositoryまたはCIへ取り込まない。
- base revision、H3実行環境のlicense review、品質referenceは未解決事項として維持する。

## Consequences

初回pushだけではH3の生成や性能改善は実行できません。選択したGPU topologyがH3をresidentに保持できることもまだ保証しません。一方、後続のGPU検証が再現性、artifact検証、license境界を迂回せず追加できる基盤になります。
