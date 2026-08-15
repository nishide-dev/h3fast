# H3 baseline runtime

- Status: Accepted for Phase 1A experiment
- Date: 2026-08-15
- Related: Issue #1

## Context

MiniMax H3対応はSGLang v0.5.16のrelease後、PR #33275で導入されました。そのため既存release versionだけを固定してもH3 pipelineを再現できません。最新PyPI版の通常依存解決はCUDA 13とPyTorch 2.11を選びますが、検証hostのdriver 555.58.02はCUDA 12系runtimeを対象とします。

hostには4基のRTX 6000 Ada Generation 48GBがありますが、GPU 0は別projectが使用中です。他jobへ干渉せず検証するには、空いているGPUから2基だけを明示選択する必要があります。

## Decision

- MiniMax H3をHugging Face commit `42ed227ee7df40d41602854ae760620d6eb651fe`へ固定する。
- 参照backend sourceをSGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`へ固定する。
- CUDA user-spaceは公式amd64 image `lmsysorg/sglang@sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f`をSingularity SIFへ変換して使用する。
- image内のrelease sourceは使用せず、固定SGLang checkoutの`python/`をread-only bindし、`PYTHONPATH`で優先する。
- 2×RTX 5090向け公式構成を基に、TP2、Ulysses1、memory mode、DiT resident 20 layers、text encoder・DiT・VAE layerwise offload、`torch.compile`無効で開始する。
- preflightは選択GPU上のcompute process、45,000 MiB未満の空きVRAM、384 GiB未満のRAM、driver、snapshot容量、source revision、runtime imageをfail-closedで拒否する。
- 選択GPUの排他性はpreflight時だけでなく、model loadから生成完了まで維持する。途中でforeign compute processを検出した試行は中断し、性能結果として採用しない。
- H3Fastが依存するSGLang面は公開`sglang serve` CLIと非同期`/v1/videos` endpointに限定する。内部Python APIをH3Fast packageからimportしない。

## Consequences

この構成はH3Fastの配布targetではなく、Phase 1Aの内部再現実験です。SIF、SGLang checkout、H3 snapshot、生成物はGit管理せず再配布しません。H3 E2Eが完走するまではRTX 6000 Adaをsupport済みまたはTier 1/2と表示しません。

release tagとsource commitを組み合わせるため、依存ABIの不一致が起こり得ます。最初の起動でimportまたはkernel互換性を確認し、不一致時は暗黙fallbackせず実験結果へ失敗理由を記録します。実機検証後に競合する依存graphを継続保守する必要が確認された場合だけ、`targets/nvidia-ada/`独立projectを追加します。

最初の実機試行はDiT staging中に別workloadが選択GPUへ入ったため中断しました。互換性検査とtext encoder loadは成功しましたが、E2E完走とはみなしません。詳細は[`docs/experiments/0001-rtx6000-ada-baseline-smoke.md`](../experiments/0001-rtx6000-ada-baseline-smoke.md)に記録します。
