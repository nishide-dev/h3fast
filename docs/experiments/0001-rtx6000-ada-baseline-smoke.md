# RTX 6000 Ada baseline smoke attempt

- Date: 2026-08-15 (Asia/Tokyo)
- Protocol: `h3fast-phase1a-baseline-v1`
- Related: Issue #1
- Outcome: Interrupted before server readiness

## Purpose

固定したMiniMax H3 FL2VA snapshotとSGLang runtimeが、2基のRTX 6000 Ada GenerationでBF16 lossless baselineを起動できるか確認する。性能値の測定ではなく、Phase 1A harnessとruntime互換性のsmoke testを目的とした。

## Fixed inputs

- H3 revision: `42ed227ee7df40d41602854ae760620d6eb651fe`
- SGLang revision: `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`
- Base image: `lmsysorg/sglang@sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f`
- SIF SHA-256: `c20b1b3c7da5b164d2783859937e4189c6265a2ad0b4f7be4b7329b7feebd2a4`
- SIF size: `15102517248` bytes
- Runtime: PyTorch `2.11.0+cu129`, CUDA `12.9`, NCCL `2.28.9`
- Topology: TP2、Ulysses1、memory mode、DiT resident 20 layers、layerwise offload、`torch.compile`無効

## Observations

1. CPU-only importで固定SGLang sourceのMiniMax H3 pipelineを読み込めた。
2. CUDA importで明示した2基だけがcontainerから認識された。
3. preflightはmodel revision、snapshot 84 files / `144051241571` bytes、host RAM、driver、GPU、SGLang revision、SIF digestを含む全項目で通過した。
4. serverはdistributed/NCCL初期化、FL2VA pipeline選択、text encoder 12 shardの読込、50 layerのoffload設定を完了した。
5. BF16 DiT 13 safetensorsのCPU staging中に、無関係なworkloadが選択GPUの一方へ新規に入った。
6. 排他条件が崩れた時点でserverを中断し、H3Fast側GPU memoryが解放されたことを確認した。

## Result

生成requestは送信しておらず、MP4、latency、quality結果はない。この試行から性能値やRTX 6000 Ada対応を主張してはならない。runtime/source互換性とDiT staging到達までは確認できたが、E2E baselineは未完了である。

## Follow-up

- 2基のGPUがload開始から生成完了まで排他的に利用できるwindowで同じprotocolを再実行する。
- 実行直前だけでなく実行中と終了時にもforeign compute processを確認する。
- E2E完走まではIssue #1をcloseせず、benchmark protocolを`draft`のまま維持する。
