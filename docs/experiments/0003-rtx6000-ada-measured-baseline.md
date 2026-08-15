# RTX 6000 Ada measured BF16 baseline

- Date: 2026-08-15 (Asia/Tokyo)
- Protocol: `h3fast-phase1a-baseline-v1`
- Related: Issue #3
- Outcome: Warmup 1回と測定3回を完走

## Purpose

固定したMiniMax H3 FL2VA snapshotとSGLang runtimeを使い、2基のRTX 6000 Ada GenerationでT2VA BF16 baselineを規定回数測定する。warmupを統計から除外し、client E2E、server inference、stage別時間、reported peak GPU memory、生成artifactを記録する。

## Fixed inputs

- H3 revision: `42ed227ee7df40d41602854ae760620d6eb651fe`
- H3 snapshot: 84 files / `144051241571` bytes
- SGLang revision: `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`
- Base image: `lmsysorg/sglang@sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f`
- SIF SHA-256: `c20b1b3c7da5b164d2783859937e4189c6265a2ad0b4f7be4b7329b7feebd2a4`
- ffprobe adapter SHA-256: `f69a957140cce3d55be043fb4f89cb7cdbf18fd45cf90ea4d4dc6b0d9daca8d0`
- Runtime: PyTorch `2.11.0+cu129`、Triton `3.6.0`、CUDA `12.9`、driver `555.58.02`
- GPU: RTX 6000 Ada Generation 48GB × 2
- Topology: TP2、Ulysses1、memory mode、DiT resident 20 layers、layerwise offload、`torch.compile`無効
- Loader: standard safetensors loader（`SGLANG_USE_RUNAI_MODEL_STREAMER=false`）
- Case: 1344×768、124 frames、24 fps、50 inference steps、seed `1101`

## Procedure

1. `serve-guarded`でsnapshot、source、runtime、media probe、host resourceと選択GPUの排他性を検証した。
2. server起動後も選択GPUを監視し、`/health`成功前にNVIDIA process queryを再検証した。
3. 同じready serverへ`smoke-001`を4回連続送信し、先頭1回をwarmupとして統計から除外した。
4. request IDに対応するSGLang performance dump、client result、artifact hash、media contractをrunごとに検証した。
5. suite完了後、hostの独立した`ffprobe 4.4.2`で4成果物を再検証した。

server起動からreadyまでは`485.707095`秒だった。この値にはmodel loadとSGLang synthetic warmupを含むが、client E2E集計には含めない。

## Results

| Run | Client E2E (s) | Server inference (s) | Text encode (s) | Denoise (s) | Decode (s) | Reported peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| warmup-001（除外） | 893.521093 | 890.724257 | 0.655331 | 851.278921 | 38.392354 | 23,376 |
| measured-001 | 889.512870 | 887.177244 | 0.643230 | 847.748445 | 38.394863 | 23,376 |
| measured-002 | 883.467506 | 880.905419 | 0.643740 | 841.481252 | 38.394567 | 23,362 |
| measured-003 | 889.495172 | 886.759141 | 0.643419 | 847.339288 | 38.394162 | 23,362 |

| Metric | min | p50 | p95 | max |
|---|---:|---:|---:|---:|
| Client E2E (s) | 883.467506 | 889.495172 | 889.511100 | 889.512870 |
| Server inference (s) | 880.905419 | 886.759141 | 887.135434 | 887.177244 |
| Denoise (s) | 841.481252 | 847.339288 | 847.707530 | 847.748445 |
| Decode (s) | 38.394162 | 38.394567 | 38.394833 | 38.394863 |
| Reported peak GPU memory (MiB) | 23,362 | 23,362 | 23,374.6 | 23,376 |

denoiseのp50はserver inference p50の約95.55%を占め、明確な支配stageだった。

## Artifact validation

warmupを含む4成果物はすべて次の値で一致した。

- MP4 size: `912408` bytes
- MP4 SHA-256: `530a6bc980cf357d0518ea60366bc09dd1868ee77f46ee3f43681ca8822f5909`
- Video: H.264、1344×768、24 fps、124 frames、5.166667秒
- Audio: AAC、32,000 Hz、stereo、5.175000秒、163 frames
- Container: 5.207000秒
- A/V duration drift: 0.008333秒

この同一性は固定caseにおける再現性の観測であり、一般的な品質同等性や無損失性の証明ではない。生成物と詳細resultはH3 Outputまたはlocal pathを含むためGit管理しない。

## Excluded attempts and harness corrections

- model loadまたは測定中に選択GPUへ外部compute processが入った試行は、guardで中断し測定値から除外した。
- model staging中にNVIDIA process queryがtimeoutする場合があった。`/proc` device holderは全GPUを列挙しただけのprocessも含むため候補としてだけ扱い、ready前にprocess queryでcompute利用を再確認するようguardを修正した。ready後はquery failureをfail-closedで扱う。
- RunAI model streamerは2回、18分以上および30分以上進捗を返さず停止したため、両試行を除外した。通常safetensors loaderではDiT shardを直ちに読み始めたため、このprotocolではRunAI streamerを明示的に無効化した。

これらの除外は成功runだけを選ぶためではなく、GPU排他性または固定loader条件を満たさない試行をprotocol外として扱うためである。

## Interpretation

規定したwarmup 1回と測定3回の性能baselineを取得し、固定caseのartifact再現性とmedia contractを確認した。次の単一最適化では、支配stageであるdenoiseと、その中で発生するlayerwise offload/placementだけを変更対象にする。

後続変更で単一`smoke-001`のexact quality reference gateは実装・実測した。一方、公開可能な10件以上のsmoke set、50件以上のregression set、知覚品質および法務確認は未完了である。したがってprotocol statusは`draft`、RTX 6000 AdaはTier未付与のままとし、外部向けにlosslessまたは品質同等を主張しない。
