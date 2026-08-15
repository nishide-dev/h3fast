# RTX 6000 Ada DiT resident 40-layer experiment

- Date: 2026-08-16 (Asia/Tokyo)
- Baseline protocol: `h3fast-phase1a-baseline-v1`
- Candidate protocol: `h3fast-phase1b-resident40-v1`
- Related: Issue #7
- Outcome: Adopted with an explicit 20-layer rollback

## Purpose

固定BF16 baselineでserver inferenceの約95.55%を占めたdenoise stageに対し、DiT 50層のGPU resident数だけを20から40へ増やす。offload量を減らす仮説を、同一case、schedule、runtime、GPU topologyおよび集計方法で単独測定する。

## Fixed inputs and changed variable

H3 revision、snapshot、SGLang commit、runtime image、driver、TP2/Ulysses1、BF16、50-step schedule、seed、codec、`torch.compile`無効、standard safetensors loader、GPU排他guardは[`0003-rtx6000-ada-measured-baseline.md`](0003-rtx6000-ada-measured-baseline.md)と同一である。

変更した値は次の1項目だけである。

- Baseline: `runtime.dit_layerwise_resident_layers = 20`
- Candidate: `runtime.dit_layerwise_resident_layers = 40`

protocolの差分がprotocol IDとこの値だけであることをunit testで検証する。実効値40はlaunch argv、server lifecycle、suite bundleで一致を確認した。protocolと起動済みserverの値が異なる場合、suiteは測定開始前に拒否する。

## Procedure

1. 2×RTX 6000 Adaの空き容量と外部compute process不在をpreflightで確認した。
2. `serve-guarded`で40層設定のserverを起動し、ready後も選択GPUの排他性を監視した。
3. 固定`smoke-001`を同じserverへ4回連続送信し、先頭1回をwarmupとして除外した。
4. measured 3回のclient、request-scoped server metrics、stage、peak memoryおよびmedia contractを集計した。
5. 全measured artifactを`exact-decoded-artifact-v1` referenceへ照合した。

server起動からreadyまでは`322.377380`秒だった。baselineの`485.707095`秒より短いが、起動時間は各protocol 1回だけの観測であり、採用の主要根拠にはしない。

## Candidate results

| Run | Client E2E (s) | Server inference (s) | Text encode (s) | Denoise (s) | Decode (s) | Reported peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| warmup-001（除外） | 883.527238 | 881.235794 | 0.653226 | 841.800606 | 38.391983 | 35,696 |
| measured-001 | 885.470775 | 883.927158 | 0.643708 | 844.493141 | 38.405997 | 35,696 |
| measured-002 | 877.485159 | 875.962151 | 0.643574 | 836.529517 | 38.408353 | 35,682 |
| measured-003 | 883.515755 | 881.937033 | 0.643477 | 842.506824 | 38.407201 | 35,682 |

| Metric | min | p50 | p95 | max |
|---|---:|---:|---:|---:|
| Client E2E (s) | 877.485159 | 883.515755 | 885.275273 | 885.470775 |
| Server inference (s) | 875.962151 | 881.937033 | 883.728146 | 883.927158 |
| Denoise (s) | 836.529517 | 842.506824 | 844.294509 | 844.493141 |
| Decode (s) | 38.405997 | 38.407201 | 38.408237 | 38.408353 |
| Reported peak GPU memory (MiB) | 35,682 | 35,682 | 35,694.6 | 35,696 |

## Baseline comparison

| Metric (p50 unless noted) | 20-layer baseline | 40-layer candidate | Change |
|---|---:|---:|---:|
| Client E2E (s) | 889.495172 | 883.515755 | -5.979417 s (-0.672%) |
| Server inference (s) | 886.759141 | 881.937033 | -4.822108 s (-0.544%) |
| Denoise (s) | 847.339288 | 842.506824 | -4.832464 s (-0.570%) |
| Decode (s) | 38.394567 | 38.407201 | +0.012634 s (+0.033%) |
| Reported peak maximum (MiB) | 23,376 | 35,696 | +12,320 MiB (+52.704%) |

E2Eとdenoiseの両p50が事前の採用条件を満たした。一方、約6秒のE2E改善に対して12,320 MiB多く使うため、これは速度と容量の明確なtrade-offである。

## Quality and artifact validation

measured 3 runはすべてreferenceと次の値で完全一致し、failed checkは0だった。

- MP4 SHA-256: `530a6bc980cf357d0518ea60366bc09dd1868ee77f46ee3f43681ca8822f5909`
- Decoded video SHA-256: `e4920b0ad49c51f2e4032c0156b6f1adeaa9955377ca7ac700f6be528c7ee836`
- Decoded audio SHA-256: `d635ad6825998a1322d964d252d53e923e9832e1566ecf6f47614f1c16ad1712`
- media metadata、A/V duration drift、artifact size: reference一致
- environment check: `ffmpeg` / `ffprobe` 2/2 pass

生成MP4、raw stream、server log、full suiteおよびquality reportは`benchmark-results/`以下のlocal artifactであり、Git管理またはuploadしない。

## Adoption and rollback

OOMなし、E2E/denoise改善、exact gate合格のため、40層設定を[`benchmarks/protocol.yaml`](../../benchmarks/protocol.yaml)の既定値として採用する。35,696 MiBのreported peakとprotocolのpreflight余裕を確保できない場合は、[`benchmarks/protocol-baseline20.yaml`](../../benchmarks/protocol-baseline20.yaml)を明示して20層へrollbackする。fallbackは自動で黙って行わず、選択したprotocolと実効値をmetadataへ残す。

この結果は単一case、単一host、測定3回のlocal A/Bである。10/50件の正式quality set、知覚品質、clean-machine再現、他task/GPU、法務確認は未完了であり、lossless、Support Tier、公開性能を主張しない。
