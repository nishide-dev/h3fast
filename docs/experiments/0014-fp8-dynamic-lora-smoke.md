# Online FP8 with dynamic LoRA: smoke and canary

- Date: 2026-08-20 (Asia/Tokyo)
- Baseline: `h3fast-phase1b-turbo-lora-12-v1`(既定balanced profile、BF16、`merge_mode: auto`)
- Candidate: `h3fast-phase1b-turbo12-fp8-v1`(同構成 + `--quantization fp8`、`merge_mode: dynamic`)
- Canary: `h3fast-phase1b-fp8-nolora-canary-v1`(FP8、LoRAなし)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Class: **Class R**(計算削減系、測定前に宣言、[ADR 0014](../decisions/0014-compute-reduction-optimization-class.md))
- Related: [Issue #55](https://github.com/nishide-dev/h3fast/issues/55), [experiment 0013](0013-turbo-lora-step-sweep.md)
- Outcome: **実行可能を確認**。FP8はVRAM −28%、速度1.17×。品質評価は未実施。

## Purpose

sm89でonline FP8量子化が実行可能かを確認する。初回試行は`merge_mode: auto`でserver起動が失敗したため、原因を特定し`dynamic`での実行可能性を検証する。

## Root cause of the auto-merge failure

初回失敗の例外は次であった。

```
File ".../multimodal_gen/runtime/layers/lora/linear.py", line 264, in _merge_lora_into_data
    data_2d[start:end].add_(chunk_delta, alpha=scale)
RuntimeError: The size of tensor a (10752) must match the size of tensor b (5376)
```

**原因はdtypeのbyte幅ではなく、FP8 runtime weightの転置layoutである。** PyTorch tensorの`shape`はdtypeでは変化しない。`Fp8LinearMethod.process_weights_after_loading()`が量子化後にCUTLASS/Marlin向けruntime表現へ明示的に転置する。

```python
qweight, weight_scale = per_token_group_quant_fp8(layer.weight, layer.weight.shape[-1])
weight_scale = weight_scale.t().contiguous()
layer.weight = Parameter(qweight.t(), requires_grad=False)
```

H3のTP2 QKV projectionで数値が一致する。

| 対象 | shape |
|---|---|
| logical local QKV weight | `[10752, 5376]` |
| online FP8後のruntime storage | `[5376, 10752]` |
| LoRA delta (B @ A) | `[10752, 5376]` |

static mergeはruntime storageを論理weightとみなして加算するため、dim 1が10752対5376で不一致となる。

**shapeだけを合わせる修正では不十分である。** `layer.weight`はscaleを伴う量子化値であり、正しいstatic mergeにはdequantize → logical orientationでdelta加算 → 再quantize → `weight_scale`再計算 → runtime orientationへ変換が必要である。現在の`merge_lora_weights()`は`weight_scale`を更新しないため、shape mismatchを回避しても数値的に壊れたweightになる。SGLangのopen PR #30487は別のweight-only FP8 Linearについて同じ判断を示し、FP8 wrapperをdynamic-onlyとして明示mergeをerrorにしている。

`_should_merge_lora_for_layers()`のmerge判断は`dynamic`以外ではDTensor/FSDPのみを確認し、量子化状態を見ない。これは確認時点のSGLang mainでも同じであり、H3Fastの引数生成やLarry Turbo LoRA固有のformat問題ではなく、**SGLangの汎用LoRA auto-merge policyがquantized runtime weightを考慮していない問題**である。

## Why dynamic is structurally compatible

dynamic経路は量子化weightへdeltaをmergeしない。base layerのquantization methodで出力を計算し、LoRAを元のprecisionで別計算して出力tensorへ加算する。

```
y = dequant(Q(W)) x + scale * B(Ax)
```

したがってbase weightはFP8 runtime形式のまま、base GEMMはFP8 kernelを使用し、LoRA A/BはBF16のまま、`weight_scale`は変更されない。**量子化対応LoRAやINT8 LoRA loaderは不要である。** ComfyUIのINT8 ConvRot LoRAはadapter自体の保存形式が特殊であるため専用loaderを要する別問題であり、標準PEFT形式のadapterを量子化baseへdynamic適用することとは区別する。

## Results

### Startup verification(全項目合格)

| 項目 | 結果 |
|---|---|
| startup | 183秒 |
| `quantization` | `fp8` |
| `lora.merge_mode` | `dynamic` |
| LoRA applied layers | **259 / 266**、shape mismatchなし |
| 除外された7層 | `video_patch_proj`、`audio_patch_proj`、`condition_proj`、`time_embedder.proj_in/out`、`final_layer.video_out/audio_out` |
| server log | `LoRA adapter(s) ... applied to 259 layers (targets: all, strengths: 1.00, merge_mode=dynamic)` |

除外された7層は、cookbookが「H3 automatically keeps its video/audio patch projections, timestep MLP, and final video/audio heads in FP32」と記載する層と一致する。

### Speed and memory(smoke-001、12 sigma points、同一seed)

| 構成 | elapsed | peak VRAM | artifact SHA-256 |
|---|---|---|---|
| BF16 + Turbo (auto) | 165.6秒 | 36,412 MiB | `e457ca45e405e10d…` |
| **FP8 + Turbo (dynamic)** | **141.5秒** | **26,204 MiB** | `4362b0670b9f6026…` |
| FP8 + LoRAなし (canary) | 134.5秒 | 22,530 MiB | `628e489da6e5618d…` |

- FP8の効果: **速度1.17×、peak VRAM −10,208 MiB(−28.0%)**
- dynamic LoRAのoverhead: **+7.0秒(約5%)、+3,674 MiB**(FP8構成のD対Cで分離)
- net VRAM benefit: FP8による削減13,882 MiBからadapter常駐分3,674 MiBを差し引いて**10,208 MiB**

### Canary(LoRA適用の証拠)

同一FP8 server条件・同一seed・同一12 pointsでLoRA on/offを比較した。

```
FP8 + Turbo : 4362b0670b9f6026…
FP8 no-LoRA : 628e489da6e5618d…
differ      : True  -> PASS
```

artifactが相違するため、起動成功だけでなく**LoRAがforwardで実際に適用されている**ことが確認できた。一致した場合は適用証拠として不合格とする基準であった。

## Limits

- **品質は未評価である。** `FP8 Q(W) + BF16 delta`は`BF16 W + BF16 delta`と同一出力ではなく、Tier 2のblind pairwiseを要する。本記録は実行可能性と速度・memoryの観測に限られる。
- 単一case(smoke-001)・単一runである。
- `denoise_steps_seconds`が空であり、「11 denoiser evaluations」を確認していない。これはFP8固有ではなく`run-formal-cases`経路がserver performance dumpを取得しないためで、BF16 turbo12でも同様である。suite経路での確認が必要。
- 4構成マトリクスのB(BF16 + dynamic)は未測定である。dynamic overheadはFP8側(D対C)で7秒と分離したが、BF16側で同程度かは未確認。
- `--quantization-ignored-layers`によるhybrid量子化は未評価である。Turbo LoRAはDiTのQKV/attention out/MLP/AdaLN/token refiner/input projection/final出力を広く対象とするため、全active LoRA targetをBF16へ戻すとFP8の利点の多くを失う見込みであり、恒久回避策としては推奨しない。
- online FP8は近似であり、cookbookが「not a consistency ground-truth mode」と明記している。

## Consequences

FP8 + LoRAを使う構成では`merge_mode: dynamic`が必須である。`auto`は量子化状態を考慮せずstatic mergeを選び、shape mismatchで起動に失敗する。protocolの`runtime.lora.merge_mode`はこの選択を機械可読に固定する。

upstream報告の価値がある最小再現が揃った。2026-08-22にSGLang upstreamへ報告した([issue #35970](https://github.com/sgl-project/sglang/issues/35970)、[PR #35975](https://github.com/sgl-project/sglang/pull/35975))。報告にあたりpinned commitではなくupstream mainのソースで再現を確認しており(commit `5c03069d4bce87c97b257ad05f9d497729a47c4f`、同一の`linear.py:264`で同一のshape mismatch)、この確認runはpinned環境外のためbenchmark記録には使用していない。提出したissue/PR本文の作業コピーは`docs/upstream/`にGit管理外で保持する。

```
online --quantization fp8 + standard BF16 LoRA
  --lora-merge-mode auto    -> shape mismatch (10752 vs 5376)
  --lora-merge-mode dynamic -> successful inference
```

望ましい修正は`auto`をquantization-awareにすること(quantized/runtime-packed weightではdynamicへ切り替え、明示`merge`はstartup早期に説明的にfail)である。dtypeだけの判定は将来壊れやすいため、`quant_method.supports_lora_merge`のようなcapability判定が望ましい。

## Reproduction

- FP8 + LoRA: `serve-guarded --protocol benchmarks/protocol-turbo12-fp8.yaml --sage-attention-path <ada-build> --lora-path <adapter-dir>`
- canary: `--protocol benchmarks/protocol-fp8-nolora.yaml`(同一構成でLoRAのみ除去)
- 生成: `run-formal-cases --split smoke --task t2va`
- 比較: 両構成のartifact SHA-256、`elapsed_seconds`、`server.peak_memory_mib`
