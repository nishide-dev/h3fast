# Denoise kernel profile: NCCL AllReduce is 24.5%, compute is already optimized

- Date: 2026-08-22 (Asia/Tokyo)
- Configuration: `balanced` profile(既定、FP8 + turbo LoRA dynamic + Sage、TP2)
- Case: smoke-001(t2va、12 sigma points = 11実効step、768p、4秒)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Related: [experiment 0017](0017-default-profile-stage-breakdown.md)(stage内訳), spec §14.2
- Outcome: **kernel最適化の余地は小さい。** 最大の単一要因はTP2のNCCL AllReduce(24.5%)である。

## Purpose

[experiment 0017](0017-default-profile-stage-breakdown.md)でdenoiseが77.3%を占めることが分かったが、その内訳(attention / MLP / AdaLN / 通信)は不明であった。kernel最適化の対象を推測なしに選定するため、denoise内部のCUDA kernel分布を測定する。

## Method

上流のprofiling手順([`sglang-diffusion-benchmark-profile`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile))に従った。

- server引数ではなく**リクエストの`profile: true`**で有効化する(`num_profiled_timesteps: 3`)
- `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`を維持
- traceは`SGLANG_DIFFUSION_TORCH_PROFILER_DIR`未設定時に`./logs`(container cwd `/outputs`)へ出力される
- kernelランキングは上流文書付属のparserを使用した

server側の`inference_time_s`は149.6秒であり、profiling有効時のオーバーヘッドを含む。

## Results

denoise 3 stepのGPU時間41.30秒(55種のkernel)の内訳。

| 分類 | kernel | 時間 | 比率 | 呼数 |
|---|---|---|---|---|
| **通信** | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | **10.12s** | **24.5%** | 432 |
| GEMM | `cutlass::Kernel2<DefaultGemmWithVisitor>` (2種) | 13.13s | 31.8% | 863 |
| attention | `qk_int_sv_f8_attn_kernel` (Sage) | 9.06s | 21.9% | 216 |
| elementwise | `vectorized_elementwise_kernel` (2種) | 3.14s | 7.6% | 2166 |
| 量子化 | `sglang::per_token_quant_fp8_warp_kernel` | 0.70s | 1.7% | 863 |
| offload | `Memcpy HtoD (Pinned -> Device)` | 0.66s | 1.6% | 343 |
| 融合norm | `_indexed_gate_bf16_kernel` | 0.55s | 1.3% | 432 |
| 融合norm | `_indexed_scale_shift_bf16_kernel` | 0.38s | 0.9% | 436 |
| 融合QK | `sglang::fused_qknorm_rope_warp` | 0.27s | 0.6% | 216 |

上位9分類で89.7%を占める。

## Classification against known fast paths

上流文書§4は「hot kernelから直接新規実装に飛ばず、既知のmainline familyに対して分類せよ」と指示する。その表に照らした結果は次である。

| 確認項目 | 判定 |
|---|---|
| `fused_inplace_qknorm_rope`が不発で、qk normとropeが別々に出ているか | **いいえ。** `sglang::fused_qknorm_rope_warp`が216回動作している |
| LayerNormとadaLN elementwiseが別々に出ているか | **いいえ。** `_indexed_scale_shift_bf16_kernel`と`_indexed_gate_bf16_kernel`という融合済みkernelが動作している |
| packed-QKVのfast-path missか(`to_q -> to_k -> to_v`) | **いいえ。** 個別のprojection呼び出しは出ていない |

**既存のfast pathはすべて機能している。** 新規融合を提案する根拠はない。

計算側も既に最適化済みである。GEMMはFP8量子化済み(`per_token_quant_fp8`が伴走している)、attentionはSageのINT8実装である。融合可能に見えるelementwiseは合計7.6%で、しかも既にvectorized実装である。

## Interpretation

**最大の単一要因はkernelではなく通信である。** NCCL AllReduceが24.5%(432回、3 stepあたり)を占める。これはTP2で層を分割していることの構造的コストであり、kernel実装の改善では減らない。

削減方向は2つある。

1. **TP1化** — AllReduceが消える。実装は`tp_size == 1`分岐を持ち(`models/dits/minimax_h3.py`のweight読み込みとall_gather)、正式にサポートされている。ただし公式cookbookは2〜8 GPU構成のみを記載し、単一GPU構成の推奨値を示していない
2. **並列方式の変更** — `--ulysses-degree` / `--ring-degree`は現在1である。sequence parallelでは通信パターンが変わる。ただしH3は現行実装でring attention時にFA以外のbackendを拒否するため、Sageとの併用に制約がある([experiment 0009](0009-sage-attention-noop.md))

TP1のVRAM見積もりは、TP2実測のpeak 31.7GB + startup resident 11.2GB ≒ **43GB**(48GB容量に対し約5GBの余裕)である。ただしactivation側が縮まない前提の粗い見積もりであり、resident層数40はTP2のsharding前提で調整された値である。

## Limits

- **単一case・3 stepのwindowである。** 11 step全体やcase間のばらつきは測定していない。
- profiling有効時のオーバーヘッドを含む(`inference_time_s` 149.6秒 対 通常時の約133秒)。kernel間の**相対比率**は有効だが、絶対時間は通常実行と一致しない。
- decodeの内訳(video VAE 対 audio decode)は未取得である。上流は`video_vae.decode_base`と`_decode_audio`のinner scopeを設定するよう求めており、本測定はdenoise既定windowのみを対象とした。
- TP1のVRAM見積もりは未検証の計算である。実測していない。
- 通信24.5%はrank0のtraceであり、rank間の非対称性は測定していない。
- H2D memcpy 1.6%はlayerwise offloadの転送だが、GPU時間としての計上であり、CPU側の待ち時間は含まない。

## Consequences

**kernel最適化は次の候補として妥当ではない。** 既存のfast pathは機能しており、計算はFP8とINT8で量子化済みである。上流文書の分類手順に従った結果、新規融合を書く根拠が見つからなかった。

次に検証すべきはTP1化である。これはkernel実装ではなく起動設定の変更であり、通信を削減する。ただしlaunchが2 GPU固定であるため実装変更を要し、VRAM収容性は実測で確認する必要がある。公式が単一GPU構成を記載していないため、推奨値の裏付けはない状態での探索になる。
