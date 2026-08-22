# Parallelism and quantization limits on 48 GB GPUs

- Date: 2026-08-22 (Asia/Tokyo)
- Configuration: `balanced` profile(既定、online FP8 + turbo LoRA dynamic + Sage)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、48 GB each、sm89)
- Related: [experiment 0018](0018-denoise-kernel-profile.md)(NCCL AllReduce 24.5%), [Issue #55](https://github.com/nishide-dev/h3fast/issues/55)
- Outcome: **TP2が必須。** AllReduceの24.5%はこのハードウェアでは削減できない。主因はtext encoderが量子化対象外であること。

## Purpose

[experiment 0018](0018-denoise-kernel-profile.md)でdenoise時間の24.5%がTP2のNCCL AllReduceであることが判明した。kernel最適化の余地は小さいため、通信を削減する並列構成を探索する。

## Attempts and outcomes

| # | 構成 | 結果 |
|---|---|---|
| 1 | TP1、1 GPU、resident 40 | **OOM**(47.09 GB使用、99.62 MiB free) |
| 2 | TP1、1 GPU、resident 20 | **OOM**(同水準) |
| 3 | TP1 × Ulysses2、2 GPU | **OOM**(両GPUが同時に枯渇、transformerロード時) |

いずれもCUDA OOMであり、実装エラーや設定ミスではない。

### 試行1・2: 単一GPU

resident層数を40から20へ半減しても空きメモリが増えなかった。**支配的なのはDiTのresident weightではない**ことを示す。

### 試行3: Ulysses sequence parallel

上流cookbookはB200 8 GPUで`TP=1, Ulysses=8`、H200 16 GPUで`TP=1, Ulysses=8, Ring=2`を推奨構成として記載する。`tp_size`と`ulysses_degree`は独立した並列軸であり、World size = TP × Ulysses × Ringである。

H3の56 attention headsはUlysses degree 2で割り切れる(56 / 2 = 28)。Sage attentionとの併用も可能である: `supports_ring_rotation`はring parallel専用のチェックであり、Ulyssesには適用されない(かつSageは`True`を返す)。[experiment 0009](0009-sage-attention-noop.md)で記録した「ring時にFA以外を拒否する」制約はringに限られ、Ulyssesとは無関係である。

しかし**TP=1ではweightが分割されない**。Ulyssesはsequence次元を分割するのみで、各GPUは全parameterを保持する。結果として両GPUが同時に枯渇した。

## Root cause

snapshot内のcomponentサイズ(BF16)。

| component | サイズ |
|---|---|
| `text_encoder` | **63 GB** |
| `transformer` (DiT) | 62 GB |

**`--quantization fp8`はtransformerにのみ適用される。** server_argsのhelp textが「Quantization method for **the transformer**」と明記している。text encoderはBF16のまま63 GBであり、layerwise offloadされていても展開時のピークが48 GBに収まらない。

resident層数の調整が効かなかったのはこのためである。私が調整していたのはDiT側のみであった。

TP2ではtext encoderも2 GPUへ分散されるため成立している。

### 上流の推奨構成が使えない理由

| GPU | VRAM | TP=1でweightを保持できるか |
|---|---|---|
| B200 | 180 GB | ✅ |
| H200 | 141 GB | ✅ |
| RTX 6000 Ada | **48 GB** | ❌ |

上流がTP=1 + Ulysses8を推奨するのは大容量GPU構成である。48 GBではweight分割なしにH3は載らない。

## Text encoder quantization: an implementation asymmetry

**H3のtext encoderを量子化する経路は、現時点のSGLangに存在しない。** ただしこれはモデルや形式の制約ではなく、実装の非対称性である。

| encoder実装 | `quant_config` | 使用モデル |
|---|---|---|
| `models/encoders/ideogram.py` | **あり**(bitsandbytes 4-bit、weight-only FP8) | `Qwen3VLTextModel` |
| `models/encoders/minimax_h3_qwen3vl.py` | **なし**(206行中に記述ゼロ) | 同じ`Qwen3VLTextModel` |

`Qwen3VLTextModel`自体は`quant_config`と`use_weight_only_fp8`を受け取る設計であり、ideogram encoderはそれを利用している。H3 encoderが同じ配線を持たないため、`--quantization`がtext encoderへ届かない。

text encoderをFP8化できれば63 GB → 約32 GBとなり、TP1やUlyssesが成立する可能性がある。VRAM制約の主因がここにあるため、影響は大きい。

**pinned SGLangへのpatchは行わない。** 再現性が壊れ、benchmark結果の比較可能性が失われる。upstreamでの対応を待つか、報告する。

## Quantization format survey

より小さい量子化でVRAMを下げる経路も調査した。いずれも本runtimeでは使用できない。

| 形式 | サイズ | 使用可否 |
|---|---|---|
| online `fp8`(現行) | DiTのみ、実効 peak 31.7 GB/GPU | ✅ 使用中 |
| `modelopt_fp4`(NVFP4) | 10.86〜18.69 GB | ❌ `get_min_capability() == 100`(Blackwell専用、本機はsm89) |
| GGUF Q4_K (pruned) | 11.4 GB | ❌ ComfyUI / llama.cpp前提。SGLangに読み込み経路なし |
| unsloth `.pt` FP8 / INT8 | 20.2 GB | ❌ H3 DiTは`safetensors.torch.safe_open`のみを使用する |
| INT8 ConvRot text encoder | 27.1 GB | ❌ 単一ファイル形式(SGLangはHFディレクトリ構造を期待)。ConvRotは`--quantization`の選択肢にない |
| `bitsandbytes` NF4 | — | 事前量子化checkpointが必要。H3 DiT側の対応記述なし |
| `mxfp4` / `mxfp8` / `mxfp4_npu` | — | ❌ ROCm MI350+ / Ascend NPU専用 |

ComfyUI形式をSGLangで読む試みについては、上流discussion [#34079](https://github.com/sgl-project/sglang/discussions/34079)が5件の非公式patchを要すると報告している。特にComfyUI形式は`fp8_value = weight / weight_scale`で保存するが、SGLangのロード経路がscaleを除去するため**重みが約90倍で読み込まれる**。

同discussionはさらに、コミュニティのpruned FP8 checkpointが**数学的に壊れている**と報告する: 2688次元のtime-embeddingが8次元のlookup tableに置換されており、方向余弦0.07〜0.63(時に負)、大きさが10〜60倍ずれ、出力がフラットなグレースケールへ崩壊する。unslothのGGUFも`_pruned`系であり、同じ問題を抱えている可能性がある。

## Consequences

**TP2が本ハードウェアでは必須である。** NCCL AllReduceの24.5%は48 GB GPUでH3を動かすための構造的コストであり、削減できない。通信を減らす構成はweight分割の放棄を伴い、それがVRAM制約に抵触する。

既定profileは現行構成(TP2、online FP8、turbo LoRA dynamic、Sage、12 sigma points)を維持する。sm89 / 48 GB × 2 における到達点はFA baseline比で約6.9×である。

`--ulysses-degree`のlaunch対応はrepositoryへ残す。実行基盤としては正しく動作しており(topology検証とhead divisibility検証を含む)、大容量GPU環境またはtext encoder量子化が可能になった時点で使用できる。

## Limits

- OOMの内訳を計測していない。text encoderが主因という判断は、componentサイズ(63 GB)、`--quantization`のスコープ(transformerのみ)、resident層数調整が無効であったことからの推論である。component別のGPUメモリ内訳は測定していない。
- Ulysses degree 2のみを試した。本機は2 GPUであり他の値は取り得ない。
- text encoder量子化の実装可能性は、ideogram encoderとの比較による推定である。H3固有の制約(dual-tower bridgeとの整合など)は検証していない。
- GGUF・`.pt`形式は読み込みを試行していない。実装のweight loader経路(`safe_open`のみ)からの判断である。
- 上流discussion #34079の内容は本環境で再現確認していない。
