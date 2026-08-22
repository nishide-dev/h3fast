# Parallelism and quantization limits on 48 GB GPUs

- Date: 2026-08-22 (Asia/Tokyo)
- Configuration: `balanced` profile(既定、online FP8 + turbo LoRA dynamic + Sage)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、48 GB each、sm89)
- Related: [experiment 0018](0018-denoise-kernel-profile.md)(NCCL AllReduce 24.5%), [Issue #55](https://github.com/nishide-dev/h3fast/issues/55)
- Outcome: **TP2が必須。** AllReduceの24.5%はこのハードウェアでは削減できない。主因はTP1でtransformer weightが分割されないことである(当初「text encoderが量子化対象外」と記述したが誤りであり、2026-08-22に訂正した)。

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

**2026-08-22訂正**: 当初この節は「text encoderがBF16の63 GBであることが主因」と記述していた。これは誤りであった。ディスク上のcomponentサイズをVRAM使用量と混同した推論であり、測定していなかった。実測値は次である。

TP2(動作する構成)のserver log。

| component | model size | consumed GPU mem |
|---|---|---|
| `text_encoder` | 3.01 GB | **3.10 GB** |
| `transformer` | 15.47 GB | 0.00 GB(layerwise offload) |
| `video_vae` | 9.7 GB | 9.73 GB |
| `audio_vae` | 0.56 GB | 0.58 GB |

**text encoderのVRAM使用は3.10 GBであり、63 GBではない。** H3はQwen3-VLの64層のうち50層までしか読み込まず(`MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50`)、`lm_head`とlayer 50以降のweightを捨てる。さらにlayerwise offloadで常駐は一部のみである。ディスク上の63 GBはVRAMへ載っていない。

FP8 text encoder(`Qwen/Qwen3-VL-32B-Instruct-FP8`、35.5 GB)を実際に投入して確認した結果、**peak VRAMは26,204 MiBで既定構成と完全に同一**であり、encoderのmodel sizeも3.01 GBで変わらなかった。text encoderは制約要因ではない。

### 実際の主因: TP1ではtransformer weightが分割されない

TP1の失敗はtransformerのロード時である。

```
Error while loading component: transformer
torch.OutOfMemoryError: ... GPU 0 has 47.37 GiB total, 99.62 MiB free
```

TP2ではtransformerのmodel sizeが**15.47 GB**であり、これは2 GPUへshardされた後の値である。TP=1では全shardが1枚へ載るため約2倍を要し、`video_vae`の9.73 GBとactivationを加えると48 GBに収まらない。

resident層数を40から20へ半減しても改善しなかったのは、**resident設定がoffload対象のDiT blockにしか効かず、ロード時に必要なweight総量を変えない**ためである。

Ulysses2(TP=1、2 GPU)も同じ理由で失敗した。Ulyssesはsequence次元を分割するのみで、weightは各GPUが全量を保持する。

### 上流の推奨構成が使えない理由

| GPU | VRAM | TP=1でweightを保持できるか |
|---|---|---|
| B200 | 180 GB | ✅ |
| H200 | 141 GB | ✅ |
| RTX 6000 Ada | **48 GB** | ❌ |

上流がTP=1 + Ulysses8を推奨するのは大容量GPU構成である。48 GBではweight分割なしにH3は載らない。

## Text encoder quantization: supported, but without effect here

**2026-08-22訂正**: 当初この節は「H3 encoderに量子化経路がなく、実装の非対称性である」と記述していた。これも誤りであった。

`MiniMaxH3Qwen3VLEncoder`はcapabilityを宣言している(`models/encoders/minimax_h3_qwen3vl.py` 51-54行)。

```python
checkpoint_quantization_capability = CheckpointQuantizationCapability(
    backend="diffusion",
    methods=frozenset({"fp8"}),
)
```

`loader/component_loaders/text_encoder_loader.py:132`がこれを読み、事前量子化checkpointの方式と突き合わせる。ideogramとは別方式(loader経由のcapability契約)で実装されており、欠落ではなかった。`grep`で`quant_config`を探しただけで結論したのが誤りである。

公式の`Qwen/Qwen3-VL-32B-Instruct-FP8`(35.5 GB、7 shards、`quant_method: fp8`)は要件を満たす。tensor名はscale 448個を除いて我々のsnapshotと完全一致し(1058個、双方向の差分0)、architectures・layers・hidden・headsも一致する。

実測の結果、**起動・生成には成功したがVRAMは変わらなかった**。

| 指標 | BF16 encoder | FP8 encoder |
|---|---|---|
| encoder model size | 3.01 GB | 3.01 GB |
| encoder GPU mem | 3.10 GB | 3.10 GB |
| peak VRAM | 26,204 MiB | 26,204 MiB |

上記のとおりVRAM制約の要因はtext encoderではないため、これを量子化しても効果がない。

### 副産物: `--text-encoder-path` override のバグ

同じFP8 checkpointで、投入方法により結果が分かれた。

| 方法 | 結果 |
|---|---|
| `--text-encoder-path <dir>` | `KeyError: Unexpected MiniMax H3 Qwen3-VL checkpoint weight: ...weight_scale_inv` |
| container内でsnapshot位置へbind mount | 起動成功 |

override経路では`quant_config`が渡らず、scale tensorを受け取るparameterが作られないためと考えられる。capability実装自体は機能する。実用上の利益がないため優先度は低いが、実在するバグである。

なおhost側snapshotをsymlinkで差し替える方法はpreflightが拒否する(`snapshot symlinks are not accepted`)。これはsnapshot同一性を保証する正しい設計である。

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

- component別のGPUメモリ内訳はserver logの報告値であり、torch側のallocator内訳やactivationのピークは分離していない。TP1でtransformer weightが約2倍になるという説明はTP2実測の15.47 GBからの推論であり、TP1でのweight総量を直接測っていない(OOMのため到達しない)。
- Ulysses degree 2のみを試した。本機は2 GPUであり他の値は取り得ない。
- FP8 text encoderの評価は単一case・単一runである。出力がBF16 encoderと一致するかは比較していない(digest比較を行っていない)。
- GGUF・`.pt`形式は読み込みを試行していない。実装のweight loader経路(`safe_open`のみ)からの判断である。
- 上流discussion #34079の内容は本環境で再現確認していない。
