# Refreshing the SGLang revision cuts VAE VRAM by 8% with bit-identical output

- Date: 2026-08-23 (Asia/Tokyo)
- Baseline (A): pinned `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`(2026-08-14)、`balanced` profile
- Candidate (B): `7d22b7a8750f53a04e41a5a5671f9a56ab6cd001`(2026-08-22)、同一 profile
- Case: smoke-001(t2va、12 sigma points = 11実効step、768p、4秒)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、48 GB each、TP2)
- Tier: **Tier 1**(revision更新のみ、測定前に宣言、[ADR 0012](../decisions/0012-tiered-optimization-verification.md))
- Related: [Issue #67](https://github.com/nishide-dev/h3fast/issues/67) Stage 1、[experiment 0020](0020-vae-residency.md)
- Outcome: **採用。** 速度同等、peak VRAM 7.99%減、生成物はbit単位で一致。

## Purpose

Issue #67 は `kitchen_int8` と sparse attention の評価を目的とするが、その前提として
**SGLang revision更新そのものの影響を分離する**ことをStage 1として要求している。
pinned revisionは2026-08-14であり、`python/sglang/multimodal_gen/` に87 commitsの差分がある。

本実験はquantization、attention backend、step数、LoRA、topologyを一切変えず、
**SGLang revisionのみ**を変更する。

## Prediction

CLAUDE.md「Predict before measuring」に従い、測定前に予測を固定した。

1. **出力はbit一致しない可能性が高い**(確信度: 中)。87 commitsのうちH3の数値経路に触るものを個別に分類し、
   `5ecd6d794`(quantized qkv scale reorder)はpre-quantized checkpointのper-row scale metadataが対象で
   online FP8には影響しないと判断した。ただし残る差分を完全には追跡できていなかった。
2. **E2E変化は −5% 〜 +5%**(確信度: 中)。意図的な最適化が既定profileの計算経路に含まれないため、改善は期待しない。
3. **起動は成功する**(確信度: 高)。新規のfail-closedは量子化checkpoint供給時のみ作用する。

予測2と3は当たった。**予測1は外れた。** 出力はbit単位で一致した。理由は後述する。

## Method

`REFERENCE_SGLANG_COMMIT` 定数を書き換えるのではなく、protocolがrevisionを固定する構造へ変更した。
preflightは `environment.software.sglang` の `git:<sha>` を期待値として検証し、
宣言のないprotocolはreference commitへfallbackする。これにより
**pinned baselineとcandidateを別identityで並存**できる。

評価中は候補protocol `h3fast-phase2-revision-candidate-v1` を用い、`unresolved` に候補である旨を明記して
revisionだけを読んで採用済みbaselineと誤認できないようにした。採用後はこのprotocolを削除し、
既定profileのprotocolを採用revisionへ移行した。テストは各protocolが採用revisionまたは
`PREVIOUS_SGLANG_COMMIT` のいずれかを指すことを検証し、第三のrevisionを未固定候補として拒否する。

両構成で `run-suite`(warmup 1 + measured 3)を実行し、`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` を維持した。

## Results

### smoke gate

Issue #67 の要求項目をすべて確認した。

| 項目 | 結果 |
|---|---|
| server startup | 成功(174秒、Aは164秒) |
| LoRA applied layers | **259/266**(Aと同一) |
| `lora.merge_mode` | `dynamic` |
| attention backend | `Using sage_attn attention backend`(DiT) |
| NaN/Inf | なし |
| media contract | 合格(measured 3 run) |

`0447ade32`(component既定backendへのfallback)により、audio_vae / video_vae が
`torch_sdpa (global backend fallback)` とログに現れるようになった。**DiTは `sage_attn` を維持している**ため
既定profileの意図は保たれている。

### Performance(measured 3 runのp50)

| metric | A: pinned | B: candidate | 差 |
|---|---|---|---|
| client E2E | 146.5秒 | 147.5秒 | +1.0秒 (+0.69%) |
| pipeline total | 145.5秒 | 146.0秒 | +0.5秒 (+0.35%) |
| `MiniMaxH3DenoisingStage` | 130.9秒 | 130.8秒 | −0.1秒 (−0.10%) |
| `MiniMaxH3DecodingStage` | 13.48秒 | 13.55秒 | +0.07秒 (+0.50%) |
| peak VRAM | 32,602 MiB | **29,996 MiB** | **−2,606 MiB (−7.99%)** |

**速度は同等である。** 差はいずれも1%未満で、単一case・3 runのp50では測定ばらつきと区別できない。
予測2の範囲内であり、revision更新に速度改善を期待していなかった判断は妥当であった。

### Memory: 原因の特定

server logがvideo VAEのload時サイズを報告する。

| 構成 | video VAE |
|---|---|
| A | `model size: 9.7 GB, consumed GPU mem: 9.73 GB` |
| B | `model size: 5.2 GB, vram: 5.20 GB` |

commit `61981e1fc`(「keep vae decoder weights in their decode dtype from load」)が原因である。
upstreamのdocstringが機構を説明している。

> The decode stage persists these frozen weights in the autocast dtype on first use
> (`prepare_autocast_linear_weights`), so **the rounding itself is already part of the output**.
> Doing it at load makes residency plans, host pins, and every host-to-device copy carry the halved size.

kill-switchは `SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST` である。

### Tier 1 gate: bit-identical

| 構成 | artifact SHA-256 |
|---|---|
| A (pinned) | `a14a14152d73baca68f14fa895158b5984ea94844b53ea03b4ac4b8484f57b7d` |
| B (candidate) | `a14a14152d73baca68f14fa895158b5984ea94844b53ea03b4ac4b8484f57b7d` |

**完全一致。** warmup 1 + measured 3 の全runが同一digestであり、
[experiment 0020](0020-vae-residency.md) のA/B双方とも一致する。

**予測1が外れた理由**は、`61981e1fc` の性質を測定前に読み切れていなかったことである。
この変更はdtype丸めの**時点**を前倒しするだけで、decode stageが初回使用時に行う丸めと同一の結果になる。
つまり計算内容は変わらない。docstringはこれを明示していたが、
私は「VAE dtypeを変える変更」を数値変更候補として分類し、
「丸めの時点が変わるだけで結果は同じ」という区別に至らなかった。

**87 commitsを挟んでbit一致した**という結果は、
H3の既定profile(online FP8 + Turbo dynamic + Sage + TP2)の数値経路が
この期間のupstream変更から実質的に隔離されていたことを示す。

## Decision

**Bを新しいpinned baselineとして採用する。** project ownerが2026-08-23に承認した。

根拠:

- 出力がbit単位で一致するため、品質差は0でありmetric実測とpairwise判定を要しない(ADR 0012)
- peak VRAMが2,606 MiB減少し、48 GBに対する余裕が16,550 MiB → 19,144 MiBへ拡大する
- 速度は同等(1%未満、ばらつきの範囲)
- Issue #67 Stage 2(`kitchen_int8` 対 FP8)のcontrolとして必要である

`REFERENCE_SGLANG_COMMIT`を`7d22b7a8`へ更新し、直前のpinを`PREVIOUS_SGLANG_COMMIT`として保持した。既定profileが参照する`protocol-turbo12-fp8.yaml`は採用revisionへ移行したが、**Phase 1の他protocolは測定時のrevisionを保持する**。採用commitへ書き換えると、その数値を生成していないrevisionで測定したと主張することになるためである。

## Limits

- 単一case(smoke-001、768p / 4秒)の測定である。
- measured 3 runのp50であり、runごとのばらつきは記録していない。E2Eの+0.69%はこの分解能では有意でない。
- fl2va / ref2va、および長尺caseでは未測定である。
- VRAM削減はvideo VAEに限定される。decoder weightをdecode dtypeで保持する変更であり、
  BF16 activationやDiT offload bufferには効かない。
- startup が164秒 → 174秒 へ増えた。VAE cast処理の追加分と考えられるが、単一測定であり切り分けていない。
- `kitchen_int8` と `sol_attn` は本実験では**有効化していない**。Stage 2以降の対象である。
- 87 commitsのうち、既定profileが通らないpath(GGUF、pruned checkpoint、quantized text encoder、
  PEFT LoRA、SM90 subblock sparse)は未検証である。

## Consequences

Issue #67 Stage 1 は完了した。Stage 2(B/C/D: FP8 対 `kitchen_int8`)へ進める前提が整った。

`kitchen_int8` の事前条件も本調査で確認済みである。

- `get_min_capability() → 75`(Turing以降)であり sm89 を満たす
- `comfy-kitchen` 0.2.31 に cp312-abi3 x86_64 wheel(58.5 MB)が存在する
- 「self-contained abi3 extension、does not link against libtorch」であり torch versionへ非依存
- **data-free quantization**(group-wise Hadamard rotation + per-output-channel absmax)であり、
  BF16 snapshotから起動時に量子化する。事前量子化checkpointを要しないためBYOW制約と整合する

`sol_attn` はPyPIに存在せず `git+https://github.com/NVlabs/Sana.git@sol-engine` からの取得となるため、
digest固定が煩雑である。Issue #67 も E/F を「原則 `speed` profile候補」と限定しており、Stage 3は分離して扱う。

[experiment 0020](0020-vae-residency.md) が記録したVAE常駐の前提(9.73 GB)は、
本revisionでは5.20 GBになる。常駐判断そのものは変わらないが、
より小さいGPUでの常駐可能性は改善している(未測定)。
