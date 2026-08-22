# Keeping the video VAE resident cuts decode by 65%

- Date: 2026-08-23 (Asia/Tokyo)
- Baseline (A): `balanced` profile、`--layerwise-offload-components dit,text_encoder,vae`
- Candidate (B): 同一構成で `--layerwise-offload-components dit,text_encoder`(video VAE常駐)
- Case: smoke-001(t2va、12 sigma points = 11実効step、768p、4秒)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、48 GB each、TP2)
- Tier: **Tier 1**(placement-only、測定前に宣言、[ADR 0012](../decisions/0012-tiered-optimization-verification.md))
- Related: [Issue #66](https://github.com/nishide-dev/h3fast/issues/66), [experiment 0017](0017-default-profile-stage-breakdown.md)
- Outcome: **採用**。E2E 15.8%短縮、decodeは65%短縮。生成物はbit単位で一致。

## Purpose

[experiment 0017](0017-default-profile-stage-breakdown.md)で既定profileのdecode stageが22.1%(38.1秒)を占めることが判明した。当初この記録はVAEを最適化対象外と判断していたが、その根拠は上流がH3で`spatial`、`spatial_shard`、patch decode modeを拒否している点であり、**decode modeの話であってplacementの話ではなかった**。

現行launchはvideo VAEもlayerwise offloadしている。約9.7 GBのweightをdecode tileごとに再転送していれば、decode時間の相当部分が転送待ちである。一方peak VRAMは26,204 MiBであり、48 GB GPUには常駐の余地が残っていた。

本実験は計算内容、量子化、attention backend、step数を一切変えず、**VAE placementのみ**を変更する。

## Method

`--offload-components`をlaunchへ追加し、`dit`を必須とするfail-closed検証を設けた(`dit_layerwise_resident_layers`はoffloadされたDiTを前提とするため)。既定値は`dit,text_encoder,vae`でpinned argvは不変である。

両構成で`run-suite`(warmup 1 + measured 3)を実行し、`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`を維持した。DiT resident layersは40のまま変更していない。

Issueは起動失敗時にresident layersを30 / 24 / 20へ下げる案(C / D / E)を用意していたが、**Bが起動したため不要であった**。

## Results

### Stage breakdown(measured 3 runのp50)

| stage | A: VAE offload | B: VAE resident | 差 |
|---|---|---|---|
| pipeline total | 172.4秒 | **145.1秒** | **−27.3秒(15.8%短縮)** |
| `MiniMaxH3DenoisingStage` | 133.3秒 (77.3%) | 130.6秒 (90.0%) | −2.7秒 |
| `MiniMaxH3DecodingStage` | 38.1秒 (22.1%) | **13.5秒 (9.3%)** | **−24.6秒(65%短縮)** |

decodeの短縮がE2E改善のほぼ全てを説明する。denoiseの−2.7秒は測定間のばらつきの範囲と考えられる(単一case、3 runのp50)。

### Memory

| 指標 | A | B | 差 |
|---|---|---|---|
| peak VRAM (p50) | 26,204 MiB | 32,602 MiB | **+6,398 MiB (+24.4%)** |
| 48 GBに対する余裕 | — | 16,550 MiB | — |

server logは`Loaded video_vae: model size 9.7 GB, consumed GPU mem: 9.73 GB`を報告する。peak増加が9.73 GBではなく6,398 MiBに留まるのは、offload時にも転送バッファとtile単位のworkspaceを確保していたためと考えられる(未検証)。

### Tier 1 gate: bit-identical

| 構成 | artifact SHA-256 |
|---|---|
| A (VAE offload) | `a14a14152d73baca68f14fa8…` |
| B (VAE resident) | `a14a14152d73baca68f14fa8…` |

**完全一致。** B内のmeasured 3 runも相互に一致した。placement変更がcompute graph、schedule、step数、precisionを保存するという主張が、per-case digestで裏づけられた。ADR 0012に従い、**品質差は0でありmetric実測とpairwise判定を要しない**。

## Decision

**video VAEを常駐させる構成を既定とする。** project ownerが2026-08-23に承認した。

`balanced` profileのlaunch既定を`--layerwise-offload-components dit,text_encoder`へ変更する。DiT resident layersは40を維持する。

## Limits

- 単一case(smoke-001、768p / 4秒)の測定である。decodeはframe数に比例するため、長尺caseでは短縮の絶対値が変わる。比率が同じかは未測定。
- measured 3 runのp50であり、runごとのばらつきは記録していない。
- fl2va / ref2vaでは未測定である。参照条件付きfamilyのdecode特性は確認していない。
- peak VRAM +6,398 MiBは48 GB機での値である。より小さいGPUでは常駐できない可能性がある。その場合はDiT resident layersを下げる余地がある(Issue #66のC / D / E案、本実験では不要であった)。
- denoiseの−2.7秒に因果的な説明はない。placementはdenoiseへ影響しないはずであり、測定ばらつきとして扱う。
- text encoderは引き続きoffloadしている。常駐させた場合の効果は未測定である(VRAM使用は3.10 GBと小さく、text encoding stageは0.6%であるため優先度は低い)。

## Consequences

既定profileのE2Eが15.8%短縮された。FlashAttention 50-step baseline比は約6.9×から**約8.0×**になる(20 case総計での再測定は未実施であり、この値はsmoke-001のstage測定からの換算である)。

decodeの比率は22.1%から9.3%へ下がり、denoiseが90.0%を占める構成になった。今後の最適化対象はdenoiseへさらに集中する。

[experiment 0017](0017-default-profile-stage-breakdown.md)の「decodeは上流制約により対象外」という判断は誤りであった。decode modeの制約とplacementの選択を混同していた。上流文書が禁じているのは前者のみである。
