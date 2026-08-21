# Default profile stage breakdown: denoise 77%, decode 22%

- Date: 2026-08-21 〜 2026-08-22 (Asia/Tokyo)
- Configuration: `balanced` profile(既定、`h3fast-phase1b-turbo12-fp8-v1`)
- Case: smoke-001(t2va、12 sigma points = 11実効step、768p、4秒)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Related: [experiment 0015](0015-fp8-default-adoption.md)(既定profileの根拠), spec §14.2
- Outcome: denoise 77.3% / decode 22.1%。同期profilingの有無で差は0.3%であった。

## Purpose

最適化候補をprofile証拠から選定するため、既定profileのstage内訳を測定する。あわせて上流のprofiling手順が警告する「同期なしでのdecode膨張」が本構成で起きているかを確認する。

## Method

`run-suite`(warmup 1 + measured 3)でsmoke-001を実行し、server側のstage時間を取得した。同一構成で2回測定した。

1. 同期なし(従来の起動)
2. `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`(新設の`--sync-stage-profiling`)

launch側の対応は本測定のために追加した。既定はOFFで、pinned argvは不変である。この環境変数はtiming帰属のみを変えcompute graphを変えないため、`runtime_settings`へ記録するがprotocolの一致検査からは除外する。

## Results

### Stage breakdown(同期あり、measured 3 runのp50)

| stage | 時間 | 比率 |
|---|---|---|
| `MiniMaxH3DenoisingStage` | 133.3秒 | **77.3%** |
| `MiniMaxH3DecodingStage` | 38.1秒 | **22.1%** |
| その他(text encoding等) | 1.0秒 | 0.6% |
| pipeline total | 172.4秒 | — |

denoiseは11 step、per-step p50 11.51秒。11 × 11.51 = 126.6秒であり、stage時間133.3秒との差6.7秒はstep外の処理である。

### 同期の有無による差

| stage | 同期なし | 同期あり | 差 |
|---|---|---|---|
| Denoising | 127.0秒 | 133.3秒 | +6.3秒 |
| Decoding | 38.0秒 | **38.1秒** | **+0.1秒(0.3%)** |

**上流が警告する2〜3倍のdecode膨張は本構成では起きていなかった。** 上流のprofiling手順は「同期がない場合はqueuedなdenoise処理が次のblocking stageへ漏れて`DecodingStage`を2〜3倍に膨張させる」と記載する。この警告自体は一般には妥当だが、本構成では該当しなかった。

推定される理由は、11実効stepまで蒸留されておりdenoiseのqueueが浅いことである。同期のオーバーヘッドはdenoise側に+6.3秒として現れており、decodeへ漏れる非同期処理がほとんど残っていない。

**この確認には実測の価値があった。** ドキュメントの警告を根拠に「decode 22%は膨張であり無効」と判断していれば、実在するボトルネックを見落としていた。仕様や上流文書から予測できる事項は最小限の答え合わせに留めるという原則(ADR 0012)は、予測が外れる場合を含めて機能している。

### 最適化余地の上限

| 前提 | 上限 |
|---|---|
| decodeを一切改善しない | **4.52×** |
| denoiseを一切改善しない | 1.29× |

denoiseが主対象であることは確かだが、decode 22%は無視できない。denoiseをさらに削るほどdecodeの相対比重は上がる(denoiseを半減させた場合decodeは約36%になる)。

## Limits

- **単一case(smoke-001)である。** 4秒768pのcaseであり、長尺・高解像度caseでの比率は未測定である。decodeはframe数に比例する一方denoiseはstep数固定であるため、長尺caseではdecodeの比重がさらに上がる可能性がある。
- **denoise内部の内訳は未取得である。** 11.51秒/stepのうちattention、MLP、AdaLN、normのどれが支配的かは不明であり、stage時間からは導出できない。kernel最適化の対象選定には`--profile-all-stages`またはtorch.profilerによる測定を要する。
- **decodeの内訳も未取得である。** このstageはvideo decodeとrank-0 audio decodeの両方を含む。上流は内訳が必要な場合に`video_vae.decode_base`と`_decode_audio`のinner scopeを設定するよう求めており、集計値を単一VAEへ帰属させてはならない。
- t2vaのみ。fl2va / ref2vaのstage比率は未測定である。
- 同期ありの測定はmeasured 3 runのp50であり、runごとのばらつきは記録していない。

## Consequences

既定構成のstage比率が判明した。次の最適化候補の選定にはdenoise内部の内訳測定が前提となる。

VAE decode modeの変更については、上流が`spatial`、`spatial_shard`、patch decodeをH3で出力不一致を理由に拒否しており、released overlapping tiled video-VAE decodeを維持する方針である(spec §14.2)。decode 22%への対処はこの制約下で検討する。audio decode側の余地は内訳測定後に判断する。

`--sync-stage-profiling`はrepositoryへ残す。本測定では差が小さかったが、denoiseのstep数が多い構成(quality profileの50 step等)では膨張が起きる可能性があり、stage帰属を要する測定では設定を維持する。
