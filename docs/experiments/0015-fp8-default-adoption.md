# Online FP8 becomes the default: 4.22× with no detected degradation

- Date: 2026-08-21 (Asia/Tokyo)
- Baseline: `h3fast-phase1b-turbo-lora-12-v1`(当時の既定balanced、BF16、12 sigma points)
- Candidate: `h3fast-phase1b-turbo12-fp8-v1`(同構成 + online FP8、`lora.merge_mode: dynamic`)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Class: **Class R**(計算削減系、測定前に宣言、[ADR 0014](../decisions/0014-compute-reduction-optimization-class.md))
- Related: [experiment 0014](0014-fp8-dynamic-lora-smoke.md)(実行可能性とcanary), [Issue #55](https://github.com/nishide-dev/h3fast/issues/55)
- Outcome: **既定採用**。quality profile比4.22×、peak VRAM −22%、blind pairwise +0.20。

## Method

t2va formal 20 caseをcandidate構成で1 rep生成し、baselineには当時の既定balanced profileの同caseを用いた(追加生成なし)。変更した次元は`quantization`とそれに伴い必須となる`lora.merge_mode`のみである。

実行可能性、LoRA適用の証拠(canary)、除外層の一致確認は[experiment 0014](0014-fp8-dynamic-lora-smoke.md)で完了している。本記録は20 caseでの速度・memory・品質評価を扱う。

判定はADR 0010のsingle-reviewer policyでreviewer `nishide-dev`が実施した。blind staging(per-case randomization、assignment封印)の20 case ballotを全件記録し復号した。

## Results

### Speed and memory(20 case、単一rep)

| 指標 | BF16 balanced | **FP8 balanced** | 差 |
|---|---|---|---|
| 総時間 | 2.23h | **2.04h** | **1.09×** |
| per-case高速化 | — | min 1.06× / p50 1.12× / max 1.17× | 全caseで改善 |
| peak VRAM (p50) | 40,661 MiB | **31,729 MiB** | **−8,932 MiB (−22.0%)** |
| quality profile比 | 3.86× | **4.22×** | — |
| artifact一致数 | — | 0 / 20 | FP8が実効している |

smoke単一caseでは1.17×であったが20 case総計では1.09×である。重いcaseではdenoise以外の比重が上がるため高速化率が下がる。**主たる価値はVRAM削減である。**

### Quality: blind human-pairwise(一次判定)

ballot `fp8-turbo12-t2va-v1`、20 case、reviewer 1名。

| 判定 | 件数 |
|---|---|
| candidate (FP8) 勝ち | 7 |
| baseline (BF16) 勝ち | 3 |
| tie | 10 |
| score | **+0.20** |

tieが半数を占め、差が判別できた10件のうち7件でFP8が選ばれた。**劣化は検出されなかった。**

数値精度を落としながらscoreが正であることは、FP8による量子化誤差が知覚品質に対して有意な劣化を生んでいないことを示す。ただし7対3は二項検定でp≈0.34であり有意ではない。結論は「FP8が優れる」ではなく「劣化が検出されなかった」である。

### Quality: objective metric(証跡)

| metric | BF16 balanced | FP8 balanced | 評価 |
|---|---|---|---|
| temporal consistency (step-LPIPS p50) | 0.0992 | **0.1391** | delta p50 **+0.0303**、20件中17件で悪化 |
| prompt adherence (delta p50) | — | **+0.0004** | 不変(range −0.0240〜+0.0113) |
| 欠損・NaN/Inf | なし | なし(20/20) | |

**pairwiseとtemporal metricが再び逆方向を示した。** pairwiseはFP8を7対3で選んだ一方、temporalは17/20で悪化しdelta p50は+0.0303である。これはexperiment 0013で記録した「temporal metricは蒸留由来の差異を検出できても知覚的劣化を判定できない」という知見が、量子化誤差についても同様に当てはまることを示す。

metricが検出しているのは「BF16構成との差異」であり、その差異が劣化かどうかを判定していない。ADR 0013がblind pairwiseを一次判定に置き、temporal metricをbudget根拠に使わないと定めた方針をここでも維持する。metricは異常検知の証跡として記録する。

なお、この不一致が繰り返し観測されたことは、temporal metricの実装が誤っているという主張ではない。metricは定義どおり隣接frame間LPIPS軌跡の差を測っており、その量が知覚品質の代理として機能しないという観測である。

## Decision

**FP8 + dynamic LoRAをH3Fastの既定生成profileとする。** project ownerが2026-08-21に承認した。

profile階梯を4段階へ拡張する。

| profile | protocol | vs quality | pairwise |
|---|---|---|---|
| `quality` | `protocol-sage.yaml` | 1.00× | +0.20(FA比較、劣化なし) |
| `bf16-balanced` | `protocol-turbo12.yaml` | 3.86× | 0.00(tie) |
| **`balanced`(既定)** | `protocol-turbo12-fp8.yaml` | **4.22×** | **+0.20** |
| `speed` | `protocol-turbo.yaml` | 4.94× | −0.25(劣化あり) |

BF16構成は`bf16-balanced`として保持する。online FP8はcookbookが明示するとおり近似であり、BF16 weightを要する用途のための選択肢を残す。

`quantization`が設定されたprofileは`lora.merge_mode: dynamic`を必須とする。`auto`はquantized runtime weightを考慮せずstatic mergeを選び起動に失敗する(experiment 0014)。この制約はprotocolのpinned identityとregression testで固定した。

## Limits

- 単一reviewer・単一repである。inter-rater reliabilityは測定できない(ADR 0010)。
- 7対3はp≈0.34で有意でない。tie 10件を含めた方向性の観測である。
- **online FP8は近似である。** cookbookは「not a consistency ground-truth mode」と明記する。bit-exact性や数値再現性を要する用途では`quality`または`bf16-balanced`を用いる。
- t2vaのみ。fl2va / ref2vaは未測定。
- `denoise_steps_seconds`は未取得であり「11 denoiser evaluations」を直接確認していない(`run-formal-cases`経路がserver performance dumpを取得しないため。BF16構成でも同様)。
- 4構成マトリクスのB(BF16 + dynamic)は未測定である。dynamic overheadはFP8側で+7.0秒・+3,674 MiBと分離したが、BF16側で同程度かは未確認。
- `--quantization-ignored-layers`によるhybrid量子化は未評価である。
- 速度はclient elapsedであり、server側stage別内訳は未集計である。

## Reproduction

- 生成: `serve-guarded --protocol benchmarks/protocol-turbo12-fp8.yaml --sage-attention-path <ada-build> --lora-path <adapter-dir>` → `verify-backend --requested sage_attn` → `run-formal-cases --task t2va`
- 判定: `prepare-human-pairwise --task t2va` → `stage-human-pairwise` → 全件`record-human-pairwise` → `check-human-pairwise`
- metric: temporal-consistency と prompt-adherence を BF16 balanced の同caseと比較
- private artifact(ballot、assignment、seed、media manifest、prompt file)はGit外のJapan-local storageに保持する
