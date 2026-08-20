# Turbo LoRA Tier 2 evaluation: 4.94× at a temporal cost

- Date: 2026-08-19 〜 2026-08-20 (Asia/Tokyo)
- Baseline: `h3fast-phase1b-sage-attn-v1`(採用済みSage構成、50 step)
- Candidate: `h3fast-phase1b-turbo-lora-v1`(同構成 + turbo LoRA、`sigma_points: 9` = 8実効step)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Base model: H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`
- Adapter: `larryvrh/MiniMax-H3-Turbo-Lora` revision `43a74557…`、`minimax_h3_turbo_v4_step600_ema.safetensors`、SHA-256 `5f3a626c…`
- Related: [ADR 0014](../decisions/0014-compute-reduction-optimization-class.md), [ADR 0013](../decisions/0013-single-operator-process-reduction.md), [experiment 0011](0011-sage-attention-tier2-adoption.md), [Issue #51](https://github.com/nishide-dev/h3fast/issues/51)
- Outcome: **profile採用**(既定はSage構成、opt-inでturbo)。Sage比4.94×、劣化はtemporalに集中。

## Class declaration

本候補は**Class R(計算削減系)**である。step蒸留は計算量そのものを削るため、ADR 0014に従い劣化ゼロを要求せず、速度比と劣化量を併記してownerが判断する。

この宣言は事後の追加である。当初はADR 0013のClass E相当の基準(temporal同水準)で測定を開始し、結果を見た後にownerの指摘で基準の誤りが判明した。基準の性質を変えた経緯を隠さないため記録する。**測定値自体は基準変更の前に確定しており、判定を通すために数値を再解釈していない。**

## Method

t2va formal 20 caseをcandidate構成で1 rep生成し、baselineには採用済みSage構成の同caseを用いた(追加生成なし)。変更した次元はLoRAと`sigma_points`のみである。

判定はADR 0010のsingle-reviewer policyでreviewer `nishide-dev`が実施。blind staging(per-case randomization、assignment封印)の20 case A/B ballotを全件記録し、assignment keyで復号した。

## Results

### 事前に未確定だった項目(Issue #51)の解消

| 項目 | 結果 |
|---|---|
| `num_inference_steps` の意味論 | `denoise_steps_seconds` が**8要素**。9 sigma points = 8実効stepであり、cookbookの`9`とadapter作者の「4〜8 step」は矛盾しない |
| H3でのLoRA e2e | 成立。schedule短縮が実効(131秒/case)し、lifecycleにweight digestが記録された |
| run-to-run決定性 | **bit-exact**。suite 4 run(warmup + measured 3)の生成物SHA-256が全一致 |
| media contract | baselineと一致(1344×768、107 frames、H.264 + AAC) |

### Speed(client elapsed、20 case、単一rep)

| 比較 | 最小 | 中央値 | 最大 | 総時間 |
|---|---|---|---|---|
| vs Sage | 4.12× | 4.82× | 5.11× | **4.94×**(8.6h → 1.74h) |
| vs FA(元baseline) | 5.16× | 8.04× | 10.12× | **8.04×**(14.0h → 1.74h) |

baselineより遅いcaseは0件。

### Quality: blind human-pairwise(一次判定)

ballot `turbo-lora-t2va-v1`、20 case、reviewer 1名。

| 判定 | 件数 |
|---|---|
| baseline (Sage) 勝ち | 10 |
| candidate (turbo) 勝ち | 5 |
| tie | 5 |
| score | **−0.25** |

tieが5件であり、差が判別可能なcaseが15件ある(Sage評価時はtie 8件)。そのうち2/3がbaseline勝ちである。reviewerの事前所感は「step蒸留で劣化するのは当然」であり、破綻ではなく程度の問題として扱う。

### Quality: objective metric(証跡)

| metric | baseline | candidate | 評価 |
|---|---|---|---|
| temporal consistency (step-LPIPS p50) | 0.0798 | **0.0909** | **約14%悪化** |
| temporal 最悪case | — | regression-018 で **+0.0917** | 局所的に大きい |
| prompt adherence (delta p50) | — | **−0.0035** | ほぼ不変(range −0.0235〜+0.0877) |
| sample間LPIPS | — | (別サンプル収束のため合否対象外) | — |
| 欠損・NaN/Inf | なし(20/20/20) | | |

**劣化はtemporalに集中し、prompt追従性はほぼ落ちていない。** これはstep蒸留の既知の弱点と一致する。adapter作者のREADMEは4 stepで「motion smear / trailing ghosting」が出ると明記しており、8 stepでも動きの滑らかさに影響が残ったと読める。pairwiseで差が出たcaseは動きの多いものに偏る傾向があり、metricと人間判定が同方向を指した。

## Decision

**profile採用**とする。project ownerが2026-08-20に承認した。

- 既定は品質重視の`benchmarks/protocol-sage.yaml`(Sage構成、劣化なし、FA比1.63×)
- opt-inで速度重視の`benchmarks/protocol-turbo.yaml`(Sage比4.94×、FA比8.04×、temporal劣化あり)
- profileの選択は明示指定でのみ行い、暗黙に切り替えない
- 速度主張は本記録の再現条件(単一rep、client elapsed、2×RTX 6000 Ada、TP2、t2va)に限定する

既定をturboへ切り替えるかは、劣化を減らした構成での再判定を待つ。劣化がtemporalに集中しているため、`sigma_points`を増やした構成(10〜12)で劣化が消えれば既定候補になり得る。その場合でもSage比3〜4×が残る見込みだが**未測定**であり、Issue #51で追跡する。

## Limits

- 単一reviewer・単一repの結果である。inter-rater reliabilityは測定できない(ADR 0010)。
- t2vaのみ。fl2va / ref2vaへの適用は各familyでの生成と判定を要する。
- pairwise 10:5:5は非tie 15件の10対5であり、二項検定でp≈0.30。統計的有意ではなく、方向性の観測である。劣化の存在はtemporal metricが独立に裏づける。
- 劣化がtemporalに集中するという解釈はcase単位の相関観察に基づき、動き量との定量的な関係は未分析である。
- audio専用metricは未実装であり、audio品質はpairwise視聴のみでカバーした。
- 8実効stepより多いsigma pointsは未測定である。
- adapterはH3派生物であり、local研究利用に限定する。再配布・同梱はしない。

## Reproduction

- 生成: `serve-guarded --protocol benchmarks/protocol-turbo.yaml --sage-attention-path <ada-build> --lora-path <adapter-dir>`(weight digestを起動前に検証)→ `run-formal-cases --task t2va`
- 判定: `prepare-human-pairwise --task t2va` → `stage-human-pairwise` → 全件`record-human-pairwise` → `check-human-pairwise`
- metric: 3種をbaseline(Sage rep1)と比較
- private artifact(ballot、assignment、seed、media manifest、prompt file)はGit外のJapan-local storageに保持する
