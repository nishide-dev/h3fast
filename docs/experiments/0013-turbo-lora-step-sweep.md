# Turbo LoRA step sweep: 12 sigma points ties the quality profile

- Date: 2026-08-20 (Asia/Tokyo)
- Baseline: `h3fast-phase1b-sage-attn-v1`(quality profile、50 step)
- Candidate: `h3fast-phase1b-turbo-lora-12-v1`(同構成 + turbo LoRA、`sigma_points: 12` = 11実効step)
- Reference: `h3fast-phase1b-turbo-lora-v1`(9 points = 8実効step、[experiment 0012](0012-turbo-lora-tier2-evaluation.md))
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Class: **Class R**(計算削減系、測定前に宣言、[ADR 0014](../decisions/0014-compute-reduction-optimization-class.md))
- Outcome: **既定採用**。blind pairwiseがtie(score 0.00)、Sage比3.86×。

## Purpose

experiment 0012で8実効stepのturbo LoRAはSage比4.94×だがpairwise −0.25(baseline勝ち)であった。劣化がtemporalに集中していたため、sigma pointsを増やせば劣化が消えるかを確認する。adapter作者のREADMEは快適圏を4〜8 stepとし「8超は過鋭化」と警告するため、12 points(11実効step)はその外側であり、劣化が消えるか別種の劣化が出るかのいずれかが観測できる。

## Method

t2va formal 20 caseをcandidate構成で1 rep生成した。baselineはquality profileの同case(追加生成なし)。8 point構成との差分は`sigma_points`のみである。生成前に`verify-backend`でSageの解決を確認した(`resolved: sage_attn`)。

判定はADR 0010のsingle-reviewer policyで、blind staging(per-case randomization、assignment封印)の20 case ballotを全件記録し復号した。

## Results

### Speed(client elapsed、20 case、単一rep)

| 構成 | 総時間 | vs quality | vs FA |
|---|---|---|---|
| quality (Sage, 50 step) | 8.58h | 1.00× | 1.63× |
| **balanced (12 points)** | **2.23h** | **3.86×** | **6.28×** |
| speed (9 points) | 1.74h | 4.94× | 8.04× |

per-caseのばらつきは小さい(vs quality 3.68〜3.93×)。12 pointsは9 pointsの0.78倍の速度で、事前予測(0.73倍)と整合する。

### Quality: blind human-pairwise(一次判定)

| 構成 | baseline勝ち | candidate勝ち | tie | score |
|---|---|---|---|---|
| 9 points (speed) | 10 | 5 | 5 | −0.25 |
| **12 points (balanced)** | **6** | **6** | **8** | **0.00** |

12 pointsはtieが8件へ増え、勝敗も6対6で**完全な引き分け**である。8 pointsで検出された劣化は12 pointsでは検出されない。

### Quality: objective metric(証跡)

| metric | 9 points | 12 points |
|---|---|---|
| temporal delta p50 | +0.0078 | +0.0096 |
| temporal 最悪 | +0.0917 | +0.0837 |
| temporal 悪化case数 | 13/20 | 12/20 |
| adherence delta p50 | −0.0035 | −0.0038 |
| 欠損・NaN/Inf | なし | なし(20/20) |

## Interpretation

**pairwiseとtemporal metricが一致しない。** pairwiseは−0.25から0.00へ改善したが、temporal deltaは実質同じ(+0.0078 → +0.0096)である。

temporal metricは隣接frame間LPIPS軌跡の差を測る。蒸留された動きがbaselineと「異なる」ことは検出できるが、それが知覚的に劣化かどうかは判定できない。reviewerは8 stepを劣化と判断し11 stepは判断しなかったが、metricはその差を捉えていない。またmetricはcase単位で符号が両方向に出るため、p50だけを見ると解釈を誤る(experiment 0012の「+14%悪化」という記述は、case単位の符号分布を伴わない要約であった)。

これはADR 0013がblind pairwiseを一次判定に置いた設計の妥当性を裏づける一方、**temporal metricをTier 2のbudget根拠に使ってはならない**ことを示す。metricは異常検知の材料として記録し、合否はpairwiseで判定する現行方針を維持する。

## Decision

**balanced(12 points)をH3Fastの既定生成profileとする。** project ownerが2026-08-20に承認した。

3 profileの階梯を`h3fast.benchmarks.profiles`へ登録する。

| profile | protocol | vs FA | pairwise |
|---|---|---|---|
| `quality` | `protocol-sage.yaml` | 1.63× | +0.20(劣化なし) |
| **`balanced`(既定)** | `protocol-turbo12.yaml` | **6.28×** | 0.00(tie) |
| `speed` | `protocol-turbo.yaml` | 8.04× | −0.25(劣化あり) |

profileの選択は明示指定でのみ行い、暗黙に切り替えない。Tier 1のexact artifact gateは`protocol.yaml`(FlashAttention、50 step)に紐づくため、placement-only最適化の比較基準としてそのまま維持する。

## Limits

- 単一reviewer・単一repである。inter-rater reliabilityは測定できない(ADR 0010)。
- score 0.00は「差がない」ではなく「平均すれば互角」である。差が判別できたcaseが12件あり、その半分でbaselineが勝っている。case単位では当たり外れがある。
- 二項検定では6対6は当然有意でないが、8 pointsの10対5もp≈0.30で有意ではない。両者の比較はtie数の変化(5→8)を含めた方向性の観測である。
- t2vaのみ。fl2va / ref2vaは未測定。
- 11実効stepより多いpointsは未測定であり、最適点を探索していない。
- audio専用metricは未実装で、audio品質はpairwise視聴のみでカバーした。
- 速度はclient elapsedであり、server側stage別内訳は未集計である。

## Reproduction

- 生成: `serve-guarded --protocol benchmarks/protocol-turbo12.yaml --sage-attention-path <ada-build> --lora-path <adapter-dir>` → `verify-backend --requested sage_attn` → `run-formal-cases --task t2va`
- 判定: `prepare-human-pairwise --task t2va` → `stage-human-pairwise` → 全件`record-human-pairwise` → `check-human-pairwise`
- metric: temporal-consistency と prompt-adherence を quality profile の同caseと比較
- private artifact(ballot、assignment、seed、media manifest、prompt file)はGit外のJapan-local storageに保持する
