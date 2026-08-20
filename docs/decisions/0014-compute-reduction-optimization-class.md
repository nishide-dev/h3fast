# ADR 0014: Judge compute-reducing optimizations by rate, not by zero degradation

- **Status:** Approved by project owner (2026-08-20)
- **Date:** 2026-08-20
- **Related:** [ADR 0012](0012-tiered-optimization-verification.md), [ADR 0013](0013-single-operator-process-reduction.md), [experiment 0012](../experiments/0012-turbo-lora-tier2-evaluation.md), [Issue #51](https://github.com/nishide-dev/h3fast/issues/51)

## Context

ADR 0013はTier 2の品質判定をblind human-pairwise一次 + objective metric証跡とし、Sage Attentionの採用で機能した。しかしその判定基準は「temporal同水準、adherence delta無視可能」であり、暗黙に**劣化ゼロ**を要求していた。

turbo LoRA(step蒸留、50 step → 8実効step)の評価でこの基準が破綻した。同一の計算を速く実行するSageと違い、step蒸留は**計算量そのものを削減する**。劣化が出るのは設計上当然であり、「劣化ゼロか」で判定すれば結果は測定前から決まっている。実際の実測はSageに対し4.94×の高速化と、pairwise 10:5:5(score −0.25)およびtemporal +14%を示した。

「劣化ゼロ」基準では不採用になるが、それはこの候補が無価値であることを意味しない。8.6時間から1.74時間への短縮は試行錯誤フェーズで決定的な差である。基準が最適化の性質と噛み合っていなかった。

## Decision

Tier 2の内部を**出力等価性の意図**で2クラスに分け、判定基準を分ける。宣言はADR 0012同様、測定前に行う。

### Class E: 等価変換系 (equivalence-preserving)

同じ計算を速く/省メモリで実行する候補。attention実装の差し替え、kernel融合、精度を変えないメモリ最適化など。

- 判定: 劣化ゼロを要求する。pairwiseで有意な劣化がなく、objective metricが同水準であること
- 例: Sage Attention(採用済み、pairwise +0.20、temporal同水準)

### Class R: 計算削減系 (compute-reducing)

計算量そのものを削る候補。step蒸留、step削減、cache/skip系、大幅な量子化など。

- 判定: **劣化ゼロは要求しない。** 速度比と劣化量を併記し、ownerが「そのレートを受け入れるか」を判断する
- 必須記録:
  - 影響task familyのformal case全件での速度比(総時間および分布)
  - blind pairwiseの勝敗内訳(tie数を含む。tieは差が判別不能なcase数を示す)
  - objective metricのdelta分布(どのmetricに劣化が集中したかを特定する)
  - 劣化が集中した性質の記述(例: temporalに集中、adherenceは不変)
- 採否は3値とする: **既定採用** / **profile採用**(opt-inで利用可、既定は別構成) / **不採用**

Class Rを不採用にするのは、速度比が小さい、劣化が破綻的、または劣化がmetricを跨いで広範な場合である。「劣化が存在する」ことだけを理由に不採用としない。

### Profile採用の要件

profileとして採用する候補は、既定構成と同じ再現性要件を満たす。protocolのpinned identity、fail-closed検証、実験記録、限界の明記。profileの選択は利用者の明示指定でのみ行い、暗黙に切り替えない(既存方針どおり)。

## Consequences

- turbo LoRAはClass Rの**profile採用**とする(experiment 0012)。既定は品質重視のSage構成、opt-inで速度重視のturbo構成。
- 「劣化が最低限なら既定にする」判断のために、劣化量を減らす方向の探索が意味を持つようになった。8実効stepでの劣化はtemporalに集中しているため、sigma pointsを増やした構成(10〜12)で劣化が消えるかは未測定であり、Issue #51で追跡する。既定切り替えの判断はその結果を待つ。
- Class宣言を測定前に行う義務が増える。ADR 0012の「tierを結果を見た後に下げてはならない」と同じ規律をclassにも適用する。

## Rejected alternatives

- **Class Rにも劣化ゼロを要求し続ける**: 測定前に結論が決まっており、計算削減系の候補を一律に排除する。プロジェクトの目的(効率化・高速化)と矛盾する。
- **速度比に対する劣化量の数値閾値を定める**(例: 「2×につきpairwise −0.1まで許容」): 現時点で根拠となるデータが2候補しかなく、恣意的な定数がgateになる。ownerの判断に委ね、判断材料の記録を必須とする方が誠実である。
- **profileを設けず既定をturboへ切り替える**: 現在の実測(pairwise −0.25、temporal +14%)は既定にするには劣化が大きい。劣化を減らした構成で再判定すべきである。
