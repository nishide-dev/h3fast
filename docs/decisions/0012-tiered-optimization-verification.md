# ADR 0012: Tier optimization verification by output-equivalence class

- **Status:** Approved by project owner (2026-08-18)
- **Date:** 2026-08-18
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16), [ADR 0008](0008-formal-quality-metric-plan.md), [ADR 0011](0011-bit-exact-repetition-exemption.md), [experiment 0008](../experiments/0008-formal-generation-determinism.md)

## Context

現行のquality契約は、すべての最適化に対してformal 60 caseとbaseline/candidate各3反復を要求する。この設計は「品質が変わり得る最適化」を前提としており、量子化やstep蒸留のようにcaseごとに劣化の出方が異なる変更では妥当である。

一方、2026-08-17のt2va 20-case実測で、DiT resident layer数の変更（20→40）が生成物をbit単位で変えないことが確認された（[experiment 0008](../experiments/0008-formal-generation-determinism.md)）。出力が同一である場合、metric値の差は統計的に小さいのではなく存在しない。したがって追加caseや追加反復は新しい情報を生まない。

実際、この事実の確認に約41時間のGPU時間を要したが、同じ結論はsmoke数caseのdigest照合で数時間以内に到達できた。安価な判定を先に行わず、最初から最大の測定を実行したことが原因である。

さらに、fl2va/ref2vaを含むformal 60 caseの完全実行は、最適化候補を1つ評価するたびに数十時間を要する。この費用は「出力が変わる最適化」には正当だが、すべての変更へ一律に課すと、安全な最適化の探索速度を不必要に下げる。

## Decision

最適化を出力等価性のclassで分け、必要な検証をtier化する。

### Tier 1: bit-exact placement最適化

compute graph、schedule、step数、precisionを変えず、offload配置、prefetch、residencyなどの実行配置だけを変更するもの。

検証手順:

1. smoke splitから1 caseを選び、baselineとcandidateで各1反復生成する。
2. per-case生成物のSHA-256を照合する。
3. 一致した場合、当該最適化はbit-exactとして採用判定できる。exact artifact gate、protocol差分test、実効setting一致、E2E/stage/memory測定は従来どおり要求する。
4. 不一致の場合、Tier 1判定は成立しない。原因を確認し、Tier 2へescalateする。

digest一致は品質差が0であることの証明であり、metric familyの実測を代替できる。この場合metric budgetの承認は不要とする。

1 caseで足りるのは、bit-exact性がcase内容ではなくcompute graphとruntime設定の性質だからである。ある1 caseでbit一致し別のcaseで不一致になる状況は、caseごとに異なるコード経路が走る場合に限られる。そうした疑いがある変更はそもそもTier 1ではない。case数を増やしても得られるのは同じ結論の反復であり、GPU時間に見合わない。

### Tier 2: 数値を変える最適化

量子化、量子化attention、step蒸留、kernel書き換え、precision変更、schedule変更など、出力bytesが変わり得るもの。

formal set全caseと[ADR 0008](0008-formal-quality-metric-plan.md)の反復・統計・family独立判定・budget承認を要求する。digestは必ず変わるため、metric実測以外に判定手段はない。[ADR 0011](0011-bit-exact-repetition-exemption.md)のbit-exact例外は適用しない。

### 共通要件

- 検証tierは事前に宣言し、根拠（compute graphを変えるか）をprotocolまたはexperiment記録へ書く。実測後にtierを下げてはならない。
- Tier 1で採用した最適化について、後にruntime、driver、GPU構成、schedule、precision、attention backendが変わる場合は、digest照合をやり直す。
- 未測定のtask family（現時点のfl2va/ref2va）は、最初の1回だけ決定性を確認する。確認後はTier 1判定に同じsmoke手順を適用できる。
- 判定に用いたcase ID、反復数、digest一致件数をexperiment記録へ残す。

## 測定の前に予測する

検証tierの判断とは別に、次を原則とする。

**仕様から予測できることを発見のために測定しない。** 公式ドキュメント、モデルカード、設定の定義から結論が導ける場合、測定は予測の答え合わせであり、最小の規模で行う。予測が外れた場合にのみ深掘りする。

この原則は2026-08-17の実測から得た。t2va 20 caseを2 protocol×2反復（80生成、約41時間）測定してbit-exact決定性と20層/40層の出力一致を確認したが、前者はH3の`quality: "lossless"`が「bit完全」と明記していること、後者はresident layer数がoffload配置のみを変えることから、いずれも事前に予測できた。同じ確証は1 case×2反復（4生成、約50分）で得られた。

一方、測定が必要な場合もある。SGLangは`sage_attn`をsupported backendとして文書化し、有効化ログも出し、pipeline validationも通したが、実際にはkernelが実行されていなかった（[experiment 0009](../experiments/0009-sage-attention-noop.md)）。仕様と実装がずれる境界、特に外部依存の統合部分では、最小限の実測で確認する。

判断基準は次のとおりである。

- 自プロジェクトが定義した契約の帰結 → 予測する。測定は不要か最小限
- 外部componentの文書化された仕様 → 予測した上で、境界を1回だけ確認する
- 数値を変える最適化の品質影響 → 予測できない。実測する

## Consequences

placement系最適化の判定費用が数十時間から数時間へ下がり、安全な最適化を高頻度に試せる。品質保証は弱まらない。digest一致は統計的推定ではなく同一性の証明であり、metric実測より強い根拠である。

数値を変える最適化の費用は変わらない。Sage Attentionやstep蒸留のような候補では引き続きformal setとmetric実測が必要であり、実装済みのmetric adapter群はそこで本来の役割を果たす。

tier宣言を事前に要求するため、「測ってから安い方へ寄せる」運用はできない。これは意図した制約である。
