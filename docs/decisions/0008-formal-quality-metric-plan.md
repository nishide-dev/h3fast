# ADR 0008: Formal quality metric plan contract

- **Status:** Accepted for contract implementation; metric selection pending
- **Date:** 2026-08-16
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16), [ADR 0005](0005-formal-quality-set-contract.md)

## Context

formal quality-set recordは6 metric familyごとにowner、implementation、version、budget、evidenceを要求する。一方、文字列だけでは、反復数、統計量、score方向、dependency pin、baseline自己変動との比較、欠損時の扱い、family間の相殺禁止を十分に固定できない。

外部metricやjudge modelを先に選ぶと、Python/CUDA互換性、license、入力条件、再現性、実際のbaseline変動を確認しないままmethodと閾値が事実上固定される。現在の段階では評価意味論と承認条件を先に固定し、実装選定は別の測定・reviewとして扱う必要がある。

## Decision

[`benchmarks/quality/formal-quality-metric-plan.json`](../../benchmarks/quality/formal-quality-metric-plan.json)をformal quality-set recordを裏付ける6-familyの詳細plan artifactとし、[`schemas/quality-metric-plan.schema.json`](../../schemas/quality-metric-plan.schema.json)とsemantic validatorを追加する。formal readinessのsource of truthは引き続き`formal-quality-set.json`とし、詳細planがapprovedでその要約と一致するまでformal recordのmetricをapprovedにしない。

- baselineとcandidateは各3反復とする
- p5、p50、p95、worst-caseを固定統計とする
- exact profileはfamilyごとにbaseline自己変動envelopeと比較し、追加toleranceを0とする
- observation欠損はfailとする
- 6 familyを独立判定し、全family合格だけを全体合格とする
- implementationはversion、40/64文字のimmutable revision、entrypoint、exact dependency pin、inputs、score方向を持つ
- approved budgetはunit、全caseの100% coverage、per-case/all-runs集計、failure policyを持つ
- approved familyはownerとHTTPS evidenceを必須とする
- planの存在またはschema validationだけをmetric approvalとして扱わない

`h3fast benchmark check-quality-metric-plan`は全familyがapprovedの場合だけ終了code 0、正当なdraft/planned状態は1、不正または矛盾するrecordは2を返す。

## Consequences

metric実装を追加するPRは、moving versionや非固定dependency、平均値だけのbudget、missing observationの黙示除外、映像metricによるaudio/A/V sync失敗の相殺を導入できない。

committed planは意図的に`draft`かつ全family `unassigned`である。[ADR 0009](0009-formal-quality-metric-selection.md)で各familyのcandidateとblockerを固定したが、実在entrypoint、runtime互換性とbaseline自己変動が未確認のため`planned`へは進めない。今後、固定baselineの自己変動を実測してbudget evidenceを作成する必要がある。このADRはmetricの品質や適切性を承認せず、formal quality set、Support Tierまたはrelease gateを通過させない。
