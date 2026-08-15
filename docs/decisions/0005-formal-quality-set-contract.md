# ADR 0005: Formal quality-set contract

- **Status:** Accepted for contract implementation; dataset approval pending
- **Date:** 2026-08-16
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16) (population/evaluation), [Issue #14](https://github.com/nishide-dev/h3fast/issues/14) (contract), [Issue #11](https://github.com/nishide-dev/h3fast/issues/11) (H3-use compliance)

## Context

固定`smoke-001`の`exact-decoded-artifact-v1`は、placementだけを変更した候補のbitwise回帰を検出できる。一方、Phase 0で必要な公開可能な10件以上のsmoke set、50件以上のregression set、知覚・audio・semantic A/V評価、prompt/reference mediaの権利reviewは代替しない。

権利review前にprompt本文やreference mediaをrepositoryへ追加すると、公開可否が未確定の素材を配布する危険がある。反対に、単一booleanだけでは、件数、coverage、rights、metric budget、ownerのどこが未完了かを検証できない。

## Decision

`benchmarks/quality/formal-quality-set.json`をformal set readinessのsource of truthとし、schema 1.0とsemantic validatorを追加する。

- recordにはprompt本文、reference media、生成media、local pathを保存しない
- caseはprompt/reference assetのSHA-256、seed、task、duration、aspect ratio、language、subject、motion、audio、reference modalityだけを記録する
- smoke 10件、regression 50件を下限とする
- T2VA、FL2VA、Ref2VA、4/5/10/15秒、landscape/square/portrait、日本語を含む複数言語、および仕様のsubject/motion/audio/reference分類をaggregateでcoverする
- caseごとのrights approval evidenceを要求する
- prompt adherence、perceptual video、temporal consistency、audio quality、A/V sync、人手pairwiseのversioned implementation、budget、owner、evidenceを要求する
- set全体のrights reviewerとquality ownerを別々に承認する
- selection method、case registry digest、exclusions、known failuresのreview状態を記録する

`h3fast benchmark check-quality-set`は、承認済みrecordだけ終了code 0、正当だが未完了のrecordは1、schemaまたは意味上矛盾するrecordは2を返す。

## Consequences

契約と未完了項目はCPU-only CIで検証できる。case registryやmediaを公開せずに設計を先行できるが、このrecordが`incomplete`の間はquality同等性、lossless性、Support Tier、公開releaseを主張できない。

実際のcase選定、rights review、GPU生成、metric実装とbudget決定は別途必要である。formal quality setのrights reviewerをH3-use compliance判断で代替せず、承認evidenceが揃うまでprotocolの`formal_quality_set_ready`を`false`に保つ。
