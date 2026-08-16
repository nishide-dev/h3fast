# ADR 0010: Human-pairwise single-reviewer policy

- **Status:** Approved by project owner (2026-08-16)
- **Date:** 2026-08-16
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16), [ADR 0009](0009-formal-quality-metric-selection.md), [experiment 0006](../experiments/0006-human-pairwise-pilot.md)

## Context

ADR 0009はhuman-pairwise candidateの残blockerとして、reviewer数、conflict handling、single-reviewer acceptabilityの未承認を挙げていた。presentation runnerとsynthetic-media pilot(experiment 0006)は完了しており、review policyだけがhuman-pairwise familyの運用開始を止めていた。

本プロジェクトは承認済みのJapan-local single-operator scope(ADR 0007)で運用され、H3 Worksと生成Outputへaccessできる人間は`nishide-dev`だけである。複数reviewerを立てるにはOutputの第三者提供が必要になり、territory inventoryの再reviewを要する。

## Decision

formal human-pairwise評価を次のsingle-reviewer policyで運用する。

1. **Reviewer:** `nishide-dev`(quality owner)1名とする。single-reviewer運用はproject ownerが2026-08-16に承認した。
2. **Conflict handling:** reviewer間conflictは構造上発生しない。case内で判断に迷う場合は`tie`を選択する。
3. **誤入力の訂正:** ballot完了前は`record-human-pairwise --overwrite`で訂正できる。completed ballotは不変とし、訂正が必要な場合は新しいseedとballotで再実施する。
4. **Bias controls:** blind staging(hidden assignment、case単位randomization)を必須とし、reviewerはreview完了までassignment key、staged fileのサイズ・timestamp等のmetadataを閲覧しない(experiment 0006のfile size side channel対策)。staging directoryの閲覧は`index.html`経由に限る。
5. **Session手順:** 60 caseを複数sessionに分割してよい。中断・再開はpending ballotの部分記録として扱い、schemaが混在状態を許容する。
6. **記録:** formal実施ではballot ID、reviewer、実施日、使用したrunner実装のimmutable revisionをexperiment記録へ残す。

## Consequences

- human-pairwise familyの残blockerは、GPU実出力によるformal ballotとimmutable implementation evidenceだけになる。
- single-reviewer運用ではinter-rater reliabilityを測定できない。formal結果には単一reviewerの判断であることを明記し、公開時の品質主張はこの限界とあわせて提示する。
- reviewerを追加する場合はOutputの第三者提供に該当するため、territory inventoryを`incomplete`へ戻して再reviewしてから本ADRを改訂する。
