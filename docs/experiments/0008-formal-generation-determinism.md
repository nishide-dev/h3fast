# Formal generation determinism and 20/40-layer output equality

- Date: 2026-08-17 (Asia/Tokyo)
- Baseline protocol: `h3fast-phase1a-baseline-v1`（DiT resident 20層）
- Candidate protocol: `h3fast-phase1b-resident40-v1`（DiT resident 40層）
- Host: 承認済みJapan-local GPU host（2×RTX 6000 Ada per server、4 GPU並列）
- Related: Issue #16, [ADR 0008](../decisions/0008-formal-quality-metric-plan.md), [ADR 0011](../decisions/0011-bit-exact-repetition-exemption.md)
- Outcome: 生成はbit-exact決定的。20層と40層の出力は全対象caseでbit単位一致。

## Purpose

formal quality setに対するbaseline自己変動を実測し、metric budget設定の前提を確立する。あわせて採用済みの40層candidateとbaselineの品質差を測定する。

## Scope

- 対象は現在生成可能なt2va 20 case（smoke 4 / regression 16）。formal setのfl2va 20件とref2va 20件はreference条件付きpayloadが未実装のため対象外とする。
- 各protocolについてrep1とrep2の2反復を独立に生成した（計80生成、うち80件が完了）。
- 固定条件はH3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`、SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`、BF16、50-step schedule、`torch.compile`無効、case固定seed。attention backendはautoで`fa`（FlashAttention）に解決された。
- 2 serverを同一host上で並列実行したため、latency・memory値はpinned単一server測定と比較しない。本実験はlatencyを主張しない。

## Method

`h3fast benchmark run-formal-cases`でprivate reviewed registryをformal quality setへdigest拘束し（prompt digest、seed、task、duration、aspect ratioの一致検証）、pinned protocolの固定生成parameterでcaseごとに生成した。各生成物のSHA-256はrunnerがper-case resultとredacted run manifestへ記録する。

判定は生成物のSHA-256照合のみで行い、media本体は参照しない。

## Results

### 反復間の決定性（自己変動）

両protocolについて、rep1とrep2の生成物SHA-256が全20 caseで完全一致した（40比較、不一致0件）。生成時刻と所要時間は反復間で異なるが、出力bytesは同一である。

### protocol間の出力一致

同一caseにおいてbaseline20とresident40の生成物SHA-256も全20 caseで一致した（20比較、不一致0件）。

| Case | duration (s) | rep1=rep2 かつ 20層=40層 digest（先頭16文字） |
|---|---:|---|
| smoke-001 | 4 | `748134a32a6cddfd` |
| smoke-004 | 15 | `5c896929828b0152` |
| smoke-007 | 10 | `0edccfb70709af8c` |
| smoke-010 | 5 | `45fe36cc0abd9c19` |

（残る16 regression caseも同様に全一致。per-case digestはprivate run manifestに記録し、本記録には代表4件のみ掲載する。）

## Interpretation

1. **baseline自己変動は0である。** 固定runtime・固定seedの下で生成はbit-exact決定的であり、metric familyのbaseline envelopeはすべて厳密に0となる。これによりmetric budgetは「baselineが自分自身とどれだけ異なり得るか」ではなく、許容する実差そのものとして設定できる。
2. **40層candidateは品質を変えていない。** 出力がbit単位で同一であるため、prompt adherence、perceptual video、temporal consistency、audio quality、A/V sync、human-pairwiseのいずれの指標でも差は厳密に0である。これはDiT resident layer数がoffload placementのみを変更しcompute graphを保存するという設計上の主張を、単一caseのexact gateから20 caseへ拡張して裏づける。
3. **3反復目は情報を持たない。** 2反復のdigest一致が証明された条件下で追加反復は同一bytesの再生成にしかならない。この根拠に基づき[ADR 0011](../decisions/0011-bit-exact-repetition-exemption.md)でmetric planへbit-exact例外を追加し、rep3を打ち切った（baseline rep3は打ち切り時点で1件生成済み、破棄可）。

## Limits

- bit-exact性はこのpinned runtime、GPU構成、生成parameterでの実測事実であり、一般保証ではない。runtime、driver、GPU、schedule、precision、attention backendのいずれかが変わる場合は再測定する。
- 数値を変える最適化（量子化、step蒸留、量子化attention等）ではbit-exact性は成立しない。それらの候補ではmetric実装による実測が必須である。
- 対象はt2va 20 caseに限る。fl2va/ref2va 40 caseの決定性と品質は未測定である。
- 生成物はH3 OutputとしてGit外のJapan-local storageに保持し、本記録にはdigestとcountのみを掲載する。

## Follow-ups

- formal quality setのfl2va/ref2va生成に必要なreference条件付きpayloadを実装し、残る40 caseを同じ手順で測定する。
- metric familyのbudgetとowner approvalは、数値を変える最適化候補を評価する時点で必要になる。bit-exact placement最適化の判定にはdigest照合で足りる。
- 次の最適化候補（量子化attention、step蒸留等）はcandidate assessmentへ追加してから、1変数ずつ実装・実測する。
