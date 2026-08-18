# ADR 0013: Reduce quality process to what a single operator can execute

- **Status:** Approved by project owner (2026-08-18)
- **Date:** 2026-08-18
- **Related:** [ADR 0008](0008-formal-quality-metric-plan.md), [ADR 0009](0009-formal-quality-metric-selection.md), [ADR 0011](0011-bit-exact-repetition-exemption.md), [ADR 0012](0012-tiered-optimization-verification.md)

## Context

枠組み全体(spec 2,234行、ADR 12本、schema 22個、benchmark CLI 22 subcommand、実装約10,000行)をreviewした。運用実態はsingle operator、Japan-local、公開release予定なし(release gateは意図的に`blocked`)である。reviewで次の3つの欠陥を確認した。

**1. Tier 2が通過不可能である。** ADR 0008のexact profileは「baseline自己変動envelopeとの比較、追加tolerance 0」を要求する。しかし生成はbit-exact決定的であることが実証済みで(experiment 0008)、自己変動は厳密に0、envelopeの幅は0である。したがってbytesを変える最適化(量子化attention、step蒸留など、Tier 2の全対象)は、品質が実際に同等でも全familyで自動的に不合格になる。この契約のままではSage Attentionを含むいかなるTier 2最適化も採用できない。

**2. `planned`昇格条件が循環している。** ADR 0009は`planned`の前提に「formal 60 caseのbaseline自己変動測定」を課すが、自己変動はbit-exact性から0であると予測でき、測定は答え合わせにしかならない。またfl2va/ref2vaの40 caseの生成には数十GPU時間を要し、その生成物は`planned`という記録状態以外の何も変えない。記録状態のためにGPU時間を要求する構造になっていた。

**3. 昇格条件が4箇所に重複している。** 同じ条件がADR 0009とspecの3段落(metric plan節、各adapter節)に別々の文言で書かれ、参照する場所によって読み取れる条件が異なった。この重複は実際にoperator(agent)の判断ミスを2回誘発した。

過剰の実例として、再配布しないlocal checkpointのlicense調査に公式ソース3箇所の確認と記録分割を行い、blockerを1つも解消しなかった作業が同日に発生している。

## Decision

### 1. Tier 2の品質判定はhuman-pairwiseを一次、objective metricを証跡とする

bytesが変わる候補(Tier 2)の品質合否は、実装済みのblind human-pairwise runner(ADR 0010のsingle-reviewer policy)による影響task familyのformal case比較で判定する。実装済みのobjective metric(perceptual-video、temporal-consistency、prompt-adherence)は全影響caseで実測して**証跡として記録**し、ownerが異常なdeltaを確認する材料とする。zero-width envelopeによる自動合否には使わない。

zero-tolerance判定はTier 1(placement-only)にのみ適用し、これは既存のper-case digest gateが担う(ADR 0012、変更なし)。

familyごとの数値budgetは、ownerが実測deltaの分布を見て閾値を設定できると判断した時点で`approved`として導入する。それまでmetric planの`approved`はTier 2の前提ではない。

### 2. `planned`の条件を実行可能なものへ縮小する

metric planの`planned`は次で足りる。

- H3Fast内のtyped entrypointとcorrectness test
- 固定runtimeの実H3出力に対する動作確認(1 case以上)

「formal 60 caseの自己変動測定」は`planned`の前提から外す。自己変動の実測は、bit-exact性が成立する限りADR 0011のdigest一致証明で代替し、bit-exactが成立しない構成が現れた場合にのみ実測へ戻す。

### 3. 昇格条件の記述を一元化する

metric familyの昇格条件はこのADRのみに置く。specは本ADRを参照し、条件本文を繰り返さない。各adapter節はmetric契約(入力、score方向、fail-closed規則)だけを記述する。

### 4. 凍結する領域

次は実装済み・検証済みのまま**凍結**し、明示のtriggerが発生するまで追加実装・追加記録・追加検証を行わない。

| 領域 | trigger |
|---|---|
| release gate機構 | 公開release(wheel/image/bundle/service)の具体的な計画 |
| territory inventory | operator、国、第三者アクセス、配布、serviceのいずれかの変更 |
| registry attestation / review chain | registry内容の変更、または外部者へのcase開示 |
| human-pairwiseのformal評価フロー | Tier 2判定での使用(runner自体は§1の一次判定手段として使う) |
| 新規schema・新規validator | 実運用で検証漏れが実害を出した事実 |

### 5. 削除は行わない

既存のコード、schema、テストはすべて動作しており、削除は解消する問題がないままchurnとregressionリスクを生む。縮小は「追加をやめる」ことで行う。

## Consequences

- Sage AttentionのTier 2評価が実行可能になる: 影響familyのformal caseをbaseline/candidate両構成で生成し、objective metric 3種を記録し、blind pairwiseでownerが判定する。
- 実装済み3 familyは本ADRの条件で`planned`にできる(実H3出力での動作確認は2026-08-18に完了: LPIPS系はbit一致入力で0.0、prompt-adherenceは20 case×2 repで自己変動0.0)。
- audio-qualityとav-syncのobjective metricは未実装のまま残る。Tier 2判定はpairwise(音声を含む視聴)がカバーし、専用metricは実害(pairwiseで判定できない事例)が出るまで実装しない。
- ADR 0008の評価意味論(反復数、統計量、欠損fail、family独立)はrecord構造として維持する。本ADRが変更するのは合否のarbiterであり、記録の形ではない。

## Rejected alternatives

- **envelopeへ人為的なtolerance定数を足す**: 根拠のない数値がgateになり、budget承認の意味が失われる。
- **audio/av-sync metricを先に実装してから Tier 2 を再開する**: 未検証の外部checkpoint(ViSQOL、Synchformer)の導入コストが、pairwiseで代替可能な判定のために先行する。順序が逆である。
- **枠組みの縮小をspec全面改訂で行う**: 差分が大きく、review不能になる。条件の一元化と参照化に留める。
