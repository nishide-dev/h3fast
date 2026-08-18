# Sage Attention Tier 2 evaluation and adoption

- Date: 2026-08-18 〜 2026-08-19 (Asia/Tokyo)
- Baseline: `h3fast-phase1b-resident40-v1`(FlashAttention、rep1、bit-exact確認済み)
- Candidate: `h3fast-phase1b-sage-attn-v1`(global `sage_attn` + text_encoder `torch_sdpa`、他は同一)
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Base model: H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`
- SGLang: pinned commit `6eb941a3…` / SageAttention: commit `d9704247…` Ada(SM89) build
- Related: [ADR 0012](../decisions/0012-tiered-optimization-verification.md), [ADR 0013](../decisions/0013-single-operator-process-reduction.md), [ADR 0010](../decisions/0010-human-pairwise-review-policy.md), [experiment 0009](0009-sage-attention-noop.md)
- Outcome: **採用**。E2E 1.63×(総時間)、blind pairwiseで品質劣化なし。

## Tier declaration

測定前に**Tier 2**を宣言した(INT8量子化attentionはoutput bytesを変える)。判定はADR 0013に従い、blind human-pairwiseを一次、objective metricを証跡とする。判定基準(temporal同水準、adherence delta無視可能、欠損/NaN fail、sample距離は合否対象外)はmetric集計前に宣言した。

## Method

t2va formal 20 case(smoke 4 + regression 16)をcandidate構成で1 rep生成した。baselineは既存のresident40 rep1(rep1/rep2 bit一致確認済み、自己変動0)。serverはguarded起動とし、生成前に`verify-backend`で`sage_attn`の遅延解決を確認した(experiment 0009のsilent fallbackを排除)。

品質判定はADR 0010のsingle-reviewer policyでreviewer `nishide-dev`が実施した。blind staging(per-case randomization、assignment封印、index.html経由の閲覧のみ)の20 case A/B ballotを全件記録し、assignment keyで復号・集計した。runner実装はPR #48/#49の版(task-scoped ballot対応)である。

## Results

### Speed(client elapsed、単一rep)

| 統計 | 高速化(FA÷Sage) |
|---|---|
| 最小 | 1.25× |
| 中央値 | 1.61× |
| 最大 | 2.14× |
| 総時間 | **1.63×**(FA 14.0h → Sage 8.6h) |

FAより遅いcaseは0件。高速化率はcaseが重いほど大きい(最軽量のsmoke-001で1.25×、長尺・高解像度で2×超)。attentionのコストが系列長に対して超線形であり、重いcaseほどattentionが支配的になるという予測と整合する。

### Quality: blind human-pairwise(一次判定)

ballot `sage-attn-t2va-v1`、20 case、reviewer 1名。

| 判定 | 件数 |
|---|---|
| candidate (Sage) 勝ち | 8 |
| baseline (FA) 勝ち | 4 |
| tie | 8 |
| score | +0.20 |

reviewerの所感は「いずれも強いて言えば程度で明確な差はない」。非tie 12件の8対4は二項検定でp≈0.19であり有意でない。結論は「Sageが優れる」ではなく「**劣化が検出されなかった**」である。

### Quality: objective metric(証跡)

| metric | 結果 |
|---|---|
| temporal consistency (step-LPIPS) | baseline p50 0.0800 / candidate p50 0.0798(同水準) |
| prompt adherence delta | p50 +0.0005、range [−0.0101, +0.0163](score域0.05〜0.19に対し無視可能) |
| sample間LPIPS | p50 0.241、range [0.050, 0.500] |
| 欠損・NaN/Inf | なし(20/20/20) |

sample間LPIPSが大きいのは劣化ではなく**別サンプルへの収束**である。INT8の数値誤差が50 stepの拡散過程で軌道を変え、異なるがもっともらしいサンプルに到達する。temporal・adherence・pairwiseのすべてが品質同等を支持する。

worst temporal delta(regression-012、+0.028)はblind判定でもbaseline勝ちであり、metricと人間判定が同一caseを指した。全体判定は変わらないが、Sageが局所的に劣るcaseが存在することは記録する。

この結果はADR 0013の判定変更の妥当性を実証する。旧契約(自己変動envelope幅0 + tolerance 0)ではsample間LPIPS 0.24で全caseが自動不合格になっていたが、人間はわずかにSage側を好んだ。

## Decision

**Sage Attention(global構成)をt2vaのTier 2最適化として採用する。** project ownerが2026-08-19に承認した。

- `benchmarks/protocol-sage.yaml`が採用構成のpinned protocolである
- FA構成(`protocol.yaml`)は今後の最適化の固定比較baselineとして維持する
- perf主張は本記録の再現条件(単一rep、client elapsed、2×RTX 6000 Ada、TP2)に限定する

## Limits

- 単一reviewer・単一repの結果である。inter-rater reliabilityは測定できない(ADR 0010)。
- t2vaのみの評価である。fl2va / ref2vaへの適用はそれぞれのformal case生成と同判定を要する。
- Sage生成のrun-to-run決定性は未確認である(baselineはbit-exact、candidateは未検証)。
- audio専用metricは未実装であり、audio品質はpairwise視聴のみでカバーした。
- 速度はclient elapsedであり、server側stage別内訳は未集計である。
- pinned SGLang commit固有のcomponent override問題(Issue #40)は未解決のまま、global指定で回避している。

## Reproduction

- 生成: `serve-guarded --protocol benchmarks/protocol-sage.yaml --sage-attention-path <ada-build>` → `verify-backend --requested sage_attn` → `run-formal-cases --task t2va`
- 判定: `prepare-human-pairwise --task t2va` → `stage-human-pairwise` → 全件`record-human-pairwise` → `check-human-pairwise`
- metric: `score-perceptual-video` / `score-temporal-consistency` / `score-prompt-adherence`を全caseでbaseline rep1と比較
- private artifact(ballot、assignment、seed、media manifest)はGit外のJapan-local storageに保持する
