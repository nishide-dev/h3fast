# Reference-conditioned families execute on real hardware

- Date: 2026-08-18 (Asia/Tokyo)
- Protocols: `h3fast-phase1b-resident40-v1`(fl2va)、`h3fast-phase1b-ref2va-v1`(ref2va)
- Host: 承認済みJapan-local GPU host（2×RTX 6000 Ada、TP2）
- Base model: H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`
- Related: [PR #43](https://github.com/nishide-dev/h3fast/pull/43), [PR #45](https://github.com/nishide-dev/h3fast/pull/45)
- Outcome: fl2va / ref2vaがいずれも実機で成立した。3つの欠陥を修正し、うち1つはserved variantの取得が前提であった。

## Purpose

PR #43で実装したfl2va / ref2vaのformal case生成が実機で動作するかを確認する。CPUテストは通っていたが、server到達後に失敗する経路は検証されていなかった。

## Method

smoke splitから各familyの1 caseを生成した。目的は配線の成否確認であり、品質評価ではない。品質はTier 2のformal評価として別途行う。

固定条件はTP2、BF16、resident 40層、attention backend `auto`である。

## Results

3つの欠陥が順に判明し、いずれも実機でのみ現れた。

### 1. task familyの上書き

`_payload()`が`task`を`"t2va"`へhardcodeしており、fl2vaを要求してもserverはt2vaとして解釈し、conditionsとの不整合でHTTP 400を返した。

silentに誤ったfamilyを生成する経路である。400で落ちなかった場合、「fl2vaとして記録されたt2va動画」が成果物になり得た。PR #32のreviewでも同じ箇所が指摘されていたが、根本原因まで到達していなかった。

### 2. reference assetのcontainer可視性

`file://` URIがhost pathを指しており、guarded server内から解決できなかった。read-only bindと固定mount path `/reference-assets`で修正した。

### 3. served variantの制約

上記2件の修正後もref2vaはschedulerに拒否された。

```
task 'ref2va' is not served by MiniMax H3 partition 'fl2va';
supported tasks: ['t2va', 'fl2va']
```

launchが`--model-variant fl2va`をhardcodeしていた。さらにローカルsnapshotはFL2VA variantしか持っておらず、Ref2VAの重み144GB・81ファイルが未取得であった。BYOWで`--include 'FL2VA/*'`のみを指定した結果である。

`model_variant`をprotocolのruntime設定へ追加し、Ref2VAをpinned revisionから取得して再実行した。

### 生成結果

| | fl2va | ref2va |
|---|---|---|
| case | smoke-002 | smoke-003 |
| 解像度 | 768×768 (square) | 768×1344 (portrait) |
| 長さ | 5.17秒 | 10.125秒 / 243 frames |
| codec | H.264 + AAC | H.264 + AAC |
| `inference_time_s` | 470.95 | 3039.81 |
| peak memory | — | 42,210 MB |
| server起動 | 178秒 | 1,542秒 |

fl2vaはdenoise 447秒 + decode 22秒である。両familyともpipeline全stageを通過し、`MiniMaxH3PartitionAdmissionStage`と`MiniMaxH3VisualEncodingStage`の通過が参照条件の解決を裏づける。

## Interpretation

ref2vaがfl2vaの6.45倍を要した。latent数の代理指標（画素数×長さ）では3.54倍であり、残る1.82倍が超線形成分である。attentionが系列長に対して線形以上に効くことと整合する。

この6.45倍は2つのcaseの単一測定であり、familyごとの固有コスト差を分離していない。解像度・長さ・familyを同時に変えているため、ref2va自体が高コストであるという主張はできない。

server起動1,542秒はfl2vaの178秒に対し8.7倍だが、これはmodel実装ではなく取得直後のcold readである。135GBを書き込んだ直後でpage cacheに乗っておらず、共有network filesystemから読んだ。再起動時には短縮されるはずだが未測定である。

## Consequences

`model_variant`をprotocolのruntime設定として宣言し、launchへ配線した。既定`fl2va`では従来のpinned argvと同一であり、既存protocolの再現性へ影響しない。

variantに対応する重みがsnapshotに無い場合、launch構築時にfail closedとする。今回はmodel resident後のschedulerまで進んでから失敗しており、TP2のロードを消費してからでないと不足が判明しなかった。

`runtime_settings`へvariantを含めたため、serverのlifecycle記録とprotocolの不一致検査が自動的にvariantを対象にする。protocolがref2vaを宣言しながらfl2va serverへ接続したrunは拒否される。

formal setの60 caseは、t2va 20とfl2va 20をFL2VA variantで、ref2va 20をRef2VA variantで生成する。両variantの重みが必要である。

## Limits

- family別1 caseの単一runであり、統計値ではない。
- 品質は評価していない。数値を変える変更ではないが、familyごとの品質はTier 2のformal評価に属する。
- 6.45倍の内訳（解像度・長さ・family）を分離していない。
- ref2vaはimage参照1件のcaseのみで、video / audio / mixed参照は未検証である。
- server起動時間はcold read条件での値である。

## Lesson

所要時間の予測に解像度を入れ忘れた。durationだけを見て「fl2vaの2倍弱」と見積もったが、実際はportrait（画素数1.75倍）と超線形性で6.45倍であった。ADR 0012の「測定の前に予測する」を適用する際、コストを決める変数を数え落とすと予測は測定の確認にならない。

client側に45分のtimeoutを設けたため、denoise 50.7分のjobでclientがSIGTERMで落ちた。serverのjobは生き残っており、job IDを直接pollingして回収した。ここでrunnerを再実行していれば2件目のjobを投入し、experiment 0009で記録した測定汚染を繰り返していた。client timeoutは最長caseの実測所要時間から決める必要がある。

CPUテストが通っても、payload契約・container内path解決・served partitionの制約は実機でしか露見しない。3つの欠陥はいずれもこの層にあった。
