# Artifact register

最終更新: 2026-08-16

この表は技術的な来歴記録であり、法的助言ではありません。公開前に適切なreviewerの承認を記録してください。

| 成果物 | 出所 | H3との関係 | License | 配布範囲 | 状態 |
|---|---|---|---|---|---|
| H3Fast source | 独自実装 | H3専用tooling。MiniMax Materialsのsource fileは未収録 | Apache-2.0 | Public | engineering inventoryとproject classification済み |
| `h3fast` wheel/sdist | H3Fast sourceからbuild | H3重み・cache・MiniMax source fileを含まない | Apache-2.0 | Public候補 | clean install済み。territory approval対象外、他のrelease gate待ち |
| CPU CI artifacts | H3Fast sourceとtest fixture | 合成fixtureのみ。H3 Materialsを扱わない | Apache-2.0 | CI内部 | 利用可 |
| Initial Runtime release gate | H3Fast独自metadata | owner、期限、evidenceとcode-only release checkを記録。H3 Materialsなし | Apache-2.0 | Public候補 | `blocked`。release/schema ownerと品質・再現性・supply-chain等が未完了 |
| Initial Runtime territory inventory | H3Fast独自metadata | H3 access、GPU、Output、実行のregion evidenceを記録。H3 Materials/Output本体なし | Apache-2.0 | Public候補 | Japan-local single-operator research scopeをself-attestationで承認。第三者提供・service・Japan外利用は再review必須 |
| Formal quality-set record | H3Fast独自metadata | 10/50件のcase identity、coverage、rights/metric approval契約。prompt本文、reference media、生成物なし | Apache-2.0 | Public候補 | review済み60件のredacted metadataと全required coverageを登録。Rights reviewer承認済み、Quality owner割当済み。metricsとformal set approvalは未完了 |
| Formal quality registry attestation | H3Fast独自metadata | Git外candidate registry全体のSHA-256、10/50件数、selection method、redaction状態。per-case digest、prompt、path、mediaなし | Apache-2.0 | Public候補 | review前のlocal 60-case candidate identityをattestした時点記録 |
| Formal quality review attestation | H3Fast独自metadata | source registry/contentのaggregate SHA-256、reviewer/owner、60件の集約rights/selection判断。per-case digest、prompt、path、mediaなし | Apache-2.0 | Public候補 | `nishide-dev`のproject-owner self-attestationとして承認済み。外部法務意見、metric、formal set、GPUまたはrelease承認ではない |
| Private quality registry | H3Fast所有または権利review対象のprompt/reference input | prompt本文、local asset path、rights evidenceを含み得る。H3 Materials/Outputは含めない | 入力ごとの権利に従う | repository外・access制御 | immutable evidence適用済みreviewed registryをlocal生成。Git管理禁止 |
| Private quality review | H3Fast独自のlocal review decision | registry/content digest、per-case digest、rights/selection decision、evidenceを含む。prompt/media/pathなし | Apache-2.0候補 | repository外・access制御 | 60件とselection判断を承認しimmutable evidence適用済み。Git/CI artifact化禁止 |
| Formal quality metric plan | H3Fast独自metadata | 6 familyの実装identity、dependency pin、入力、score方向、budget、evidence契約。prompt/media/Outputなし | Apache-2.0 | Public候補 | evaluation semanticsを固定した`draft`。全family unassigned、method/owner/budget approval未完了 |
| Formal quality metric candidate assessment | H3Fast独自metadataと外部projectの公開metadata | 6 familyの候補、固定source/model revision、license scope、採用条件、blocker。外部code/model/checkpoint本体、prompt/media/Outputなし | H3Fast metadataはApache-2.0。候補componentは各記録のlicenseに従う | Public候補 | SigLIP2/LPIPS/H3Fast ballotをcandidate、ViSQOL/Synchformerをblockedとして記録。実装・runtime確認・自己変動・承認は未完了 |
| Private human-pairwise ballot / assignment / seed / media manifest / staging | H3Fast独自のlocal review artifact | formal case ID、blind A/B判断、hidden assignment、salt、seed、media path/digest、blinded media copyを含む | Apache-2.0候補(media自体は入力の権利に従う) | repository外・access制御 | schema、prepare/stage/record/check CLI、commitment検証とaggregate scorerを実装。synthetic media pilotでworkflow検証済み([experiment 0006](../experiments/0006-human-pairwise-pilot.md))、single-reviewer policy承認済み([ADR 0010](../decisions/0010-human-pairwise-review-policy.md))。formal reviewは未完了。Git/CI artifact化禁止 |
| Compiled quality metadata | private registry compilerのredacted出力 | prompt/referenceのSHA-256と公開可能metadataのみ。本文、path、mediaなし | Apache-2.0 | Public候補 | 60件をformal recordへ採用。rights/coverage検証済み、metricとformal set approval待ち |
| LPIPS AlexNet backbone checkpoint | download.pytorch.org `alexnet-owt-7be5be79.pth`(SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`) | 外部pretrained weights。H3 Materials/Outputなし | torchvision projectが配布。checkpoint自体のlicense scopeは未確認 | ローカル実験限定・再配布なし | digest固定・offline検証をadapterで強制。lpips 0.1.4同梱linear weightsはBSD-2-Clause pinned revisionに従う |
| SigLIP2 model snapshot | `google/siglip2-base-patch16-256` revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`(7 fileのSHA-256 manifestで固定) | 外部pretrained model。H3 Materials/Outputなし | HF metadataはApache-2.0表記。再配布前に個別review | ローカル実験限定・再配布なし | manifest検証をadapterで強制。offline loadと意味的方向性をlocal CPUで確認済み |
| SageAttention build | thu-ml/SageAttention commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`、Ada(SM89)向けlocal build、wheel SHA-256 `bae8c1a02a5b3246cb73a1f8bfc86549c48092d7ae010cd1c1f935479915e850` | 外部attention kernel。H3 Materials/Outputなし | Apache-2.0 | ローカル実験限定・再配布なし | runtime imageへ同梱せず外部pathからbind/PYTHONPATHで注入。runtime image digestは不変。Tier 2実測は未完了 |
| SGLang adapter metadata | 独自実装 | SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`との互換性とmedia probeを記録。コードcopyなし | Apache-2.0 | Public | 20層baselineと40層candidateをlocal実測済み。単一case exact gate合格、support未付与 |
| SGLang source checkout | sgl-project/sglang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e` | H3 pipelineを含む外部runtime source。repositoryには未収録 | Apache-2.0ほか依存license | ローカル実験限定・再配布なし | 固定checkoutでE2E実機検証済み。依存license review未完了 |
| SGLang CUDA 12.9 SIF | `lmsysorg/sglang` amd64 manifest `sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f` | H3重みを含まない外部runtime image | image内licenseに従う | ローカル実験限定・再配布なし | 固定digestのSIFでE2E実機検証済み。再配布・SBOM・license review未完了 |
| MiniMax H3 FL2VA snapshot | `MiniMaxAI/MiniMax-H3` commit `42ed227ee7df40d41602854ae760620d6eb651fe` | MiniMax H3 Works | MiniMax H3 Community License | 配布禁止 | 84 files / `144051241571` bytesをlocal検証済み。権利・地域review未完了 |
| H3 derivative weights | 未作成 | Model Derivative候補 | MiniMax H3 Community License | 配布禁止 | 対象外 |
| AdaLN cache / LoRA / quantized weights | 未作成 | Model Derivative候補 | 未確定 | 配布禁止 | 対象外 |
| Benchmark outputs | 固定H3 baselineおよび40層candidateで作成 | H3 Output | H3 license・AUP review必須 | 非公開・Git管理禁止 | 各protocolでwarmup 1回・測定3回済み。candidate measured 3成果物はexact gate合格。配布review待ち |
| Exact quality reference metadata | local Benchmark outputsから生成したdigest・media metadata | H3 Output自体を含まず、Outputのhashとstream metadataを記録 | H3 license・AUP review必須 | Public候補 | 単一caseのplacement-only gateで実測検証済み。一般品質・配布review待ち |
| Benchmark protocols | H3Fast独自metadata | 固定revision、runtime、case、quality methodを記録。prompt本文1件を含む | Apache-2.0候補。promptの公開権reviewが必要 | Public候補 | 20層baselineと40層既定protocolをschema検証済み。formal set未完成 |

## Initial boundary decision

- H3Fast repositoryへMiniMax H3のコード、設定、重み、documentationをcopyしない。
- 初期CIはH3をdownload・import・実行しない。
- SGLangはruntime dependencyにせず、version検出とadapter契約だけを実装する。
- H3を用いるGPU CI、benchmark、変換は、Applicable Territoryと対象環境のreview完了後に別途追加する。
- 独立codeのsource storage、CPU CI、global配布にはH3 territory gateを適用しない。境界変更時は[ADR 0006](../decisions/0006-independent-code-license-boundary.md)に従って再分類する。
