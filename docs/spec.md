# H3 Fast Distribution Specification

- **文書名:** MiniMax H3 高速・効率化派生版 配布仕様
- **略称:** H3 Fast Distribution Spec
- **状態:** Draft v0.33（fl2va/ref2va生成を実装、GPU実測とTier 2評価が未完了）
- **最終外部調査日:** 2026-08-16 (Asia/Tokyo)
- **最終更新日:** 2026-08-16 (Asia/Tokyo)
- **対象:** MiniMax H3-Base FL2VA / Ref2VA を基礎とする高速化・効率化ランタイムおよび派生モデル
- **想定読者:** モデル研究者、GPUカーネル開発者、MLOps/SRE、配布・法務担当、サービス開発者

> **重要:** 本仕様は技術・運用設計であり、法的助言ではない。MiniMax H3 Community Licenseは地域制限、派生モデル、再配布、Hosted Service、商用利用およびAcceptable Use Policyに独自条件を持つ。派生重みまたはサービスを公開する前に、最新版の原文を法務担当と確認し、必要に応じてMiniMaxから書面による許可を得ること。

---

## 0. 規範用語

本書では次の用語を使用する。

- **MUST / 必須:** 満たさなければ本仕様準拠とみなさない。
- **MUST NOT / 禁止:** 実施してはならない。
- **SHOULD / 推奨:** 正当な理由がある場合のみ逸脱可能。逸脱理由を文書化する。
- **MAY / 任意:** 実装または運用判断に委ねる。

### 0.1 文書の適用範囲

- 本書のMUST/MUST NOTは、明示されたPhaseの成果物にだけ適用する。将来Phaseの要件を現在のリリースゲートへ混在させない。
- JSON、TOML、CLI、APIの例に含まれる`<...>`はプレースホルダーであり、そのまま実装値として使用してはならない。
- 外部プロジェクトの「現行」挙動は本書の最終調査日時点の参考情報である。実装とリリースでは、依存先のcommitまたはversionを固定し、固定した版の挙動をテストで確認する。
- 法的要件は本書だけで充足したと判断してはならない。特にMiniMax H3 Community Licenseの地域制限は、開発者、CI、配布元、利用者、Hosted ServiceおよびOutputの利用場所を含めて別途確認する。

---

## 1. エグゼクティブサマリー

H3高速版は「Hugging Faceへ重みを置くプロジェクト」ではない。完成時の製品構成は次の独立した配布物を想定するが、すべてが現在実装・公開済みという意味ではない。

1. **公開ランタイム:** Python SDK、変換CLI、Tritonカーネル、SGLang統合、フォールバック実装
2. **モデル成果物:** 派生重み、LoRA、量子化重み、AdaLNキャッシュ、最適化マニフェスト
3. **再現可能な実行環境:** 重みを含まないOCIコンテナ
4. **サービス境界:** SGLang互換の非同期動画API、認証、レート制限、安全対策
5. **供給網情報:** 署名、SBOM、SLSA provenance、チェックサム、ベンチマーク結果
6. **配布権限管理:** ライセンス同意、地域・組織確認、監査可能な承認記録

初期公開候補は、次の方式に限定する。

> **推奨MVP: Public Runtime + Bring Your Own Weights (BYOW)**
> 利用者が公式H3を自身の権限と責任で取得し、将来の`h3fast convert`でローカル変換する。`convert`はv0.11時点では未実装であり、派生重みの一般公開はMiniMaxおよび法務担当から配布条件の確認を得るまで行わない。

BYOWは重みの再配布を避ける配布方式であり、H3を利用・変換する権利を付与したり、地域制限やAUPを回避したりする仕組みではない。H3Fastが独立実装しMiniMax Materialsを含まないsource、wheel、schema、CLI、protocolおよびmetadataはApache-2.0成果物として扱い、その保存・CI・配布へH3のterritory gateを適用しない。公式H3コード、設定、重みその他のMaterialsを`h3fast`へコピーまたはadaptする変更はこの分類を継承せず、同じ変更で再分類する。

初期実装は次の順序で進める。

1. **Phase 0:** ライセンス境界、固定baseline、対象GPU、測定方法を確定する。
2. **Phase 1A:** 単一Python packageでmanifest、検証、doctor、benchmark harnessを実装し、固定したSGLang版を参照backendとしてbaselineを再現する。
3. **Phase 1B:** プロファイル結果から支配的なbottleneckを特定し、効果を単独測定できる最初の最適化を一つ実装する。

### 1.1 現在の実装状態

2026-08-16時点のrepository実装を次に限定して扱う。将来PhaseのCLI、package、container、service例を現行機能として実行してはならない。

| 区分 | 状態 | 現在の範囲 |
|---|---|---|
| 単一Python distribution | 実装済み | `src/h3fast`、Python 3.12、CPU import、wheel/sdist build |
| BYOW入力検証 | 実装済み | 明示されたlocal snapshotのrevision・構造・任意の全file SHA-256検証。download/convertは行わない |
| Manifest検証 | 実装済み | `h3fast_manifest.json`とper-file checksumのfail-closed検証 |
| 診断 | 実装済み | Pythonとoptional SGLangの診断。GPU初期化やmodel downloadは行わない |
| Release gate | 実装済み・blocked | 独立Initial Runtimeのrequired checkとrelease/schema approvalをrecord化。全承認前は終了code 1 |
| Territory inventory | Japan-local scope承認済み | `nishide-dev`が宣言済みJapan内machine/storageで行うsingle-operator local researchに限定。第三者提供、service、Japan外利用または環境変更は再review |
| Benchmark harness | 実装済み | 固定SGLang/SIF、preflight、guard、非同期T2VA client、stage集計、local bundle |
| Quality gate | 限定実装 | 固定1 caseのplacement-only exact decoded artifact gate |
| Formal quality set | 60-case rights metadata・metric candidate登録済み、incomplete | review済み10 smoke / 50 regressionのredacted metadata、coverage、immutable rights evidenceを登録。Rights reviewer承認済み、Quality ownerは`nishide-dev`へ割当済み。6-family candidate assessment、human ballot contract、offline presentation runnerを実装しsynthetic-media pilotを完了し、single-reviewer policyを承認した（ADR 0010）。perceptual-video (LPIPS)、temporal-consistency (adjacent-frame LPIPS trajectory)、prompt-adherence (SigLIP2) adapterを実装したが、metric実測/承認、formal set approval、GPU評価は未完了 |
| 最初の最適化 | 実測済み | DiT resident 20→40層。40層を既定、20層を明示rollbackとする |
| Converter / derivative weights | 未実装 | 法務・artifact分類・正式quality gate後の将来Phase |
| H3Fast kernel / quantization / cache | 未実装 | profile根拠と個別correctness/quality benchmarkが必要 |
| Public wheel | build可能・未公開 | release、供給網、license gate未完了 |
| OCI / Hosted API | 未実装・未公開 | securityを含む各release gate未完了 |

実装監査の詳細は[`docs/audits/0001-phase1b-completion.md`](audits/0001-phase1b-completion.md)を参照する。

SGLangを初期の参照実行系とし、H3 pipelineや非同期video API全体を最初から再実装しない。`h3fast`は検証可能な最適化recipe、manifest、診断、benchmark、および必要なkernel/backend差分を所有する。SGLangの内部APIへ依存する箇所はadapterに隔離し、対応commitを固定する。

派生重みを配布できる状態になった後も、ランタイム、重み、コンテナは分離してリリースする。重みをコンテナへ内包してはならない。

---

## 2. スコープ

### 2.1 対象

本仕様が対象とするもの:

- H3-Base FL2VA
  - Text-to-Audio-Video (`t2va`)
  - First/Last-Frame-to-Audio-Video (`fl2va`)
- H3-Base Ref2VA
  - 画像、動画、音声参照による `ref2va`
  - Ref2VAとして提供されるVideo-to-Video用途
- 次の最適化を単独または組み合わせた派生物
  - AdaLN事前計算
  - FP8 / INT8 / W4A8等の量子化
  - Few-step LoRA、蒸留Student
  - Sparse Attention
  - Feature Cache / Cross-step Cache
  - Triton/CUDAカーネル融合
  - VAEのチャンク化・キャッシュ
  - 並列化・配置最適化

### 2.2 対象外

- MiniMaxのHosted H3-Context-IRそのものの再配布
- 公開されていないH3-Regenerate-2Kの再実装または重み配布
- MiniMax公式APIをローカル実装と誤認させる表示
- MiniMaxから許諾されていない地域へのH3 WorksまたはModel Derivativesの配布
- H3と無関係な汎用動画生成プラットフォーム全体の仕様
- Phase 1における独自queue、課金、tenant管理、moderation serviceまたはSGLang互換server全体の再実装
- 実測前に複数GPU vendor、全量子化方式、全task familyを同時にTier 1対応すること

H3公式モデルカード上、完全なシステムはContext-IR、H3-Base、Regenerate-2Kから構成され、ローカル公開されている中心部分はH3-Baseである。本製品はローカルH3-Base高速版であり、公式完全システムと同等であると表示してはならない。

---

## 3. 基本設計判断

### 3.1 配布モード

| モード | 内容 | 公開条件 | 推奨時期 |
|---|---|---|---|
| A. BYOW Runtime | 公開コードとローカル変換・検証CLI。利用者が公式重みを取得 | コード由来と地域適用のレビュー後に検討可能。H3の利用権を代替しない | MVP |
| B. Controlled Weights | 派生重みを手動承認で提供 | MiniMax/法務確認、地域・利用者審査、同意記録が必須 | Phase 2 |
| C. Hosted API | 運営者が重みを保持しAPI提供 | 利用規約、安全対策、違反報告、監査、地域制御が必須 | Phase 3 |
| D. Enterprise On-Prem | 顧客環境へ署名済みコンテナと重みを納品 | 契約、地域・再配布制限、サポート境界が必須 | Phase 3 |

### 3.2 推奨製品構成

```text
Public Git Repository
  └─ h3fast source, converter, Triton kernels, tests

PyPI
  └─ h3fast wheel / sdist

OCI Registry
  └─ accelerator-specific runtime images, no weights

Model Registry
  ├─ metadata-only model cards for BYOW
  └─ controlled derivative-weight repositories after authorization

Artifact/Entitlement Service (optional but recommended for controlled weights)
  ├─ license acceptance
  ├─ organization and territory review
  ├─ time-limited signed download URL
  └─ audit log

Serving Layer
  └─ SGLang-compatible /v1/videos asynchronous API
```

### 3.3 重みとコードの分離

- ランタイムの更新と重みの更新は独立して行えること。
- カーネル修正のために100GB超の重みを再配布してはならない。
- モデル更新のためにコンテナ脆弱性修正版を作り直す必要がない構成にすること。
- コンテナ内にH3重み、HFアクセストークン、顧客入力、生成物を焼き込んではならない。

### 3.4 初期アーキテクチャ判断

- 参照backendは、immutable commitまたはreleaseへ固定したSGLangとする。
- H3公式repositoryまたはSGLangからコードをコピーする場合、出所、license、変更内容をファイル単位で記録する。コピーせず依存として利用できる場合は依存を優先する。
- 最適化は一度に一つずつ追加し、baselineとの差をkernel単体、stage、E2E、品質の各層で測定する。
- `exact`は「bitwise identical」を意味しない。許容する数値差と品質回帰gateをbenchmark protocolで定義するまでは、外部向けにlosslessまたは無損失と表現しない。
- 初期GPU、task family、解像度、durationはPhase 0で一つのTier 1候補へ絞る。未検証の組み合わせはExperimentalまたはUnsupportedと表示する。

---

## 4. ライセンス・コンプライアンス仕様

### 4.0 成果物分類を先に行う

公開可否は「重みを含むか」だけで判断しない。各成果物について、少なくとも次を`docs/compliance/artifact-register.md`へ記録する。

| 項目 | 記録内容 |
|---|---|
| 成果物 | source、wheel、container、cache、adapter、weight、service、benchmark data等 |
| 出所 | 独自実装、MiniMax Materials、SGLang、その他第三者 |
| H3との関係 | H3非依存、H3専用だが独自実装、Materialsを含む、Model Derivative候補 |
| 適用license | license名、取得元、immutable revision、取得日、digest |
| 配布範囲 | 公開、地域・利用者制御、社内限定、配布禁止 |
| 承認 | reviewer、判断日、根拠または書面許諾 |

`h3fast`という名称がMiniMaxの公式製品・提携製品であるとの誤認を招かないかも、公開前の商標・表示レビュー対象とする。

### 4.0.1 独立code境界

MiniMax H3 Community Licenseの`Materials`はMiniMaxが公開したH3本体とDocumentationを対象とし、`MiniMax H3 Works`はMaterials、Model Derivativesおよびそれらの派生物を対象とする。H3Fastが所有し独立実装したcodeは、H3専用またはH3互換であることだけを理由にH3 Worksへ分類しない。

現行repositoryとwheelではMiniMax source、weight、configuration、Documentationのcopyを検出していないため、source、schema、CLI、wheel、sdist、合成fixtureおよびH3 Worksを含まないmetadataをApache-2.0成果物として分類する。このproject decisionの根拠、guardrail、再review条件は[ADR 0006](decisions/0006-independent-code-license-boundary.md)と[`docs/compliance/h3-license-boundary-review.md`](compliance/h3-license-boundary-review.md)をsource of truthとする。

この分類は、H3 snapshotの取得・実行、Model Derivative、OutputまたはH3 Worksを組み込むserviceへH3のlicense条件が適用されないという意味ではない。また、SGLangその他の第三者成果物のlicenseをApache-2.0へ置き換えない。

### 4.1 現行ライセンスから得られる重要な制約

2026-08-16時点のMiniMax H3 Community Licenseでは、少なくとも次をリリース設計へ反映する必要がある。

1. Applicable Territoryは世界からEU、英国、韓国、米国を除いた地域として定義されている。
2. Model Derivativesの定義は広く、修正、蒸留、中間表現の転移、H3出力による合成データ学習などを含む。
3. 第三者へ配布する場合、ライセンス原文、変更表示、指定NOTICEが要求される。
4. 商用製品・サービスではUIへ「MiniMax H3」を表示する条件がある。
5. 年間売上2,000万米ドル超の商用製品・サービスには事前の書面承認条件がある。
6. Hosted Serviceでは、利用者を保護的な利用規約へ拘束し、安全策、報告経路、調査・停止対応を維持する必要がある。
7. AUPは更新され得るため、リリース時だけでなく継続監視が必要である。
8. H3 WorksまたはそのOutputsをApplicable Territory外で利用、再配布、表示することも禁止対象に含まれる。BYOWやOutputだけの移動でこの制限は消えない。
9. H3 Worksを組み込んだ製品・サービスへのアクセス提供前に、Hosted Serviceに限らず、各recipient/userをSection VおよびAUPと同等以上に保護的な強制可能な条件へ拘束する要件がある。

### 4.2 独立code release gateとH3-use gate

独立実装のPublic Runtimeと、H3 Worksを扱う実行・artifact・serviceは別々に判定する。

独立codeのrelease stateは[`compliance/release-gates/initial-runtime.json`](../compliance/release-gates/initial-runtime.json)へ記録する。`h3fast release check`はcode boundary、quality、reproducibility、supply chain、notice、support target、converter、GPU E2E、public benchmarkおよびrelease/schema approvalが完了した場合だけ成功する。H3 territory approvalはこのcode-only release判定のrequired checkにしない。

H3 snapshotへのaccess、保存、変換、GPU実行、Output利用またはH3 Worksを組み込むserviceのregion stateは[`compliance/territories/initial-runtime.json`](../compliance/territories/initial-runtime.json)へ記録する。`h3fast compliance check-territories`はH3-related flowのregion、operator、owner、evidenceと必要なapprovalが揃った場合だけ成功する。現行approvalは[ADR 0007](decisions/0007-japan-local-h3-use-scope.md)に従い、`nishide-dev`が宣言済みJapan内machine/storageで行うsingle-operator local researchだけを対象とする。2026-08-17に同一条件の2台目Japan-local GPU host(`gpu-benchmark-host-secondary`)を追加した。このhostは固定protocolの測定に使用せず、探索的実行に限定し、その測定値をpinned baseline/candidateと比較しない。第三者access、Hosted Service、derivative/Output配布、Japan外利用、operator/machine/storage変更前にはrecordを`incomplete`へ戻す。source storage、H3を扱わないCPU CI、global source/package accessは`h3_relation: none`、`decision: not-applicable`とする。

H3 Works、Model Derivative、またはそれらを組み込んだ製品・サービスを第三者へ提供する前には、該当する次の項目を満たす。

- [x] 利用するH3ライセンス版のURL、取得日、SHA-256を記録した
- [x] 現行Japan-local H3-related flowのApplicable Territoryと実行・保存・Output利用地の整合性をowner申告で確認した
- [ ] 対象成果物がModel Derivativeに該当する前提でレビューした
- [x] 公開runtimeへMiniMax Materialsをコピーしていないかengineering inventoryを実施した
- [x] 現行の独立source、wheel、schema、CLIおよびmetadataをApache-2.0境界へ分類した
- [ ] `LICENSE`, `NOTICE`, `MODIFICATIONS.md`を同梱した
- [ ] 第三者依存物のライセンスを収集した
- [ ] Qwen3-VL由来部分のApache-2.0表示を確認した
- [ ] Hosted Serviceの場合、利用規約、AUP、報告窓口、安全策、違反対応手順を用意した
- [ ] 商用表示要件をUI/APIドキュメントに反映した
- [ ] 売上条件に該当する可能性を財務・法務担当が確認した
- [ ] 除外地域の利用者に対する個別許諾の有無を記録した
- [x] 現行local scopeのH3 access、GPU、H3 artifact storage、実行およびOutput利用をJapanとして確認した
- [ ] downstream利用者を拘束する条件と、その提示・同意記録方法を確認した

### 4.3 HF Gated Modelの位置付け

Hugging FaceのGated Modelはアクセス申請、追加フィールド、手動承認を提供するが、標準機能だけで法的所在地、実際のダウンロード地点、VPN、再配布を完全に保証するものではない。また、アクセスは組織単位ではなく個人ユーザーへ付与される。

したがって:

- **自動承認をMUST NOT使用する。**
- HFのGateだけを地域制限の唯一の技術的対策としてMUST NOT扱う。
- 申請フォームには法人名、居住・設立国、利用地域、用途、再配布有無、ライセンス同意を含める。
- 企業利用は個人アカウントの承認だけでなく、別途組織契約または承認記録へ結び付ける。
- MiniMaxの確認が得られない場合、HFにはメタデータと変換方法のみを置き、派生重みを置かない。

### 4.4 BYOWモード

BYOWモードでは、公開側は公式H3重みを再配布せず、次のみを提供する。

- 公式モデルIDとrevisionを指定する変換CLI
- 元ファイルが利用者のローカル環境に存在することを確認する処理
- ライセンス確認を促す明確な表示
- 変換された成果物のローカル保存
- 外部アップロードを標準で無効化

CLIはライセンス順守を保証するものではないが、少なくとも誤配布を助長しない設計にする。

追加要件:

- CLIはH3を利用できる地域・権利があると自動判定してはならない。
- 公式repositoryからの自動downloadは初期MVPに含めず、`--source-dir`で利用者が明示したlocal snapshotだけを入力とする。
- input snapshotのrepository ID、immutable commit、対象ファイルdigestを記録する。commitを確認できない入力は`unverified-source`としてfail closedすることを既定とする。
- 変換物は既定でlocal filesystemにのみ書き込み、publish/upload機能は別Phase・別コマンドとする。

---

## 5. リリース成果物

### 5.1 必須成果物一覧

| 成果物 | 配布先 | 重みを含む | 署名/来歴 | 必須 |
|---|---|---:|---:|---:|
| ソースコード | GitHub等 | いいえ | Git tag署名、release attestation | MUST |
| Python wheel/sdist | PyPI | いいえ | Trusted Publishing / PyPI attestation | Public Runtime releaseでMUST |
| OCI runtime image | GHCR等 | いいえ | Cosign、SBOM、provenance | 配布するPhaseでMUST |
| Model Card | Hugging Face等 | 原則いいえ | revision固定 | 公開成果物が生じた時点でMUST |
| 派生重み | Controlled Registry | はい | SHA-256、provenance、承認記録 | 条件付き |
| AdaLN cache | Model Registry | H3由来データを含み得る | モデルと同じ管理 | 条件付き |
| benchmark bundle | local/controlled storage。公開は権利review後 | いいえ | 結果JSONと環境manifest | Phase 0から内部保存MUST。公開はrelease gate後 |
| Helm chart | OCI/GitHub | いいえ | tag/release署名 | SHOULD |
| ComfyUI node | GitHub/Registry | いいえ | release署名 | MAY |

### 5.2 ソースモノレポと`uv workspace`方針

初期実装は現在の`src/h3fast`を使う**単一distribution**として管理する。module境界と公開APIが安定する前に、`core`、`kernels`、`runtime`、`server`を別distributionへ分割しない。Tritonがoptionalである間は遅延importとbackend dispatchでCPU import可能性を保つ。

次のいずれかが実測・運用上必要になった時点で、**単一Gitリポジトリ内の`uv workspace`**へ移行する。

- native extensionのbuild backendまたはwheel matrixを分離する必要がある
- kernel packageをH3以外から独立利用する
- serverとSDKのrelease cadenceまたは依存解決が分かれる
- license、所有チーム、公開範囲をdistribution単位で分ける必要がある

移行後は、コア実装、Tritonカーネル、公開SDK、サーバーを同時に変更・検証しながら、配布単位と依存境界を明確にする。

ただし、`uv workspace`は全メンバーで単一の`uv.lock`と依存解決を共有し、workspace全体の`requires-python`も各メンバーの積集合となる。したがって、互いに競合するPyTorch、CUDA、ROCm、SGLang、TritonまたはPython環境を一つのworkspaceへ押し込んではならない。

workspace移行後は次を目標構成とする。

```text
Git monorepo
├─ Main uv workspace
│  ├─ h3fast-core
│  ├─ h3fast-kernels
│  ├─ h3fast runtime/CLI
│  └─ h3fast-server
│
└─ Independent uv target projects
   ├─ NVIDIA Hopper production
   ├─ NVIDIA Blackwell production
   ├─ ROCm production
   ├─ PyTorch/Triton nightly experiments
   └─ paper-reproduction environments
```

workspace移行後の規則:

- メインworkspaceはMUST単一の`uv.lock`をリポジトリへcommitする。
- GPUターゲット別の独立projectは、それぞれ専用の`pyproject.toml`と`uv.lock`をMUST持つ。
- CUDA/ROCm/nightly等の競合環境はメインworkspaceのmemberへMUST NOT追加する。
- `.python-version`とCI・コンテナで使用する`uv`版をMUST pinする。
- ロック更新を伴うPull Requestでは、依存差分とGPU targetへの影響をMUST記載する。
- PyPI利用者へ`uv.lock`を強制してはならない。lockfileは開発、CI、コンテナ、公式benchmarkの再現性を担保する内部契約である。

### 5.3 リポジトリ構成

現在の単一distribution実装は次の構成を標準とする。存在しないkernel、target、test階層を先に作らない。

```text
h3fast/
├── pyproject.toml
├── uv.lock
├── .python-version
├── src/h3fast/
│   ├── cli.py
│   ├── exceptions.py
│   ├── manifest/
│   ├── backends/
│   │   └── sglang.py
│   ├── diagnostics/
│   ├── benchmarks/
│   ├── compliance/
│   └── release/
├── tests/                     # CPU unit/schema/package contract tests
├── benchmarks/
│   └── quality/
├── compliance/
│   ├── release-gates/
│   └── territories/
├── schemas/
├── runtime/
│   └── ffprobe.py             # pinned SGLang image用local adapter
├── docs/
└── .github/workflows/ci.yml
```

workspaceへ移行する場合の目標構成は次の通りとする。

```text
h3fast/
├── pyproject.toml                 # virtual workspace root
├── uv.lock                        # main workspace lock
├── .python-version
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── SECURITY.md
├── CONTRIBUTING.md
│
├── packages/
│   ├── h3fast-core/
│   │   ├── pyproject.toml
│   │   └── src/h3fast_core/
│   │       ├── config/
│   │       ├── model/
│   │       ├── conversion/
│   │       ├── quantization/
│   │       ├── sparse_attention/
│   │       ├── caching/
│   │       └── provenance/
│   │
│   ├── h3fast-kernels/
│   │   ├── pyproject.toml
│   │   └── src/h3fast_kernels/
│   │       ├── triton/
│   │       ├── reference/
│   │       ├── fallback/
│   │       ├── tuning/
│   │       └── dispatch.py
│   │
│   └── h3fast-runtime/
│       ├── pyproject.toml
│       └── src/h3fast/
│           ├── pipeline/
│           ├── backends/
│           ├── cli/
│           ├── diagnostics/
│           └── safety/
│
├── apps/
│   └── h3fast-server/
│       ├── pyproject.toml
│       └── src/h3fast_server/
│
├── integrations/
│   ├── sglang/
│   ├── diffusers/
│   └── comfyui/
│
├── targets/                       # main workspaceから除外
│   ├── nvidia-hopper/
│   │   ├── pyproject.toml
│   │   └── uv.lock
│   ├── nvidia-blackwell/
│   │   ├── pyproject.toml
│   │   └── uv.lock
│   ├── rocm/
│   │   ├── pyproject.toml
│   │   └── uv.lock
│   └── nightly/
│       ├── pyproject.toml
│       └── uv.lock
│
├── experiments/                  # workspaceから除外、原則非配布
├── benchmarks/
├── schemas/
├── tests/
│   ├── unit/
│   ├── kernel/
│   ├── generation/
│   ├── security/
│   └── performance/
├── docker/
├── deploy/helm/
├── docs/
└── scripts/
```

`integrations/`は、依存関係がメインworkspaceと整合するものだけをmember化する。特定SGLang版、ComfyUI環境、または別PyTorch系列を強制するintegrationは、独立projectまたは外部リポジトリへ分離する。

### 5.4 パッケージ境界

この節はworkspace移行後のdistribution境界を定める。移行前は同じ責務を`h3fast`内のmodule境界として適用する。

#### 5.4.1 `h3fast-core`

責務:

- H3設定、manifest、schema
- checkpoint変換とAdaLN事前計算
- Sparse maskおよびCache policyのアルゴリズム
- 量子化recipeと感度情報
- provenance、hash、互換性判定
- GPU kernelに依存しない参照ロジック

制約:

- SGLang、FastAPI、ComfyUIを依存へMUST NOT含める。
- CPU-only環境でimportおよび主要unit testが可能であること。
- Tritonのimportをmodule import時に必須としないこと。

#### 5.4.2 `h3fast-kernels`

責務:

- Triton kernel source
- PyTorch参照実装
- backend dispatchとcapability detection
- tuning profileとkernel cache key
- Dense/vendor fallback
- kernel correctness test utility

制約:

- Triton sourceのみを含む段階ではpure Python packageとして構築可能であること。
- CUDA/C++拡張を追加する場合、`uv_build`へ無理に載せず、`scikit-build-core`等のPEP 517 backendを用いる別distributionへの分離をSHOULD検討する。
- import時にGPU初期化、Triton compile、autotuneをMUST NOT実行する。

#### 5.4.3 `h3fast` (`h3fast-runtime` source project)

責務:

- 利用者向けPython API
- `h3fast convert/generate/verify/doctor/warmup/benchmark` CLI
- pipeline orchestration
- model/runtime compatibility check
- backend選択、fallback、diagnostics

公開distribution名は`h3fast`とし、内部workspace directory名は`h3fast-runtime`としてよい。

#### 5.4.4 `h3fast-server`

責務:

- `/v1/videos` API
- queue、job state、authentication、rate limit
- object storage、metrics、trace
- moderation、C2PA、Hosted Service固有機能

サーバーはSDKの依存を内包するが、SDKはサーバーへ依存してはならない。`h3fast-server`をPyPI公開するか、OCI専用applicationとするかはPhase 3開始前に決定する。

### 5.5 workspace root設定

workspace移行後のルートprojectは配布パッケージにせず、workspace、共通development dependencies、tooling policyだけを管理するvirtual projectとする。初期の単一distributionではルートproject自体が`h3fast`であるため、`package = false`を設定しない。

```toml
[project]
name = "h3fast-workspace"
version = "0.0.0"
requires-python = ">=3.11,<3.13"
dependencies = []

[tool.uv]
package = false

[tool.uv.workspace]
members = [
  "packages/*",
  "apps/*",
]
exclude = [
  "targets/*",
  "experiments/*",
  "vendor/*",
]

[dependency-groups]
dev = [
  "mypy",
  "pre-commit",
  "pytest>=8",
  "pytest-xdist>=3",
  "ruff",
]
benchmark = [
  "pandas",
  "pyarrow",
  "psutil",
]
docs = [
  "mkdocs-material",
  "mkdocstrings[python]",
]
```

workspace移行後の要件:

- workspace rootは`[tool.uv] package = false`をMUST設定する。
- rootへruntime依存を集約して、各packageの公開metadataを空洞化してはならない。
- 各memberは自身が直接importする依存を自身の`project.dependencies`へMUST宣言する。
- workspace内の別packageを偶然importできる状態に依存してはならない。uv workspaceはPython import isolationを保証しないため、package単独testをCIで実施する。

### 5.6 workspace member間依存

この節はworkspace移行後にのみ適用する。

workspace memberへの依存は、標準metadataの依存名と`tool.uv.sources`の`workspace = true`を併記する。

```toml
[project]
name = "h3fast"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "h3fast-core==0.1.0",
  "h3fast-kernels==0.1.0",
  "safetensors",
  "huggingface-hub",
  "torch",
]

[project.scripts]
h3fast = "h3fast.cli:main"

[tool.uv.sources]
h3fast-core = { workspace = true }
h3fast-kernels = { workspace = true }

[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"
```

- workspace member依存はMUST明示する。
- 初期リリースでは`h3fast-core`、`h3fast-kernels`、`h3fast`を同一release versionへ揃えることをSHOULD推奨する。
- version同期はuvが自動実行するものではないため、release scriptで全member、manifest、container labelを一括検証・更新する。
- 公開wheelを検証するときはworkspace sourceを使わず、clean environmentでPyPI相当の成果物だけからinstall testを行う。

### 5.7 GPUターゲット別独立project

PyTorchはCPU、CUDA、ROCm等で異なるwheelとindexを使用し、Triton、SGLang、FlashAttention系にも厳しい互換性がある。以下を一つのworkspace lockで同時解決しようとしてはならない。

- CPU-only開発環境
- NVIDIA Hopper向け安定版
- NVIDIA Blackwell向け安定版
- ROCm向け安定版
- PyTorch/Triton nightly
- 古い論文再現環境

`targets/<target>/pyproject.toml`は、初期構成ではrepository root、workspace移行後は該当packagesをpath dependencyとして参照し、ターゲット固有のPyTorch indexと厳密なversionを管理する。実機検証していないtarget directoryやlockfileを先回りして作らない。

```toml
[project]
name = "h3fast-target-nvidia-hopper"
version = "0.0.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "h3fast",
  "h3fast-core",
  "h3fast-kernels",
  "torch==<tested-version>",
  "triton==<tested-version>",
  "sglang==<tested-version>",
]

[tool.uv]
package = false
environments = [
  "sys_platform == 'linux' and platform_machine == 'x86_64'",
]

[tool.uv.sources]
h3fast = { path = "../../packages/h3fast-runtime", editable = true }
h3fast-core = { path = "../../packages/h3fast-core", editable = true }
h3fast-kernels = { path = "../../packages/h3fast-kernels", editable = true }
torch = { index = "pytorch-target" }

[[tool.uv.index]]
name = "pytorch-target"
url = "https://download.pytorch.org/whl/<tested-backend>"
explicit = true
```

要件:

- `<tested-version>`と`<tested-backend>`はreleaseごとに実測済み値へ置換する。
- PyTorch用indexは`explicit = true`とし、意図しないpackageが代替indexから解決されるのを防ぐ。
- `torch-backend=auto`等のpreview/自動選択機能を公式production lock生成にMUST NOT使用する。開発者用補助としてはMAY使用する。
- target lockのhashをOCI label、benchmark environment、provenanceへMUST記録する。
- production imageは対象targetの`uv.lock`を正とし、メインworkspace lockだけでGPU環境の再現性を主張してはならない。

限定されたCPU/CUDA切替だけで、同じ依存グラフを保てる場合はoptional dependencyと`tool.uv.conflicts`を使う方式もMAY採用する。ただしH3Fastのproduction GPU matrixでは、独立target projectを既定とする。

### 5.8 build backend方針

- v0.11の単一`h3fast` distributionはHatchling `>=1.27,<2`を使用し、wheel/sdistとclean installを検証する。backend変更だけを目的に移行しない。
- workspace分割後のpure Python `h3fast-core`、`h3fast-kernels`、`h3fast`には`uv_build`をSHOULD検討する。
- `uv_build`のversionには互換上限をMUST設ける。
- CUDA/C++/Rust extensionを含むpackageには、extension対応PEP 517 backendをMUST使用する。
- FlashAttention、DeepGEMM等、installed PyTorchと同じABIへbuildする必要がある依存はtarget projectで管理する。
- build isolationをworkspace全体で無効化してはならない。必要なpackageだけに限定し、可能ならuvの追加build dependency機能を用いて、build時とruntimeのPyTorch versionを一致させる。
- source buildが必要な依存は、production release前にwheel化するか、署名済みbuilder imageで再現可能にする。

### 5.9 開発コマンド

初期の単一distributionでの標準操作は次の通りとする。

```bash
uv sync --locked --all-groups
uv run h3fast --help
# `blocked` recordでは終了code 1が正常。release workflowだけがcode 0を要求する。
uv run h3fast release check --record compliance/release-gates/initial-runtime.json
# `incomplete` recordでは終了code 1が正常。これはH3-related flow用でありcode-only releaseの前提ではない。
uv run h3fast compliance check-territories --record compliance/territories/initial-runtime.json
# `incomplete` recordでは終了code 1が正常。case/rights/metric approval前にcode 0を要求しない。
uv run h3fast benchmark check-quality-set --record benchmarks/quality/formal-quality-set.json
uv run pytest --cov=h3fast
uv run ruff format --check .
uv run ruff check .
uv run ty check src/
uv build --no-sources
```

workspace移行後の操作は次の通りとする。

```bash
# main workspaceを同期
uv sync --all-groups

# lockfileを更新せず同期
uv sync --locked --all-groups

# package単位でcommandを実行
uv run --package h3fast-core pytest packages/h3fast-core
uv run --package h3fast-kernels pytest packages/h3fast-kernels
uv run --package h3fast h3fast --help
uv run --package h3fast-server h3fast-server --help

# 公開packageを個別build
uv build --package h3fast-core
uv build --package h3fast-kernels
uv build --package h3fast

# ターゲット環境
cd targets/nvidia-hopper
uv sync --locked
uv run h3fast doctor
```

- 日常コマンドは`uv run`を標準とし、shellごとの手動virtualenv activateを必須にしない。
- `uv add`による依存変更後は、main lockおよび影響するtarget lockを更新する。
- benchmark、schema generation、release helperのような再現性が必要なscriptは、workspace dependency groupまたはPEP 723 script metadataを使用する。

### 5.10 Dockerでのworkspace利用

Docker buildでは依存layerとsource layerを分離する。workspaceの場合、最初のdependency syncではmember sourceがまだ存在しないため、公式推奨に従い`--frozen --no-install-workspace`を使用し、全source copy後に`--locked`で整合性を検証する。

概念例:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM <gpu-base-image-by-digest> AS builder

COPY --from=ghcr.io/astral-sh/uv:<pinned-version> /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# dependency layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev

COPY . /app

# lockfileと全member metadataを検証し、非editableでinstall
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --package h3fast-server
```

production GPU imageでは、上記のmain workspace lockではなく、該当する`targets/<target>/uv.lock`を使用するDockerfileまたはbuild contextを標準とする。

追加要件:

- uv image/binaryはversionまたはdigestでMUST pinする。
- production環境にuv自体を残す必要がない場合は、builderから`.venv`だけをruntime stageへcopyする。
- uv cacheとTriton runtime cacheを混同しない。
- build時のuv cache、Triton JIT cache、model cacheをOCI layerへ無制限に残してはならない。
- production installは`--no-editable`をMUST使用する。

### 5.11 CI/CDとuv

v0.11は`.github/workflows/ci.yml`でMain project CPU CIとClean-wheel CIを実装済みである。Target GPU CIとRelease CIは未実装であり、対象targetまたは公開releaseを追加する変更で導入する。

完成時のCIは次の層に分ける。

1. **Main workspace CPU CI**
   - `uv sync --locked`
   - package単位のunit/type/import test
   - workspace全体のlint/schema/license test
   - 各公開packageの`uv build --package`
2. **Target GPU CI**
   - `targets/<target>/uv.lock`から同期
   - GPU capability、kernel correctness、E2E smoke、performance regression
3. **Clean-wheel CI**
   - workspace sourceを使わず、build済みwheelのみを新規環境へinstall
   - 公開metadataと依存宣言の漏れを検出
4. **Release CI**
   - uv version、Python version、main/target lock digestをattestationへ記録
   - dependency orderに従ってpackageをpublish

GitHub Actionsでは`astral-sh/setup-uv`をcommit SHAとuv versionの双方でpinする。cache keyには少なくともOS、architecture、Python、uv version、`uv.lock` hashを含める。GPU CIのcache keyにはtarget lock、driver/CUDA、GPU architectureを追加する。

### 5.12 package公開とversioning

初期のPyPI distributionは`h3fast`一つとする。workspace分割条件を満たした後は次を想定する。

| Distribution | Import | 公開 | 備考 |
|---|---|---:|---|
| `h3fast-core` | `h3fast_core` | 条件付き | 純粋な共通ロジック |
| `h3fast-kernels` | `h3fast_kernels` | 条件付き | Triton sourceとfallback |
| `h3fast` | `h3fast` | MUST | SDK/CLI |
| `h3fast-server` | `h3fast_server` | MAY | OCI専用でもよい |

- workspace分割直後は同期versionをSHOULD使用し、互換性の理解が進んだ後に独立versionへ移行してよい。
- 公開順序は`core` → `kernels` → `h3fast` → `server`とする。
- 各wheelの`Requires-Dist`がworkspace外でも成立することをMUST確認する。
- `tool.uv.sources`は開発時のsource overrideであり、公開wheelの依存契約の代わりではない。
- release tag、全package version、runtime manifest、OCI labelはMUST一致する。

### 5.13 パッケージ分割の追加条件

初期段階で次を個別packageへ分けてはならない。

```text
h3fast-cache
h3fast-quantization
h3fast-rope
h3fast-adaln
h3fast-manifest
h3fast-converter
```

新しいpackageを作るのは、少なくとも一つを満たす場合とする。

- 独立して公開・versioningする必要がある
- 依存関係またはbuild backendが大きく異なる
- ライセンス境界を分離する必要がある
- 別チームまたは別release cadenceが所有する
- H3以外のモデルから独立利用される
- native extensionやvendor SDKが必要である

ファイル数やmodule数が増えただけではpackage分割理由としない。
---

## 6. モデルパッケージ仕様

### 6.1 リポジトリ分割

異なる用途・重み・量子化方式は別リポジトリとする。

推奨例:

```text
org/H3-Fast-FL2VA-BF16
org/H3-Fast-FL2VA-FP8
org/H3-Fast-FL2VA-8Eval-FP8
org/H3-Fast-Ref2VA-BF16
org/H3-Fast-Ref2VA-FP8
org/H3-Fast-Ref2VA-8Eval-FP8
```

一つの巨大repoに全task family、全precision、全Studentを混在させない。利用者が必要な成果物だけを取得できることを優先する。

### 6.2 必須ファイル

```text
model-repo/
├── README.md
├── LICENSE
├── NOTICE
├── MODIFICATIONS.md
├── model_index.json
├── h3fast_manifest.json
├── optimization_config.json
├── compatibility.json
├── checksums.sha256
├── provenance.intoto.jsonl
├── benchmarks/
│   ├── benchmark_results.json
│   ├── benchmark_environment.json
│   └── quality_report.md
├── transformer/
│   ├── config.json
│   ├── model-00001-of-000NN.safetensors
│   └── model.safetensors.index.json
├── text_encoder/
├── visual_vae/
├── audio_vae/
└── optional_artifacts/
    ├── adaln_cache/
    ├── adapters/
    └── tuning_profiles/
```

### 6.3 シリアライズ

- 重みは**SafetensorsをMUST使用**する。
- Pickle、`.pt`、`.pth`、実行可能なPython objectを重み配布形式としてMUST NOT使用する。
- モデルrepoに任意コードを入れて`trust_remote_code=True`を必須にする設計をSHOULD NOT採用する。
- 実行コードは署名済みPython packageへ分離する。
- 重みは再開可能なダウンロードに適したshardへ分割する。標準shard上限は5 GiBを推奨する。

### 6.4 `h3fast_manifest.json`

必須フィールド例:

```json
{
  "schema_version": "1.0",
  "artifact_id": "h3fast-h3-base-fl2va-8eval-fp8-v1.0.0",
  "artifact_type": "model_derivative",
  "base_model": "MiniMaxAI/MiniMax-H3",
  "base_revision": "<immutable-hf-commit-sha>",
  "task_family": "fl2va",
  "runtime": {
    "name": "h3fast",
    "requires": ">=1.0.0,<1.2.0",
    "tested_versions": ["1.0.0", "1.1.0"]
  },
  "components": [
    {
      "name": "transformer",
      "format": "safetensors",
      "dtype": "float8_e4m3fn",
      "index": "transformer/model.safetensors.index.json"
    }
  ],
  "license": {
    "name": "MiniMax H3 Community License Agreement",
    "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
    "captured_at": "2026-08-15T00:00:00+09:00",
    "sha256": "<license-file-digest>"
  },
  "build": {
    "source_revision": "<h3fast-git-sha>",
    "recipe_digest": "sha256:<digest>",
    "provenance_file": "provenance.intoto.jsonl"
  }
}
```

`base_revision`へ`main`や可変tagを指定してはならない。

directoryまたはlogical componentへ単一の`sha256`を付けてはならない。各ファイルの相対path、byte size、SHA-256は`checksums.sha256`またはmanifest内のfile inventoryへ記録し、shard index自体も検証対象に含める。`runtime.requires`はPEP 440互換specifierとして解釈し、`tested_versions`は実際にE2E検証したversionだけを列挙する。

### 6.5 `optimization_config.json`

```json
{
  "schema_version": "1.0",
  "profile": "balanced",
  "denoising": {
    "sigma_points": 9,
    "model_evaluations": 8,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0
  },
  "adaln": {
    "precomputed": true,
    "cache_artifact": "optional_artifacts/adaln_cache/cache.safetensors",
    "schedule_bound": true
  },
  "attention": {
    "backend": "h3fast_block_sparse_v1",
    "fallback": "sdpa",
    "mass_threshold": 0.995
  },
  "feature_cache": {
    "enabled": true,
    "algorithm": "adaptive_residual_v1"
  },
  "precision": {
    "weights": "fp8",
    "activations": "bf16",
    "vae": "bf16",
    "sensitive_layers": "bf16"
  }
}
```

AdaLNキャッシュはmode、step count、flow shift、audio flow shift、条件ノイズ等へ結び付く。異なるスケジュールの要求へキャッシュを流用してはならない。SGLangの現行H3実装もスケジュール不一致を拒否する設計であり、本ランタイムも同様にfail closedとする。

### 6.6 `compatibility.json`

```json
{
  "schema_version": "1.0",
  "tested": [
    {
      "accelerator": {
        "vendor": "nvidia",
        "model": "H100 SXM",
        "architecture": "sm90",
        "memory_gib": 80,
        "count": 4
      },
      "software": {
        "runtime": "1.0.0",
        "pytorch": "<tested-version>",
        "triton": "<tested-version>",
        "cuda": "<tested-version>"
      },
      "driver": {
        "minimum": "<minimum-version>",
        "tested": "<tested-version>"
      },
      "status": "tier-1"
    }
  ],
  "unsupported": [],
  "fallback_rules": []
}
```

「動くはず」ではなく、実機でE2E生成を完了した組み合わせのみを`tested`へ記載する。
driver要件は曖昧な`<exact>`ではなく、比較方法をschemaで定義した`driver_min`と実測した`driver_tested`を分ける。GPU名だけでなくvendor、architecture、VRAM、GPU countを機械可読な別fieldにする。

---

## 7. Python SDK / CLI仕様

### 7.1 パッケージ

現在検証済みなのはsource checkoutとbuild済みlocal wheelである。

```bash
uv sync --locked --all-groups
uv run h3fast --help
uv build --no-sources
```

Public Runtime release後は`uv add h3fast`または`pip install h3fast`を想定するが、v0.11時点ではPyPI公開済みコマンドではない。公開前に別名packageを誤って取得しないよう、現行setup手順として提示してはならない。

将来の開発・サーバー用途では、検証済みtarget projectまたは公式OCI imageを優先する。GPUスタックを任意のextraだけで完全に再現できると表現してはならない。次はtarget作成後に検証する構想例であり、現在は実行できない。

```bash
cd targets/nvidia-hopper
uv sync --locked
uv run h3fast doctor
```

- `h3fast`本体はCPU環境でもimport可能であること。
- GPU依存物はoptional dependencyまたはtarget projectへ分離する。
- PyTorch、Triton、SGLangの範囲を無制限に許可せず、検証済み範囲を指定する。
- main workspace lockとGPU target lockを混同しない。
- lockfile、Python/uv version、完全なテスト環境manifestを提供する。

### 7.2 変換CLI

この節はInitial Runtime releaseへ向けた将来契約である。v0.11に`convert` commandは存在せず、次の例は実行不可である。

```bash
h3fast convert \
  --base-model MiniMaxAI/MiniMax-H3 \
  --base-revision <immutable-sha> \
  --variant fl2va \
  --source-dir /models/MiniMax-H3 \
  --output-dir /models/H3-Fast-FL2VA-FP8 \
  --precision fp8 \
  --precompute-adaln \
  --schedule configs/50step.json \
  --emit-provenance
```

CLI要件:

- 入力ファイルのhashを計算する。
- 元revisionと変換recipeをmanifestへ記録する。
- 出力をSafetensorsで保存する。
- 途中失敗時に不完全成果物を完成品として扱わない。
- 変換結果を外部へ自動uploadしない。
- `--publish`は別コマンドとし、明示的な承認を要求する。
- ライセンス同意を装うチェックボックスだけで配布可能と判断しない。

### 7.3 検証CLI

現在実装済みの検証commandは次である。

```bash
h3fast inspect-snapshot /models/MiniMax-H3 \
  --variant fl2va \
  --base-revision <immutable-sha>
h3fast verify-model /models/H3-Fast-FL2VA-FP8
h3fast doctor
```

`verify-image`とmodel-aware `doctor`は将来契約であり、v0.11では実行できない。

```bash
h3fast verify-image ghcr.io/org/h3fast@sha256:<digest>
h3fast doctor --model /models/H3-Fast-FL2VA-FP8
```

検証内容:

- 全ファイルhash
- manifest schema
- base revision
- runtime互換性
- AdaLN schedule
- accelerator capability
- Triton backend可否
- fallback可否
- LICENSE/NOTICE存在
- 署名・provenance

---

## 8. Tritonカーネル配布仕様

### 8.1 基本方針

TritonはH3全体の再実装ではなく、次のH3固有・非標準パターンへ使用する。

優先対象:

1. Block Sparse Attention
2. QK-Norm + 3D MM-RoPE + layout変換 + quantize融合
3. AdaLN + Norm + gate + residual融合
4. Cache判定とreduction
5. Token packing / gather / scatter
6. 小さなelementwise chainの融合

大型Dense GEMM、標準Dense Attention、一般的な3D Convは、測定で優位性が確認されない限りcuBLAS/CUTLASS、Flash/Sage系、cuDNNへ任せる。

### 8.2 配布形式

- Triton kernel sourceをPython wheelへ含める。
- 初回JITを標準経路とする。
- 検証済みshape向けのtuning profileを同梱する。
- AOTバイナリは特定環境向けの任意追加物とし、唯一の実行方法にしない。
- kernelはPyTorch custom operatorとして登録し、`torch.compile`、FakeTensor、export等との統合を阻害しない設計を推奨する。

### 8.3 JITキャッシュ

標準キャッシュキーに次を含める。

```text
kernel source digest
h3fast runtime version
Triton version
PyTorch version
GPU vendor and architecture
CUDA/ROCm runtime
compile flags
shape bucket
precision profile
```

キャッシュパス例:

```text
/var/cache/h3fast/triton/<cache-key>/
```

異なるGPU世代やTriton版のcacheを互換と仮定してはならない。

### 8.4 Warmup

```bash
h3fast warmup \
  --model /models/H3-Fast-FL2VA-FP8 \
  --profiles 768p-5s-16x9,768p-10s-9x16 \
  --backend triton
```

- 本番request pathで大規模autotuneをMUST NOT実行する。
- 未知shapeは安全な既存configまたはDense fallbackへ落とす。
- `triton.autotune`は開発・prewarm jobで使用し、選択結果をtuning manifestへ保存する。

### 8.5 フォールバック

```text
H3Fast Triton sparse kernel
  ↓ unsupported shape/arch/error
Vendor-optimized sparse/attention backend
  ↓ unsupported
PyTorch SDPA / dense reference
```

- fallbackは黙って品質profileを変えてはならない。
- response metadataとmetricsへ実際のbackendを記録する。
- `exact` profileで近似Sparseへfallbackしてはならない。
- fallback不能の場合は明確なエラーを返す。

### 8.6 カーネル正当性

各kernelについて次をテストする。

- 参照PyTorch実装との数値差
- BF16、FP16、FP8等の対応dtype
- 可変長packed sequence
- zero-lengthまたは境界条件
- 代表shapeと端数shape
- NaN/Inf伝播
- 不正indexへの防御
- non-contiguous tensorの扱い
- GPU世代別結果
- Dense fallbackとの生成回帰

---

## 9. OCIコンテナ仕様

### 9.1 イメージ原則

- 重みをMUST NOT含める。
- base imageはdigestでpinする。
- 主要GPU世代・vendorごとに別の検証済みimageを提供する。
- `latest` tagを本番例でMUST NOT使用する。
- 本番デプロイ例はOCI digestを使用する。
- root以外のユーザーで実行する。
- 可能な範囲でread-only root filesystemを使用する。
- compilerやdebug toolをproduction imageから除く。
- build imageとruntime imageをmulti-stageで分離する。

タグ例:

```text
ghcr.io/org/h3fast:1.0.0-nvidia-hopper
ghcr.io/org/h3fast:1.0.0-nvidia-blackwell-dc
ghcr.io/org/h3fast:1.0.0-nvidia-blackwell-geforce
ghcr.io/org/h3fast:1.0.0-rocm-cdna
```

実際の対応は`compatibility.json`を正とする。

### 9.2 起動例

次はOCI distribution実装後の構想例である。v0.11ではH3Fast OCI imageと`h3fast serve`は未実装・未検証であり、digestのplaceholderを実行値として使用してはならない。

```bash
docker run --rm --gpus all \
  --ipc=host \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=8g \
  -v /srv/h3-models:/models:ro \
  -v /srv/h3-cache:/var/cache/h3fast:rw \
  -v /srv/h3-outputs:/outputs:rw \
  -p 8000:8000 \
  ghcr.io/org/h3fast@sha256:<image-digest> \
  h3fast serve --model /models/H3-Fast-FL2VA-FP8
```

### 9.3 署名とSBOM

各imageについて:

- Cosignによる署名をMUST提供する。
- build provenanceをMUST提供する。
- SPDXまたはCycloneDX SBOMをMUST提供する。
- source revision、base image digest、wheel digestをattestationへ含める。
- 重大な脆弱性が見つかったimageはrevocation/denylistへ追加する。

### 9.4 Kubernetes

推奨構成:

```text
initContainer
  └─ entitlement確認、重みの取得、hash検証

Persistent Volume
  ├─ model weights (read-only to server)
  └─ Triton/compile cache

Main Container
  └─ h3fast / SGLang server

Sidecar or platform service
  ├─ metrics/log forwarding
  └─ optional C2PA signing
```

- imageはtagでなくdigestで指定する。
- HF token、object-store token、C2PA signing keyを通常の環境変数へ平文で埋め込まない。
- Kubernetes Secretは最小権限とし、可能なら外部Secret Store/KMSを用いる。
- admission policyで署名・attestationを検証することを推奨する。

---

## 10. モデル取得と権限管理

### 10.1 BYOW

次はconverter実装後に利用者が明示的に取得・変換する構想例である。v0.11はmodelを自動downloadせず、`h3fast convert`も未実装である。H3の取得前に利用権とApplicable Territoryを確認する。

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision <immutable-sha> \
  --include 'model_index.json' 'FL2VA/*' \
  --local-dir /models/MiniMax-H3

h3fast convert \
  --source-dir /models/MiniMax-H3 \
  --variant fl2va \
  --output-dir /models/H3-Fast-FL2VA-FP8 \
  --precision fp8
```

ダウンロード対象を明示し、不要なFL2VA/Ref2VA/Diffusersコピーをまとめて取得しない。

### 10.2 Controlled Weights

派生重みを配布する場合、推奨フローは次の通り。

```text
申請
  ↓
本人/組織・地域・用途確認
  ↓
ライセンス/AUP/再配布条件への同意
  ↓
手動またはポリシー承認
  ↓
短時間の署名付きURLまたは個別token
  ↓
取得ログ・artifact digest記録
  ↓
アクセス取消と更新通知が可能な状態
```

- IP geolocationだけで法的所在地を判定しない。
- 承認後の永久公開URLを発行しない。
- object storageのURLは短時間で失効させる。
- ダウンロードされた重みへ個人情報を埋め込むwatermarkは、プライバシーと安全性を検討せず実施しない。
- アクセス取消は将来の取得を止めるものであり、既取得コピーを技術的に消去できると表現しない。

---

## 11. Serving API仕様

### 11.1 互換性方針

固定したSGLang版のH3実装が採用している非同期`/v1/videos` endpointを基本互換面とする。「OpenAI-compatible」という表示だけに依存せず、request/response schemaとHTTP挙動をcontract testで固定する。H3Fast固有fieldは`h3fast` namespace内に置き、未知fieldをSGLangへ透過するか拒否するかを明示する。

Phase 3のtarget endpoint surface:

```text
POST /v1/videos
GET  /v1/videos/{id}
GET  /v1/videos/{id}/content
GET  /v1/models/{id}/capabilities
GET  /healthz
GET  /readyz
GET  /metrics
```

Phase 1ではSGLangが提供する先頭3 endpointを利用し、`capabilities`、health、metricsはadapterまたはsidecarで追加してよい。独自job queueとcontent storageの再実装はPhase 3まで必須としない。

### 11.2 リクエスト例

```json
{
  "model": "org/H3-Fast-FL2VA-8Eval-FP8",
  "prompt": "A quiet street at dawn with natural synchronized ambience.",
  "seconds": 5,
  "task": "t2va",
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0
  },
  "num_outputs_per_prompt": 1,
  "num_inference_steps": 9,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 1101,
  "h3fast": {
    "profile": "balanced",
    "require_backend": null,
    "allow_fallback": true,
    "emit_content_credentials": true
  }
}
```

互換性のため`seconds`と`target.duration_seconds`を併記する場合、両者は同値でなければならず、不一致requestはHTTP 422で拒否する。内部のcanonical fieldは`target.duration_seconds`とする。`num_inference_steps`はsigma grid point数で、H3のdenoiser model evaluation数は通常その1少ないため、API metadataには両方を明示する。例の9は8-eval adapterを前提としており、対応adapterがないbase modelへ暗黙適用してはならない。

### 11.3 再現性メタデータ

job statusまたは完成responseに次を含める。

```json
{
  "id": "vid_...",
  "status": "completed",
  "reproducibility": {
    "model_artifact_id": "h3fast-h3-base-fl2va-8eval-fp8-v1.0.0",
    "model_revision": "<immutable-sha>",
    "runtime_version": "1.0.0",
    "image_digest": "sha256:<digest>",
    "optimization_config_digest": "sha256:<digest>",
    "kernel_backend": "h3fast_block_sparse_v1",
    "fallbacks_used": [],
    "seed": 1101
  },
  "content_credentials": {
    "embedded": true,
    "manifest_id": "<id>"
  }
}
```

### 11.4 Profile

| Profile | 許可する最適化 | 意味 |
|---|---|---|
| `exact` | 配置、通信、kernel fusion、graph capture。アルゴリズム近似なし | 参照計算と同じ数学的処理を意図 |
| `balanced` | FP8、保守的Sparse、適応Cache | 品質予算内で高速化 |
| `fast` | Few-step Student/LoRA、強いCache/Sparse | 最大速度。別model IDを推奨 |

同じmodel IDのままサーバー側判断で50-stepから4-stepへ変更してはならない。

`exact`という名称は計算グラフ上の意味を示し、bitwise一致を保証しない。数値許容差と生成品質gateが確定するまで、API文書では`reference` profileという名称を使用してもよい。FP8、Sparse、Cache、Few-step adapterを利用するprofileは、実際に有効化した最適化とfallbackをresponse metadataへ列挙する。

---

## 12. 入力メディア・セキュリティ

動画生成APIではモデル自体より、URI取得、FFmpeg、画像デコーダー、巨大入力、SSRFが攻撃面になりやすい。

### 12.1 URI

- Public APIで任意の`file://` URIをMUST NOT許可する。
- `http(s)`取得は許可ドメインまたは署名付きobject storageへ制限することを推奨する。
- loopback、link-local、metadata endpoint、private networkへのSSRFを遮断する。
- redirect後の宛先も再検証する。
- Content-Length、実ダウンロード量、timeoutを制限する。

### 12.2 メディア処理

- 入力ファイルをMIME文字列だけで信頼しない。
- codec、duration、resolution、sample rate、channel数、frame数に上限を設ける。
- FFmpeg/decoderはpatch済みversionへpinし、隔離されたprocessで実行する。
- zip bomb、decompression bomb、極端な可変フレームレートを拒否する。
- 一時ファイルはjob終了後に削除する。

### 12.3 ログとプライバシー

- prompt、画像、動画、音声本文を標準でログへ残さない。
- request ID、hash、サイズ、処理時間、policy resultを記録する。
- デバッグ目的のコンテンツ保存は明示的opt-in、期限、アクセス制御を必要とする。
- 顧客入力をbenchmarkや学習へ無断転用してはならない。

### 12.4 安全対策

Hosted Serviceは少なくとも次を持つ。

- 入力・出力moderation
- なりすまし、未成年、性的コンテンツ、権利侵害等のpolicy
- rate limitとabuse detection
- 利用者の同意と利用規約
- 報告フォーム
- repeat violatorの停止
- インシデント調査と証跡

---

## 13. 生成物の来歴とAI表示

H3 LicenseのAUPは、Outputをpublic environmentへ配布する場合にmachine-generatedであることの明確かつ目立つ開示を要求している。したがって公開機能ではAI生成表示をMUSTとする。C2PA Content Credentialsはその表示を補強する推奨手段だが、C2PAだけで利用者が実際に表示を見ることやlicense順守を保証したと扱ってはならない。

### 13.1 最低要件

生成物またはsidecar manifestへ次を記録する。

- AI生成であること
- H3 Fastのmodel artifact ID
- H3由来であること
- 生成日時
- サービスまたは署名主体
- 編集・再エンコード履歴が分かる範囲

公開サービスのUI、download画面、share pageおよびAPI documentationには、少なくともAI生成であることとMiniMax H3由来であることを人が認識できる形で表示する。埋め込みmetadataが除去された場合にも必要な表示が残る運用経路を用意する。

### 13.2 C2PA

- C2PA 2.4等の現行仕様を参照する。
- `digitalSourceType`でAI/ML生成を表現する。
- 本番署名鍵はfilesystemへ直接置かずKMS/HSMを使用する。
- C2PA metadataは除去され得るため、絶対的なDRMまたは真偽判定とは説明しない。
- 必要に応じてsoft bindingやmanifest repositoryを検討する。

---

## 14. 品質・性能ベンチマーク

### 14.1 ベンチマーク原則

- 最大速度だけを掲載してはならない。
- baseline、各最適化単独、組合せを段階別に掲載する。
- cold startとwarm startを分ける。
- E2E時間とdenoiser時間を分ける。
- GPU、driver、CUDA/ROCm、PyTorch、Triton、runtime、model revisionを記録する。
- prompt/seed/input条件を固定し、再現可能にする。

### 14.2 必須比較

```text
BF16 50-step reference
+ lossless kernel fusion / graph capture
+ FP8
+ conservative sparse attention
+ adaptive cross-step cache
8-eval BF16
8-eval FP8
8-eval FP8 + sparse + cache
```

上記は最適化が揃った段階の最終比較matrixであり、Phase 0/1Aで未実装の行を要求しない。各Pull Requestでは「固定BF16 reference」と「そのPull Requestが変更する最適化」のA/B比較をMUST実施し、複数最適化を同時に有効化した結果だけを提示してはならない。

### 14.3 性能指標

- E2E latency
- Time to first accepted job / queue delay
- Denoising latency per evaluation
- VAE encode/decode時間
- text/reference encoder時間
- Peak GPU memory
- Host memory
- GPU utilization
- 通信時間
- cold compile時間
- warmup後時間
- jobs/hour
- energy/job（取得可能なら）

### 14.4 品質指標

- Prompt adherence
- VBench系の動画品質
- LPIPS / DINO / SSIM / PSNR（用途に応じる）
- optical-flow整合性、flicker
- 顔・人物identity
- OCR・小文字
- 参照画像/動画保持
- audio quality（FAD、CLAP等）
- A/V sync、lip sync、event sync
- stereo consistency
- 人手pairwise評価

### 14.5 テストセット

正式な品質主張とPhase 2以降では最低200ケースを推奨し、次を含める。Phase 0では公開可能な10件以上のsmoke setと、代表条件を層化した50件以上のregression setから開始してよい。ただし件数、選定方法、除外、失敗例を結果と共に公開する。

Phase 0 formal setのmachine-readable契約は[`benchmarks/quality/formal-quality-set.json`](../benchmarks/quality/formal-quality-set.json)と[`schemas/formal-quality-set.schema.json`](../schemas/formal-quality-set.schema.json)で固定する。prompt本文、reference media、生成物、local pathはこのrecordへ含めない。case registryのURI/digest、caseごとのprompt/reference digest、rights evidence、selection method、exclusions、known failures、metric plan、rights reviewer、quality ownerが揃い、次のcoverageをaggregateで満たした場合だけ`approved`にできる。

prompt本文とreference assetのlocal pathを含む入力は[`schemas/private-quality-registry.schema.json`](../schemas/private-quality-registry.schema.json)に従うprivate registryとしてrepository外で管理する。registry全体のdigest、件数、selection methodだけを公開する場合は[`schemas/quality-registry-attestation.schema.json`](../schemas/quality-registry-attestation.schema.json)に従い、per-case digestも含めない。完了したreviewのaggregate evidenceは[`schemas/quality-registry-review-attestation.schema.json`](../schemas/quality-registry-review-attestation.schema.json)に従い、source registry/content digest、reviewer、role、selection判断、集約件数だけを公開する。`h3fast benchmark compile-quality-registry`はregistry file自体、prompt UTF-8 bytes、reference asset bytesのSHA-256を計算し、formal record candidateへdigestと公開可能metadataだけを出力する。次をMUSTとする。

- private registry、reference asset、prompt本文またはlocal pathをGit、CI artifact、logへ追加しない
- registry URIは権限制御されたregistryまたはredacted digest attestationのimmutable HTTPS identityとし、local pathや一時URLを公開recordへ記録しない
- T2VAはreferenceなし、FL2VA/Ref2VAは1件以上の実在assetを要求し、複数modalityは`mixed` coverageとして記録する
- per-case rights status/evidenceをcompilerが推測せず、approvedの場合はHTTPS evidenceを要求する
- registry digestとcaseが変わるたびにset-levelのrights reviewer/quality owner承認を`unassigned`へ戻し、旧承認を引き継がない
- outputがregistry/template inputを上書きしないよう拒否し、一時fileのformal semantic validation成功後にだけatomic replaceする
- compiler成功はcase、rights、metricまたはset approvalを意味しない。formal validatorが終了code 0になるまでcandidateを正式recordとして採用しない
- rights/quality approval前はprivate registry、reference assetに加えてper-case digestを持つcandidate formal recordもcommit/uploadせず、registry全体のredacted attestationだけを公開できる

`h3fast benchmark prepare-quality-review`は、registry全体とreview対象contentを別々のSHA-256へ固定し、caseごとのprompt/reference digest、rights/selection decision、evidence欄を持つlocal-only checklistを生成する。review instanceは[`schemas/private-quality-review.schema.json`](../schemas/private-quality-review.schema.json)に従い、private registryと同様にGit、CI artifact、logへ追加してはならない。reviewerは元registryのpromptとreference assetをローカルで確認し、各caseのrights/selection decision、selection method・exclusions・known failuresの判断、HTTPS evidence、review時刻を明示する。

`h3fast benchmark apply-quality-review`は、source registry digestとreview対象content digestが一致し、全selection判断と全caseのrights/selection判断が`approved`で、各caseにHTTPS rights evidenceがある場合だけ、承認済み状態を持つ新しいprivate registryをatomicに出力する。`pending`または`rejected`が一つでもある場合は終了code 1とし、reviewed registryを生成しない。command生成物やoperator実行そのものを人手の権利判断として扱わず、reviewerによる実質確認をMUSTとする。

2026-08-16に`nishide-dev`がRights reviewerとQuality ownerの役割を受諾し、固定された60件すべてのrights/selection、selection method、exclusions、known failuresを承認した。公開証跡は[`formal-quality-review-attestation.json`](../benchmarks/quality/formal-quality-review-attestation.json)へaggregate digestと件数だけを記録し、project-owner self-attestationであって外部法務意見ではないことを明示する。この承認はmetric method/budget、formal set全体、GPU評価、Support Tierまたはreleaseを承認しない。

承認証跡のimmutable commit URLをprivate checklistへ適用した後、review済みregistryから60件のredacted metadataを[`formal-quality-set.json`](../benchmarks/quality/formal-quality-set.json)へ登録した。10 smoke / 50 regression、全required coverage、prompt/reference digest、case rights evidence、selection method/exclusions/known failures reviewをmachine-readableに検証できる。Rights reviewerは承認済みとし、Quality ownerは`nishide-dev`へ割り当てたまま`pending`とする。case noteはreview前candidateの文言をaudit fidelityのため保持し、現在の権利状態は`rights_status`とimmutable `rights_evidence`を正とする。

6 metric familyの計画は[`benchmarks/quality/formal-quality-metric-plan.json`](../benchmarks/quality/formal-quality-metric-plan.json)と[`schemas/quality-metric-plan.schema.json`](../schemas/quality-metric-plan.schema.json)で固定する。`h3fast benchmark check-quality-metric-plan`は、次を満たす場合だけ終了code 0を返す。

- prompt adherence、perceptual video、temporal consistency、audio quality、A/V sync、人手pairwiseが重複なく1件ずつ存在する
- 各familyにowner、version、immutable revision、entrypoint、exact dependency pin、入力、score方向、budget、HTTPS evidenceがある
- baseline/candidateを各3反復し、p5/p50/p95/worst-caseを記録する
- bit-exact決定性が独立2反復のper-case digest一致で証明された場合に限り、任意の`deterministic_generation_exemption`（policy、検証反復数、owner、日付、evidenceを必須）で追加反復を省略できる。反復要件そのものは3のまま変更せず、数値を変える最適化やbit-exactが成立しない条件では適用しない（[ADR 0011](decisions/0011-bit-exact-repetition-exemption.md)）
- exact profileはbaseline自己変動envelope外へのabsolute/relative toleranceを0とする
- 全familyを独立判定し、映像の合格でaudioまたはA/V syncの失敗を相殺しない
- observation欠損はfailとし、各familyでformal set全caseの100% coverageを要求する

committed metric planは意図的に`draft`かつ全family `unassigned`とする。これは評価規則だけを固定し、外部metric、model、閾値、ownerまたはquality approvalを推測しないためである。method選定後も、固定baseline上の自己変動を実測し、implementationとbudgetのevidenceをreviewするまで`approved`へ変更してはならない。設計判断は[`docs/decisions/0008-formal-quality-metric-plan.md`](decisions/0008-formal-quality-metric-plan.md)に記録する。

候補調査は[`benchmarks/quality/formal-quality-metric-selection.json`](../benchmarks/quality/formal-quality-metric-selection.json)と[`schemas/quality-metric-selection.schema.json`](../schemas/quality-metric-selection.schema.json)へ分離する。SigLIP2、LPIPS、ViSQOL、Synchformerとproject-owned human ballotについて、固定revision、license scope、入力、score方向、採用条件、blockerを記録する。このartifactは候補選定の再現用であり、実装identity、runtime互換性、checkpoint rights、baseline自己変動またはquality approvalを証明しない。H3Fast entrypointとcorrectness testが存在し、固定runtimeで実機確認するまでformal metric planを`planned`へ変更しない。設計判断は[`docs/decisions/0009-formal-quality-metric-selection.md`](decisions/0009-formal-quality-metric-selection.md)に記録する。

human-pairwise candidateは[`schemas/private-human-pairwise-ballot.schema.json`](../schemas/private-human-pairwise-ballot.schema.json)と[`schemas/private-human-pairwise-assignment.schema.json`](../schemas/private-human-pairwise-assignment.schema.json)でlocal-only contractを固定する。`prepare-human-pairwise`はformal set bytesのSHA-256へ60 caseを拘束し、A/B assignmentをcaseごとにrandomizeしてballotとassignment keyを別fileへ0600で新規作成する。seedは32文字以上かつgroup/otherから読めないlocal fileだけから読み、CLI引数、ballotまたはstdoutへ含めない。assignmentはsalt付きcommitmentとseed digestを持ち、review完了までreviewerへ開示しない。

blind media提示は`stage-human-pairwise`が行う。case_idをbaseline/candidate media fileへ対応付けるprivate media manifestを[`schemas/private-human-pairwise-media.schema.json`](../schemas/private-human-pairwise-media.schema.json)で固定し、pending ballot、assignment digest、ballotとassignment双方のcommitment相互一致、manifestのformal set digest、全media fileのSHA-256、pair内suffixの一致と英数字構成を検証してから、assignment keyに従い`<case_id>/a.<ext>`・`b.<ext>`を新規staging directory(0700)へcopyする。同梱する`index.html`は相対参照のみで、prompt、source識別子、absolute path、外部リソースを含めない。検証またはcopyの失敗時はstagingを残さず、filesystem失敗も終了code 2へ正規化する。

selectionは`record-human-pairwise`がcaseごとに記録する。pending ballotだけを対象とし、重複case、不正なselection値、group/otherから読めるballot/assignmentを拒否し、記録済みcaseの変更には明示flagを要求し、全case記録時にのみ`completed`と完了時刻を原子的に設定する(0600維持)。ballot schemaはselection入力済みcaseと未入力caseが混在する記録途中のpending状態を許容する。`check-human-pairwise`はcompleted ballot、assignment file digest、全commitment、formal set digest、全caseの固定順coverageを検証する。欠損、abstain、重複、改ざんまたはstale assignmentは終了code 2とし、aggregateにはbaseline/candidate win、tieと`(candidate_wins - baseline_wins) / case_count`だけを出力する。prompt、media、local path、seed、assignmentまたはper-case判断をstdoutへ出力しない。ballot、assignment、seed、media manifest、stagingはGit、CI artifact、共有logへ追加してはならない。

2026-08-16に、Git外のsynthetic media 60 caseでprepare→stage→record→checkの全工程pilotを完了した。assignment keyを参照しない知覚代理判定が全60 caseでground truthと一致し、blind割当・復号・集計を検証した([`docs/experiments/0006-human-pairwise-pilot.md`](experiments/0006-human-pairwise-pilot.md))。これはworkflow検証でありformal human evaluationまたはquality approvalではない。review policyは[`docs/decisions/0010-human-pairwise-review-policy.md`](decisions/0010-human-pairwise-review-policy.md)で固定する。quality owner `nishide-dev`によるsingle-reviewer運用とし、blind stagingとmetadata非閲覧を必須、迷う場合は`tie`、completed ballotは不変で再実施は新ballotとする。GPU実出力によるformal実測、immutable implementation revisionとformal evidenceが揃うまでhuman-pairwiseはcandidate、formal metric planは`unassigned`のままとする。

最初の外部metric adapterはperceptual-video (LPIPS)とする。`score-perceptual-video`は、依存を`quality-metrics` dependency group(lpips 0.1.4、torch 2.11.0 CPU wheel、torchvision 0.26.0)へ隔離し、packageのCPU importとwheelのruntime依存ゼロを維持する。AlexNet backbone checkpointは自動downloadせず、SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`へ固定したlocal fileだけを検証して読み込む。両入力はffmpegでRGB24へdecodeし、frame数・解像度・frame rateの完全一致をfail-closedで要求してtemporal resamplingを行わず、評価はtorch thread数1へ固定した単一process決定性で行い、per-frame LPIPSのmeanとmax、使用digest、thread数、依存versionをreportへ記録する。非有限値、decode失敗、構築中の予期しないcheckpoint追加、quality-metrics group欠如はValidationError(終了code 2)へ正規化する。同一入力→0、摂動単調性、単一process決定性、契約不一致、checkpoint改ざん・破損、decode失敗の帰属はcorrectness testで固定済みである。formal media contractと固定H3 runtimeでの実機確認、baseline自己変動、backbone checkpointのlicense scope確認が完了するまでcandidateのままとし、formal metric planは`unassigned`のまま変更しない。

temporal-consistencyはproject-owned契約`adjacent-frame-lpips-trajectory-v1`で固定する。`score-temporal-consistency`は各動画の隣接frame間LPIPS列をtrajectoryとし、index対応するstepの絶対差のmeanとmaxを比較する(lower-is-better)。scene cutは除外せず、candidateが保持したcutは相殺され、消失・移動したcutはdeltaとして現れる。timestamp alignmentはperceptual-videoと同一契約(解像度・frame rate・frame数の完全一致、resamplingなし)で、2 frame未満の入力は拒否する。static scene→0、flicker/cut除去の検出、決定性、decode失敗の帰属、非有限値failはcorrectness testで固定済みである。同じbackbone checkpoint契約とquality-metrics groupを共有し、実機確認とbaseline自己変動が完了するまでcandidateのままとする。

prompt-adherenceは契約`siglip2-base-patch16-256-cosine-v1`で固定する。`score-prompt-adherence`は、pinned SigLIP2 snapshot(revision `3f9f96cb…`、7 fileのSHA-256 manifestで検証、自動downloadなし、manifest外のfile検出で拒否)をoffline loadし、最大16 frameを固定則で一様サンプリングして、L2正規化したtext/image featureのcosine類似度のmeanとminを出力する(higher-is-better)。prompt本文はgroup/otherから読めないprivate local fileとして供給し、そのbytesのSHA-256がformal caseの`prompt_sha256`と一致しない場合は拒否する。prompt本文、path、mediaはstdoutへ出力しない。pinned snapshotのconfigはmodel_type `siglip`(FixRes変種)であり、digest manifestが同一性を固定した上でarchitecture classを検証する。torch thread数1の決定性、非有限値fail、依存欠如の正規化はほかのadapterと同一契約である。pinned snapshotのoffline loadと意味的方向性(一致promptと不一致promptのscore分離)はlocal CPUで実機確認済みである。formal media contractと固定H3 runtimeでの確認、baseline自己変動が完了するまでcandidateのままとする。

formal caseの生成は`run-formal-cases`が行う。private reviewed registry(group/other不可読)をcommitted formal quality setへfail-closedで拘束し、全60 caseの固定順一致、prompt本文のSHA-256とformal caseの`prompt_sha256`の一致、seed・task・duration・aspect ratioの一致を検証してから、pinned protocolのtemplate caseから固定生成parameter(short_edge、sigma_points、flow_shift、audio_flow_shift、conditions)を採り、caseごとのpayloadをguarded serverへ送る。repetitionごとのoutput directoryへper-case result(JSON)とmediaを保存し、artifact digest検証つきのresumeで中断再開できる。redactedなrun manifest(case_id、artifact名、SHA-256、size、経過時間のみ。prompt・digest・pathなし)を書き出し、これがmetric評価とhuman-pairwise media manifestの入力になる。runnerはt2va、fl2va、ref2vaのpayloadを構築する。task選択は`--task`で行い、選択したtaskのcaseだけを対象とする。silentなtask置換は行わず、未対応のtask familyは明示的なerrorとする。

reference条件は公式cookbookの`conditions`契約に従い、`{type, uri, role}`を与える。fl2vaは`role: keyframe`と`frame_index`(first=`0`、last=`-1`)の2条件を要求し、2件でない場合や画像以外の場合はfail closedとする。ref2vaは`role: reference`と`type: image|video|video_audio|audio`を用い、mixed caseは複数条件として展開する。asset pathはregistryからの相対解決を許し、実在しないassetは拒否する。`uri`は`file://`形式で与え、生成物と同様にasset本体はGit外のJapan-local storageに置く。resumeは現在のformal case prompt digestとprotocol_idへ一致するresultだけを再利用する。stdoutへはcount・manifest digestだけを出力する。生成mediaはH3 OutputとしてGit外に保持し、承認済みJapan-local scope内でのみ実行する。

2026-08-17に固定runtimeでt2va 20 caseを両protocol2反復ずつ生成し、rep1/rep2の生成物SHA-256が全40比較で一致してbit-exact決定性を確認した。同一caseにおける20層baselineと40層candidateの出力も全20 caseでbit単位一致し、placement変更がcompute graphを保存するという主張を単一caseのexact gateから20 caseへ拡張して裏づけた。これによりbaseline自己変動は0、40層candidateの品質差も0である。詳細と限界は[`docs/experiments/0008-formal-generation-determinism.md`](experiments/0008-formal-generation-determinism.md)に記録する。bit-exact性は当該pinned条件での実測事実であり一般保証ではなく、fl2va/ref2va 40 caseは未測定である。

この結果を受けて、最適化の検証費用を出力等価性のclassでtier化する（[ADR 0012](decisions/0012-tiered-optimization-verification.md)）。compute graph、schedule、step数、precisionを保存するplacement-only最適化はTier 1とし、protocolごとにsmoke 4 case以上をper-case digestで照合する。全一致は品質差0の証明であり、当該変更についてmetric実測とbudget承認を代替する。1 caseでも不一致があればTier 2へescalateする。量子化、量子化attention、step蒸留、kernel書き換え、precision/schedule変更などbytesが変わり得るものはTier 2とし、formal set全caseと[ADR 0008](decisions/0008-formal-quality-metric-plan.md)の反復・統計・family独立判定・budget承認を要求する。検証tierは測定前に宣言し、結果を見た後に下げてはならない。

2026-08-18に、formal生成で観察された発話音声の非言語性がprompt書式に起因することを探索的に確認した（[`docs/experiments/0007-japanese-dialogue-probe.md`](experiments/0007-japanese-dialogue-probe.md)）。H3のmodel cardが示す`<d>[言語] テキスト</d>`書式で台詞本文を与えた条件では日本語音声が明瞭に生成され、同一seed・同一場面記述で台詞本文を与えない現行formal様式では言語として成立しなかった。現行のplacement最適化判定はbaselineとcandidateが同一promptを使うため影響を受けない。Tier 2最適化でaudio qualityとA/V syncを実測する際は、台詞が非言語のままではmetricが発話の明瞭さを評価できない可能性があるため、formal set改訂の要否をその時点で判断する。改訂はprompt digestを変えるためrights reviewとregistry再構築を伴う。

最初のTier 2候補としてSage Attention（INT8量子化attention）を追加する。protocolの`runtime.attention_backend`（`auto` / `fa` / `sage_attn`、既定`auto`）を1変数として導入し、既定値では従来のpinned argvと同一の起動を保つ。SageAttentionはread-onlyのruntime imageへ同梱せず、pinned commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`からAda（SM89）向けにbuildした成果物を外部pathへ置き、bindとPYTHONPATHで注入する。これによりruntime image digestは不変のまま、SageAttention側をcommitとwheel digestで独立に固定できる。`benchmarks/protocol-sage.yaml`はresident40との差分をattention backendだけに限定し、数値が変わるためexact artifact quality gateを持たない。採用判定は[ADR 0012](decisions/0012-tiered-optimization-verification.md)のTier 2に従い、formal setとmetric実測、family別budget承認を要求する。SageAttentionはApache-2.0であり、H3 Materialsを含まない外部依存としてartifact registerへ分類する。`sage_attn`はSGLang公式ドキュメントがCUDA/MUSA対応backendとして記載し、pinned commit `d9704247…`を推奨installとして明示しているものである。component単位の指定は公式書式の`--component-attention-backends <component>=<backend>`を用いる。

2026-08-18の測定で、component-scopedなbackend指定がMiniMax H3へ届かないことを確認した。SGLangのcomponent overrideはtransformerロード中だけ有効なContextVarだが、H3はattention backendの解決を最初のforwardまで遅延するため、解決時点でcontextが終了しており指定が失われる。この構成では生成物がFA baselineとbit単位で一致し、比較が成立しない。global指定（`--attention-backend sage_attn`と`--component-attention-backends text_encoder=torch_sdpa`の併用、ring-degree 1）では遅延解決後もSageが選ばれ、出力がFAと異なる（[experiment 0009](experiments/0009-sage-attention-noop.md)）。launchはこのglobal構成を出す。

component loaderのログやpipeline validationの成功は、実装が各attention moduleへ設定された証拠にならない。このため`h3fast benchmark verify-backend`をfail-closedで追加した。guarded serverログの最終resolutionを実際に使われたbackendとみなし、要求と一致しないrun、および証拠のないrunを拒否する。Tier 2判定にはbackend検証を通過したrunだけを使用する。数値を変えるはずの最適化でdigestが一致した場合は、品質劣化ゼロではなく最適化が無効である可能性を先に疑う。

同一host上で2つのguarded serverを並列運用する場合は`--master-port`で分散rendezvous portを分離する。これはcompute graph・scheduleへ影響しない起動設定であり、既定(None)では従来のpinned argvと同一である。並列生成中のlatency・memory測定値はpinned単一server環境と比較しない。

- T2VA / FL2VA / Ref2VA
- 4秒、5秒、10秒、15秒
- 横長、正方形、縦長
- 日本語を含む複数言語
- 顔、手、文字、製品、複数人物
- 激しい運動、静止画面、カメラ移動
- 会話、環境音、音楽、無音に近い場面
- 画像、動画、音声、混合参照

### 14.6 品質ゲート

固定した単一閾値だけで「無損失」を宣言しない。次の方法を採用する。

1. 同じbaselineを同じ環境で複数回実行し、非決定性の分布を測る。
2. `exact` profileは、各品質指標がbaseline同士の変動包絡内にあることを要求する。
3. `balanced`と`fast`は、リリースごとに明示した品質予算を超えないことを要求する。
4. 映像指標が合格しても、音声またはA/V同期が悪化した場合は不合格とする。
5. 平均値だけでなくp5/p50/p95とworst-caseサンプルを確認する。

2026-08-15時点の最初のplacement-only A/Bには、`exact-decoded-artifact-v1`を使用する。測定3回がcontainer、RGB24 decoded video、PCM decoded audio、media metadataで一致する場合だけreferenceを作成し、candidateは同じartifact identityとmedia contractを要求する。prompt本文、生成物、local pathはreferenceへ含めない。method、decode format、`ffmpeg`/`ffprobe` version、protocol/case/request digestを固定する。

このgateは[`benchmarks/quality/exact-smoke-001-reference.json`](../benchmarks/quality/exact-smoke-001-reference.json)の単一caseに限定する。10件以上のsmoke set、50件以上のregression set、知覚・audio・semantic A/V品質指標を代替せず、一般的なlossless性または品質同等性の根拠にしてはならない。設計判断は[`docs/decisions/0004-exact-quality-gate.md`](decisions/0004-exact-quality-gate.md)に記録する。

formal set contractは[`docs/decisions/0005-formal-quality-set-contract.md`](decisions/0005-formal-quality-set-contract.md)に記録する。`h3fast benchmark check-quality-set --record benchmarks/quality/formal-quality-set.json`は、60件以上のmetadata、全coverage、per-case rights evidence、versioned metric/budget、rights/quality approvalが揃った場合だけ終了code 0を返す。recordの存在やschema validation成功だけを品質承認として扱わない。

### 14.7 Phase 0 baseline protocol

実装開始前にbenchmark protocolとschemaを作り、少なくとも次を固定する。

- base model repositoryとimmutable commit、task family
- SGLang/PyTorch/Tritonのversionまたはcommit
- GPU model/count、driver、CUDA、host CPU/RAM
- prompt、seed、condition artifact digest、resolution、duration、schedule
- cold/warmの定義、warmup回数、測定回数、集計方法
- E2Eと各stageの開始・終了点
- 出力artifact、stdout/stderr、metrics、失敗を保存する規則
- 品質比較のreference生成方法と許容差のversion

固定20層baselineは`benchmarks/protocol-baseline20.yaml`、実測後に採用した40層設定は`benchmarks/protocol.yaml`へ記録する。両protocolはquality method `exact-decoded-artifact-v1`、reference ID、RGB24/PCM decode format、baseline measured run数、`formal_quality_set_path`を共有し、正式なquality setが未完成であることも機械可読な`formal_quality_set_ready: false`として維持する。このbooleanはformal quality-set validatorが成功した後だけ`true`へ変更する。

性能に影響するruntime設定はprotocolが所有する。launch plan、server lifecycle、suite bundleは同じ実効値を記録し、protocolと起動済みserverの値が一致しないsuiteを開始してはならない。fallbackでresident layer数を黙って下げず、20層rollbackはbaseline protocolを明示的に選択する。

baselineを一度も再現できていないGPUをTier 1候補として最適化しない。

---

## 15. H3固有の実装注意

### 15.1 Sparse Attention

H3公式は学習後半でnative sparse attentionを導入したと説明しているが、初期オープンリリースには実装が含まれていない。したがって独自Sparse backendは公式実装と同一と表示してはならない。

Sparse実装は:

- 3D MM-RoPEを維持する。
- 映像・音声・テキスト・参照tokenの境界を保持する。
- text/reference/global tokenを誤ってpruneしない。
- 映像と同期音声間のcross-modal blockを保護する。
- maskだけをDense Attentionへ渡す「見かけ上の疎化」ではなく、実際に不要blockを計算しないkernelを使用する。

### 15.2 AdaLN Cache

H3公式は約13Bパラメータ分のAdaLN branch出力を推論時に事前計算可能としている。SGLangの現行ガイドは、このcacheを実験的経路とし、固定sampling scheduleへ結び付けている。

- cache artifactは派生重み相当の管理対象として扱う。
- schedule fingerprintを必須にする。
- 量子化checkpointと未量子化cacheを混用しない。
- mismatch時は拒否し、黙って通常経路へ変更する場合もmetadataへ記録する。

### 15.3 `torch.compile`

SGLangの現行H3ガイドでは、推奨lossless presetはDiTをeagerに保ち、現在の`torch.compile`経路は数値出力を変えるためground truth生成に用いないとしている。

本仕様では:

- `torch.compile`を既定でlosslessとみなさない。
- backend/versionごとに品質回帰を実施する。
- graph break、compile cache、shape bucketを測る。
- 明示フラグで有効化し、結果metadataへ記録する。

### 15.4 H3-specific optimization reference

NVIDIA SANA TeamのSol Engine H3ページは、stock checkpointに対し、kernel fusion/graph capture、sparse attention、cross-step residual reuseを組み合わせたH3固有の推論時高速化を報告している。これは本仕様のbenchmark分解とplugin architectureの重要な参考になる。ただし、公開ページの主張を自製ランタイムの性能保証として引用してはならず、同条件で再測定すること。

---

## 16. CI/CD仕様

### 16.1 Pull Request CI

v0.11のCPU-only Pull Request CIは次を実装済みとする。

- pinned uv versionで`uv sync --locked`が成功
- build済みwheelだけを用いたclean install test
- Ruff format/lintと`ty` type check
- coverage閾値付きunit test
- 全JSON schemaのmeta-schema検証と、commit済みprotocol/reference/release gateのinstance検証
- Initial Runtime release/territory/formal-quality recordのsemantic validationとfail-closed CLI test
- repository外の一時環境でCPU import

Public Runtime release前には次を追加し、完了するまでこのrepositoryのCIをrelease gate済みと扱わない。

- dependency/license scan
- secret scan
- release artifactのlicense/notice検証
- 配布対象がある場合のcontainer build、SBOM、vulnerability scan
- 実装したbackend/kernelのreference fallback test
- main projectまたは追加済みtargetごとのlockfile差分検証

GPU CIは未実装であり、Tier付与またはGPU runtime release前に次を追加する。

- 少なくとも代表1 GPUでkernel correctness
- nightlyまたはmerge queueで複数GPU世代
- E2E短尺generation smoke test

### 16.2 Release CI

```text
signed Git tag
  ↓
source checkout by immutable SHA
  ↓
wheel/sdist build in isolated runner
  ↓
unit + kernel + E2E + benchmark gates
  ↓
PyPI Trusted Publishing
  ↓
OCI build and push
  ↓
SBOM + provenance + Cosign signature
  ↓
model manifest/checksum generation
  ↓
controlled model publication
  ↓
clean-room installation verification
```

### 16.3 PyPI

- `uv build --package <name>`で各distributionを個別buildする。
- `uv publish`または同等のTrusted Publishing対応経路を使用する。
- 長期API tokenではなくTrusted Publishing/OIDCを使用する。
- wheelとsdistを両方検証する。
- release tagとpackage versionの一致を確認する。
- package attestationを保持する。

### 16.4 Provenance

- SLSA provenance形式またはGitHub Artifact Attestationsを利用する。
- binary、wheel、container、model conversion outputのsubject digestを記録する。
- model変換の入力にbase model file digestsを含める。
- tenantが自由に書き換えられる手動JSONだけをprovenanceと呼ばない。

---

## 17. バージョニング

### 17.1 Runtime

Semantic Versioningを使用する。

```text
h3fast 1.2.3
        │ │ └ patch: numerical/performance bug fix, compatible
        │ └── minor: backward-compatible feature/backend
        └──── major: API/model manifest/kernel ABI break
```

### 17.2 Model artifact

モデルIDは意味を読める形式にする。

```text
h3fast-h3-base-{task}-{evals}-{precision}-v{model-major}.{model-minor}
```

例:

```text
h3fast-h3-base-fl2va-8eval-fp8-v1.0
```

- 重みが変わった場合、同じrevision/tagを上書きしない。
- model repoのcommit SHAをimmutable revisionとして公開する。
- tuning JSONだけ変えた場合もartifact digestを更新する。

### 17.3 互換性

Runtimeとmodelの互換性はmanifestで機械判定する。

```text
installed_runtime satisfies model.runtime.requires
```

version比較は文字列比較ではなくPEP 440として行う。範囲外はwarningではなく、既定で起動拒否とする。範囲内でも`tested_versions`にないversionは`compatible-untested`として明示する。`--allow-untested-runtime`は実験用に限り、結果metadataへ必ず記録する。

---

## 18. 可観測性

### 18.1 Metrics

最低限:

```text
h3fast_requests_total
h3fast_request_duration_seconds
h3fast_queue_duration_seconds
h3fast_stage_duration_seconds{stage=...}
h3fast_gpu_memory_bytes
h3fast_host_memory_bytes
h3fast_kernel_backend_total{backend=...}
h3fast_kernel_fallback_total{reason=...}
h3fast_triton_compile_seconds
h3fast_cache_hit_ratio{cache=...}
h3fast_generation_failures_total{reason=...}
h3fast_policy_blocks_total{policy=...}
```

### 18.2 Logs

構造化JSONで次を記録する。

- request/job ID
- tenant IDまたは匿名化ID
- model artifact ID
- runtime/image digest
- input sizes、duration、形式
- stage timings
- backend/fallback
- policy結果
- error code

prompt本文やメディアを標準ログへ含めない。

### 18.3 Trace

OpenTelemetry等で:

```text
queue
→ input fetch
→ media decode
→ context encode
→ denoise steps
→ visual VAE
→ audio VAE
→ mux
→ moderation
→ C2PA signing
→ upload
```

を分離する。

---

## 19. サポートマトリクス

### 19.1 Support Tier

| Tier | 意味 |
|---|---|
| Tier 1 | 各releaseでE2E、品質、性能、障害回帰を実機検証 |
| Tier 2 | E2E smokeとkernel correctnessを検証。性能SLOは限定 |
| Experimental | コミュニティまたはbest effort。fallback前提 |
| Unsupported | 起動拒否または明確な非対応表示 |

将来のproduction候補:

- NVIDIA H100/H200: Tier 1候補
- NVIDIA B200/GB200/B300: Tier 1候補
- RTX 5090等: Tier 2候補
- AMD Instinct MI300X/MI355X: Tier 2候補
- Apple Silicon/consumer Radeon: ExperimentalまたはUnsupported

これはSGLang等の外部対応実績を参考にした候補であり、H3Fastの実機CIが完了するまで正式サポートと表示してはならない。

Phase 0のローカル検証候補は、2×RTX 6000 Ada Generation 48GB、FL2VA/T2VA、768p、5秒とする。これはSGLangの2×RTX 5090 layerwise-offload recipeを基にしたExperimental構成であり、H3 E2E、peak memoryおよび品質確認が完了するまでTierを付与しない。4 GPU構成は空きGPUを同時確保できる環境で別途検証する。

2026-08-15に固定T2VA caseを2×RTX 6000 Adaでwarmup 1回と測定3回完走した。client E2E p50は889.495秒、server inference p50は886.759秒、reported peak GPU memoryは最大23,376 MiBだった。denoise p50は847.339秒でserver時間の約95.55%を占めた。4成果物はMP4 SHA-256、size、media contractが一致し、後続のexact reference比較でもmeasured 3回が合格した。これは単一caseの再現性であり一般的な品質同等性またはlossless性を証明しないため、Tierは付与せずprotocol statusを`draft`のまま維持する。条件と結果は[`docs/experiments/0003-rtx6000-ada-measured-baseline.md`](experiments/0003-rtx6000-ada-measured-baseline.md)と[`docs/experiments/0004-exact-quality-reference.md`](experiments/0004-exact-quality-reference.md)に記録する。

---

## 20. Release Checklist

### 20.1 Code

- [ ] Git tagが署名済み
- [ ] uv versionとPython versionがpin済み
- [ ] main `uv.lock`とproduction target `uv.lock`がcommit済み
- [ ] `uv sync --locked`とclean-wheel installが成功
- [ ] workspace member versionとrelease manifestが一致
- [ ] source revisionが固定
- [ ] wheel/sdistのclean install成功
- [ ] Triton sourceとfallbackを含む
- [ ] API/schema migration noteがある
- [ ] SECURITY.mdと脆弱性連絡先がある

### 20.2 Model

- [ ] base model revisionがcommit SHA
- [ ] Safetensorsのみ
- [ ] 全shard checksumあり
- [ ] manifest/schema validation成功
- [ ] LICENSE/NOTICE/MODIFICATIONSあり
- [ ] optimization configが実装と一致
- [ ] AdaLN schedule fingerprintが一致
- [ ] benchmark bundleあり
- [ ] model cardに限界・既知問題あり

### 20.3 Container

- [ ] base image digest固定
- [ ] weight/tokenなし
- [ ] non-root
- [ ] SBOMあり
- [ ] vulnerability scan済み
- [ ] Cosign署名あり
- [ ] provenanceあり
- [ ] digest指定のdeployment exampleあり

### 20.4 Service

- [ ] 認証・rate limit
- [ ] SSRF防御
- [ ] media size/codec上限
- [ ] moderation
- [ ] report channel
- [ ] terms/AUP consent
- [ ] retention policy
- [ ] C2PA/AI表示
- [ ] incident response runbook

### 20.5 License

- [ ] 最新原文とhashを取得
- [x] 現行Japan-local single-operator research scopeの地域レビュー
- [ ] 派生物レビュー
- [ ] 商用条件レビュー
- [ ] MiniMax許可文書を必要に応じて保管
- [ ] downstream noticeと利用規約を確認

---

## 21. 段階的ロードマップ

| Phase | 2026-08-16時点の状態 | 次のgate |
|---|---|---|
| Phase 0 | 独立code境界、Japan-local H3-use scope、60-case rights/selection reviewとredacted metadata登録は解決。metric plan、6-family candidate assessment、human ballot/scorer contract、offline presentation runner、synthetic-media pilotとsingle-reviewer policy承認済み | metric adapter/owner/budget approvalと実測、formal set approval |
| Phase 1A | 内部technical path実装済み | clean machineでGPU baseline再現、package release確認 |
| Phase 1B | 最初のplacement最適化を実測・採用済み | clean machine再現、formal quality、release supply chain |
| Phase 2以降 | 未着手 | Phase 0とInitial Runtime release gateの完了 |

### Phase 0: 法務・基準線

- H3 licenseとAUPをH3-related flowへ適用する条件の確認
- H3 access・実行・Output地域と、対象外のsource/CPU CIを分離したartifact registerの作成
- H3公式コードをcopy、dependency、独立実装のどれで扱うかの決定
- H3-Base BF16 baselineの固定
- prompt/seed benchmark suite作成
- source、runtime、modelのlicense境界決定（独立sourceはADR 0006で解決済み）
- 最初のtask family、GPU topology、解像度、durationを一つずつ決定

### Phase 1A: Reproducible BYOW Baseline

内部成果物（このPhaseだけでは公開releaseを要求しない）:

- 単一の`h3fast` packageと`uv.lock`
- manifest/schema、`verify`、`doctor`、benchmark harness
- 固定SGLang版へのadapter
- local snapshotのrevisionとdigestを固定するBYOW検証経路
- BF16 baseline bundle

2026-08-15時点で、固定runtimeと2×RTX 6000 Adaにおけるwarmup 1回・測定3回のlocal BF16 baseline bundleは作成済みである。支配stageはdenoiseと特定し、単一`smoke-001`のplacement-only exact quality gateも実測済みである。2026-08-16に10/50件の正式quality set契約とprivate registry compilerを追加し、Git外に10 smoke / 50 regression candidateとsynthetic referenceを生成してregistry全体digestをattestした。さらにregistry/content digestへ拘束したlocal-only rights/selection review workflowと、6 familyを独立判定するformal metric plan contractを実装した。同日、`nishide-dev`が60件すべてのrights/selection、selection method、exclusions、known failuresをreviewし、aggregate承認証跡を記録した。immutable evidenceをprivate registryへ適用し、review済み60件のredacted metadataと全required coverageをformal recordへ登録した。続いて6 familyの候補、固定revision、license scope、採用条件とblockerをcandidate assessmentへ記録し、human-pairwiseのprivate ballot/key schema、blind assignment commitment、欠損時failとaggregate scorerを実装した。同日、offline A/B presentation runner(media manifest検証、blind staging、selection記録CLI)を追加し、synthetic mediaによる60-case pilotでprepare→stage→record→checkの全工程を検証し、single-reviewer review policyを承認した。metric owner/budget approval、formal set approvalとGPU実測は未完了である。clean machineでの再現、formal setの承認・実測、公開可否の確認は完了条件として残る。

完了条件:

- 公式重みの再配布なし
- clean machineで固定baselineを再現可能
- wheelのclean installとCPU importが成功
- source/license境界がartifact registerに記録済み

### Phase 1B: First Measured Optimization

- profile結果から選んだbottleneckを一つ最適化
- 数値kernel/algorithmを変更する場合はPyTorch reference実装とcorrectness testを追加し、全最適化に安全なfallbackまたは明示的unsupported errorを用意
- stage/E2E/品質のA/B benchmark
- 効果がない最適化を既定経路へ入れない
- 必要になった場合のみTriton、target別lock、weightless OCI imageを追加
- code-only releaseの品質・再現性・supply-chain gateを満たした時点で最初のPublic Runtime release候補とする。H3-related実行・artifactには別途H3-use gateを適用する

2026-08-16に最初の単一変数最適化としてDiT resident layer数を20から40へ増やし、2×RTX 6000 Adaでwarmup 1回・測定3回を完走した。20層baselineに対し、client E2E p50は889.495秒から883.516秒へ0.672%、denoise p50は847.339秒から842.507秒へ0.570%改善した。measured 3成果物はexact decoded artifact gateをすべて通過したため、40層を既定protocolへ採用した。

reported peak GPU memory最大値は23,376 MiBから35,696 MiBへ12,320 MiB（52.704%）増えた。空きメモリ要件を満たせない場合は20層baseline protocolへ明示的にrollbackする。結果は[`docs/experiments/0005-rtx6000-ada-resident40.md`](experiments/0005-rtx6000-ada-resident40.md)に記録する。単一case・単一host・測定3回の結果であり、Tierまたは公開性能主張へ拡張しない。

このplacement変更は既存BF16計算graph、schedule、step数を変えないため、新しい数値kernel向けPyTorch referenceは該当しない。20層protocolをreference/rollbackとし、protocol差分test、実効setting一致、E2E/stage A/B、exact decoded artifact gateでcorrectnessを確認した。

### Phase 2: Controlled Derivative Weights

- MiniMax/法務の確認
- 手動承認gateまたはentitlement service
- FP8、AdaLN、8-eval等を別repoで配布
- access auditとrevocation

### Phase 3: Hosted API / Enterprise

- SGLang-compatible video API
- queue、tenant、billing、SLO
- moderation、reporting、incident process
- C2PA signing
- signed Helm chart / on-prem bundle

### Phase 4: Upstream / Ecosystem

- SGLangへのbackend upstream提案
- Diffusers/ComfyUI integration
- 他backendとのplugin ABI
- 公開benchmark leaderboard

---

## 22. Initial Runtime Release Definition of Done

初回production runtime releaseは、次をすべて満たした時点で完成とする。文書の正式化条件と実装の完成条件を混同しない。

1. BYOWで公式H3から派生成果物を再現できる。
2. 変換物の入力revision、recipe、digestを追跡できる。
3. Triton backendを含む場合、非対応環境で安全にfallbackまたは明確に停止する。
4. 配布対象にOCI imageを含む場合、重みなしの署名済みimageで固定SGLang APIを起動できる。
5. 代表GPUでE2E動画・音声生成試験を通過する。
6. baselineと最適化版の品質・性能差が公開される。
7. LICENSE、NOTICE、変更履歴、第三者ライセンスが揃う。
8. 派生重みの一般公開は、承認前には行われない。
9. Hosted Serviceを提供する場合、AUP、安全策、報告、監査を備える。
10. 利用者がmodel revision、runtime version、および配布する場合はimage digestを固定できる。
11. 初期の単一`uv.lock`と、作成済みの各Tier 1 GPU target lockから環境を再現できる。
12. 公開wheelがrepository外のclean environmentでinstall・importできる。

本仕様をDraftからApprovedへ変更するには、Phase 0のblocker、schema owner、初期support target、参照backend revision、release承認者が明記され、未解決事項にownerと期限が設定されていなければならない。

---

## 23. 参考資料

### 23.1 MiniMax H3一次資料

1. **MiniMax H3 Model Card**
   https://huggingface.co/MiniMaxAI/MiniMax-H3

2. **MiniMax H3 Community License Agreement**
   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

3. **MiniMax H3 License Q&A**
   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md

4. **MiniMax H3 GitHub repository**
   https://github.com/MiniMax-AI/MiniMax-H3

5. **MiniMax H3 announcement / system overview**
   https://www.minimax.io/blog/minimax-h3

### 23.2 H3推論・高速化

6. **SGLang MiniMax-H3 deployment guide** — task、topology、AdaLN cache、`/v1/videos` API、benchmark
   https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3

7. **Sol Engine for MiniMax-H3** — H3固有のfusion、graph capture、sparse attention、cross-step cacheの事例
   https://nvlabs.github.io/Sana/Sol-Engine/H3/

8. **StreamDiffusionV2** — streaming scheduler、rolling cache、Stream-VAEの参考
   https://streamdiffusionv2.github.io/

9. **MiniMax Sparse Attention paper**
   https://arxiv.org/abs/2606.13392

10. **MiniMax Sparse Attention implementation**
    https://github.com/MiniMax-AI/MSA

### 23.3 Triton / PyTorch

11. **Triton documentation**
    https://triton-lang.org/

12. **Triton installation and supported-distribution entry point**
    https://triton-lang.org/main/getting-started/installation.html

13. **Triton autotune API**
    https://triton-lang.org/main/python-api/generated/triton.autotune.html

14. **Triton block-scaled FP4/FP8 matmul tutorial**
    https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

15. **PyTorch: User-Defined Triton Kernels with `torch.compile`**
    https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

16. **PyTorch custom operator guidance**
    https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html

17. **PyTorch compile caching**
    https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

### 23.4 Hugging Face配布

18. **Gated Models**
    https://huggingface.co/docs/hub/models-gated

19. **Model Cards**
    https://huggingface.co/docs/hub/model-cards

20. **Model Release Checklist**
    https://huggingface.co/docs/hub/model-release-checklist

21. **Model downloading / `hf download`**
    https://huggingface.co/docs/hub/models-downloading

22. **Pickle scanning and model-file security**
    https://huggingface.co/docs/hub/security-pickle

23. **Hugging Face Hub security**
    https://huggingface.co/docs/hub/security

24. **Fine-grained access tokens**
    https://huggingface.co/docs/hub/security-tokens

### 23.5 Python / OCI / 供給網

25. **PyPI Trusted Publishing**
    https://docs.pypi.org/trusted-publishers/

26. **PyPI Trusted Publishing security model**
    https://docs.pypi.org/trusted-publishers/security-model/

27. **GitHub Artifact Attestations**
    https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds

28. **SLSA Provenance v1**
    https://slsa.dev/provenance/v1

29. **SLSA Specification v1.2**
    https://slsa.dev/spec/v1.2/

30. **Cosign: Signing Containers**
    https://docs.sigstore.dev/cosign/signing/signing_with_containers/

31. **Cosign: Verifying Signatures**
    https://docs.sigstore.dev/cosign/verifying/verify/

32. **OCI Image Manifest Specification**
    https://specs.opencontainers.org/image-spec/manifest/

33. **OCI Image Index Specification**
    https://specs.opencontainers.org/image-spec/image-index/

34. **OCI Distribution Specification**
    https://specs.opencontainers.org/distribution-spec/

35. **SPDX**
    https://spdx.dev/

36. **CycloneDX**
    https://cyclonedx.org/

### 23.6 Kubernetes / 運用

37. **Kubernetes: image tags and immutable digests**
    https://kubernetes.io/docs/concepts/containers/images/

38. **Kubernetes Secret good practices**
    https://kubernetes.io/docs/concepts/security/secrets-good-practices/

39. **Kubernetes security checklist**
    https://kubernetes.io/docs/concepts/security/security-checklist/

40. **Artifact attestation enforcement with Kubernetes admission controller**
    https://docs.github.com/actions/security-guides/enforcing-artifact-attestations-with-a-kubernetes-admission-controller

### 23.7 生成物の来歴

41. **C2PA Content Credentials Specification 2.4**
    https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html

42. **C2PA specifications index and AI/ML guidance**
    https://spec.c2pa.org/specifications/specifications/2.4/index.html

43. **C2PA command-line tool**
    https://opensource.contentauthenticity.org/docs/c2patool/

44. **Content Credentials implementation guidance**
    https://spec.c2pa.org/specifications/specifications/2.4/guidance/Guidance.html

### 23.8 uv / Python workspace

45. **uv: Using workspaces** — 単一lockfile、member操作、workspace適用条件と非適用条件
    https://docs.astral.sh/uv/concepts/projects/workspaces/

46. **uv: Managing dependencies** — `workspace = true`、editable member、path dependency
    https://docs.astral.sh/uv/concepts/projects/dependencies/

47. **uv: Project configuration** — virtual project、`tool.uv.package = false`、build isolation
    https://docs.astral.sh/uv/concepts/projects/config/

48. **uv: Using uv with PyTorch** — accelerator別index、marker、optional dependency
    https://docs.astral.sh/uv/guides/integration/pytorch/

49. **uv: Using uv in Docker** — workspaceの中間layer、`--no-install-workspace`、non-editable install
    https://docs.astral.sh/uv/guides/integration/docker/

50. **uv: The uv build backend** — `uv_build`の対象とnative extension時の制約
    https://docs.astral.sh/uv/concepts/build-backend/

51. **uv: Building and publishing packages** — workspace package単位のbuild/publish
    https://docs.astral.sh/uv/guides/package/

52. **uv: GitHub Actions integration** — `setup-uv`、version pin、cache、PyPI公開
    https://docs.astral.sh/uv/guides/integration/github/

---

## 24. 未解決事項

Public Runtimeのrelease判断またはPhase 2開始前に、少なくとも次のBlockerを決定する必要がある。Issue [#11](https://github.com/nishide-dev/h3fast/issues/11)のH3-use complianceはJapan-local single-operator researchに限定して[ADR 0007](decisions/0007-japan-local-h3-use-scope.md)で解決した。release approverとschema ownerは未指名であり、独立code境界は[ADR 0006](decisions/0006-independent-code-license-boundary.md)で決定済みとする。

1. **Resolved for current H3 use / tracked in #11:** `nishide-dev`によるJapan-local single-operator researchとしてdevelopment host、GPU host、benchmark Output storage、runtime execution、Output useをowner申告とともに承認した。第三者access、Hosted Service、derivative/Output配布、Japan外利用またはoperator/machine/storage変更前にはinventoryを`incomplete`へ戻す。この承認は独立sourceのreleaseや将来のH3-use scopeを承認しない。
2. **Resolved for independent code:** 現在の公開repositoryとwheelにH3公式source fileのcopyは検出されていない。H3Fast source、schema、CLI、wheelおよび独自documentationはApache-2.0境界へ分類した。将来BYOW converterへ公式code、configurationまたはDocumentationを取り込む場合は再reviewする。
3. **Partially resolved / tracked in #16:** 初期ローカル候補（FL2VA/T2VA、768p、5秒、2×RTX 6000 Ada 48GB）は、warmup 1回と規定3回のBF16 baseline測定、stage集計、memory capacity、media contract、単一caseのplacement-only exact quality gate、およびDiT resident 20→40層の最初のA/Bを確認した。formal quality-set schema/compilerに加え、Git外の10 smoke / 50 regression candidateとsynthetic referenceを生成してregistry全体digestと件数をattestし、digest拘束されたprivate review workflowと6-family metric plan contractを実装した。`nishide-dev`による60件すべてのrights/selection review、immutable evidenceのprivate registry適用、redacted per-case metadataとcoverageの登録も完了した。6-family candidate assessmentへ候補、固定revision、license scopeと採用条件を記録し、human-pairwise ballot/key/scorer contract、offline presentation runner、synthetic-media pilotとsingle-reviewer policy承認を完了した。metric adapter/owner/budget approval、formal set approval、知覚・audio・semantic A/V実装とGPU実測は引き続きBlockerとする。4 GPU構成は空きGPU確保後に別途検証する。
4. **Resolved for Phase 1A:** 参照backendをSGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`へ固定し、SGLangの公開CLI `sglang serve`と非同期`/v1/videos`だけをadapter境界とする。根拠とruntime imageは[`docs/decisions/0002-h3-baseline-runtime.md`](decisions/0002-h3-baseline-runtime.md)に記録する。
5. MiniMaxが派生重みのHF手動gate配布を十分と認めるか。
6. Sparse Attentionの方式と公式Sparse実装公開後の移行戦略。
7. FP8の保存形式と各backend間の互換性。
8. AdaLN cacheを別artifactとするかmodel repoへ含めるか。
9. C2PA signing主体と証明書/KMS運用。
10. Hosted Serviceのmoderation providerと人手review手順。
11. Benchmark prompt setを公開可能にするか、権利処理が必要か。
12. Phase 3でSGLang serverをそのまま利用するか、独自serverを維持するか。
13. **Partially resolved:** Python 3.12、uv `0.11.2`、Hatchling `>=1.27,<2`を固定した。Python patchとbuild backend exact versionをrelease artifactでどこまで固定するか。
14. Tier 1 targetごとのPyTorch/Triton/SGLang indexとlock更新責任者。
15. `h3fast-server`をPyPI公開するか、OCI専用applicationとするか。
16. native CUDA/C++ extensionが必要になった場合のdistribution名とbuild backend。
17. Public Runtime release前にdependency/license scan、secret scan、artifact notice検証をどのCIで必須化するか。
18. 固定20層baselineと40層candidateをclean machineで再現し、host固有状態を排除できるか。
19. **Partially resolved:** independent-code classification ownerと限定H3-use compliance ownerは`nishide-dev`とした。release approver、schema ownerと正式deadlineは未決定。

次の作業順序は、(1) Issue #16でcandidate assessmentに従いmetric adapterを1 familyずつ追加してbaseline自己変動とfixed 20/40-layerを実測しformal setを承認、(2) 項目18のclean-machine再現、(3) 項目17のrelease supply-chain gate、(4) release/schema ownerの決定とする。H3-related runは項目1の限定scope内だけで行い、scope変更前に再reviewする。これらのrelease check完了前にPublic Runtime releaseまたはSupport Tier付与へ進まず、別途承認するまでPhase 2 derivative配布へ進まない。
