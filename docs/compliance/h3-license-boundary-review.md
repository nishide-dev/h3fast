# H3 license and source boundary review

- Review date: 2026-08-16 (Asia/Tokyo)
- Status: Independent-code classification accepted; Japan-local H3-use scope approved
- Tracking: [Issue #11](https://github.com/nishide-dev/h3fast/issues/11)
- Coordination owner: `nishide-dev`
- H3-use reviewer: `nishide-dev` (project-owner self-attestation; not external legal advice)
- Release approver: Unassigned
- Target decision date: 2026-08-31 (provisional)

この文書はengineering inventoryとproject classificationの記録であり、法的助言ではない。独立codeのrelease stateは[`compliance/release-gates/initial-runtime.json`](../../compliance/release-gates/initial-runtime.json)、H3-related flowのregion evidenceは[`compliance/territories/initial-runtime.json`](../../compliance/territories/initial-runtime.json)をsource of truthとする。境界判断は[ADR 0006](../decisions/0006-independent-code-license-boundary.md)に従い、ownerまたはevidenceがないH3-related flowを推測で承認済みに変更しない。

## Pinned primary sources

| Source | Immutable URL | Retrieved | SHA-256 |
|---|---|---|---|
| MiniMax H3 Community License Agreement | [LICENSE at `42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE) | 2026-08-16 | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |
| MiniMax H3 License Q&A | [Q&A at `42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/docs/QA-about-License.md) | 2026-08-16 | `c39dcfc5dc3e546918509b57709db826a9b1945311bffaa01e80501101b8abe4` |

取得時点では両fileとも`main`と固定revisionのdigestが一致した。License/AUPは更新され得るため、この一致を将来の同一性の根拠には使用しない。

## Engineering findings

License本文から、release設計に直接関係する事項を次のように整理した。

- Applicable TerritoryはEU、英国、韓国、米国を除く世界と定義される。
- MaterialsはMiniMax H3とMiniMaxが公開したDocumentationを含む。
- MiniMax H3にはweightだけでなく、公開されたsoftware、algorithm、model code、inference-supporting codeも含まれる。
- MiniMax H3 WorksはMaterials、Model Derivatives、その派生物を含む。
- H3 WorksまたはOutputの利用、配布、表示にはterritory制限があり、BYOWで消滅しない。
- 第三者へH3 Worksまたは組込み製品へのaccessを提供する場合、downstream termsとsafeguardが必要になる。
- Q&Aの最初の見出しはEU、英国、韓国、米国「に限定」とも読めるが、License本文とQ&A本文の説明はこれらをrestricted regionとして扱う。判断ではLicense本文を優先し、この表現差はMiniMaxまたはlegal reviewerへ確認する。

## Public repository boundary

2026-08-16のGit追跡対象とbuild済みwheelを対象にengineering inventoryを行った。

| Boundary | Finding | Release classification |
|---|---|---|
| `src/h3fast` | H3Fast独自のvalidation、diagnostics、benchmark orchestration。MiniMax source fileのcopyなし | Apache-2.0、Public |
| `schemas` | H3Fast独自artifact/protocol/report/release contract | Apache-2.0、Public |
| `runtime/ffprobe.py` | 固定SGLang containerからmedia metadataを得るlocal adapter | Apache-2.0、Public |
| `benchmarks/*.yaml` | model ID、immutable revision、実測条件、H3Fast所有promptを記録 | Apache-2.0候補。prompt公開権reviewは別途必要 |
| exact quality reference | digest、stream metadata、制約のみ。H3 Output本体とlocal pathなし | metadata-only Public |
| `docs` | 独自の設計・実測・一次資料要約。MiniMax Documentation fileのcopyなし | Apache-2.0、Public |
| local H3 snapshot | Git管理外。H3 Worksとしてlocal-only、再配布禁止 | MiniMax H3 Community License対象 |
| benchmark media | Git管理外。H3 Outputとしてlocal-only、再配布禁止 | License/AUP review対象 |
| SGLang checkout/SIF | Git管理外。固定identityのみ記録 | 第三者license/SBOM review待ち |

文字列scanでは公式License本文の複製をsource/package内に検出しなかった。`docs/spec.md`等に存在する名称・短い要約・参照linkは独自のengineering documentationとして扱う。このinventoryとLicenseのMaterials定義に基づき、現行の公開source、wheel、schemaおよびdocumentationはH3 WorksではなくApache-2.0成果物と分類する。この分類は法的助言ではなく、新たにMiniMax Materialsをcopyまたはadaptした場合は再reviewする。

## Territory classification

| Location or flow | Current evidence | Decision |
|---|---|---|
| Developer location used to access or run H3 | project ownerがdevelopment machineとremote access元をJapanと申告 | Approved for declared Japan-local scope |
| GPU benchmark host | hardware evidenceに加え、project ownerがphysical countryをJapan、operatorを`nishide-dev`と申告 | Approved for declared Japan-local scope |
| CPU-only GitHub Actions runner | H3をdownload、import、実行せず合成fixtureだけを扱う | Not applicable |
| H3Fast source/artifact storage | 独立Apache-2.0成果物だけを保存する | Not applicable |
| Local benchmark Output storage | Japan-local research environmentに保持し、Gitと第三者accessから除外 | Approved for declared Japan-local scope |
| Public source/package distribution | H3 Worksを含まない独立codeだけを配布する | Not applicable |
| H3 runtime and Output use | `nishide-dev`によるJapan内のlocal実行・確認に限定し、第三者提供なし | Approved for declared Japan-local scope |

IP geolocation、checkbox、HF gate、timezoneは法的所在地や実利用地の十分な証明として扱わない。

GitHubの公式資料では、GitHub.com dataは既定で米国に保存される。現行CIのstandard Ubuntu runnerはAzure上で動作するが、`runs-on: ubuntu-latest`は特定regionを固定しない。現行workflowとrepositoryはH3をdownload、import、実行せず、H3 Materials、Output、benchmark artifactを保存しない。このためGitHub storage、CPU CIおよびglobal source accessにはH3のterritory decisionを適用しない。将来GPU CIやH3 artifact uploadを追加する場合は、この判断を継承せず対象flowをterritory inventoryへ追加または再分類する。

- [GitHub Enterprise Cloud with data residency](https://docs.github.com/en/enterprise-cloud@latest/admin/data-residency/about-github-enterprise-cloud-with-data-residency)
- [GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [GitHub Actions artifact and log retention](https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization)

## Approved H3-use scope and re-review conditions

現行approvalのsource of truthは[ADR 0007](../decisions/0007-japan-local-h3-use-scope.md)と[project-owner declaration](https://github.com/nishide-dev/h3fast/issues/11#issuecomment-5303694328)である。Japan内の宣言済みmachine/storageにおける`nishide-dev`のsingle-operator local researchだけを対象とする。

次の変更前にはinventoryを`incomplete`へ戻して再reviewする。

- Japan外からのaccess、実行、storageまたはOutput表示
- third-party access、Hosted Service、public demoまたはremote API
- H3 Works、Model Derivative、weight、benchmark artifactまたはOutputの提供・配布
- operator、machine、storage providerまたはphysical countryの変更

`h3fast compliance check-territories --record compliance/territories/initial-runtime.json`の成功はこの限定scopeだけを承認し、独立codeのrelease、formal quality、Support Tier、derivative distributionまたはserviceを承認しない。
