# H3 license and source boundary review

- Review date: 2026-08-16 (Asia/Tokyo)
- Status: Independent-code classification accepted; H3-use approval blocked
- Tracking: [Issue #11](https://github.com/nishide-dev/h3fast/issues/11)
- Coordination owner: `nishide-dev`
- Legal reviewer: Unassigned
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
| Developer location used to access or run H3 | timezone情報だけでは物理所在地を証明できない | Blocked |
| GPU benchmark host | hardware evidenceはあるが物理所在地・運用主体の記録なし | Blocked |
| CPU-only GitHub Actions runner | H3をdownload、import、実行せず合成fixtureだけを扱う | Not applicable |
| H3Fast source/artifact storage | 独立Apache-2.0成果物だけを保存する | Not applicable |
| Local benchmark Output storage | repository外であること以外の所在地・access control記録なし | Blocked |
| Public source/package distribution | H3 Worksを含まない独立codeだけを配布する | Not applicable |
| H3 runtime users and Output use | 対象利用者、所在地、実行地、Output利用地が未定 | Blocked |

IP geolocation、checkbox、HF gate、timezoneは法的所在地や実利用地の十分な証明として扱わない。

GitHubの公式資料では、GitHub.com dataは既定で米国に保存される。現行CIのstandard Ubuntu runnerはAzure上で動作するが、`runs-on: ubuntu-latest`は特定regionを固定しない。現行workflowとrepositoryはH3をdownload、import、実行せず、H3 Materials、Output、benchmark artifactを保存しない。このためGitHub storage、CPU CIおよびglobal source accessにはH3のterritory decisionを適用しない。将来GPU CIやH3 artifact uploadを追加する場合は、この判断を継承せず対象flowをterritory inventoryへ追加または再分類する。

- [GitHub Enterprise Cloud with data residency](https://docs.github.com/en/enterprise-cloud@latest/admin/data-residency/about-github-enterprise-cloud-with-data-residency)
- [GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [GitHub Actions artifact and log retention](https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization)

## Remaining H3-use decisions

H3-related flowのownerと必要なreviewerは、少なくとも次をevidence URL、decision timestamp、適用範囲とともにrecordへ記録する。

1. H3 snapshotへaccessするdevelopment host、GPU host、storage、実行、Output利用のregionがApplicable Territoryと整合するか。
2. Excluded Territoryが含まれる場合、MiniMaxの個別書面licenseが必要か、その承認記録は何か。
3. License Q&Aの見出しとLicense本文のterritory表現差をどの根拠で解釈するか。
4. H3 Worksを組み込むservice、派生成果物、Output、downstream termsへ必要な対応。
5. `H3Fast`名称が公式製品・提携製品との誤認を生まないための表示とtrademark対応。

これらが未承認の間、`h3fast compliance check-territories --record compliance/territories/initial-runtime.json`は終了code 1を返し、H3-related flowを新たに実行・提供する判断へ使用してはならない。独立codeの`h3fast release check`はterritory inventoryと別であり、quality、reproducibility、supply-chain等のremaining blockerだけで判定する。
