# H3 license and source boundary review

- Review date: 2026-08-16 (Asia/Tokyo)
- Status: Engineering evidence complete; legal and release approval blocked
- Tracking: [Issue #11](https://github.com/nishide-dev/h3fast/issues/11)
- Coordination owner: `nishide-dev`
- Legal reviewer: Unassigned
- Release approver: Unassigned
- Target decision date: 2026-08-31 (provisional)

この文書はengineering inventoryと一次資料の記録であり、法的助言または公開承認ではない。承認状態のsource of truthは[`compliance/release-gates/initial-runtime.json`](../../compliance/release-gates/initial-runtime.json)とし、ownerまたはevidenceがない項目を推測で承認済みに変更しない。

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
| `src/h3fast` | H3Fast独自のvalidation、diagnostics、benchmark orchestration。MiniMax source fileのcopyなし | Apache-2.0候補、legal classification待ち |
| `schemas` | H3Fast独自artifact/protocol/report/release contract | Apache-2.0候補 |
| `runtime/ffprobe.py` | 固定SGLang containerからmedia metadataを得るlocal adapter | Apache-2.0候補 |
| `benchmarks/*.yaml` | model ID、immutable revision、実測条件、H3Fast所有promptを記録 | prompt公開権とlicense boundary review待ち |
| exact quality reference | digest、stream metadata、制約のみ。H3 Output本体とlocal pathなし | metadata-only Public候補、legal review待ち |
| `docs` | 独自の設計・実測・一次資料要約。MiniMax Documentation fileのcopyなし | Apache-2.0候補、要約範囲のlegal review待ち |
| local H3 snapshot | Git管理外。H3 Worksとしてlocal-only、再配布禁止 | MiniMax H3 Community License対象 |
| benchmark media | Git管理外。H3 Outputとしてlocal-only、再配布禁止 | License/AUP review対象 |
| SGLang checkout/SIF | Git管理外。固定identityのみ記録 | 第三者license/SBOM review待ち |

文字列scanでは公式License本文の複製をsource/package内に検出しなかった。`docs/spec.md`等に存在する名称・短い要約・参照linkはengineering documentationであり、この結果だけで非侵害またはH3 Works非該当を保証しない。

## Territory inventory requiring evidence

| Location or flow | Current evidence | Decision |
|---|---|---|
| Developer location | timezone情報だけでは物理所在地を証明できない | Blocked |
| GPU benchmark host | hardware evidenceはあるが物理所在地・運用主体の記録なし | Blocked |
| GitHub Actions runner | workflowは`ubuntu-latest`だが実行regionを契約資料で固定していない | Blocked |
| GitHub source/artifact storage | public GitHub利用。保存・replication地域を確認していない | Blocked |
| Local benchmark Output storage | repository外であること以外の所在地・access control記録なし | Blocked |
| Public source distribution | global accessを前提とするため、H3 Works非該当判断が必要 | Blocked |
| Initial users and Output use | 対象利用者、所在地、実行地、Output利用地が未定 | Blocked |

IP geolocation、checkbox、HF gate、timezoneは法的所在地や実利用地の十分な証明として扱わない。

## Required decisions

Legal reviewerとrelease approverは、少なくとも次をevidence URL、decision timestamp、適用範囲とともにrecordへ記録する。

1. H3Fastの独自source、wheel、schema、documentationがMiniMax H3 Worksに該当しないか。該当する場合は適用条件と配布可能地域。
2. 開発、CI、storage、配布、実行、Output利用の全regionがApplicable Territoryと整合するか。
3. Excluded Territoryが含まれる場合、MiniMaxの個別書面licenseが必要か、その承認記録は何か。
4. License Q&Aの見出しとLicense本文のterritory表現差をどの根拠で解釈するか。
5. `h3fast`名称、MiniMax H3表示、NOTICE、downstream termsへ必要な対応。

これらが未承認の間、`h3fast release check --record compliance/release-gates/initial-runtime.json`は終了code 1を返す。終了codeを無視したreleaseを行わない。
