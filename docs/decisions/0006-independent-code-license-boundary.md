# ADR 0006: Separate independent code from H3-use licensing

> **Operational state update:** H3-use territory inventoryのJapan-local限定承認は[ADR 0007](0007-japan-local-h3-use-scope.md)で行った。本ADRの独立code境界は引き続き有効である。

- Status: Accepted
- Date: 2026-08-16
- Owner: `nishide-dev`
- Evidence: [H3 license boundary review](../compliance/h3-license-boundary-review.md)

## Context

MiniMax H3 Community Licenseの`Materials`は、MiniMaxが公開したH3本体、software、algorithm、weight、inference-supporting codeおよびDocumentationを対象とする。`MiniMax H3 Works`はMaterials、Model Derivativesおよびそれらの派生物である。一方、H3Fastの公開repositoryとbuild済みwheelにMiniMaxのsource、weight、configurationまたはDocumentationのcopyは検出されていない。

従来のrelease recordは、独立実装のsource保存・CPU CI・global配布と、H3 snapshotの利用・GPU実行・Output利用を同じterritory approvalでblockしていた。この扱いでは、H3 Worksを含まないApache-2.0 codeまでH3の地域制限対象であるかのように扱われる。

## Decision

H3Fastが所有し独立実装したsource、schema、CLI、wheel、sdist、合成test fixtureおよびH3 Worksを含まないmetadataはApache-2.0成果物として分類する。H3専用またはH3互換であることだけを理由にMiniMax H3 Worksへ分類しない。

次のflowにはMiniMax H3 Community Licenseのterritory decisionを適用しない。

- H3 Materialsを扱わないsource storageとpublic source distribution
- H3をdownload、importまたは実行しないCPU-only CI
- H3 Materialsを含まないwheel、sdistおよびその配布
- H3 Output本体を含まない独自schema、protocolおよびmetadata

次のflowは引き続きH3 license、Applicable TerritoryおよびAUPの確認対象とする。

- H3 snapshotの取得、保存、検証、変換または実行
- MiniMax source、configurationまたはDocumentationのcopy・改変
- Model Derivative、量子化weight、LoRA、cacheその他の派生成果物
- H3を用いるGPU benchmarkと、そのOutputの保存、表示または配布
- H3 Worksを組み込む製品、serviceまたはHosted Service

独立codeのrelease gateはterritory inventoryの完了を要求しない。territory inventoryはH3-related flowを実行または提供する前の別gateとしてfail closedを維持する。

## Guardrails

- 新たにMiniMax Materialsを取り込む変更は、この分類を自動的に継承しない。artifact registerとboundary reviewを同じ変更で更新する。
- 公式実装を参照してkernel、adapterまたはconverterを作る場合は、copy・adaptationの有無とprovenanceを記録する。
- BYOWはH3の利用権、地域制限、AUPまたはOutput制限を変更しない。
- 公開packageはH3 weight、Model Derivative、Outputまたはlocal model pathを含めない。
- `H3Fast`および`MiniMax H3`の表示は、公式製品または提携製品との誤認を避ける。
- SGLangその他の第三者成果物には、それぞれのlicenseとnoticeを適用する。

## Consequences

GitHub source storage、global source access、CPU-only Actionsおよびcode-only package accessはterritory inventoryで`not-applicable`になる。本ADR採択時に未確認だったdevelopment host、GPU host、runtime execution、Output storageおよびOutput useは、後続のADR 0007でJapan-local single-operator researchに限定して承認した。

Initial Runtime releaseはterritory approvalではblockしないが、quality set、clean-machine reproduction、supply-chain、artifact notice、support target、converterおよびGPU E2E等のrelease gateは引き続き満たす必要がある。
