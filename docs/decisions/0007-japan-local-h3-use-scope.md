# ADR 0007: Limit approved H3 use to Japan-local research

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tracking:** [Issue #11](https://github.com/nishide-dev/h3fast/issues/11)
- **Declaration:** [Project-owner territory declaration](https://github.com/nishide-dev/h3fast/issues/11#issuecomment-5303694328)

## Context

MiniMax H3 Community LicenseはApplicable TerritoryをEU、英国、韓国、米国を除く世界と定義し、H3 WorksとOutputの利用、実行、保存、配布、表示をterritory条件の対象にしている。従来のinventoryはtimezoneやrunner名から所在地を推測せず、development host、GPU host、benchmark Output storage、runtime execution、Output useを`unknown`としていた。

Project ownerは、現行のdevelopment workspace、remote access元、GPU machineおよびH3関連artifactを扱うresearch environmentがJapan内にあり、operatorは`nishide-dev`であると申告した。現時点でH3 weights、H3 Works、benchmark Outputを第三者へ提供しない。

## Decision

現行Phase 1A/1BのH3利用を、`nishide-dev`がJapan内の宣言済みmachineとstorageで行うsingle-operator local researchに限定して、MiniMax H3 Community LicenseのApplicable Territory内として承認する。

この承認はproject ownerによるself-attestationであり、外部弁護士による法的意見ではない。独立したH3Fast source、CPU CI、code-only package配布はADR 0006どおりH3-use territory gateの対象外である。

次のいずれかを行う前にterritory inventoryを`incomplete`へ戻し、新しいoperator、location、artifact、recipientとlicense条件をreviewする。

- Japan外からのH3 access、実行、保存またはOutput表示
- 第三者へのH3 Works、weight、Model Derivative、benchmark artifactまたはOutputの提供
- Hosted Service、shared runtime、public demoまたはremote API
- public Output、派生weight、containerまたはH3 Materialsを含むartifactの配布
- operator、machine、storage providerまたはphysical countryの変更

## Consequences

`h3fast compliance check-territories`は、宣言済みのJapan-local scopeに対して成功できる。これはPublic Runtime release、Support Tier、formal quality equivalence、Hosted Serviceまたはderivative distributionの承認を意味しない。それらは個別のrelease、quality、supply-chain、downstream termsとterritory gateを引き続き要求する。
