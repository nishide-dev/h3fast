# First optimization target

- Status: Accepted and measured
- Date: 2026-08-15
- Measured: 2026-08-16
- Related: Issues #3 and #7

## Context

固定BF16 baselineの測定3回では、server inference p50が`886.759141`秒、denoise p50が`847.339288`秒だった。denoiseはserver時間の約95.55%を占める。一方、text encode p50は`0.643419`秒、decode p50は`38.394567`秒であり、最初にこれらを最適化してもE2Eへの寄与は限定される。

baselineはDiT 50層のうち20層だけをGPU residentとし、残りをlayerwise offloadする。reported peak GPU memoryは最大`23,376` MiBだった。ただしこの値だけから追加resident layer数の安全上限は確定できず、process別peakと物理GPU全体の使用量を同一視してはならない。

## Decision

最初の単一最適化は、denoise中のDiT layerwise placementとする。

- baselineの数値演算、50-step schedule、seed、codec、TP2/Ulysses1、`torch.compile`設定は変更しない。
- 変更変数は`dit-layerwise-resident-layers`だけに限定する。
- 最初のcandidateは20層から40層への増加とし、preflight後のloadでOOMまたは安全余裕不足になった場合は結果を失敗として保存し、30層へ下げた別candidateとして扱う。失敗値を成功比較へ混ぜない。
- candidateごとにwarmup 1回、測定3回をbaselineと同じguard・case・集計方法で実行する。
- 採用判断はdenoiseとclient E2Eの両方が改善し、品質reference gateとmedia contractを通過することを必須とする。
- 改善しない、OOMになる、外部GPU processを検出する、または品質gateに失敗するcandidateは既定値にしない。

## Rationale

支配stageへ直接作用し、1変数のA/Bとして説明でき、重み変換や量子化を伴わないため、最初の最適化として切り分けやすい。固定caseのbaseline 4成果物はbitwise一致したため、このplacement-only candidateにはまずartifact完全一致を厳格gateとして適用できる。ただし、この1 caseの完全一致を一般的な品質保証へ拡張しない。

## Consequences

resident layer数をprotocolで設定し、launch plan、server lifecycle、suite bundleへ同じ実効値を記録する。protocolと起動済みserverの値が異なるsuiteは拒否する。baseline artifactから作るlocal quality reference gateがない状態ではcandidateの性能測定を採用しない。

この判断はTriton kernel、quantization、sparse attention、cache、compileを同時に導入しない。placement A/Bの結果が否定的でも、測定記録を残してbaselineへ戻す。

## Outcome

2×RTX 6000 Adaで40層candidateをwarmup 1回、測定3回実行した。20層baselineに対してclient E2E p50は`889.495172`秒から`883.515755`秒へ0.672%、denoise p50は`847.339288`秒から`842.506824`秒へ0.570%改善した。OOMと外部GPU processはなく、3 measured runすべてがexact decoded artifact gateを通過したため、事前条件に従い40層を既定protocolとして採用する。

代償としてreported peak GPU memoryの最大値は`23,376` MiBから`35,696` MiBへ12,320 MiB（52.704%）増えた。空きメモリ要件を満たせない環境では[`benchmarks/protocol-baseline20.yaml`](../../benchmarks/protocol-baseline20.yaml)を明示して20層へrollbackする。これは固定1 case、単一hostでの小幅な改善であり、Support Tierまたは一般的な性能主張には使用しない。詳細は[`docs/experiments/0005-rtx6000-ada-resident40.md`](../experiments/0005-rtx6000-ada-resident40.md)に記録する。
