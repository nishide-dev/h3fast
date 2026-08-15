# First optimization target

- Status: Accepted for the next Phase 1B experiment
- Date: 2026-08-15
- Related: Issue #3

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

resident layer数を設定・記録できるlaunch interfaceと、baseline artifactから作るlocal quality reference gateを先に実装する。品質gateがない状態ではcandidateの性能測定を採用しない。

この判断はTriton kernel、quantization、sparse attention、cache、compileを同時に導入しない。placement A/Bの結果が否定的でも、測定記録を残してbaselineへ戻す。
