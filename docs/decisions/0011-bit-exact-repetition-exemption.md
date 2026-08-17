# ADR 0011: Allow a bit-exact determinism exemption for repetitions

- **Status:** Approved by project owner (2026-08-17)
- **Date:** 2026-08-17
- **Related:** [Issue #16](https://github.com/nishide-dev/h3fast/issues/16), [ADR 0008](0008-formal-quality-metric-plan.md), [experiment 0008](../experiments/0008-formal-generation-determinism.md)

## Context

ADR 0008のformal metric planは`baseline_repetitions: 3`と`candidate_repetitions: 3`を固定した。目的はbaseline自己変動をmetric budget設定前に実測することであり、「変動が0であると仮定しない」という設計意図に基づく。

t2va 20 caseの実測で、両protocolのrep1とrep2の生成物SHA-256が全40比較で一致し、生成がbit-exact決定的であることが証明された（[experiment 0008](../experiments/0008-formal-generation-determinism.md)）。この条件下では3反復目は同一bytesの再生成にしかならず、約27 GPU時間を消費して新しい情報を生まない。

一方、単に反復数を2へ下げると、bit-exactが成立しない最適化（量子化、step蒸留、量子化attention等）でも反復が不足する。反復要件そのものを緩めるのではなく、証明を伴う例外として扱う必要がある。

## Decision

metric planの`evaluation`へ任意フィールド`deterministic_generation_exemption`を追加する。`baseline_repetitions`と`candidate_repetitions`は3のまま変更しない。

例外recordは次を必須とする。

- `policy`: `bit-exact-digest-match-v1`固定
- `verified_repetitions`: 2以上の整数（digest一致を確認した独立反復数）
- `owner`: 証明をreviewしたowner
- `verified_at`: ISO 8601日付
- `evidence`: per-case digest一致を記録したevidenceの配列（1件以上）

validatorとschemaはこれらの欠落、policy不一致、反復数不足、空evidenceを拒否する。フィールドを持たないplanは従来どおり3反復を要求する。committed planは現時点でこのフィールドを持たない。

例外の適用範囲は、証明された条件（runtime、driver、GPU構成、schedule、precision、attention backend、生成parameter）に限る。いずれかが変わる場合、または数値を変える最適化を評価する場合は、例外を適用せず実測する。1 caseでもdigest不一致があれば例外は成立しない。

## Consequences

placement-only最適化の判定はdigest照合で完結し、bit-exact条件下での余剰反復を省略できる。今回はrep3を打ち切り、約27 GPU時間を他の作業へ回した。

例外はmetric実装、budget、owner approvalのいずれも代替しない。数値を変える最適化候補ではmetric familyごとの実測が引き続き必要であり、その時点でbudget承認が要件となる。

fl2va/ref2vaの40 caseは決定性未測定であり、この例外の証明範囲に含まれない。
