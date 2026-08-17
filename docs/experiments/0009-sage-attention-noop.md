# Sage Attention is selected but never executed for the H3 DiT

- Date: 2026-08-18 (Asia/Tokyo)
- Baseline protocol: `h3fast-phase1b-resident40-v1`（DiT resident 40層、FlashAttention）
- Candidate protocol: `h3fast-phase1b-sage-attn-v1`（同一構成、DiTのattention backendのみsage_attn）
- Host: 承認済みJapan-local GPU host（2×RTX 6000 Ada）
- Related: [Issue #40](https://github.com/nishide-dev/h3fast/issues/40), [ADR 0012](../decisions/0012-tiered-optimization-verification.md)
- Outcome: Sage kernelは一度も実行されない。Tier 2評価は成立せず、候補をblockedへ戻す。

## Purpose

ADR 0012の最初のTier 2候補としてSage Attention（INT8量子化attention）の速度と品質を測定する。

## Method

pinned SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`と、SGLangが要求するSageAttention commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`のAda（SM89）buildを使用した。runtime imageへ同梱せず外部pathをbindしPYTHONPATHで注入する。

DiTだけへ適用するため`--component-attention-backends.transformer=sage_attn`を用いた（server全体へ適用するとtext_encoderのattention layerが`sage_attn`を拒否して起動失敗する）。

固定条件はH3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`、TP2、BF16、50 steps、seed 12000、smoke-001、768p、4秒である。

`sitecustomize.py`で`sageattention.sageattn`とbackendが直接importした束縛の両方をラップし、呼び出し回数を記録する使い捨てprobeを用いた。

## Results

| 観測項目 | 結果 |
|---|---|
| serverログ | `Using sage_attn attention backend`、`Using sage_attn backend for component: transformer` |
| `sageattn` 実呼び出し回数 | **0** |
| 生成物SHA-256 | `748134a32a6cddfd…`（FA baselineとbit単位で一致） |
| E2E所要 | 716.4秒 vs FA 762.7秒（1.06倍） |

## Interpretation

Sage kernelは実行されていない。生成物がbit単位で一致することと呼び出し回数0は同じ結論を指し、1.06倍の所要差は測定誤差の範囲である。

serverログの「backend有効化」表示はbackendオブジェクトの生成を報告するものであり、推論経路で実際に使用された証拠ではない。

## Ruled out

- **kernel不良ではない**: Ada build済みSageAttentionはH3相当形状（S=2048、H=56、head_dim=128、NHD、bf16）でSDPA参照と異なる出力を返す（max_err 0.0156、mean_err 0.0011、NaN/Infなし）。
- **暗黙のfallbackではない**: 基底DiTの`_supported_attention_backends`はSAGE_ATTNを含み、selectorは要件不足時にValueErrorを送出する設計である。
- **text_encoder制約ではない**: DiTへスコープした状態での結果である。
- **backend実装の分岐差ではない**: `forward_varlen`の両分岐がいずれも`sageattn`を呼ぶ。

推定原因は、H3 DiTが`_attention_impl`を設定する経路と実際にforwardで使用する経路の間でcomponent単位のbackend制約が反映されていないことだが、未確定である。詳細と追跡は[Issue #40](https://github.com/nishide-dev/h3fast/issues/40)で扱う。

## Consequences

Sage AttentionのTier 2評価は成立しない。候補としてはblockedへ戻し、原因が解消するまでformal setによるmetric実測へ進まない。

`benchmarks/protocol-sage.yaml`とlaunch/protocolの`attention_backend`対応はrepositoryへ残す。実行基盤としては正しく動作しており、upstream側が解決した時点で再測定に使える。既定`auto`では従来のpinned argvと同一の起動を保つため、既存protocolの再現性には影響しない。

## Limits

- 単一case、単一seed、単一構成での観測である。
- pinned SGLang commit固有の問題であり、他versionでの挙動は未確認である。
- upstream実装の内部原因は特定していない。

## Lesson

数値を変えるはずの最適化でdigestが一致した場合、それは「品質劣化ゼロ」ではなく「最適化が効いていない」ことを疑う。ADR 0012のTier判定でdigest照合を先に行う運用が、速度計測だけでは見逃していた誤りを捕捉した。
