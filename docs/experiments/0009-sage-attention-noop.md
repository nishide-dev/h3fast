# Sage Attention comparison does not execute: evaluation blocked, cause unconfirmed

- Date: 2026-08-18 (Asia/Tokyo)
- Baseline protocol: `h3fast-phase1b-resident40-v1`（DiT resident 40層、FlashAttention）
- Candidate protocol: `h3fast-phase1b-sage-attn-v1`（同一構成、DiTのattention backendのみsage_attn）
- Host: 承認済みJapan-local GPU host（2×RTX 6000 Ada）
- Related: [Issue #40](https://github.com/nishide-dev/h3fast/issues/40), [ADR 0012](../decisions/0012-tiered-optimization-verification.md)
- Outcome: Sage kernelは一度も実行されず比較が成立しない。原因は未確定で、SGLang upstream regressionが有力候補。評価不能として扱う。

## Purpose

ADR 0012の最初のTier 2候補としてSage Attention（INT8量子化attention）の速度と品質を測定する。

## Method

pinned SGLang commit `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`と、SGLangが要求するSageAttention commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`のAda（SM89）buildを使用した。runtime imageへ同梱せず外部pathをbindしPYTHONPATHで注入する。

DiTだけへ適用するため`--component-attention-backends.transformer=sage_attn`を用いた（server全体へ適用するとtext_encoderのattention layerが`sage_attn`を拒否して起動失敗する）。

固定条件はH3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`、TP2、BF16、50 steps、seed 12000、smoke-001、768p、4秒である。

`sitecustomize.py`で`sageattention.sageattn`とbackendが直接importした束縛の両方をラップし、呼び出し回数を記録する使い捨てprobeを用いた。

**この計測手法には欠陥がある。** SGLangはTP2でworker processを分離するため、`sitecustomize.py`はprobeを読み込んだprocessにしか適用されず、実際に推論を行うworkerへ届いていない可能性が高い。後続の4段階probeでは`get_attn_backend`の呼び出しすら記録されず（必ず呼ばれるはずの経路）、probeが推論processへ適用されていないことが裏づけられた。したがって`sageattn`呼び出し0回は「使われていない証拠」ではなく「計測できていない」可能性がある。

一方、生成物SHA-256がFA baselineとbit単位で一致する事実はprobeに依存しない一次観測であり、Sage比較が成立していないという結論自体は変わらない。

## Results

| 観測項目 | 結果 |
|---|---|
| serverログ | `Using sage_attn attention backend`、`Using sage_attn backend for component: transformer` |
| `sageattn` 実呼び出し回数 | **0**（ただし計測手法に欠陥あり。下記参照） |
| 生成物SHA-256 | `748134a32a6cddfd…`（FA baselineとbit単位で一致） |
| E2E所要 | 716.4秒 vs FA 762.7秒（1.06倍） |

## Interpretation

Sage kernelは実行されていない。生成物がbit単位で一致することと呼び出し回数0は同じ結論を指し、1.06倍の所要差は測定誤差の範囲である。

serverログの「backend有効化」表示はbackendオブジェクトの生成を報告するものであり、推論経路で実際に使用された証拠ではない。

したがってこの測定は「SageAttentionの効果が小さい」ことを示さない。**SageAttention比較が成立していない**、すなわち評価不能である。速度・品質のいずれについてもSageAttentionの性能を主張してはならない。

## Ruled out

- **kernel不良ではない**: Ada build済みSageAttentionはH3相当形状（S=2048、H=56、head_dim=128、NHD、bf16）でSDPA参照と異なる出力を返す（max_err 0.0156、mean_err 0.0011、NaN/Infなし）。
- **公式サポート外ではない**: SGLang公式ドキュメント（`docs/sglang-diffusion/attention_backends`）は`sage_attn`をCUDA/MUSA対応backendとして記載し、本実験が使用したcommit `d9704247…`を推奨installとして明示している。
- **暗黙のfallbackではない**: 基底DiTの`_supported_attention_backends`はSAGE_ATTNを含み、selectorは要件不足時にValueErrorを送出する設計である。
- **text_encoder制約ではない**: DiTへスコープした状態での結果である。
- **backend実装の分岐差ではない**: `forward_varlen`の両分岐がいずれも`sageattn`を呼ぶ。

推定原因は、H3 DiTが`_attention_impl`を設定する経路と実際にforwardで使用する経路の間でbackend選択が接続されていないことだが、**未確定である**。

現時点では「H3Fast固有の問題でSGLang側は正常」とは判断できない。むしろupstream regressionが有力候補である。pinned commit `6eb941a3…`は、H3のattention admissionをbackend capability由来へ変更したupstream PR #33707（`AttentionRequirements(packed_varlen=True)`導入、`_attention_impl`経由実行への変更）を含み、Diffusion backend fallback scopeを修正したPR #34891より前の状態にある。疑っている接続部分はPR #33707の変更対象そのものである。

原因の確定にはrequested / selected / installed / executedの分離計測、全`MiniMaxH3Attention._attention_impl`の直接観測、4 revision比較、H3Fastを介さないdirect SGLang最小再現、CUDA kernel traceが必要である。詳細な調査計画と追跡は[Issue #40](https://github.com/nishide-dev/h3fast/issues/40)で扱う。

## Consequences

Sage AttentionのTier 2評価は成立しない。候補としてはblockedへ戻し、「効果なし」ではなく「現在のH3/SGLang構成では評価不能または未接続」と記録する。再開条件（requested/selected/installed/executedの一致、Sage `forward_varlen`のcall count、CUDA traceでのkernel確認、FA baselineとの非bit一致、warmup後の複数run、SGLang/SageAttention revisionのmanifest固定）はIssue #40に定める。

`benchmarks/protocol-sage.yaml`とlaunch/protocolの`attention_backend`対応はrepositoryへ残す。実行基盤としては正しく動作しており、upstream側が解決した時点で再測定に使える。既定`auto`では従来のpinned argvと同一の起動を保つため、既存protocolの再現性には影響しない。

## Limits

- 単一case、単一seed、単一構成での観測である。
- pinned SGLang commit固有の問題であり、他versionでの挙動は未確認である。
- upstream実装の内部原因は特定していない。
- **本実験は非公式なcomponent指定書式を用いた。** 公式ドキュメントが示す書式は`--component-attention-backends transformer=sage_attn`（space区切りの`component=backend`）であるのに対し、本実験は`--component-attention-backends.transformer=sage_attn`（dot区切り）を用いた。後者はSGLangがunknown argsから拾う互換経路で`server_args`へは反映されたが、実装経路まで正しく届くかは未検証である。公式書式での再測定が必要である。
- installed（各`MiniMaxH3Attention._attention_impl`の実class）は未計測である。

## Lesson

外部componentを扱う際は、公式ドキュメント、次にIssue/PR、最後にsourceの順で確認する。source読解から使用方法を推測すると、実装は分かっても設計意図・既知の問題・正しい書式は分からない。本実験では公式ドキュメントの`attention_backends`ページを確認しないまま非公式書式で測定し、さらにSageAttentionが公式サポート対象であることを一時的に誤認した。

数値を変えるはずの最適化でdigestが一致した場合、それは「品質劣化ゼロ」ではなく「最適化が効いていない」ことを疑う。ADR 0012のTier判定でdigest照合を先に行う運用が、速度計測だけでは見逃していた誤りを捕捉した。

さらに、backend選択ログはrequestedまたはselectedを示すだけで、installedやexecutedの証拠にならない。benchmarkはrequested / selected / installed / executedの一致を検証し、実行0回をwarningではなくfailureとして扱うべきである（防御的変更をIssue #40で追跡）。
