# H3Fast

> **Status: active development / not released.** APIとschemaは予告なく変更されます。
> 公開wheelもcontainerも配布しておらず、単一operatorのlocal研究用途を前提としています。

H3FastはローカルのMiniMax H3-Base推論を高速化する研究プロジェクトです。公式H3の重みやコードは含みません（Bring Your Own Weights）。

速度を出すこと自体より、**「速くした」と証拠つきで言える状態を保つこと**を重視しています。各最適化はpinned protocolで固定し、A/B測定と品質判定を経てから既定へ入ります。

## 測定結果

2×RTX 6000 Ada（48 GB、sm89）、TP2での実測です。FlashAttention 50-step baselineに対し、t2va formal 20 caseで**約6.9倍**（14.0時間 → 2.04時間）。その後のVAE常駐化でsmoke-001のE2Eがさらに15.8%短縮しています（20 case総計での再測定は未実施）。

| 段 | 最適化 | 効果 | 品質判定 |
|---|---|---|---|
| 1 | DiT resident 40層 | — | bit-exact（digest全一致） |
| 2 | Sage Attention（INT8） | 1.63× | 劣化なし（pairwise +0.20） |
| 3 | Turbo LoRA 11 step | 3.86× | 引き分け（6勝6敗8tie） |
| 4 | online FP8 | 4.22× | 劣化なし（+0.20）、VRAM −22% |
| 5 | video VAE常駐 | E2E 15.8%短縮 | bit-exact（digest一致） |

品質判定はblind human-pairwise（single-reviewer、[ADR 0010](docs/decisions/0010-human-pairwise-review-policy.md)）が一次で、objective metricは証跡として記録します。数字の根拠と限界は各[experiment記録](docs/experiments/)にあります。

### 生成profile

用途に応じて4段階から選びます。既定は`balanced`です。

| profile | vs quality | pairwise | 用途 |
|---|---|---|---|
| `quality` | 1.00× | +0.20 | 品質基準・数値再現性が要る場合 |
| `bf16-balanced` | 3.86× | 0.00 | BF16 weightが要る場合 |
| **`balanced`（既定）** | **4.22×** | +0.20 | 通常運用 |
| `speed` | 4.94× | −0.25 | 高速反復（劣化あり） |

```bash
uv run h3fast benchmark profiles
```

profileの選択は明示指定のみで、暗黙に切り替わりません。

## 到達した上限

sm89 / 48 GB × 2 での探索は尽きています。denoise内部のkernel分布を測定した結果、最大の単一要因は計算ではなく**TP2のNCCL AllReduce（24.5%）**でした（[experiment 0018](docs/experiments/0018-denoise-kernel-profile.md)）。

| 試した経路 | 結果 |
|---|---|
| kernel融合 | 既存fast pathが全て機能中。新規融合の根拠なし |
| VAE placement | **有効だった**（decode 65%短縮、[experiment 0020](docs/experiments/0020-vae-residency.md)） |
| TP1 / Ulysses（通信削減） | transformer weightが分割されず48 GBに載らない |
| `subblock_sparse` | compute capability 9.0/10.0のみ。sm89は拒否 |
| NVFP4 | capability 100（Blackwell専用） |
| GGUF / 事前量子化`.pt` | H3 DiTはsafetensorsのみ読む |
| FP8 text encoder | 効果ゼロ（VRAM使用は3.10 GBで制約要因ではない） |

AllReduce 24.5%は48 GB GPUでH3を動かすための構造的コストです。詳細は[experiment 0019](docs/experiments/0019-parallelism-and-quantization-limits.md)にあります。

## 提供するもの

- ローカルH3 snapshotの構造・revision検証、manifest/checksum検証
- 再現可能なbenchmark protocol、GPU preflight、guarded server、非同期client
- **fail-closedな実行検証**: attention backendの実解決、LoRA適用、served variant、量子化設定をprotocolと突き合わせ、silentなfallbackを拒否
- blind human-pairwise runner（blind staging、assignment封印、集計）
- objective metric adapter 3種（LPIPS、隣接frame LPIPS trajectory、SigLIP2）
- 承認前にfail closedするmachine-readable release gate
- CPU-only環境でimport可能な単一Python package

silentなfallbackを拒否する仕組みは実際に機能しました。Sage Attentionはログ上「有効」と表示されながら実行されておらず、生成物digestの照合で初めて判明しています（[experiment 0009](docs/experiments/0009-sage-attention-noop.md)）。

## Setup

Python 3.12とuvを使用します。

```bash
uv sync --locked
uv run h3fast --help
```

主要な入口は次の3つです。個別のオプションは`--help`を参照してください。

```bash
# 環境診断
uv run h3fast doctor

# snapshot検証（--base-revisionは40文字のimmutable commit SHA）
uv run h3fast inspect-snapshot /models/MiniMax-H3 --variant fl2va --base-revision <sha>

# guarded serverの起動からformal case生成まで
uv run h3fast benchmark serve-guarded --protocol benchmarks/protocol-turbo12-fp8.yaml ...
uv run h3fast benchmark run-formal-cases --task t2va ...
```

## Quality checks

```bash
uv sync --locked --all-groups
uv run pytest --cov=h3fast
uv run ruff format --check .
uv run ruff check .
uv run ty check src/
uv build --no-sources
```

## License and scope

このrepositoryのsource、schema、CLI、およびH3 Materialsを含まないwheelは独立実装のApache-2.0成果物です。MiniMax公式プロジェクトまたは提携製品ではありません。

**BYOWはH3の重みを再配布しない方式にすぎず、H3の取得・実行・変換・Outputに対するMiniMax H3 Community Licenseの地域・用途等の制限を免除しません。** H3を取得・利用する前に最新の原文を確認してください。

Initial Runtimeのpackage releaseは品質・再現性・supply-chainが未完了のため承認されていません。H3-use approvalはJapan内の宣言済みmachine/storageで`nishide-dev`が行うlocal researchに限定され、第三者提供、Hosted Service、Japan外利用には引き継がれません。

- 製品・配布仕様: [`docs/spec.md`](docs/spec.md)
- 設計判断: [`docs/decisions/`](docs/decisions/)
- 実測記録: [`docs/experiments/`](docs/experiments/)
- license境界: [`docs/compliance/h3-license-boundary-review.md`](docs/compliance/h3-license-boundary-review.md)
- territory scope: [`compliance/territories/initial-runtime.json`](compliance/territories/initial-runtime.json)
- release gate: [`compliance/release-gates/initial-runtime.json`](compliance/release-gates/initial-runtime.json)
