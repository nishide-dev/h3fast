# H3Fast

H3Fastは、ローカルのMiniMax H3-Base推論を再現可能な方法で高速化・効率化するための研究・ランタイムプロジェクトです。

現在はPhase 1Aの基盤段階です。公式H3の重みやコードは含めず、次を提供します。

- ローカルH3 snapshotの構造・revision検証
- 派生成果物manifestとchecksumの検証
- Python、SGLang、GPU環境の診断
- 再現可能なbenchmark protocol、GPU preflight、非同期benchmark client
- CPU-only環境でimport可能な単一Python package

製品・配布仕様は[`docs/spec.md`](docs/spec.md)、開発規約は[`AGENTS.md`](AGENTS.md)を参照してください。

## Status

このrepositoryはPhase 1Aの実装段階です。内部実験として固定SGLang sourceとSingularity runtimeを使うbaseline harnessを提供しますが、モデル変換、Triton kernel、Hosted APIはまだ提供しません。2基のRTX 6000 Adaでは固定T2VA caseをwarmup 1回・測定3回完走し、client E2E p50 889.495秒を記録しました。単一caseのlocal baselineであり、一般的な品質、lossless性、Tier 1/2 supportを示す公開benchmarkではありません。

BYOWはH3の重みを再配布しない方式ですが、MiniMax H3 Community Licenseの地域・用途・Output等の制限を免除するものではありません。H3を取得・利用する前に、必ず最新の原文を確認してください。

## Development setup

Python 3.12と固定したuvを使用します。

```bash
uv sync --locked
uv run h3fast --help
```

## Commands

環境を診断します。

```bash
uv run h3fast doctor
uv run h3fast doctor --json
```

ローカルsnapshotを検証します。`--base-revision`には40文字のimmutable commit SHAが必要です。

```bash
uv run h3fast inspect-snapshot /models/MiniMax-H3 \
  --variant fl2va \
  --base-revision <immutable-hf-commit-sha>
```

派生成果物のmanifestとchecksumを検証します。

```bash
uv run h3fast verify-model /models/H3-Fast-FL2VA-FP8
```

benchmark protocolの構造を検証します。

```bash
uv run h3fast benchmark validate-protocol benchmarks/protocol.yaml
```

固定runtime、SGLang source、明示したローカルsnapshot、選択GPUを検査します。選択GPUにcompute processがある場合は失敗します。

```bash
uv run h3fast benchmark preflight \
  --snapshot models/MiniMax-H3 \
  --gpus 1,2 \
  --sglang-source runtime-cache/sglang \
  --runtime-image runtime-cache/sglang-v0.5.16-cu129-runtime.sif \
  --ffprobe-adapter runtime/ffprobe.py \
  --output benchmark-results/preflight.json
```

起動argvをJSONで確認できます。実験では`serve-guarded`を使用し、preflight後も選択GPUを監視します。起動したserver process tree以外のcompute processを検出するとserverを停止し、失敗理由をJSONへ記録します。これらはH3の利用権を判定せず、重みも取得しません。

```bash
uv run h3fast benchmark plan-launch \
  --snapshot models/MiniMax-H3 \
  --gpus 1,2 \
  --sglang-source runtime-cache/sglang \
  --runtime-image runtime-cache/sglang-v0.5.16-cu129-runtime.sif \
  --ffprobe-adapter runtime/ffprobe.py \
  --server-output outputs/server
```

次のcommandを実行し、`server-ready` eventが出るまで待ちます。

```bash
uv run h3fast benchmark serve-guarded \
  --snapshot models/MiniMax-H3 \
  --gpus 1,2 \
  --sglang-source runtime-cache/sglang \
  --runtime-image runtime-cache/sglang-v0.5.16-cu129-runtime.sif \
  --ffprobe-adapter runtime/ffprobe.py \
  --server-output benchmark-results/server \
  --preflight-output benchmark-results/preflight.json \
  --guard-report benchmark-results/server-failure.json \
  --lifecycle-report benchmark-results/server-lifecycle.json
```

準備完了後、別shellから規定caseを実行します。生成完了まで`serve-guarded`を終了しないでください。単一の互換性smokeには`run-case`、protocolのwarmup・測定回数を実行してstage metricsを集計する場合は`run-suite`を使います。

固定SGLang imageには`ffprobe` CLIが含まれないため、H3Fastはimage内のPyAVを使う限定的な互換adapterをread-only bindします。preflightはSIFとadapterのSHA-256をprotocolの固定値と照合し、launch planにもadapter digestを記録します。

```bash
uv run h3fast benchmark run-case \
  --case-id smoke-001 \
  --output-dir benchmark-results/smoke-001
```

```bash
uv run h3fast benchmark run-suite \
  --case-id smoke-001 \
  --output-dir benchmark-results/measured-baseline \
  --server-output benchmark-results/server \
  --server-lifecycle-report benchmark-results/server-lifecycle.json \
  --server-guard-report benchmark-results/server-failure.json
```

`run-suite`はSGLangのrequest ID付きperformance dumpを検証し、warmupを集計から除外して、測定runのmin／p50／p95／maxと支配的stageをJSONへ保存します。生成動画、個別result、server metricsおよびbundleは`benchmark-results/`以下のローカル成果物であり、Git管理しません。

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src/
uv run pytest --cov=h3fast
uv build --no-sources
```

## License

H3Fastが独自に実装したコードはApache License 2.0で提供します。このlicenseはMiniMax H3、Model Derivatives、重み、Outputその他のMiniMax Materialsに対する権利を付与しません。
