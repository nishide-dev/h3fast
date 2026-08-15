# H3Fast

H3Fastは、ローカルのMiniMax H3-Base推論を再現可能な方法で高速化・効率化するための研究・ランタイムプロジェクトです。

現在はPhase 1Aの基盤段階です。公式H3の重みやコードは含めず、次を提供します。

- ローカルH3 snapshotの構造・revision検証
- 派生成果物manifestとchecksumの検証
- Python、SGLang、GPU環境の診断
- 再現可能なbenchmark protocolの検証基盤
- CPU-only環境でimport可能な単一Python package

製品・配布仕様は[`docs/spec.md`](docs/spec.md)、開発規約は[`AGENTS.md`](AGENTS.md)を参照してください。

## Status

このrepositoryは実装初期段階です。モデル変換、生成、Triton kernel、Hosted APIはまだ提供しません。性能や品質に関する公開結果もまだありません。

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
