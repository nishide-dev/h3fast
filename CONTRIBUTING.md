# Contributing to H3Fast

貢献前に[`AGENTS.md`](AGENTS.md)と[`docs/spec.md`](docs/spec.md)を確認してください。

## Communication

- commitの件名、Issueタイトル、Pull Requestタイトルは英語にします。
- commit本文、Issue本文、Pull Request本文、review内容は日本語にします。
- Pull RequestタイトルはConventional Commits形式を推奨します。

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check src/
uv run pytest --cov=h3fast
uv build --no-sources
```

性能変更には固定baselineとのA/B測定、実行環境、raw result、品質回帰を含めてください。モデル重み、生成media、token、cacheをcommitしないでください。
