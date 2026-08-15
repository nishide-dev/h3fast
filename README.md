# H3Fast

H3Fastは、ローカルのMiniMax H3-Base推論を再現可能な方法で高速化・効率化するための研究・ランタイムプロジェクトです。

現在はPhase 1Bの最初の実測最適化まで完了しています。公式H3の重みやコードは含めず、次を提供します。

- ローカルH3 snapshotの構造・revision検証
- 派生成果物manifestとchecksumの検証
- Python、SGLang、GPU環境の診断
- 再現可能なbenchmark protocol、GPU preflight、非同期benchmark client
- private registryからprompt/pathを除いたmetadataを生成するformal quality compilerとfail-closed検査
- 承認前にfail closedするmachine-readable release gate
- CPU-only環境でimport可能な単一Python package

製品・配布仕様は[`docs/spec.md`](docs/spec.md)、開発規約は[`AGENTS.md`](AGENTS.md)、Phase 1B完了監査は[`docs/audits/0001-phase1b-completion.md`](docs/audits/0001-phase1b-completion.md)を参照してください。

## Status

このrepositoryはPhase 1Bの実装段階です。内部実験として固定SGLang sourceとSingularity runtimeを使うbenchmark harnessを提供しますが、モデル変換、Triton kernel、Hosted APIはまだ提供しません。2基のRTX 6000 AdaではDiT resident layerを20から40へ増やした単一変数A/Bを実施し、client E2E p50を889.495秒から883.516秒へ、denoise p50を847.339秒から842.507秒へ改善しました。一方、reported peak GPU memoryは最大23,376 MiBから35,696 MiBへ増加しています。単一caseのlocal測定であり、一般的な品質、lossless性、Tier 1/2 supportを示す公開benchmarkではありません。

このrepositoryのsource、schema、CLIおよびH3 Materialsを含まないwheelは独立実装のApache-2.0成果物です。MiniMax公式プロジェクトまたは提携製品ではありません。一方、BYOWはH3の重みを再配布しない方式にすぎず、H3の取得・実行・変換・Outputに対するMiniMax H3 Community Licenseの地域・用途等の制限を免除しません。H3を取得・利用する前に、必ず最新の原文を確認してください。

Initial Runtimeのpackage releaseは品質、再現性、supply-chain等が未完了のため現在承認されていません。これは独立sourceのApache-2.0公開可否とは別です。H3-use approvalはJapan内の宣言済みmachine/storageで`nishide-dev`が行うlocal researchだけに限定され、第三者提供、Hosted Service、Japan外利用には引き継がれません。一次資料とsource boundaryは[`docs/compliance/h3-license-boundary-review.md`](docs/compliance/h3-license-boundary-review.md)、限定scopeは[`compliance/territories/initial-runtime.json`](compliance/territories/initial-runtime.json)、formal quality setは[`benchmarks/quality/formal-quality-set.json`](benchmarks/quality/formal-quality-set.json)、package release stateは[`compliance/release-gates/initial-runtime.json`](compliance/release-gates/initial-runtime.json)を参照してください。

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

Initial Runtime packageのrelease gateを検査します。現在のrecordは未指名approvalと未完了checkを含むため、JSON reportを出力して終了code 1を返すのが正常です。終了code 0は全required checkとrelease/schema approvalが証拠付きで完了した場合だけ返します。H3-use territory approvalはこのcode-only release gateには含めません。

```bash
uv run h3fast release check \
  --record compliance/release-gates/initial-runtime.json
```

H3 snapshotへのaccess、GPU実行、H3 artifact storage、runtime execution、Output利用のterritory inventoryを検査します。現在のrecordはJapan内の宣言済みmachine/storageで`nishide-dev`が行うsingle-operator local researchについて終了code 0を返します。H3を扱わないsource storage、CPU CI、global source/package accessは`not-applicable`です。このcommandは独立codeの配布、第三者提供、Hosted Service、Japan外利用または将来scopeの可否を判定しません。

```bash
uv run h3fast compliance check-territories \
  --record compliance/territories/initial-runtime.json
```

benchmark protocolの構造を検証します。

```bash
uv run h3fast benchmark validate-protocol benchmarks/protocol.yaml
uv run h3fast benchmark validate-protocol benchmarks/protocol-baseline20.yaml
```

formal quality setの件数、層化coverage、rights evidence、metric plan、approvalを検査します。現在のcommitted recordはprompt本文やmediaを含まず、caseと承認が未登録のため終了code 1を返します。Git管理外には10 smoke / 50 regressionのcandidate registryがあり、公開可能な全体digestと件数だけを[`formal-quality-registry-attestation.json`](benchmarks/quality/formal-quality-registry-attestation.json)へ記録しています。不正または矛盾するrecordは終了code 2です。

```bash
uv run h3fast benchmark check-quality-set \
  --record benchmarks/quality/formal-quality-set.json
```

権利review対象のpromptとreference assetはGit管理外のprivate registryへ置き、公開可能なdigest metadataだけを生成します。compilerはprompt本文、asset path、media本体を出力へコピーせず、reference assetをSHA-256で固定します。`--registry-uri`には権限制御されたregistryまたはprompt/mediaを含まないdigest attestationのimmutable HTTPS identityを指定し、出力はまず一時fileでsemantic validationしてからatomicに置換します。

```bash
uv run h3fast benchmark compile-quality-registry \
  --registry /secure/h3fast/quality.private-quality-registry.json \
  --registry-uri https://registry.example.invalid/h3fast/quality-v1 \
  --output /tmp/formal-quality-set.candidate.json
```

private registryのschemaは[`schemas/private-quality-registry.schema.json`](schemas/private-quality-registry.schema.json)です。registry、prompt、reference asset、per-case digestを含むcandidate formal recordをrights/quality approval前にcommitまたはuploadしてはなりません。全体registry digestだけのattestationは[`schemas/quality-registry-attestation.schema.json`](schemas/quality-registry-attestation.schema.json)に従い、本文、path、mediaを含めません。

`benchmarks/protocol.yaml`は実測で採用した40層resident設定です。`benchmarks/protocol-baseline20.yaml`は固定Phase 1A baselineと、メモリ不足時に明示的に選ぶrollback設定です。launch、guard、suiteはprotocolの実効値を共有し、server lifecycleとsuite bundleへ記録します。

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
  --protocol benchmarks/protocol.yaml \
  --snapshot models/MiniMax-H3 \
  --gpus 1,2 \
  --sglang-source runtime-cache/sglang \
  --runtime-image runtime-cache/sglang-v0.5.16-cu129-runtime.sif \
  --ffprobe-adapter runtime/ffprobe.py \
  --server-output benchmark-results/resident40-server
```

次のcommandを実行し、`server-ready` eventが出るまで待ちます。

```bash
uv run h3fast benchmark serve-guarded \
  --protocol benchmarks/protocol.yaml \
  --snapshot models/MiniMax-H3 \
  --gpus 1,2 \
  --sglang-source runtime-cache/sglang \
  --runtime-image runtime-cache/sglang-v0.5.16-cu129-runtime.sif \
  --ffprobe-adapter runtime/ffprobe.py \
  --server-output benchmark-results/resident40-server \
  --preflight-output benchmark-results/resident40-preflight.json \
  --guard-report benchmark-results/resident40-server-failure.json \
  --lifecycle-report benchmark-results/resident40-server-lifecycle.json
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
  --protocol benchmarks/protocol.yaml \
  --case-id smoke-001 \
  --output-dir benchmark-results/resident40 \
  --server-output benchmark-results/resident40-server \
  --server-lifecycle-report benchmark-results/resident40-server-lifecycle.json \
  --server-guard-report benchmark-results/resident40-server-failure.json
```

`run-suite`はSGLangのrequest ID付きperformance dumpを検証し、warmupを集計から除外して、測定runのmin／p50／p95／maxと支配的stageをJSONへ保存します。生成動画、個別result、server metricsおよびbundleは`benchmark-results/`以下のローカル成果物であり、Git管理しません。

測定3回がbitwise安定した`smoke-001`から、placement-only変更用のexact quality referenceを生成できます。referenceはcontainer、RGB24へdecodeした映像、PCMへdecodeした音声を別々にSHA-256化し、media metadataとA/V duration driftのp5／p50／p95を記録します。prompt本文、生成物、local pathは保存しません。

```bash
uv run h3fast benchmark build-quality-reference \
  --suite benchmark-results/measured-baseline/h3fast-phase1a-baseline-v1-smoke-001-suite.json \
  --protocol benchmarks/protocol-baseline20.yaml \
  --reference-id h3fast-phase1a-exact-smoke-001-v1 \
  --output benchmark-results/exact-smoke-001-reference.json
```

candidate suiteを固定referenceへ照合します。reportにはrun別の映像・音声・container check、candidate分布、worst caseを保存します。

```bash
uv run h3fast benchmark check-quality \
  --reference benchmarks/quality/exact-smoke-001-reference.json \
  --suite benchmark-results/resident40/h3fast-phase1b-resident40-v1-smoke-001-suite.json \
  --protocol benchmarks/protocol.yaml \
  --output benchmark-results/resident40/quality-report.json
```

上記は採用した40層candidateへreferenceを適用した検証済みcommandです。baselineを再測定する場合はsuiteとprotocolの両方を20層用へ切り替えます。

このgateは固定1 caseのplacement-only回帰検出に限定します。formal quality-set recordの存在だけでも、10件以上のsmoke set、50件以上のregression set、知覚品質指標、一般的なlossless性やSupport Tierを示しません。referenceとcandidateは同じ`ffmpeg`／`ffprobe` versionを使用する必要があります。

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
