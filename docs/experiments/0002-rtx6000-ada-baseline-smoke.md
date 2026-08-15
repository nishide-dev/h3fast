# RTX 6000 Ada guarded baseline smoke

- Date: 2026-08-15 (Asia/Tokyo)
- Protocol: `h3fast-phase1a-baseline-v1`
- Related: Issue #1
- Outcome: Completed one E2E smoke case

## Purpose

固定したMiniMax H3 FL2VA snapshotとSGLang runtimeで、2基のRTX 6000 Ada Generationを使うT2VA requestをAPI送信からMP4 downloadまで完走させる。これはruntime互換性とPhase 1A harnessのsmoke testであり、3回測定による性能baselineや品質gateではない。

## Fixed inputs

- H3 revision: `42ed227ee7df40d41602854ae760620d6eb651fe`
- H3 snapshot: 84 files / `144051241571` bytes
- SGLang revision: `6eb941a34cb100b708a42ed1d26d2bdefafbd01e`
- Base image: `lmsysorg/sglang@sha256:29f0f645122be1799a594c15907d81da326dbbe6ccd6395710a07a4292125a5f`
- SIF SHA-256: `c20b1b3c7da5b164d2783859937e4189c6265a2ad0b4f7be4b7329b7feebd2a4`
- ffprobe adapter SHA-256: `f69a957140cce3d55be043fb4f89cb7cdbf18fd45cf90ea4d4dc6b0d9daca8d0`
- Runtime: PyTorch `2.11.0+cu129`, CUDA `12.9`, NCCL `2.28.9`, PyAV `16.1.0`
- GPU: RTX 6000 Ada Generation 48GB × 2、driver `555.58.02`
- Topology: TP2、Ulysses1、memory mode、DiT resident 20 layers、layerwise offload、`torch.compile`無効
- Case: 1344×768、124 frames、24 fps、50 inference steps、seed `1101`

## Procedure

1. `serve-guarded`でpreflightを実行し、GPU 1・2が空いていること、snapshot、source revision、SIF、media probe、RAM、storageを検証した。
2. model loadからrequest完了まで、server process tree以外のGPU compute processを2秒間隔で監視した。
3. `/health`が200になるまで待ち、`run-case`でprotocolの`smoke-001`だけを送信した。
4. APIが`completed`を返した後にcontent endpointからMP4をdownloadし、SHA-256を計算した。
5. hostの独立した`ffprobe 4.4.2`でstream構成を再検証し、中間frameをローカルで目視した。
6. serverを停止し、選択GPU上からH3Fast processが解放されたことを確認した。

## Observations

- 全6 componentのload: 約9分7秒
- DiT 61.7 GiB CPU staging: 356.32秒、177.4 MiB/s
- server warmup: 76.80秒
- text encode: 0.6515秒
- latent preparation: 0.0203秒
- denoise: 841.1566秒
- decode: 38.2359秒
- SGLang pixel generation: 881.29秒
- client E2E: 883.5091秒
- reported peak GPU memory: 23,376 MiB
- foreign GPU process detection: なし
- guard failure report: なし

## Artifact validation

- MP4 size: `912408` bytes
- MP4 SHA-256: `530a6bc980cf357d0518ea60366bc09dd1868ee77f46ee3f43681ca8822f5909`
- Video: H.264、yuv420p、1344×768、24 fps、124 frames、5.166667秒
- Audio: AAC、32,000 Hz、stereo、5.175000秒
- Container: MP4 family、5.207000秒
- A/V duration drift: 0.008333秒
- Visual sanity: 中間frameに破損・黒画面はなく、固定promptと整合する街路の夜明けを確認した

MP4、result JSON、preflight JSON、抽出frameはH3 Outputまたはlocal pathを含むためGit管理せず、`benchmark-results/`または一時領域に保持する。

## Interpretation

固定したSGLang source/runtimeと2×RTX 6000 Adaで、H3 FL2VAのT2VA E2E互換性を1 case確認できた。固定imageには`ffprobe` CLIがないため、image内PyAVを使う限定adapterをread-only bindし、SGLangの最終media contractを通過させた。host `ffprobe`による独立検証も同じstream metadataを確認した。

この結果は単一のsmoke runであり、性能分布、品質回帰、lossless性、Tier 1/2 supportを示さない。protocolの`measured_runs: 3`とquality reference setは未実施で、protocol statusは`draft`のままとする。

## Follow-up

- stage timingを機械可読に収集し、warm serverで規定3回を測定する。
- quality reference setと比較方法を確定し、映像・音声の回帰gateを追加する。
- 最適化前にbaseline profileから支配的bottleneckを特定する。
