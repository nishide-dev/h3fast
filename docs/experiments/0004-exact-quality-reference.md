# Exact quality reference validation

- Date: 2026-08-15 (Asia/Tokyo)
- Reference: `h3fast-phase1a-exact-smoke-001-v1`
- Method: `exact-decoded-artifact-v1`
- Related: Issue #5
- Outcome: Baseline measured 3 runs passed

## Purpose

固定BF16 baselineのmeasured 3成果物から、次のplacement-only A/Bに使うexact referenceを生成し、同じsuiteをcandidateとして照合して映像・音声・containerのgate全体を実データで検証する。

## Environment

- `ffmpeg 4.4.2-0ubuntu0.22.04.1`
- `ffprobe 4.4.2-0ubuntu0.22.04.1`
- Video decode: RGB24 rawvideo
- Audio decode: PCM signed 16-bit little-endian、32,000 Hz、stereo
- Source: `h3fast-phase1a-baseline-v1` / `smoke-001` measured 3 runs

## Reference observations

3 runすべてで次が一致した。

- MP4 SHA-256: `530a6bc980cf357d0518ea60366bc09dd1868ee77f46ee3f43681ca8822f5909`
- MP4 size: `912408` bytes
- Decoded video SHA-256: `e4920b0ad49c51f2e4032c0156b6f1adeaa9955377ca7ac700f6be528c7ee836`
- Decoded audio SHA-256: `d635ad6825998a1322d964d252d53e923e9832e1566ecf6f47614f1c16ad1712`
- Video: H.264、yuv420p、1344×768、24 fps、124 frames、5.166667秒
- Audio: AAC、32,000 Hz、stereo、163 frames、5.175秒
- Container duration: 5.207秒
- A/V duration drift: 約0.008333秒

全数値分布は3 runで同値だったため、min、p5、p50、p95、maxも同値である。

## Validation

生成したreferenceを同じmeasured suiteへ適用し、3 runの全checkがpassした。

- environment checks: 2/2 pass
- per-run checks: 20/20 × 3 pass
- failed checks: 0
- worst-case run: `measured-001`、failed checks 0（同率先頭）

unit testでは、audio decoded hashの変更、media tool version差、baseline run間のartifact差、保存後のartifact改変を不合格またはreference生成拒否として確認した。

## Artifact handling

commitするreferenceはdigest、protocol identity、media metadata、baseline分布、制約だけを含む。prompt本文、生成MP4、raw decoded stream、artifact path、local cache pathは含めない。full quality reportと生成物は`benchmark-results/`または一時領域へ保持し、Git管理しない。

## Interpretation

この結果により、固定`smoke-001`のplacement-only candidateへ適用する機械可読なexact gateを使用できる。映像と音声のdecoded identityを別々に確認するため、次のresident-layer A/Bでcontent regressionを検出できる。

単一caseかつ同一artifactを要求する狭いgateであり、一般的な知覚品質、prompt adherence、A/V semantic sync、lossless性、10/50件の正式quality set、Support Tierを証明しない。
