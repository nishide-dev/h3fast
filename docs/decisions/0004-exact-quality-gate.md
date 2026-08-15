# Exact quality gate for the first placement experiment

- Status: Accepted for Phase 1B placement experiments
- Date: 2026-08-15
- Related: Issue #5

## Context

固定`smoke-001`のwarmup 1回と測定3回は、MP4 sizeとcontainer SHA-256が一致した。次の最適化はDiT resident layer数だけを変更するplacement-only比較であり、数値演算、sampling schedule、seed、codecを変えない。一方、仕様が求める10件以上のsmoke set、50件以上のregression set、知覚・audio品質指標をすべて生成するには別の計画と長時間のGPU枠が必要である。

container SHAだけでは映像と音声を独立して判定した記録にならず、再mux等のcontainer差とdecoded content差も区別できない。

## Decision

最初のplacement A/Bに限り、`exact-decoded-artifact-v1`を必須gateとする。

- baseline measured runを3回以上要求し、container、decoded video、decoded audio、media metadataが全runで一致しなければreference生成を拒否する。
- 映像は`rgb24` rawvideo、音声はsource sample rate/channel数の`pcm_s16le`へdecodeし、streaming SHA-256を計算する。
- container hash/size、映像hash、音声hash、codec、pixel format、resolution、frame rate、frame数、sample rate、channel数、duration、A/V driftをcandidate runごとに照合する。
- prompt本文、artifact、local pathをreference/reportへ保存しない。protocol caseはpromptをSHA-256へ置換してfingerprint化する。
- `ffmpeg`と`ffprobe`のversionをreferenceへ固定し、candidate側が異なる場合は不合格とする。
- candidate reportにはp5/p50/p95とworst-case runを含める。
- quality checkが不合格または実行不能なcandidateは性能が改善しても採用しない。

## Rejected alternatives

- **Container SHAだけ:** decoded映像・音声の独立checkを示せないため不採用。
- **Media metadataだけ:** frame内容やaudio sampleの変化を検出できないため不採用。
- **単一PSNR/SSIM閾値:** baseline変動包絡が0であり、今回のplacement-only変更にはdecoded identityの方が強い。一般品質へ拡張できないため将来の知覚gateを置き換えない。
- **生成MP4をGitへ登録:** H3 Outputの配布reviewとrepository sizeの制約に反するため禁止。

## Consequences

metadata-only referenceはGit管理するが、baseline/candidate MP4とraw reportはローカルに保持する。今回のgate通過を一般的なquality equivalence、lossless、Support Tierと表現しない。

10/50件の正式set、Prompt adherence、VBench、LPIPS/DINO、FAD/CLAP、A/V semantic sync、人手評価はBlockerとして残す。数値近似を導入する`balanced`／`fast` profileではこのexact gateだけを品質予算として使用しない。
