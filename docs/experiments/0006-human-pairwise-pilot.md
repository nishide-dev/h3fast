# Human-pairwise offline A/B presentation pilot

- Date: 2026-08-16 (Asia/Tokyo)
- Ballot: `human-pairwise-pilot-2026-08-16-001`
- Reviewer: `nishide-dev`
- Related: Issue #16, [ADR 0009](../decisions/0009-formal-quality-metric-selection.md)
- Outcome: Workflow validated end to end. Not formal quality evidence.

## Purpose

human-pairwise ballot/key contract(2026-08-16実装)に欠けていたoffline A/B presentation runnerとselection記録CLIを、synthetic mediaによるpilotで全60 caseにわたり検証する。blind割当、digest検証、集計の正しさをGPU実出力なしで確認する。

## Scope and limits

- media はGPU実出力ではなく、Git外に生成したsynthetic動画(ffmpeg `testsrc2` 128x72 12fps 1s + sine音声、H.264/AAC)である。baselineは有彩色、candidateは脱彩度と高音側toneで視覚・聴覚的に区別可能とした。
- 本pilotはworkflow検証であり、formal quality evidence、baseline/candidate品質比較、formal metric plan変更のいずれでもない。
- ballot、assignment key、seed、media manifest、staging、mediaはすべて`benchmarks/quality/private/`配下(Git外)に置いた。

## Procedure

1. `pilot_media`(使い捨てscript)でformal 60 caseぶんのsynthetic media 120本と、per-file SHA-256を持つprivate media manifestを生成した。
2. `prepare-human-pairwise`で0600のpending ballotとhidden assignment keyを新規作成した(seedは`secrets.token_hex(32)`、0600)。
3. `stage-human-pairwise`がballot/assignment/manifestのdigestとcommitmentを検証し、assignment keyに従い`<case_id>/a.mp4`・`b.mp4`へblind copyし、相対参照のみの`index.html`を生成した(staged 120 file)。
4. reviewer目視の決定的代理として、assignment keyを一切参照せず、staged fileのffmpeg `signalstats` SATAVGで脱彩度側を判定し、`record-human-pairwise`で60件のselectionをCLI記録した。60件目で自動的に`completed`へ遷移した。
5. `check-human-pairwise`が完了ballotを検証・集計した。

## Results

- Aggregate: `candidate_wins=60, baseline_wins=0, ties=0, score=1.0, complete=true`
- 脱彩度側(=candidateのground truth)を選び続けた結果と集計が全60 caseで一致し、per-case randomization(A/B順はcaseごとに混在)、commitment検証、unblind集計の正しさをend-to-endで確認した。
- staged mediaはH.264 video + AAC audioとしてffprobeで再生可能性を確認し、`index.html`は60 sectionで`baseline`/`candidate`/absolute path/外部URLを含まない。
- 誤記録訂正(`--overwrite`)と二重記録拒否はunit/CLI testで検証済み。

## Observations

1. **File size side channel:** synthetic candidateは脱彩度のため圧縮後サイズが系統的に小さく、staged file sizeから素性を推測できる。formal実施ではreviewerへfile size・metadata閲覧を避ける手順を明記するか、presentation runnerでのサイズ正規化を検討する。
2. **Directory permissions on network mounts:** staging directoryは`mkdir(mode=0o700)`で作成するが、本環境のnetwork mountはdirectory modeを強制上書きする(fileの0600は保持)。formal実施はpermissionが効くlocal filesystemで行う。
3. **Recording burden:** 60件のCLI記録は機械実行で問題ないが、人手では1件ずつのcommand入力が負担になる。formal前にreview policy(reviewer数、休憩、順序)とあわせて運用手順を確定する。

## Follow-ups

- reviewer count、conflict handling、single-reviewer acceptabilityのreview policy承認(ADR 0009 blocker)。
- GPU実出力によるformal ballotとimmutable implementation evidenceの記録。
- formal metric planのhuman-pairwiseは引き続き`unassigned`のまま維持する。
