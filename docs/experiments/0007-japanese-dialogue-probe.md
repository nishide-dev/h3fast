# Japanese dialogue prompt probe

- Date: 2026-08-17 (Asia/Tokyo)
- Host: Japan-local secondary GPU host (RTX PRO 6000 Blackwell), operator `nishide-dev`
- Related: Issue #16, [territory inventory](../../compliance/territories/initial-runtime.json)
- Status: Planned; results pending
- Outcome: Exploratory only. Not a benchmark measurement and not formal quality evidence.

## Purpose

固定protocolのformal生成で観察された「発話音声が特定言語として成立しない」現象について、モデル能力の限界ではなくprompt書式の問題であることを確認する。

MiniMax H3のmodel cardは台詞を`<d>[Language] text</d>`形式で明示する書式を示し、安定対応言語にJapaneseを含む。現在のformal quality setのpromptは「明瞭な短い会話を同期させる」のように会話の存在だけを記述し、台詞本文を与えていない。この差が原因であるという仮説を、最小の使い捨て生成で検証する。

## Scope and limits

- 固定benchmark protocolの測定ではない。latency、memory、品質のいずれもpinned baseline/candidateと比較しない。
- 使用GPUと並列条件が異なるため、生成時間は参考値としても記録しない。
- formal quality set、metric plan、release gateのいずれの状態も変更しない。
- 生成物はH3 OutputとしてGit外のJapan-local storageに保持し、digestと観察のみを記録する。

## Method

同一の場面記述に対し、台詞指定の有無だけを変えた2条件を生成して聴取比較する。

1. Baseline条件: 台詞タグなし(現行formal setのprompt様式)
2. Probe条件: `<d>[Japanese] ...</d>`で台詞本文を明示

その他の生成parameter(解像度、duration、step数、seed)は両条件で同一にする。

## Results

（実施後に記録）

## Follow-ups

（実施後に記録）
