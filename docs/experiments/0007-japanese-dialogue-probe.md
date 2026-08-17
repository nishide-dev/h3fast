# Japanese dialogue prompt probe

- Date: 2026-08-17 (Asia/Tokyo)
- Host: 承認済みJapan-local GPU host（2×RTX 6000 Ada、pinned runtime）, operator `nishide-dev`
- Related: Issue #16, [territory inventory](../../compliance/territories/initial-runtime.json)
- Status: Completed 2026-08-18
- Outcome: 仮説を確認。台詞タグの有無が発話の言語成立を決める。Exploratory only、benchmark測定でもformal quality evidenceでもない。

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

## Conditions

同一のseed 12000、768p、16:9、4秒、50 steps、固定protocolの生成parameterで、prompt末尾だけを変えた。

- 条件A（現行formal様式）: 場面記述 + 「明瞭な短い会話を同期させる。」（台詞本文なし）
- 条件B（公式書式）: 場面記述 + 「人物がカメラに向かって話す。<d>[Japanese] おはようございます。今日はとても良い天気ですね。</d>」

## Results

生成物digest（SHA-256先頭16文字）:

| 条件 | digest | mean volume | 中間無音長 |
|---|---|---:|---:|
| A: 台詞タグなし | `a3008bcb34e2ef24` | -27.3 dB | 0.52 s |
| B: `<d>[Japanese]` タグ | `a859fc421ed59015` | -25.6 dB | 1.09 s |

digestが異なるため、台詞タグは無視されず出力へ影響している。条件Bの中間無音が長いことは、与えた台詞が2文であることと整合する。

Operator `nishide-dev`による聴取結果:

- 条件A: 日本語として聞き取れない（特定言語として成立しない音列）
- 条件B: 「おはようございます。今日はいい天気ですね。」と明確に聞き取れる

## Interpretation

formal生成で観察された発話音声の非言語性は、モデルの日本語能力の限界ではなく、prompt書式の問題である。H3のmodel cardが示す`<d>[言語] テキスト</d>`書式で台詞本文を与えれば、日本語音声は明瞭に生成される。

現行formal quality setのdialogue系promptは「会話がある」ことだけを記述し台詞本文を与えていないため、モデルは発話内容を発明するしかない。これがaudio品質の共通の床として全caseに存在する。

## Impact on the formal quality set

現在進行中のformal評価への影響は限定的である。baselineとcandidateは同一promptを使うため、placement最適化の品質差判定は成立する（実際にbit単位一致が確認済み）。

ただしTier 2最適化（量子化、step蒸留等）でaudio qualityやA/V syncのmetricを実測する際、台詞が非言語のままではmetricが本来評価すべき「発話の明瞭さ」を測れない可能性がある。将来formal setを改訂する場合、dialogue系caseへ`<d>[言語] テキスト</d>`書式で台詞本文を与えることを検討する。この改訂はprompt digestを変えるためrights reviewとregistry再構築を伴う。

## Limits

- 単一case、単一seed、各条件1生成の探索的比較であり、統計的評価ではない。
- 聴取は単一operatorの主観判定である。
- 他言語（中国語、英語等）での同様の効果は未確認である。
- 生成物はH3 OutputとしてGit外のJapan-local storageに保持し、本記録にはdigestと観察のみを掲載する。

## Follow-ups

- formal quality set改訂の要否は、Tier 2最適化のaudio metric実測を開始する時点で判断する。改訂する場合はrights reviewとregistry再構築が必要になる。
- 台詞書式は上流MiniMax skillsの`h3-prompt-writing`にも記載がある。これはH3 MaterialsであるためGit管理外に置く（AGENTS.md、`.gitignore`）。書式そのものは公開model cardに記載された事実として本記録で参照する。
