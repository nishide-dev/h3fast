# Balanced profile applies to fl2va; ref2va blocked by storage I/O

- Date: 2026-08-21 (Asia/Tokyo)
- Profile: `balanced`(既定、online FP8 + turbo LoRA dynamic + 12 sigma points + Sage)
- Protocols: `h3fast-phase1b-balanced-fl2va-v1`、`h3fast-phase1b-balanced-ref2va-v1`
- Host: 承認済みJapan-local GPU host(2×RTX 6000 Ada、TP2)
- Related: [experiment 0015](0015-fp8-default-adoption.md)(既定profileの根拠), [experiment 0010](0010-reference-conditioned-families.md)(reference familyの初回検証)
- Outcome: **fl2vaへ適用可(20 case、同一caseで4.8×)**。ref2vaは storage I/O 律速で起動せず未検証。

## Purpose

既定のbalanced profileはt2vaで評価・採用した。reference条件付きtask family(fl2va / ref2va)にも同じ構成が適用できるかを確認する。

turbo LoRAのREADMEは「both text-to-video and image-to-video work」と記載し、base modelとして量子化variant(`int8_convrot`、`pruned_fp8`)も対象とする。またLoRAをrun timeで適用する設定を「sharpest, recommended」とし、これは本profileの`merge_mode: dynamic`に相当する。fl2vaはfirst/last frame条件であり、この記載の範囲内と判断できる。**ref2vaについてはREADMEに記載がなく未知である。**

## Method

既定profileのprotocolから、変更する次元を最小にした2つのprotocolを作成した。

- fl2va: `base_model.task_family` / `task` を fl2va へ。**servedなvariantは既定と同じFL2VA**
- ref2va: 上記に加え `runtime.model_variant: ref2va`(Ref2VA partitionが必要)

quantization、LoRA identity、merge mode、sigma points、attention backendは既定から変更していない。

## Results: fl2va

### 品質評価は実施していない

project ownerの判断により、fl2vaでは実行可能性と速度のみを記録する。最適化の中身(FP8 + turbo LoRA + 12 points)はt2vaのformal 20 caseで既に品質評価済みであり([experiment 0015](0015-fp8-default-adoption.md))、本記録で変えているのはtask familyのみである。fl2va固有のリスクはkeyframe条件との整合性であり、これは生成契約の検証で確認する。

**したがって本記録はfl2vaの品質について何も主張しない。**

### 生成契約(全20 caseでfail-closedに確認)

| 項目 | 結果 |
|---|---|
| `task` | 20/20が `fl2va`(silentなt2va置換なし) |
| `conditions` | 20/20が2件(`role: keyframe`、`frame_index` 0 / -1) |
| LoRA適用 | 259層、`merge_mode=dynamic` |
| `quantization` | `fp8` |
| attention backend | `verify-backend`で `resolved: sage_attn`、`verified: true` |
| server起動 | 278秒 |

LoRA適用層数259はt2va構成と同一である。adapterがtask familyに依存せずDiTへ適用されていることを示す。

### Speed and memory(20 case、単一rep)

| 指標 | 値 |
|---|---|
| 総時間 | **1.15h** |
| per-case | p50 173秒(min 83 / max 701) |
| peak VRAM (p50) | 27,310 MiB |

同一case(smoke-002)での過去実測との比較:

| 構成 | elapsed |
|---|---|
| Sage 50 step([experiment 0010](0010-reference-conditioned-families.md)) | 470.95秒(server `inference_time_seconds`) |
| balanced profile | **97.9秒**(client elapsed) |

**約4.8×**であり、t2vaで測定した4.22×と同等以上である。ただし比較の左辺はserver側inference時間、右辺はclient elapsedであり、厳密な同一指標ではない。client elapsedはserver時間より長くなるため、実際の比率は4.8×以上と考えられるが、正確な値は同一指標での再測定を要する。

per-caseのばらつき(83〜701秒)はcaseのduration(4〜15秒)と解像度に由来する。

## Results: ref2va — 未検証

Ref2VA variantのserver起動が`--startup-timeout`の3600秒で失敗した。1時間かけて`Ref2VA/text_encoder`のロードが0%から進まなかった。CUDA error、OOM、partition拒否、shape mismatchのいずれも記録されていない。

原因はstorage I/Oである。

| 測定 | 値 |
|---|---|
| Ref2VA重みの読み込み速度 | **84.9 MB/s**(`dd`実測) |
| Ref2VA weights | 135 GB、29 safetensors、`.incomplete`なし(健全) |
| 共有filesystem使用率 | 99% |

84.9 MB/sで135GBを読むには約26分を要し、text_encoder単体でも相当な時間がかかる。[experiment 0010](0010-reference-conditioned-families.md)では同じvariantが1,542秒で起動しているため、当時よりI/Oが遅い(他利用者の負荷またはfilesystem満杯)。fl2vaが278秒で起動したのはFL2VA variantがpage cacheに残っていたためである。

**これは最適化・実装・profile設計とは無関係な環境要因である。** 重みは健全で、protocolはvalidatorを通過している。

### ref2vaで未検証のまま残る項目

- turbo LoRAがRef2VA DiT(`transformer_ref`)へ適用されるか。README に記載がなく、fl2vaの259層と同じになるかも不明である
- FP8 + dynamic LoRAがRef2VA partitionで動作するか
- 4種の参照経路(image 5 / video 5 / audio 5 / image+audio 5 case)がbalanced構成で動作するか
- 15秒caseの所要時間

## Limits

- fl2vaの品質は評価していない。上記のとおり意図的な範囲外である。
- 単一repである。
- 4.8×という比較は指標が異なる(server inference時間 対 client elapsed)。
- fl2vaのbaseline 20 caseは存在しない。t2vaと異なり、fl2vaでSage 50 stepの全case生成を行っていない。
- ref2vaは完全に未検証である。実行可能かどうかも分からない。
- `denoise_steps_seconds`は未取得(`run-formal-cases`経路の制約、既知)。

## Consequences

balanced profileはfl2vaで使用できる。`protocol-balanced-fl2va.yaml`をpinned protocolとしてrepositoryへ残す。

`protocol-balanced-ref2va.yaml`も残すが、**未検証である**ことを明記する。storage I/Oが回復した環境で再試行する。profile registryへref2vaを登録しない(実行可能性が未確認のため)。
