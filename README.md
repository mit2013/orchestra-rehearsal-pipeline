# アマオケ練習録音 自動編集パイプライン

ZOOM M4 MicTrak の 4ch/32bit float 録音(1回の練習で約3時間)を、取り込みから配布・通知まで
自動化する。手作業で2〜3時間かかっていた工程を、確認作業を除けば数分で回せるようにするのが目的。

```
取り込み → チャンネル結合 → TAKE連結 → 不要区間の候補提案 → [人が確認] → トリミング
  → 正規化 → ミックス → 曲目単位エクスポート(WAV/MP3・タグ)
  → Box / Google Drive へアップロード → 通知文言生成 + LINE 通知
```

260802(2合奏ブロック)・260726(3合奏ブロック)の2データセットで全工程を実行・検証済み。

**次フェーズ**: LINE グループへの自動送信(グループIDの取得が別途必要)、
指揮者の発言をカットしたダイジェスト版の作成。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy soundfile matplotlib mutagen requests \
    google-auth-oauthlib google-api-python-client google-auth-httplib2
```

ffmpeg / ffprobe が PATH にあること(Homebrew 版で確認済み)。

### 資格情報

プロジェクト直下の `.env` に置く。**すべて gitignore 対象**:

```
BOX_CLIENT_ID / BOX_CLIENT_SECRET           Box アプリ (User Authentication / OAuth 2.0)
GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET     Google アプリ (デスクトップ)
LINE_CHANNEL_ACCESS_TOKEN / LINE_MY_USER_ID LINE Messaging API
```

取得したトークンは `.box_tokens.json` / `.google_tokens.json` に保存される(パーミッション 600、
これも gitignore 対象)。Box の Redirect URI は Developer Console に
`http://localhost:8888/callback` を**一字一句そのまま**登録しておくこと。
Google はデスクトップアプリなので事前登録が不要。

## 使い方

```bash
.venv/bin/python pipeline.py ingest  --date 260802          # TAKE走査・検証 + session_config.json 生成
.venv/bin/python pipeline.py merge   --date 260802          # Tr1+Tr2ステレオ化 + TAKE連結
.venv/bin/python pipeline.py propose --date 260802 --splits 2   # 境界候補・プレビュー・波形画像

# ここで preview_clips/ を聴き、waveform.png を見て confirmed.json を修正する

.venv/bin/python pipeline.py apply   --date 260802          # keep区間だけ書き出し

# ここで session_config.json を確認・編集 (source / mix_ratio / ext_lr_map など)

.venv/bin/python pipeline.py normalize --date 260802        # ラウドネス正規化 (-20 LUFS)
.venv/bin/python pipeline.py mix       --date 260802        # 最終ファイル生成
.venv/bin/python pipeline.py export    --date 260802        # 配布用 WAV/MP3 + タグ
.venv/bin/python pipeline.py box-upload    --date 260802    # MP3 を Box へ(パスワード付き)
.venv/bin/python pipeline.py gdrive-upload --date 260802    # WAV+MP3 を Drive へ
.venv/bin/python pipeline.py notify    --date 260802        # 文言生成 + LINE 通知
```

`all` で ingest + merge + propose を通しで実行できる。中間ファイルは残るので、
失敗しても途中から再開できる(作り直したいときだけ `--force`)。

**人が確認する箇所は2つだけ**: `confirmed.json` の境界と、`session_config.json` の設定。
それ以外は再実行しても同じ結果になる。

## 出力

```
output/260802/
  ingest.json          # TAKE一覧・検証結果・機種・channel_groups・TAKE境界の絶対時刻
  raw_merged_ext.wav   # 外部マイク(Tr1=L / Tr2=R)結合、フル尺
  raw_merged_int.wav   # 内蔵マイク(TrMic)結合、フル尺
  features_ext.npz     # フレーム特徴量キャッシュ(propose の再実行が速くなる)
  candidates.json      # 境界候補(提案そのもの。書き換えない)
  confirmed.json       # 編集用コピー。ここを直して apply に渡す        ← 手で編集
  session_config.json  # LRマッピング・ミックス・団体名・Box設定など      ← 手で編集
  analysis.json        # 閾値・ペナルティ・チューニング検出位置などの内訳
  preview_clips/       # 各境界の前後 ±15 秒(16bit WAV)
  waveform.png         # 区間色分け + 合奏らしさスコアの可視化
  loudness.json        # normalize / mix のラウドネス測定値
  messages.txt         # notify が生成する通知文言(①②③)
  trimmed/
    01_合奏1_ext.wav        # apply の出力(無加工)
    01_合奏1_ext_norm.wav   # normalize の出力(-20 LUFS。まだ 0 dBFS を超えうる)
    01_合奏1_final.wav      # mix の出力(コンプ+リミッター済み。エクスポート元)
  export/
    260802_前半.wav         # 配布用 WAV(_final.wav とビット同一)
    260802_前半.mp3         # 配布用 MP3(320kbps、ID3タグ付き)
    260802_前半_ラウドネス調整版.mp3   # --variant を付けたときの別版
```

`confirmed.json` と `session_config.json` だけが git 追跡対象。ほかは再生成できるので除外。

容量が厳しくなったら、`raw_merged_*.wav`・`features_*.npz`・`*_norm.wav` は削除してよい
(`*_norm.wav` は保持した `trimmed/` 本体から `normalize` だけで作り直せる。
`raw_merged_*` を消した場合は `merge` からやり直しになるが、`confirmed.json` が残っていれば
境界の確認作業は不要)。

## セッション設定 (`session_config.json`)

`ingest` 時に既定値で生成され、以降パイプラインは上書きしない(`confirmed.json` と同じ扱い)。

```json
{
  "ext_lr_map": "normal",
  "normalize_scope": "date",
  "source": "ext_only",
  "mix_ratio": {"ext": 0.6, "int": 0.4},
  "orchestra": "Windrose Sinfonie Orchester",
  "concert_date": "{concert_date}",
  "box_parent_folder_id": "BOX_PARENT_FOLDER_ID"
}
```

| キー | 値 | 意味 |
|---|---|---|
| `ext_lr_map` | `normal` / `swapped` | `normal` = Tr1→L, Tr2→R。配線ミスのセッションは `swapped` |
| `normalize_scope` | — | **使われない。** ラウドネス正規化に変わり不要になった。互換のため残置 |
| `source` | `ext_only` / `int_only` / `mix` | 最終ファイルの作り方。既定は `ext_only` |
| `mix_ratio` | 例 `{"ext":0.6,"int":0.4}` | `source=mix` のときだけ使用 |
| `orchestra` | 団体名 | MP3タグと Drive のフォルダ名に使う |
| `concert_date` | `yyyymmdd` | Box のパスワード生成に使う。空だと `box-upload` が止まる |
| `box_parent_folder_id` | Box のフォルダID(数字) | 空ならルート直下。**フォルダ名ではなくID** |

`orchestra` / `concert_date` / `box_parent_folder_id` の既定値はプロジェクト直下の
`pipeline_defaults.json` に置く。`ingest` 時に `session_config.json` へコピーされ、
**既に値があれば上書きしない**。ある時期は同じ団体の練習が続く運用なので、団体が変わったら
`pipeline_defaults.json` を書き換えれば以降の新規 `ingest` に反映される。特定の日付だけ
別扱いにしたい場合はその日付の値を直接編集する。

**レコーダ機種はこのファイルには持たない。** 機種と `channel_groups` は取り込み時に決まる
情報で、`ingest.json` が唯一の情報源である。二重に持つと食い違いうるため、
`session_config.json` は「取り込み後にユーザーが調整する設定」だけを持つ。

`ext_lr_map` は **`merge` の時点で**適用される。それより下流(propose/apply/normalize/mix)は
常に正しい L/R が揃っている前提でよい。変更したら `merge --groups ext --force` で
外部マイク側だけ作り直せばよく、内蔵マイク側は影響を受けない。

## レコーダプロファイル

ファイル探索だけが機種依存なので、そこを `orchpipe/recorder_profiles/` に切り出してある。

| プロファイル | 状態 | 備考 |
|---|---|---|
| `zoom-m4` | **本実装** | `{date}_{番号}.TAKE/` に Tr1/Tr2/TrMic。260802・260726 で検証済み |
| `zoom-f3` | スタブ | 呼ぶと `NotImplementedError`。想定ファイル配置は docstring に記載 |
| `single-file` | スタブ | 同上。MP3入力時は事前WAV変換が必要、というメモを docstring に記載 |

`channel_groups`(例 `{"ext": ["Tr1","Tr2"], "int": ["TrMic"]}`)が、どのトラックがどの系統に
属するかを表す。2トラックで1系統ならモノラル2本を左右に組み、1トラックならそれ自体がステレオ、
と `merge` が解釈する。

**`--recorder` は `ingest`(と `all`)でしか指定できない。** 取り込み時に決まった機種と
`channel_groups` は `output/{date}/ingest.json` に記録され、`merge` 以降の各段はそれを読む。
下流で機種を指定し直せると取り込み時と食い違う恐れがあるうえ、系統名をコード側に
埋め込む必要が生じるため、あえて指定できないようにしてある。`propose --source` や
`merge/apply --groups` に渡せる値も `ingest.json` の `channel_groups` から決まり、
未知の系統を指定すると利用可能な一覧を添えて停止する。

## 不要区間の判定ロジック

単純な無音検出は使っていない。`orchpipe/features.py` が粒度に依存しない特徴量を出し、
`orchpipe/segment.py` がそれに練習の構造(音出し → チューニング → 合奏 → 休憩)を
かぶせる、という二層構成になっている。

**フレーム単位(64ms 窓 / 16ms ホップ)**: RMS、スペクトルフラックス、フラットネス、
重心、純音性、ピーク周波数(放物線補間つき)。ffmpeg のパイプからストリーミングで
処理するので、巨大な WAV をメモリに載せない。

**窓単位(既定 20 秒 / ホップ 5 秒)**への集約で、指示書の手がかりを数値化する:

| 特徴量 | 音出し・休憩 | 合奏 |
|---|---|---|
| `pulse_clarity`(オンセット包絡の自己相関ピーク) | 低(各自バラバラ) | 高(全員が揃う) |
| `dyn_range`(RMS の p90−p10) | 低(メリハリが乏しい) | 高 |
| `silence_ratio` | ほぼ 0(強音が途切れない) | 中程度(指揮者が止める) |
| `flatness` | 高(雑音的) | 低 |

これらを頑健化(中央値/MAD)して重み付き合成したものが「合奏らしさスコア」
(平滑化は既定で無効。理由は後述)。判定は **2状態のビタビ探索**で行う — 窓ごとのスコア合計と
状態切り替え回数のトレードオフを全体最適で解くので、合奏中の一時的な落ち込み(長い強奏など)
では切り替わらず、実際の休憩のような「深く長い谷」だけが境界になる。
`--splits N` を指定すると、合奏区間がちょうど N 個になる遷移ペナルティを二分探索する。

閾値は大津の方法から、2クラス中心間距離の 25% ぶんだけ低いほうへずらしてある。
**本物の合奏を削る誤りは、音出しが少し残る誤りよりはるかに痛い**ので、意図的に
「残す側」に倒している。

**チューニング**は「純音性が高く・ピッチが安定し・スペクトル変化が小さい」フレームの
持続として検出し、合奏開始の境界をその終端にスナップさせる。ただし検出は万能ではないので、
補助的な手がかりとして ±120 秒の範囲でのみ使う。

最後に、keep 区間を前後 `--guard`(既定 120 秒)だけ外側へ広げる。境界推定の残差は
どうしても残るが、**本物の演奏を削る誤りは取り返しがつかず、不要区間が少し混じる誤りは
聴き飛ばせばよいだけ**なので、意図的に「残す側」へ倒している。

### 境界精度について(260802 での実測)

正解(目視確認)と提案の差:

| 境界 | 提案 | 正解 | 誤差 | 向き |
|---|---|---|---|---|
| 合奏1 開始 | 00:08:17 | 00:08:42 | −25s | 安全側 |
| 合奏1 終了 | 01:32:36 | 01:30:49 | +107s | 安全側 |
| 合奏2 開始 | 01:39:02 | 01:41:54 | −172s | 安全側 |
| 合奏2 終了 | 03:04:17 | 03:04:00 | +18s | 安全側 |

**演奏を削った量: 0 秒**(残差はすべて不要区間を少し含む方向)。

初期実装ではスコアに 300 秒の中央値フィルタをかけていたが、これが最大の誤差要因だった
(合奏1開始が 12 分以上ずれ、演奏を計 982 秒削っていた)。短時間の変動を無視する役目は
ビタビの遷移ペナルティがすでに担っており、平滑化を重ねると同じ仕事を二重にやったうえで
境界の時間分解能だけを失う。**平滑化は既定で無効**(`--smooth 0`)。60 秒を超えると
再び大きくずれることを実測で確認している。

また `--min-remove` の既定は 1.0 分。この録音の片付けは 1.6 分しかなく、以前の既定
(2.5 分)では正しい境界が構造的に採用され得なかった。

なお上記の数値は 1 セッション・4 境界での実測にすぎない。平滑化を外す判断は
「ビタビと役割が重複している」という機構的な理由からも支持されるが、`--guard` の
120 秒という値は他のセッションで見直す余地がある。

## 正規化とミックス(ラウドネス)

**正規化**は統合ラウドネス基準(既定 **-20 LUFS**、`--target-lufs`)。ピーク正規化ではない。
ピークは1サンプルの話で、人が感じる音量とは別物だからである。260829 でその差がはっきり出た:
`normalize_scope: date` がその日の最大ピーク(合奏3の +11.43 dBFS)を基準に全ブロック共通の
-12.43 dB を掛けた結果、打楽器の鳴らなかった合奏1まで一緒に下げられ、ブロック間に **8.6 LU**
の差が残った。

| 260829(旧・ピーク正規化) | 統合ラウドネス | レンジ | トゥルーピーク |
|---|---|---|---|
| 前半(合奏1) | -34.3 LUFS | 27.7 LU | -10.3 dBFS |
| 中盤(合奏2) | -25.7 LUFS | 24.5 LU | -1.1 dBFS |
| 後半(合奏3) | -28.5 LUFS | 24.5 LU | -1.0 dBFS |

ブロックごとに目標ラウドネスへ合わせれば、この差は原理的に生じない。したがって
`normalize_scope` は**使われなくなった**(既存の設定ファイルを読めるようにキーだけ残す)。

基準値の算出範囲と適用範囲を分けているのは従来どおり。`--guard` で外側に広げた部分
(音出しや休憩の話し声が混入しうる)は測定から除外し、ゲイン自体はブロック全体に一律で
掛ける。除外幅は `--ref-margin`(既定120秒)。測定値は `loudness.json` に残る。

**ミックス**は合成とマスタリングを行う。`ext_only`/`int_only` は正規化済みファイルを
そのまま素材にし、`source=mix` は指定比率で加算合成する。そのうえで、**すべての source で
共通に**ダイナミクス処理を1回だけ通す。

    ゲイン → コンプレッサ(2:1、緩く) → トゥルーピークリミッター(-1 dBTP)

| コンプのパラメータ | 値 | 意図 |
|---|---|---|
| レシオ | 2:1 | 明確に「かかっている」と分からない程度 |
| しきい値 | 目標ラウドネス +6 dB(既定 -14 dBFS) | 平均的な合奏には触らない |
| アタック | 100 ms | 打楽器の立ち上がりの質感を残す |
| リリース | 1000 ms | フレーズ単位で戻る。ポンピングを避ける |
| ニー | 6 dB(ソフト) | しきい値付近の不連続をなくす |

順序が重要である。コンプで打楽器の立ち上がりを先にならしておけば、リミッターはほとんど
動かずに済む。リミッターを先に置くと、打楽器のピークだけがリミッターに当たって音色が変わる。
リミッターは安全弁であり、常時動作させるものではない。

コンプを通すとラウドネスが下がるので、「測って一度ゲインを当てる」だけでは目標に乗らない。
そこで測定を2段にしている:素の統合ラウドネス I0 から暫定ゲイン g0 を出し、g0 を当てて
コンプまで通したときの I1 を測って g1 = g0 + (目標 - I1) を最終ゲインとする。測定は
`-f null` に捨てるので実時間の 1/280 程度で終わる。書き出し後に統合ラウドネスと
トゥルーピークを検証し、目標から 1 LU 以上ずれるか天井を超えたら失敗として止める。

素材のピークは **0 dBFS を超えている**(260829 の外部マイクで +11.43 dBFS、260726 で +9.7 dBFS)。
32bit float なのでファイル上は壊れていないが、固定小数点で書き出せば激しくクリップする。
`_norm.wav` はまだリミッターを通していないので 0 dBFS 超のままでよく、天井を保証するのは
`mix` のリミッターである。

旧実装にあった「合成ピークが安全閾値を超えたら全体をスケールダウンする」処理は廃止した。
リミッターが天井を保証するので不要であり、一律スケールダウンはブロック間の音量差を戻してしまう。

## エクスポート(曲目単位)

1合奏ブロック = 1トラックとして書き出す。トラックタイトルはブロック数から自動採番する:

| ブロック数 | タイトル |
|---|---|
| 2(休憩1回) | 前半 / 後半 |
| 3(休憩2回) | 前半 / 中盤 / 後半 |
| その他 | コマ1 / コマ2 / … |

ブロック数は `confirmed.json` の `action: keep` の区間数で判定し、`*_final.wav` の本数と
食い違っていればエラーで止める(`mix` のやり直し漏れを検出するため)。

WAV は 32bit float のまま実体コピーするので `_final.wav` とビット単位で同一。
MP3 は `libmp3lame` 320kbps 固定で、ID3タグを `mutagen` で書き込む:

| タグ | 値 | 例 |
|---|---|---|
| `TALB` アルバム | 日付文字列 | `260802` |
| `TIT2` タイトル | 自動採番 | `前半` |
| `TPE1` / `TPE2` アーティスト | `session_config.json` の `orchestra` | `Windrose Sinfonie Orchester` |
| `TRCK` トラック番号 | 通し番号/総数 | `1/2` |
| `TDRC` 年 | 日付の先頭2桁に `20` を前置 | `2026` |

`export` は再エンコードをスキップした場合でもタグは毎回書き直す。団体名を変えたときは
`export` を再実行するだけでよく、MP3 の再エンコードは走らない。

## 現場前処理(帰宅前に MP3 を配る)

素材は3時間で約 8GB(32bit float)あり、モバイル回線では送れない。そこで **iPhone
上で 320kbps MP3 を1本だけ作り(「プロキシ」)、それだけを母艦へ送る**。母艦は
届いた MP3 で境界提案からブロック切り出し、Box への配布までを済ませる。Drive 用の
WAV は帰宅後に原本から作る。

```bash
# 母艦: 現場で流すスクリプトを用意しておく(その日の TAKE 名を埋め込む)
.venv/bin/python pipeline.py field-script --date 260829

# 現場: M4 を File Transfer モードにして iPhone に接続(M4 は電池駆動)
#       microSD の TAKE を iPhone にコピーし、a-Shell で
#       sh field_master.sh   -> 260829_proxy.mp3(3時間で約 413MiB)

# 母艦: 置き場を見張り、届いたら受け取って境界レビューの手前まで進める
.venv/bin/python pipeline.py field-watch --date 260829 --dir ~/Library/Mobile\ Documents/... --splits 3

# ここで review_page.html を Artifact として公開し、iPhone で境界を確定する
.venv/bin/python pipeline.py review-apply --date 260829 --input <ページから取り出したJSON>

.venv/bin/python pipeline.py field-export --date 260829
.venv/bin/python pipeline.py box-upload   --date 260829
.venv/bin/python pipeline.py notify       --date 260829
```

`field-watch` は `field-receive` → `propose` → `review-page` をまとめたもので、
一つずつ実行してもよい。転送方式(iCloud Drive を監視するか Tailscale で置くか)は
まだ決めていないが、どちらも「所定のフォルダにファイルが現れる」点は同じなので
この形なら両方に乗る。

プロキシに載せるのは**クリップを避けるための固定ゲイン(-14 dB)だけ**である。
ラウドネス正規化・コンプ・リミッターは、境界が決まったあとに母艦がブロックごとに
当てる(`field-export`)。境界が決まる前に単一ゲインを確定させると、ブロック間の
音量差(260829 で 9.7 dB)がそのまま残ってしまうためである。固定ゲインは可逆なので、
母艦は測定の前に +14 dB を戻す。

260829 で原本経由と比べた結果:

| | 結果 |
|---|---|
| ブロックごとのゲイン | **0.01 dB まで一致**(+4.00 / -5.70 / -2.50 dB) |
| 配布 MP3 のラウドネス | 統合・レンジ・トゥルーピークとも一致 |
| 境界提案 | **5区間すべて同一時刻**。チューニング検出も一致 |
| 無音閾値 | -42.985 → -56.984 dBFS(固定ゲイン -14 dB ぶんちょうど) |
| 音の差 | 位置ずれ 0.00 ms / 相関 0.9998 以上 / 残差は最悪 -34.6 dB(打楽器) |

代償は MP3 の符号化が1世代増えることで、`-c copy` では切り出せない。詳細と検証は
`orchestra_recording_pipeline_field_preprocess.md` を参照。

`field-proxy` は母艦側で同じプロキシを作る(検証用、および iPhone が使えないときの
代替)。`apply` にプロキシを渡すとエラーで止まる ― ブロック WAV は原本から切るため。

### 律速は境界提案だった

通しで走らせると、投入から境界レビューのページが出るまで 25 秒だった。転送も符号化も
律速ではない。**律速は `propose` の境界が当たらないこと**である。260829 の自動提案は
合奏1が 94.6 分(合奏1と休憩と合奏2をまたぐ)で、レビューページで秒単位に直せる幅を
超えていた。対策は `--tuning-first`(下記)。

## 境界レビューのページ

```bash
.venv/bin/python pipeline.py review-page  --date 260829     # HTML を組み立てる
# Artifact として公開 -> iPhone で頭と尻を聴き、境界を動かして保存
.venv/bin/python pipeline.py review-apply --date 260829 --input state.json
```

各ブロックの頭と尻について、境界の**前後45秒**を切り出して埋め込む。判定
(よい / 切れている / 余分が長い)を押すだけでなく、**±5 / ±15 / ±30 秒のボタンで
その場で境界を動かせる**。動かした結果の時刻と長さは表に出る。保存すると
`artifact` capability でページ自身が新しい版になるので、その JSON を
`review-apply` に渡せば `confirmed.json` に反映される(元は `.review_bak` に控える)。

音源は結合済み WAV でもプロキシ MP3 でもよい。クリップはモノラル 96kbps で、
6本を埋め込んで約 8MiB(Artifact の上限は 16MiB)。

## チューニングを起点にした境界提案(`--tuning-first`)

スコアからの区切り(合奏らしさ + Viterbi)は、休憩の話し声や長い部分練習で崩れる。
一方**チューニングは合奏の直前に必ず現れ、合奏の中には現れない**。

```bash
.venv/bin/python pipeline.py propose --date 260829 --splits 3 --tuning-first
```

260829 での結果:

| ブロック | `--tuning-first` | 人が聴いた判定 |
|---|---|---|
| 合奏1 | 00:01:20 – 00:44:21 | 頭・尻とも「よい」 |
| 合奏2 | 00:47:11 – 01:34:51 | 頭・尻とも「よい」 |
| 合奏3 | 01:42:18 – 03:00:04 | 頭・尻とも「よい」 |

開始はレビューページで人が確かめた位置と一致する。終了は `--guard` を外側へ
足しているぶん実際の演奏の終わりより +19〜+101 秒あとになるが、その範囲は
聴いてもらった結果すべて音出しで、曲は含まれていなかった。設計どおり
「削りすぎない」側に出ている。

### チューニングの手前に置く余裕は 2 秒

`TUNING_PRE_ROLL_S` は **2 秒**。以前は 5 秒だった。260829 の試聴では3ブロックとも
「検出位置ちょうどまで詰めてよい」という判定で、5 秒だと手前の音出しと無音が
ブロックの頭に入るだけだった。

ただし**ちょうどにはしない。** 同じ試聴で1件、A が想定より 1 秒ほど早く聞こえたと
報告があった。実測でも検出位置の 0.5 秒手前で A 成分の比率が 0.279 まで上がっている
箇所がある(検出位置では 0.105)。`tuning.py` は倍音で頭を前へ伸ばすので報告される
開始はおおむね音の手前に来るが、常にではない。1 秒のばらつきを吸収できる 2 秒を残す。

チューニング開始そのものを合奏の開始とし、終了は「次のチューニングの手前で合奏らしさが
最後に閾値を超えた窓の終わり」とする。

**既定にはしていない。** 検証できたのが 260829 の1日だけだからである。チューニング
検出の件数が `--splits` と食い違うときは従来の経路に落ちる。

## クラウドアップロード

用途が違うので、Box と Google Drive で共有の性格を変えている。

| | Box | Google Drive |
|---|---|---|
| 対象 | MP3 のみ | WAV + MP3 |
| 宛先 | 団員全体 | 動画担当(WAV)+ 個別共有(MP3) |
| パスワード | あり(`{concert_date}{password_suffix}`) | なし(Drive に概念がない) |
| ダウンロード | **不可**(ストリーミングのみ) | **可**(素材として使うため) |
| 重複回避 | SHA-1 比較 | MD5 比較 |

### Box

```
{box_parent_folder_id}/{date}/    ← MP3 のみ
```

フォルダ単位で共有リンクを1つ設定する(パスワード保護・ダウンロード不可・リンクを
知っている人のみ)。パスワードは `{concert_date}{password_suffix}` で、Box の要件(8文字以上・
数字か記号を含む)を満たす。

MP3 は 100MB を超えるので**チャンク分割アップロード**を使う(Box の単純アップロードは
50MB まで)。セッション作成 → パートごとに SHA-1/base64 の digest と content-range を
付けて PUT → 全体の SHA-1 を digest にして commit、という流れ。commit が 202 を返したら
`Retry-After` に従って再試行する。

**Box のリフレッシュトークンはローテーションする**(1回使うと無効になり、新しい値が返る。
有効期限60日)。更新のたびに新しい値を保存し直さないと、しばらくして認証が切れる。

### Google Drive

```
orchestra-recording-pipeline/     ← 自動化専用のルート
  {orchestra}/                    ← 例: Windrose_Sinfonie_Orchester
    {date}/
      WAV/   ← 動画担当向け
      MP3/   ← 個別共有向け
```

`{orchestra}` は `session_config.json` の `orchestra` からスペースをアンダースコアに
置き換えて**自動生成**する(表記揺れを防ぐため手入力・ハードコードしない)。

共有リンクは**2つ**設定する。`{date}` フォルダ(WAV+MP3、動画担当向け)と
`{date}/MP3` フォルダ(個別共有向け)。どちらも「リンクを知っている全員が閲覧者」で、
親から子への継承に頼らずフォルダごとに個別設定している。

スコープは `drive.file`(このアプリが作成したファイルのみ)。手動作成したフォルダは
そもそも見えないので、自動化専用ルートと手作業のフォルダが混ざる心配がない。

1.8GB 級の WAV を扱うため、16MiB チャンクの再開可能アップロードを使い、通信エラーは
指数バックオフで最大8回まで再開する(実測で `BrokenPipeError` からの復帰を確認済み)。

### 変更なしファイルのスキップ

アップロード前に、同名ファイルがあればそのハッシュ(Box は `sha1`、Drive は `md5Checksum`)を
API から取得し、ローカルのハッシュと比較する。一致すればアップロードしない(新バージョンも
作らない)。260802 で 3.89 GiB が全件スキップされ、移行と検証を含めて11秒で完了した実績がある。

## 通知

```bash
.venv/bin/python pipeline.py notify --date 260802
```

3種類の文言を生成し、標準出力と `output/{date}/messages.txt` に書き出したうえで、
まとめて LINE で自分宛に push する。

| | 宛先 | 使うリンク |
|---|---|---|
| ① | 動画担当(動画担当) | Drive の `{date}` フォルダ(WAV+MP3) |
| ② | 団員グループ | Box の共有リンク + パスワード |
| ③ | **【任意・通常は送らない】** | Drive の `{date}/MP3` フォルダ(ダウンロード可) |

**リンクはローカルにキャッシュしない。** 実行のたびに Box / Drive 双方の API へ
問い合わせて最新のリンクを取得する。パスワードだけは Box API が返さないので
`{concert_date}{password_suffix}` の式からその場で計算する。

LINE 通知は付加的な機能なので、トークン未設定・ネットワークエラー・API エラーの
いずれでも例外を投げず、標準出力と `messages.txt` の生成は必ず完了させる
(失敗理由は標準出力に出す)。エンドポイントは `api.line.me` であって `line.me` ではない。

現状 LINE で自動送信するのは**自分宛の完了通知のみ**。グループへの自動送信は
グループIDの取得(Webhook を使った一度きりのブートストラップ)が必要なため次フェーズで、
それまでは `messages.txt` から手動でコピーして送る運用。

## 品質について

- 結合・トリミング・エクスポートの WAV はすべて `pcm_f32le`(入力と同一)で、
  再エンコードによる劣化はない。トリミング結果・`export/*.wav` が元データと
  ビット一致することを実測で確認済み。
- チャンネル対応は `join=...:map=0.0-FL|1.0-FR` で明示し、さらに `merge` の実行時に
  結合後の L/R と元の Tr1/Tr2 を実データで突き合わせて検証する(不一致なら停止)。
- 3時間素材のステレオ f32 は 4GB 前後で WAV の 4GB 制限に接するため、`-rf64 auto` を
  指定してある(必要なときだけ RF64 で書き出す。260726 の 3.19 時間で実際に発動した)。

## 将来フェーズへの転用

`features.py` は「粗い区切り」に特化していない。同じ `extract_frame_features` の結果を
`aggregate_windows(ff, win_s=2.0, hop_s=0.5)` のように細かい粒度で呼べば、
「合奏内の指揮者停止単位でのチャプター分割」や「指揮者発言のみカット」にそのまま使える。
区間の意味づけ(合奏/音出し/休憩のラベル)だけが `segment.py` 側に閉じている。

## 主なオプション

### `propose`

| オプション | 既定 | 意味 |
|---|---|---|
| `--splits N` | なし | 最終的に何分割(合奏いくつ)にしたいかのヒント |
| `--source ext\|int` | `ext` | 解析に使う系統 |
| `--win` / `--hop` | 20 / 5 秒 | 解析窓長・ホップ |
| `--smooth` | 0 秒(無効) | スコアの平滑化長。**上げると境界がずれる**ので通常は 0 のまま |
| `--guard` | 120 秒 | keep 区間を外側へ広げる安全マージン |
| `--min-keep` / `--min-remove` | 8 / 1.0 分 | 区間の最小長。`--splits` はこれに制限される |
| `--penalty` | 12 | `--splits` 未指定時の切り替えペナルティ(大きいほど区間が減る) |
| `--pad` | 15 秒 | プレビューの前後幅 |

### その他

| コマンド | オプション | 既定 | 意味 |
|---|---|---|---|
| `merge` / `apply` | `--groups` | 全系統 | 対象系統を限定(例 `ext`) |
| `export` | `--variant` | なし | 版名。ファイル名末尾と ID3 タイトルに入る(例 `ラウドネス調整版`) |
| `normalize` | `--target-lufs` | -20 LUFS | 目標の統合ラウドネス |
| `normalize` | `--ref-margin` | 120 秒 | 基準ラウドネスの算出から除外する前後の長さ |
| `mix` | `--target-lufs` | -20 LUFS | 目標の統合ラウドネス |
| `mix` | `--true-peak` | -1 dBTP | トゥルーピークの上限 |
| `mix` | `--comp-ratio` | 2 | コンプレッサのレシオ |
| `mix` | `--comp-threshold-offset` | 6 dB | しきい値を目標ラウドネスから何 dB 上に置くか |
| `mix` | `--comp-attack` | 100 ms | コンプのアタック |
| `mix` | `--comp-release` | 1000 ms | コンプのリリース |
| `mix` | `--comp-knee` | 6 dB | コンプのニー幅 |
| `propose` | `--tuning-first` | — | チューニングを合奏の開始として区間を組む |
| `field-watch` | `--dir` | 必須 | プロキシの置き場を見張る |
| `field-watch` | `--stable` | 15 秒 | サイズがこの秒数変わらなければ書き込み完了とみなす |
| `field-watch` | `--receive-only` | — | 受け取るだけで propose / review-page を走らせない |
| `review-page` | `--pre` / `--post` | 45 秒 | 境界の前後に含める長さ |
| `review-apply` | `--input` | 必須 | ページから取り出した JSON |
| `field-script` | `--takes` | 2 | その日の TAKE 数(`ingest.json` があればそちらが優先) |
| `field-script` / `field-proxy` / `field-export` | `--gain` | -14 dB | プロキシに載せる固定ゲイン |
| `field-receive` | `--input` | 必須 | 受け取ったプロキシ MP3 |
| `field-receive` | `--move` | — | コピーではなく移動する |
| `field-export` | `--variant` | なし | 版名。`export` と同じ |
| `field-export` | `--target-lufs` / `--true-peak` / `--ref-margin` | mix と同じ | マスタリングの設定 |
| `box-upload` | `--auth-timeout` | 300 秒 | 初回認証でブラウザ操作を待つ秒数 |
| `gdrive-upload` | `--auth-timeout` | 無制限 | 同上 |
| `notify` | `--no-line` | — | LINE への push を行わない |
| 全コマンド | `--root` / `--force` | — | 作業ディレクトリ / 中間ファイルの作り直し |
