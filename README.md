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

.venv/bin/python pipeline.py normalize --date 260802        # ピーク正規化 (-1 dBFS)
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
  messages.txt         # notify が生成する通知文言(①②③)
  trimmed/
    01_合奏1_ext.wav        # apply の出力(無加工)
    01_合奏1_ext_norm.wav   # normalize の出力(ピーク -1 dBFS)
    01_合奏1_final.wav      # mix の出力(エクスポート/アップロード元)
  export/
    260802_前半.wav         # 配布用 WAV(_final.wav とビット同一)
    260802_前半.mp3         # 配布用 MP3(320kbps、ID3タグ付き)
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
| `normalize_scope` | `date` / `block` | 正規化の基準を取る範囲。既定は `date` |
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

## 正規化とミックス

**正規化**はピーク正規化のみ(目標 -1 dBFS)。基準値の算出範囲と適用範囲を分けているのが要点で、
`--guard` で外側に広げた部分(音出しや休憩が混入しうる)は基準の計算から除外し、ゲイン自体は
ブロック全体に一律で掛ける。除外幅は `--ref-margin`(既定120秒)。

基準を取る範囲は `normalize_scope` で決まる:

- **`date`(既定)**: 日付・系統ごとに、全ブロックの本編ピークの**最大値**を基準にし、
  そこから求めた**単一のゲインをその系統の全ブロックに適用**する。合奏1が大音量で
  合奏2が静かだった、という**ブロック間の音量差が演奏どおりに保たれる**。
- **`block`**: ブロックごとに個別のゲインを求める。全ブロックが -1 dBFS に揃うため、
  静かな曲が大きく持ち上がってブロック間の相対音量が失われる。

同じ練習の連続した音源として聴くことを考えると `date` が自然なので既定にしている。
`block` は「1ブロックだけ極端に小さく録れてしまった」ような場合の逃げ道として残してある。
どのブロックが基準ピークを与えたか、系統ごとの単一ゲインはいくつかは実行時にログへ出る。

実データでは 32bit float 録音のピークが **0 dBFS を超えていた**(260726 の外部マイクで +9.7 dBFS)。
32bit float なのでファイル上は壊れていないが、固定小数点で書き出せば激しくクリップする。
つまりこの正規化は「音量を揃える」だけでなく、後段のエクスポートを成立させるために必要な工程である。

**ミックス**は `source` に従う。`ext_only`/`int_only` は正規化済みファイルをそのままコピーするだけで、
合成しないためピーク超過は起こりえない。`source=mix` のときだけ加算合成し、合成ピークが安全閾値
(既定 -1 dBFS)を超えていたら**一律の線形ゲインで下げる**(コンプ・リミッタは使わない)。

なお指示書には「位相の重なりでどちらの原音のピークよりも高い合成ピークが生まれる」とあるが、
比率の合計が 1.0 以下(6:4、8:2 など)の凸結合では三角不等式より
`|w1·a + w2·b| ≤ w1·|a| + w2·|b| ≤ max(peak_a, peak_b)` が常に成り立ち、超過は数学的に起こりえない。
実際に超過するのは比率の合計が 1.0 を超える設定(例 `ext=1.0, int=0.8`)である。比率を固定しない
仕様である以上その設定は取りうるので安全処理は必要だが、発動条件は位相ではなく比率の合計である。
合成データで両方を実測し、6:4 では -1.42 dBFS(調整なし)、1.0:0.8 では +3.66 dBFS を検出して
-4.66 dB 下げ、正確に -1.00 dBFS に収まることを確認している。

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
| `normalize` | `--target` | -1 dBFS | 目標ピーク |
| `normalize` | `--ref-margin` | 120 秒 | 基準ピークの算出から除外する前後の長さ |
| `mix` | `--safe-peak` | -1 dBFS | 合成後に超えてはならないピーク |
| `box-upload` | `--auth-timeout` | 300 秒 | 初回認証でブラウザ操作を待つ秒数 |
| `gdrive-upload` | `--auth-timeout` | 無制限 | 同上 |
| `notify` | `--no-line` | — | LINE への push を行わない |
| 全コマンド | `--root` / `--force` | — | 作業ディレクトリ / 中間ファイルの作り直し |
