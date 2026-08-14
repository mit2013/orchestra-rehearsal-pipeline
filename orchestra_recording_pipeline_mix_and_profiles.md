# アマオケ練習録音 自動編集パイプライン ― 正規化・ミックス設計 & レコーダプロファイル抽象化

## 0. 位置づけ
フェーズ1(取り込み→結合→トリミング)は260802・260726の2データセットで検証済み、実用域にあると判断済み。本指示書は、フェーズ2(曲目単位エクスポート・MP3タグ埋め込み等)に進む前段として、以下3点を実装する。

1. トラック単位の正規化(ext/intそれぞれ独立)
2. 外部マイク・内蔵マイクのミックス(設定可能な比率、安全なピーク処理込み)
3. LRマッピング・ミックス比率・レコーダ機種の外部設定化、および将来の他機種(ZOOM F3、単一ファイル入力)対応のためのレコーダプロファイル抽象化(インターフェースのみ、実装はZOOM M4のみ)

**重要な制約**: この変更は、既存のZOOM M4向けの挙動(260802・260726で検証済みのファイル探索・結合・トリミング結果)を一切変えてはならない。実装後、両日付のデータで`ingest`〜`apply`までを再実行し、結果が変わらないことを確認すること(リグレッションチェック)。

## 1. 正規化(normalize)
- 対象: `apply`で出力済みの`trimmed/`配下の各ブロックext/intファイル
- 方式: ピーク正規化、目標値は-1dBFS(true peak想定、後段のミックス時の安全マージンとして)
- 基準値の算出範囲: guard(既定120秒)で外側に広げた部分(音出し・休憩が混入している可能性がある)は正規化の基準計算から除外し、本編とみなせる中央部分のみを使う。ただし正規化処理自体はブロック全体(guard分含む)に適用する(基準の算出範囲と適用範囲は分ける)
- 出力: `trimmed/01_合奏1_ext_norm.wav` 等(非破壊。元のext/intファイルは残す)
- 備考: ピーク正規化以外の方式(RMS/LUFS等)は今回は採用しない。聴いてみて物足りなければ次回以降に見直す

## 2. ミックス(mix)
- 対象: normalize済みのext/intファイル
- 設定は`output/{date}/session_config.json`に保存する(日付ごとに1ファイル、`ingest`時にデフォルト値で生成し、`confirmed.json`と同様にユーザーが手で編集できる):

```json
{
  "ext_lr_map": "normal",
  "source": "ext_only",
  "mix_ratio": {"ext": 0.6, "int": 0.4}
}
```

- `source`: `ext_only` / `int_only` / `mix`。**デフォルトは`ext_only`**
- `mix_ratio`: `source`が`mix`の時のみ使用。比率は6:4、8:2等、今後試す想定なので固定値にしないこと
- `ext_lr_map`: `normal`(Tr1=L, Tr2=R) / `swapped`(Tr1=R, Tr2=L)。配線ミスは録音セッション単位の話なので、日付ごとに上書きできること

処理内容:
- `source=ext_only` / `int_only`: 対応するnormalize済みファイルをそのまま採用(単純コピーまたはリネーム)。合成しないためピーク超過のリスクはない
- `source=mix`: ext_norm・int_normを指定比率で加算合成し、合成後のピークを検出する。0dBFS(または-1dBFS等の安全閾値)を超えていたら、**信号全体を線形にスケールダウン**して安全域に収める。コンプレッサーやリミッターのようなダイナミクスを変える処理は使わないこと(単純な一律ゲイン調整のみ)
  - 外部マイクと内蔵マイクは同じ音を別位置で拾っている相関の高い信号のため、単純合成すると位相の重なりでどちらの原音のピークよりも高い合成ピークが生まれることがある。この安全処理は「万が一」ではなく構造的に必要
- 出力: `trimmed/01_合奏1_final.wav` 等(この先のWAV/MP3への最終エクスポートの元になるファイル)

## 3. LRマッピングの適用箇所
- `ext_lr_map`は`merge`ステップ(Tr1+Tr2→外部ステレオを組み立てる箇所)で適用する。これより下流(propose/apply/normalize/mix)は常に正しいL/Rが揃っている前提でよい
- `merge`実行時に`session_config.json`の`ext_lr_map`を読み込み、`swapped`ならTr1/Tr2を入れ替えて合成する

## 4. レコーダプロファイル抽象化
`orchpipe/recorder_profiles/`(新設)に、以下のインターフェースを定義する:

```python
class RecorderProfile:
    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        """指定ディレクトリ・日付から、TAKE単位のファイル群を発見する"""
    channel_groups: dict[str, list[str]]  # 例: {"ext": ["Tr1", "Tr2"], "int": ["TrMic"]}
    needs_take_concat: bool  # 複数TAKEの連結が必要か
```

- **`ZoomM4Profile`(本実装)**: 現在`orchpipe/ingest.py`にある`find_take_dirs()`のロジックをそのまま移植する。`{date}_{num}.TAKE/`フォルダ、`channel_groups={"ext": ["Tr1","Tr2"], "int": ["TrMic"]}`、`needs_take_concat=True`
- **`ZoomF3Profile`(スタブのみ、呼び出すとNotImplementedErrorで分かりやすく停止)**: ZOOM F3公式マニュアルで確認したところ、フォルダを掘らずルート直下に`{date}_{番号}.WAV`(ステレオ1ファイルモード)または`{date}_{番号}_Tr1.WAV`+`{date}_{番号}_Tr2.WAV`(モノラル2ファイルモード)が作られる。内蔵マイクがないため`channel_groups`は`ext`のみになる想定。これらは実装時の参考として docstring に残すが、今回はコード本体は実装しない
- **`SingleFileProfile`(スタブのみ、同様にNotImplementedError)**: 単一のWAVまたはMP3ファイルをそのまま渡すケース。`channel_groups`は1つ、`needs_take_concat=False`。MP3の場合、後続の特徴量抽出(numpy/soundfile)のために事前にffmpegでWAVへ変換する必要がある、という実装メモのみdocstringに残す
- `pipeline.py ingest --recorder zoom-m4 --date ...` のように`--recorder`引数でプロファイルを選択する。デフォルトは`zoom-m4`
- 今回は`ZoomM4Profile`のみ実データ(260802・260726)で検証し、他2つは「未実装であることが呼び出した瞬間に分かる」状態にしておく

## 5. CLIコマンド(更新後の全体像)
```bash
python pipeline.py ingest --date 260802 --recorder zoom-m4
python pipeline.py merge --date 260802
python pipeline.py propose --date 260802 --splits 2
# confirmed.json を確認・編集
python pipeline.py apply --date 260802
# session_config.json を確認・編集(source, mix_ratio, ext_lr_map)
python pipeline.py normalize --date 260802
python pipeline.py mix --date 260802
```

## 6. 今回の範囲外
- 曲目単位でのエクスポート分割(セットリスト入力方法は未決定)
- WAV/MP3形式への最終エクスポート、MP3タグ埋め込み
- クラウドアップロード・通知
- `ZoomF3Profile` / `SingleFileProfile` の本実装
- ピーク正規化以外の方式(RMS/LUFS等)の検討

## 7. 検証してほしいこと
- `ext_lr_map=swapped`を指定した場合に、実際にL/Rが入れ替わって出力されることを合成データで検証する
- `source=mix`, `mix_ratio=6:4`で、位相加算によりピークが単体より上がるケースを人工的に作り(例えば同じ波形を時間を揃えてコピーし足す等)、安全スケールダウンが正しく発動することを検証する
- 260802・260726の既存データに対し、デフォルト設定(`source=ext_only`)でnormalize/mixを通しても、フェーズ1の成果物(trimmed済みファイル)や既存の検証結果を壊さないことを確認する(セクション0のリグレッションチェック)
