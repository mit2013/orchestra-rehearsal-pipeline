"""単一ファイル入力プロファイル(スタブ。今回は未実装)。

呼び出すと NotImplementedError で明示的に停止する。

想定する入力は、TAKE 分割のない WAV または MP3 が1本だけあるケース
(他人からもらった音源、別機材で録ったものなど)。

実装上の注意:

- `channel_groups` は系統1つのみ、`needs_take_concat=False`。
  連結が不要なので `merge` は実質パススルー(またはコピー)になる。
- **MP3 が入力の場合、事前に ffmpeg で WAV(pcm_f32le)へ変換する必要がある。**
  後続の特徴量抽出は numpy / soundfile 経由でサンプルを直接読むため、
  圧縮フォーマットのままでは扱えない。変換結果を output/{date}/ に置いて
  以降はそれを入力とみなすのが素直。
- 単一ファイルなので `ext`/`int` の区別がなく、正規化は1系統に対してのみ行う。
  session_config.json の `source` は `ext_only` 相当に固定され、`mix` は使えない。
"""

from __future__ import annotations

from pathlib import Path

from .base import RecorderProfile, TakeInfo


class SingleFileProfile(RecorderProfile):
    name = "single-file"
    channel_groups = {"main": ["Main"]}
    needs_take_concat = False

    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        raise NotImplementedError(
            "SingleFileProfile は未実装です。現在実装済みのプロファイルは zoom-m4 のみです。"
            "(MP3入力時の事前WAV変換などの実装メモは"
            " orchpipe/recorder_profiles/single_file.py の docstring に記載してあります)"
        )
