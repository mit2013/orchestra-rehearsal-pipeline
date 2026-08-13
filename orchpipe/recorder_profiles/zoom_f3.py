"""ZOOM F3 プロファイル(スタブ。今回は未実装)。

呼び出すと NotImplementedError で明示的に停止する。実装時の参考として、
公式マニュアルで確認したファイル配置をここに残しておく。

ファイル配置(M4 と違い TAKE フォルダを掘らず、ルート直下に置かれる):

    ステレオ1ファイルモード:
        {date}_{番号}.WAV                 ステレオ1本

    モノラル2ファイルモード:
        {date}_{番号}_Tr1.WAV             CH1 (モノラル)
        {date}_{番号}_Tr2.WAV             CH2 (モノラル)

実装上の注意:

- F3 には内蔵マイクがないため `channel_groups` は "ext" のみになる想定。
  したがって session_config.json の `source` は実質 `ext_only` 固定で、
  `mix` は使えない(指定されたら分かりやすく落とすこと)。
- 上記2モードのどちらであるかは、ルート直下に `_Tr1`/`_Tr2` 付きのファイルが
  あるかどうかで判別できる。両モードが混在した日付は想定しなくてよい。
- TAKE フォルダがないだけで、番号によるインクリメント分割は M4 と同じなので
  `needs_take_concat=True` のままでよい。
"""

from __future__ import annotations

from pathlib import Path

from .base import RecorderProfile, TakeInfo


class ZoomF3Profile(RecorderProfile):
    name = "zoom-f3"
    channel_groups = {"ext": ["Tr1", "Tr2"]}
    needs_take_concat = True

    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        raise NotImplementedError(
            "ZoomF3Profile は未実装です。現在実装済みのプロファイルは zoom-m4 のみです。"
            "(実装に必要なファイル配置の情報は orchpipe/recorder_profiles/zoom_f3.py の"
            " docstring に記載してあります)"
        )
