"""ZOOM M4 MicTrak プロファイル(本実装)。

260802・260726 の実データで検証済み。ファイル配置は次のとおり:

    {date}_{番号}.TAKE/
        {date}_{番号}_Tr1.WAV     外部マイク CH1  (モノラル)
        {date}_{番号}_Tr2.WAV     外部マイク CH2  (モノラル)
        {date}_{番号}_TrMic.WAV   内蔵マイク      (ステレオ)

番号はSDカードのファイルサイズ上限によるインクリメントで、時系列順。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..util import PipelineError
from .base import RecorderProfile, TakeInfo

TAKE_DIR_RE = re.compile(r"^(?P<date>\d{6})_(?P<num>\d{3})\.TAKE$")


class ZoomM4Profile(RecorderProfile):
    name = "zoom-m4"
    channel_groups = {"ext": ["Tr1", "Tr2"], "int": ["TrMic"]}
    needs_take_concat = True

    def find_take_dirs(self, root_dir: Path, date: str) -> list[tuple[int, Path]]:
        """`{date}_{番号}.TAKE` フォルダを日付で厳密に絞り込み、番号順に返す。

        ディレクトリ名の日付部分と `date` の完全一致を要求するので、同じ親ディレクトリに
        複数の練習日のTAKEが並んでいても取り違えない。
        """
        found: list[tuple[int, Path]] = []
        for p in sorted(root_dir.iterdir()):
            if not p.is_dir():
                continue
            m = TAKE_DIR_RE.match(p.name)
            if m and m.group("date") == date:
                found.append((int(m.group("num")), p))
        found.sort(key=lambda t: t[0])
        return found

    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        takes: list[TakeInfo] = []
        for number, d in self.find_take_dirs(root_dir, date):
            stem = f"{date}_{number:03d}"
            takes.append(
                TakeInfo(
                    date=date,
                    number=number,
                    dir=d,
                    files={t: d / f"{stem}_{t}.WAV" for t in self.track_names},
                )
            )
        if not takes:
            raise PipelineError(f"{root_dir} に {date}_XXX.TAKE フォルダが見つかりません")
        return takes

    def relative_files(self, date: str, number: int) -> dict[str, str]:
        """iPhone にコピーしたときの相対パス。M4 は TAKE フォルダごとコピーする。"""
        stem = f"{date}_{number:03d}"
        return {t: f"{stem}.TAKE/{stem}_{t}.WAV" for t in self.track_names}
