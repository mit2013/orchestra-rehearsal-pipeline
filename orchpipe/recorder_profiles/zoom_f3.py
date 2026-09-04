"""ZOOM F3 プロファイル(本実装)。

260704 の実データ(32bit float / 48kHz / ステレオ / 5 TAKE)で検証済み。

M4 と違い **TAKE フォルダを掘らず、ルート直下にファイルを置く**。録音モードが
2種類あり、ファイル名で見分けられる。

    ステレオ1ファイルモード:
        {date}_{番号}.WAV                 ステレオ1本

    モノラル2ファイルモード:
        {date}_{番号}_Tr1.WAV             CH1 (モノラル)
        {date}_{番号}_Tr2.WAV             CH2 (モノラル)

番号はファイルサイズ上限によるインクリメントで、時系列順。F3 は FAT32 の 2GB で
切るので、3時間の練習でも 3〜5 本に分かれる。

## 系統は ext だけ

F3 には内蔵マイクがない。したがって `channel_groups` は `ext` のみで、
`session_config.json` の `source` は実質 `ext_only` 固定になる。`mix_ratio` を
指定しても混ぜる相手がいないので効かない。

## 左右の入れ替え

ステレオ1ファイルモードでは、L/R はファイルの中で既に決まっている。外部マイクの
配線が逆だった日は `ext_lr_map: swapped` を指定すると、`merge` が
`pan=stereo|c0=c1|c1=c0` で入れ替える(モノラル2本を組む場合の `join` の
map 指定と同じ結果になる)。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..util import PipelineError, log
from .base import RecorderProfile, TakeInfo

STEREO_RE = re.compile(r"^(?P<date>\d{6})_(?P<num>\d{3})\.WAV$", re.IGNORECASE)
MONO_RE = re.compile(r"^(?P<date>\d{6})_(?P<num>\d{3})_(?P<track>Tr[12])\.WAV$", re.IGNORECASE)

STEREO_TRACK = "Stereo"
MONO_TRACKS = ["Tr1", "Tr2"]


class ZoomF3Profile(RecorderProfile):
    name = "zoom-f3"
    # 既定はステレオ1ファイル。`discover` がモードを見て必要なら差し替える。
    channel_groups = {"ext": [STEREO_TRACK]}
    needs_take_concat = True

    def __init__(self) -> None:
        # クラス属性を書き換えないよう、インスタンスに持たせ直す。
        self.channel_groups = {"ext": [STEREO_TRACK]}
        self.mode = "stereo"

    # -- モードの判別 ------------------------------------------------------

    def _scan(self, root_dir: Path, date: str) -> tuple[dict[int, Path], dict[int, dict[str, Path]]]:
        """ルート直下を1回だけ走査し、ステレオ版とモノラル版の候補を両方集める。"""
        stereo: dict[int, Path] = {}
        mono: dict[int, dict[str, Path]] = {}
        if not root_dir.is_dir():
            raise PipelineError(f"{root_dir} がありません")
        for p in sorted(root_dir.iterdir()):
            if not p.is_file():
                continue
            m = MONO_RE.match(p.name)
            if m and m.group("date") == date:
                mono.setdefault(int(m.group("num")), {})[m.group("track").title()] = p
                continue
            m = STEREO_RE.match(p.name)
            if m and m.group("date") == date:
                stereo[int(m.group("num"))] = p
        return stereo, mono

    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        stereo, mono = self._scan(root_dir, date)

        # 両モードが混在した日付は想定しない。多いほうを採り、警告を出す。
        if stereo and mono:
            log(f"警告: {date} にステレオ1ファイル({len(stereo)}本)とモノラル2ファイル"
                f"({len(mono)}本)が混在しています。数の多いほうを使います。")
            if len(mono) >= len(stereo):
                stereo = {}
            else:
                mono = {}

        if mono:
            incomplete = sorted(n for n, f in mono.items() if set(f) != set(MONO_TRACKS))
            if incomplete:
                missing = ", ".join(
                    f"{date}_{n:03d}_" + "/".join(sorted(set(MONO_TRACKS) - set(mono[n])))
                    for n in incomplete
                )
                raise PipelineError(
                    f"モノラル2ファイルモードですが、片側しかない TAKE があります: {missing}.WAV"
                )
            self.mode = "mono2"
            self.channel_groups = {"ext": list(MONO_TRACKS)}
            return [
                TakeInfo(date=date, number=n, dir=None, files=dict(mono[n]))
                for n in sorted(mono)
            ]

        if stereo:
            self.mode = "stereo"
            self.channel_groups = {"ext": [STEREO_TRACK]}
            return [
                TakeInfo(date=date, number=n, dir=None, files={STEREO_TRACK: stereo[n]})
                for n in sorted(stereo)
            ]

        raise PipelineError(
            f"{root_dir} に日付 {date} の F3 のファイルが見つかりません。\n"
            f"  ステレオ1ファイルモードなら {date}_001.WAV、\n"
            f"  モノラル2ファイルモードなら {date}_001_Tr1.WAV / {date}_001_Tr2.WAV を探します。\n"
            "  SD カードのルート直下にあるか、日付の綴りが合っているか確認してください。"
        )

    # -- 現場スクリプト用 --------------------------------------------------

    def relative_files(self, date: str, number: int) -> dict[str, str]:
        """iPhone にコピーしたときの相対パス。F3 はフォルダを掘らない。"""
        if self.mode == "mono2":
            return {t: f"{date}_{number:03d}_{t}.WAV" for t in MONO_TRACKS}
        return {STEREO_TRACK: f"{date}_{number:03d}.WAV"}
