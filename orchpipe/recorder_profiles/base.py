"""レコーダプロファイルのインターフェース定義。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TakeInfo:
    """1 TAKE ぶんのファイル群。

    `files` はトラック名(レコーダが付ける呼び名。ZOOM M4 なら "Tr1"/"Tr2"/"TrMic")から
    実ファイルへの対応。どのトラックがどの系統(ext/int)に属するかは
    `RecorderProfile.channel_groups` 側が持つ。
    """

    date: str
    number: int
    files: dict[str, Path]
    dir: Path | None = None

    def as_dict(self) -> dict:
        return {
            "date": self.date,
            "number": self.number,
            "dir": str(self.dir) if self.dir is not None else None,
            "files": {k: str(v) for k, v in self.files.items()},
        }


class RecorderProfile:
    """レコーダ機種ごとのファイル配置を抽象化する基底クラス。

    サブクラスは以下を定義する:

    - ``name``            : `--recorder` で指定する識別子
    - ``channel_groups``  : 系統名 -> トラック名の並び。
                            例) {"ext": ["Tr1", "Tr2"], "int": ["TrMic"]}
                            2要素なら2本のモノラルを左右に組んでステレオ化、
                            1要素ならそのファイルがすでにステレオ、と解釈する。
    - ``needs_take_concat``: 複数TAKEの連結が必要か
    - ``discover()``      : ディレクトリと日付から TAKE 群を発見する
    """

    name: str = ""
    channel_groups: dict[str, list[str]] = {}
    needs_take_concat: bool = True

    def discover(self, root_dir: Path, date: str) -> list[TakeInfo]:
        """指定ディレクトリ・日付から、TAKE単位のファイル群を発見する。

        1件も見つからない場合は、その機種のファイル配置に即した説明を含む
        `PipelineError` を送出すること。何をどこに探しに行ったのかは機種ごとに
        まったく違う(M4はTAKEフォルダ、F3はルート直下のファイル、single-fileは
        単一ファイル)ため、メッセージは呼び出し側ではなく各プロファイルが持つ。
        """
        raise NotImplementedError

    def relative_files(self, date: str, number: int) -> dict[str, str]:
        """TAKE 番号から、コピー先で期待される相対パスを返す。

        現場前処理では母艦に原本が無い状態でスクリプトを組み立てるので、実ファイルを
        見ずにパスを作れる必要がある。`discover` と同じ命名規則をここにも書く。
        """
        raise NotImplementedError

    # -- 以下は共通のヘルパ ------------------------------------------------

    @property
    def track_names(self) -> list[str]:
        """全系統のトラック名を並び順で平坦化したもの。"""
        out: list[str] = []
        for names in self.channel_groups.values():
            out.extend(names)
        return out

    def group_of(self, track: str) -> str | None:
        for group, names in self.channel_groups.items():
            if track in names:
                return group
        return None
