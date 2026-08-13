"""レコーダ機種ごとの差異を吸収する層。

パイプラインの上流(どのファイルがどのTAKEの、どのチャンネル群に属するか)だけが
機種によって変わる。それ以降(結合・境界検出・トリミング・正規化・ミックス)は
機種を問わず同じ処理でよい。その境目をこのモジュールで切っている。

今回実装するのは `ZoomM4Profile` のみ。他機種はインターフェースの形だけ用意し、
呼び出した瞬間に未実装だと分かるようにしてある。
"""

from __future__ import annotations

from .base import RecorderProfile, TakeInfo
from .single_file import SingleFileProfile
from .zoom_f3 import ZoomF3Profile
from .zoom_m4 import ZoomM4Profile

PROFILES: dict[str, type[RecorderProfile]] = {
    "zoom-m4": ZoomM4Profile,
    "zoom-f3": ZoomF3Profile,
    "single-file": SingleFileProfile,
}

DEFAULT_PROFILE = "zoom-m4"


def get_profile(name: str) -> RecorderProfile:
    """名前からプロファイルのインスタンスを得る。"""
    from ..util import PipelineError

    try:
        cls = PROFILES[name]
    except KeyError:
        raise PipelineError(
            f"未知のレコーダ機種です: {name!r}(利用可能: {', '.join(sorted(PROFILES))})"
        ) from None
    return cls()


__all__ = [
    "RecorderProfile",
    "TakeInfo",
    "ZoomM4Profile",
    "ZoomF3Profile",
    "SingleFileProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "get_profile",
]
