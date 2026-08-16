"""ステージC: ダイジェスト版の作成。

ステージAで `playing` と判定された区間だけを、前後にマージンを付けて抜き出し、
時系列順に結合する。結合点にはクロスフェードを入れてクリックノイズを避ける。

マージンで隣接区間と重なった場合は統合する。指示書 C-3 の
「迷ったら: マージンで演奏を削りすぎるより多めに残す」に従い、
マージンは前後 3 秒を既定とし、統合の閾値も緩めにしてある。
"""

from __future__ import annotations

from pathlib import Path

from ..util import FFMPEG, PipelineError, log, run
from .states import StateSpan

MARGIN_S = 3.0
CROSSFADE_S = 0.075   # 75ms。指示書の 50〜100ms の中間
MIN_KEEP_S = 1.0      # これより短い断片は繋いでも聴き取れないので捨てる


def playing_ranges(spans: list[StateSpan], total: float,
                   margin: float = MARGIN_S) -> list[tuple[float, float]]:
    """`playing` 区間にマージンを付け、重なりを統合した範囲リスト。"""
    raw = [(max(0.0, s.start - margin), min(total, s.end + margin))
           for s in spans if s.label == "playing"]
    if not raw:
        return []
    raw.sort()
    merged = [list(raw[0])]
    for a, b in raw[1:]:
        if a <= merged[-1][1]:          # 重なる/接する → 統合
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_KEEP_S]


def build_digest(src: Path, ranges: list[tuple[float, float]], dst: Path,
                 crossfade: float = CROSSFADE_S) -> dict:
    """範囲を切り出してクロスフェード結合する。出力は 32bit float。"""
    if not ranges:
        raise PipelineError(f"{src.name}: playing 区間がありません")

    dst.parent.mkdir(parents=True, exist_ok=True)
    total_kept = sum(b - a for a, b in ranges)

    # 1本しかないなら素直に切り出すだけ(クロスフェード不要)
    if len(ranges) == 1:
        a, b = ranges[0]
        run([FFMPEG, "-hide_banner", "-v", "error", "-y",
             "-ss", f"{a:.3f}", "-t", f"{b-a:.3f}", "-i", str(src),
             "-c:a", "pcm_f32le", "-rf64", "auto", str(dst)],
            desc=f"    {dst.name}(1区間)")
        return {"n_ranges": 1, "kept_seconds": total_kept}

    # acrossfade は2入力ずつなので、フィルタグラフで数珠つなぎにする。
    # 各区間は crossfade ぶん重ねて繋ぐため、実尺は (n-1)*crossfade だけ短くなる。
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-y"]
    for a, b in ranges:
        cmd += ["-ss", f"{a:.3f}", "-t", f"{b-a:.3f}", "-i", str(src)]

    chains = []
    prev = "0:a"
    for i in range(1, len(ranges)):
        out = f"x{i}"
        chains.append(f"[{prev}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out}]")
        prev = out
    cmd += ["-filter_complex", ";".join(chains), "-map", f"[{prev}]",
            "-c:a", "pcm_f32le", "-rf64", "auto", str(dst)]
    run(cmd, desc=f"    {dst.name}({len(ranges)}区間をクロスフェード結合)")
    return {"n_ranges": len(ranges), "kept_seconds": total_kept,
            "crossfade_loss": (len(ranges) - 1) * crossfade}
