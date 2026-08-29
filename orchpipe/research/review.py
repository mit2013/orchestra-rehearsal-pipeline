"""ダイジェストの境界レビュー基盤。

`digest2.wav` を通しで聴くのは 40〜60 分かかるうえ、確認したいのは全体ではなく
**採用範囲の始まりと終わり**である。そこで各範囲の境界について、前後の文脈を含む
短いクリップと、記入用の CSV を書き出す。

クリップは**ダイジェストからではなく元のブロックから**切り出す。ダイジェストでは
カットの向こう側が既に失われており、「切りすぎ / 残しすぎ」を判断できないため。
クリップの中で、実際に採用されている範囲がどこから(どこまで)なのかは
`cut_offset_sec` 列で分かるようにしてある。

ステージF-1/F-3 は実施が見送られたため実装されていない。本モジュールはその
「候補抽出 → クリップ生成 → 記入用シート」という構成を、境界レビューに絞って
作り直したものである。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..util import FFMPEG, fmt_time, log, run

# クリップに含める前後の文脈
PRE_S = 6.0
POST_S = 6.0


@dataclass
class ReviewClip:
    clip_id: str
    block: str
    kind: str            # "start" | "end"
    range_index: int
    block_time: float    # 元ブロック上のカット位置
    digest_time: float   # ダイジェスト上での対応位置
    clip_start: float
    clip_end: float
    context_before: str  # カットの手前にあるラベル
    context_after: str   # カットの先にあるラベル
    play_gap: float      # カット位置から、確定した演奏までの秒数
    range_dur: float     # この範囲の長さ
    path: Path

    @property
    def cut_offset(self) -> float:
        """クリップの先頭からカット位置までの秒数。"""
        return self.block_time - self.clip_start


def _label_at(spans, t: float) -> str:
    for s in spans:
        if s.start <= t < s.end:
            return s.label
    return "-"


def digest_positions(ranges: list[tuple[float, float]], crossfade: float) -> list[tuple[float, float]]:
    """各範囲が、ダイジェスト上のどこに来るか (開始, 終了) を返す。"""
    out, acc = [], 0.0
    for i, (a, b) in enumerate(ranges):
        if i > 0:
            acc -= crossfade
        out.append((acc, acc + (b - a)))
        acc += b - a
    return out


def build(
    src: Path,
    block: str,
    ranges: list[tuple[float, float]],
    spans,
    outdir: Path,
    crossfade: float,
    pre_s: float = PRE_S,
    post_s: float = POST_S,
    total: float | None = None,
) -> list[ReviewClip]:
    """各範囲の始まり・終わりについてクリップと CSV を書き出す。"""
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.wav"):
        old.unlink()

    dpos = digest_positions(ranges, crossfade)
    clips: list[ReviewClip] = []

    # 範囲は playing 区間をマージン+予備拍で広げたものなので、カット位置そのものは
    # たいてい silence や unclear の中にある。レビューする側が知りたいのは
    # 「そこから何秒後に演奏が始まるのか(何秒前に終わったのか)」なので、
    # 確定した playing 区間との距離を別に出しておく。
    plays = sorted((s.start, s.end) for s in spans if s.label == "playing")

    def gap_to_play(t: float, kind: str) -> float:
        if not plays:
            return float("nan")
        if kind == "start":
            after = [ps for ps, _pe in plays if ps >= t - 0.01]
            return (min(after) - t) if after else float("nan")
        before = [pe for _ps, pe in plays if pe <= t + 0.01]
        return (t - max(before)) if before else float("nan")

    for i, ((a, b), (da, db)) in enumerate(zip(ranges, dpos), start=1):
        for kind, t, dt in (("start", a, da), ("end", b, db)):
            c0 = max(0.0, t - pre_s)
            c1 = t + post_s if total is None else min(total, t + post_s)
            if c1 - c0 < 1.0:
                continue
            cid = f"{i:03d}_{kind}"
            dst = outdir / f"{cid}_{fmt_time(t).replace(':', '-')}.wav"
            clips.append(ReviewClip(
                clip_id=cid, block=block, kind=kind, range_index=i,
                block_time=t, digest_time=dt, clip_start=c0, clip_end=c1,
                # 「手前」「先」は常に元ブロックの時間軸で見る
                context_before=_label_at(spans, max(0.0, t - 1.0)),
                context_after=_label_at(spans, t + 1.0),
                play_gap=gap_to_play(t, kind), range_dur=b - a,
                path=dst,
            ))

    log(f"  {block}: {len(ranges)} 範囲 -> レビュークリップ {len(clips)} 本")
    for c in clips:
        run([FFMPEG, "-hide_banner", "-v", "error", "-y",
             "-ss", f"{c.clip_start:.3f}", "-t", f"{c.clip_end - c.clip_start:.3f}",
             "-i", str(src), "-c:a", "pcm_f32le", str(c.path)])
    return clips


def write_sheet(clips: list[ReviewClip], dst: Path) -> None:
    """記入用の CSV。正解ラベルとメモの列は空で出す。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "クリップID", "ファイル名", "ブロック", "種別", "範囲番号",
            "元ブロックの時刻", "ダイジェスト上の時刻",
            "クリップ内のカット位置[秒]", "演奏までの距離[秒]", "範囲の長さ[秒]",
            "カットの手前", "カットの先",
            "判定(ok/切りすぎ/残しすぎ)", "メモ",
        ])
        for c in clips:
            w.writerow([
                c.clip_id, c.path.name, c.block,
                "始まり" if c.kind == "start" else "終わり", c.range_index,
                fmt_time(c.block_time), fmt_time(c.digest_time),
                f"{c.cut_offset:.1f}",
                ("" if c.play_gap != c.play_gap else f"{c.play_gap:.1f}"),
                f"{c.range_dur:.1f}", c.context_before, c.context_after,
                "", "",
            ])
    log(f"  レビューシート: {dst}")
