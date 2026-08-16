"""ステージB: 合奏開始(チューニング開始)のオンセット検出。

**trimmed/ ではなく結合ファイル(raw_merged_ext.wav)上で検出する。**

指示書 B-3 は「ステージAで細分化した tuning 区間のうち」と書かれているが、
ステージAは `trimmed/` を対象にしており、そこには次の問題がある:

- 正解データ(B-0)は結合ファイルのタイムライン基準で与えられる
- 真のチューニング開始がトリミング境界より**前**にある場合、`trimmed/` には
  そもそも含まれていないので検出しようがない(260802 は境界＝概算時刻なので
  この可能性が現実にある)

したがって、検出は結合ファイル上で、各ブロックの開始付近を探索窓として行う。
使う判定ロジック自体はステージAと同じ `research/tuning.py` である。

オンセットの時間分解能は、`features.py` のフレームホップ(16ms)で決まる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import features as F
from ..util import fmt_time, log
from .tuning import TuningEvent, detect_tuning_events

# 探索窓。基準時刻の前後にこれだけ見る。
SEARCH_BACK_S = 180.0
SEARCH_FWD_S = 180.0


@dataclass
class OnsetResult:
    block: str
    reference: float | None       # 正解(あれば)
    detected: float | None        # 検出したチューニング開始
    error: float | None           # detected - reference
    event: TuningEvent | None
    candidates: list[TuningEvent]

    def to_json(self) -> dict:
        return {
            "block": self.block,
            "reference": self.reference,
            "detected": self.detected,
            "error": None if self.error is None else round(self.error, 2),
            "event": None if self.event is None else self.event.to_json(),
            "candidates": [e.to_json() for e in self.candidates],
        }


def detect_onsets(
    merged: Path,
    anchors: list[tuple[str, float]],
    *,
    references: dict[str, float] | None = None,
    back: float = SEARCH_BACK_S,
    fwd: float = SEARCH_FWD_S,
) -> list[OnsetResult]:
    """結合ファイル上で、各アンカー(ブロック開始の目安)付近のチューニング開始を求める。

    `anchors` は [(ブロック名, 目安時刻秒), ...]。
    `references` に正解時刻を渡すと誤差も計算する。
    """
    log(f"特徴量を抽出: {merged.name}")
    ff = F.extract_frame_features(merged, progress_every=1e9)
    wf = F.aggregate_windows(ff, win_s=1.0, hop_s=0.5)
    events = detect_tuning_events(ff, wf.silence_db)
    log(f"  チューニング候補(全体): {len(events)} 件")

    results: list[OnsetResult] = []
    for name, anchor in anchors:
        lo, hi = anchor - back, anchor + fwd
        cands = [e for e in events if lo <= e.start <= hi]
        # 探索窓内でスコア(純度 × 長さ)が最も高いものを採用する。
        # 実データでは本物のチューニングが純度 0.61〜0.79・長さ 22〜74 秒で、
        # 紛らわしい周辺区間(純度 0.02〜0.47)と明確に分離できた。
        best = max(cands, key=lambda e: e.score) if cands else None
        ref = (references or {}).get(name)
        err = None if (best is None or ref is None) else best.start - ref
        results.append(OnsetResult(
            block=name, reference=ref,
            detected=None if best is None else best.start,
            error=err, event=best, candidates=cands,
        ))
        if best is None:
            log(f"  {name}: 探索窓 {fmt_time(max(0,lo))}〜{fmt_time(hi)} に候補なし")
        else:
            msg = (f"  {name}: 検出 {fmt_time(best.start)} "
                   f"(長さ{best.duration:.0f}s 純度{best.purity:.2f} "
                   f"{best.peak_hz:.0f}Hz score={best.score:.0f})")
            if ref is not None:
                msg += f"  正解 {fmt_time(ref)}  誤差 {err:+.1f}秒"
            log(msg)
    return results
