"""チューニング検出(研究ブランチ版)。

main の `segment.tuning_frames` は「純音性が高く・ピッチが安定し・スペクトル変化が
小さい」という条件だけで判定していた。そのため合奏中のロングトーンを誤検出し、
逆に本物のチューニング開始を取り逃がしていた(main の report.md §5 の既知の弱点)。

本モジュールは実データの診断にもとづいて設計し直したもの。効いた点は3つ:

1. **基準音 A の「倍音」まで見る(最重要)**
   静かなオーボエ単独音では、部屋鳴りマイクだと基音より上位倍音が強く出る。
   実測では 260802 合奏1 のチューニング開始直後、支配ピークは **1317Hz**
   (A=439Hz の第3倍音)であって 440Hz ではなかった。A のオクターブだけを
   見ていた旧実装はここで完全に取り逃がしていた。オクターブ × 倍音1〜5 を
   参照集合にすることで、同じ区間の充足率が 4% → 92% に跳ね上がった。

2. **ピッチ安定性の条件を外した**
   ピークが基音と倍音の間を行き来するため「不安定」と判定され、本物の
   チューニングが落ちていた。倍音マッチ自体がピッチを保証しているので、
   この条件は冗長かつ有害だった。

3. **「純度」で選ぶ**
   区間内で判定条件を満たすフレームの割合を純度と呼ぶ。実データでは
   本物のチューニングが 0.61〜0.79、紛らわしい周辺区間が 0.02〜0.47 と
   明確に分離した。長さ(12秒以上)と併せると、6時間の素材から7件しか
   拾わない選択性が得られる。

main の `features.py` は一切変更していない。`FrameFeatures` を入力に取るだけ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import FrameFeatures

# オーケストラの基準音 A(442Hz 前後で運用される)
A_CENTER_HZ = 442.0
# 参照するオクターブ(A2〜A6 相当)
A_OCTAVES = (-2, -1, 0, 1, 2)
# 参照する倍音。基音が弱く倍音が支配的になる場合を拾うために必須。
A_HARMONICS = (1, 2, 3, 4, 5)
# 参照周波数として意味のある帯(features の解析帯域に合わせる)
A_MIN_HZ, A_MAX_HZ = 50.0, 4200.0

# --- 実データのスイープで決めた既定値(5/5 が ±3秒以内)--------------------
DEFAULT_TOLERANCE_CENT = 40.0   # 基準からの許容ずれ
DEFAULT_TONAL_TH = 0.20         # 純音性の下限
DEFAULT_FLUX_PCT = 45           # スペクトル変化の上位何%を「定常」とみなすか
DEFAULT_EVENT_GAP = 1.5         # この間隔以内の途切れは同一イベントとみなす
DEFAULT_MIN_DURATION = 12.0     # チューニングとみなす最短の長さ
DEFAULT_MIN_PURITY = 0.55       # 区間内で条件を満たすフレームの割合の下限

EPS = 1e-12


@dataclass
class TuningEvent:
    start: float
    end: float
    duration: float
    purity: float           # 区間内の条件充足率。本物ほど高い
    peak_hz: float          # 区間内のピーク周波数の中央値
    level_db: float         # 区間内の RMS 中央値
    level_drop_db: float    # 直後の音量低下量(参考値。判定には使わない)

    @property
    def score(self) -> float:
        """選択に使う指標。純度と長さの積(長く純度が高いものを本物とみなす)。"""
        return self.purity * self.duration

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "purity": round(self.purity, 3),
            "peak_hz": round(self.peak_hz, 2),
            "level_db": round(self.level_db, 2),
            "level_drop_db": round(self.level_drop_db, 2),
            "score": round(self.score, 2),
        }


def a_reference_frequencies() -> np.ndarray:
    """A のオクターブ × 倍音 からなる参照周波数の集合。"""
    refs = []
    for o in A_OCTAVES:
        for k in A_HARMONICS:
            f = A_CENTER_HZ * (2.0 ** o) * k
            if A_MIN_HZ < f < A_MAX_HZ:
                refs.append(f)
    return np.array(sorted(set(refs)))


_REFS = a_reference_frequencies()


def cents_from_nearest_a(hz: np.ndarray) -> np.ndarray:
    """各周波数が、A のいずれかの倍音から何セント離れているか(絶対値)。"""
    hz = np.maximum(np.asarray(hz, dtype=np.float64), 1.0)
    best = np.full(hz.shape, np.inf)
    for ref in _REFS:
        best = np.minimum(best, np.abs(1200.0 * np.log2(hz / ref)))
    return best


def _runs(mask: np.ndarray) -> list[list[int]]:
    out: list[list[int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    return out


def tuning_frame_mask(
    ff: FrameFeatures,
    silence_db: float,
    *,
    tolerance_cent: float = DEFAULT_TOLERANCE_CENT,
    tonal_th: float = DEFAULT_TONAL_TH,
    flux_pct: int = DEFAULT_FLUX_PCT,
) -> np.ndarray:
    """フレームごとに「A の音が鳴っている」かの真偽値列。"""
    near_a = cents_from_nearest_a(ff.peak_hz) < tolerance_cent
    tonal = ff.tonal > tonal_th
    active = ff.rms_db > (silence_db + 4.0)
    steady = ff.flux < max(float(np.percentile(ff.flux, flux_pct)), EPS)
    return near_a & tonal & active & steady


def detect_tuning_events(
    ff: FrameFeatures,
    silence_db: float,
    *,
    tolerance_cent: float = DEFAULT_TOLERANCE_CENT,
    tonal_th: float = DEFAULT_TONAL_TH,
    flux_pct: int = DEFAULT_FLUX_PCT,
    event_gap: float = DEFAULT_EVENT_GAP,
    min_duration: float = DEFAULT_MIN_DURATION,
    min_purity: float = DEFAULT_MIN_PURITY,
    drop_window: float = 6.0,
) -> list[TuningEvent]:
    """チューニング区間を検出する。"""
    fps = ff.fps
    n = len(ff.times)
    if n == 0:
        return []

    mask = tuning_frame_mask(ff, silence_db, tolerance_cent=tolerance_cent,
                             tonal_th=tonal_th, flux_pct=flux_pct)

    events: list[list[int]] = []
    gap_frames = int(round(event_gap * fps))
    for a, b in _runs(mask):
        if events and a - events[-1][1] <= gap_frames:
            events[-1][1] = b
        else:
            events.append([a, b])

    drop_frames = int(round(drop_window * fps))
    out: list[TuningEvent] = []
    for a, b in events:
        dur = (b - a) / fps
        if dur < min_duration:
            continue
        purity = float(np.mean(mask[a:b]))
        if purity < min_purity:
            continue
        after = ff.rms_db[b: min(n, b + drop_frames)]
        level = float(np.median(ff.rms_db[a:b]))
        drop = level - float(np.percentile(after, 20)) if len(after) else 0.0
        out.append(TuningEvent(
            start=float(ff.times[a]), end=float(ff.times[b - 1]), duration=dur,
            purity=purity, peak_hz=float(np.median(ff.peak_hz[a:b])),
            level_db=level, level_drop_db=drop,
        ))
    return out
