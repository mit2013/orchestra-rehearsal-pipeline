"""チューニング(基準音Aの合わせ)の検出。

## この実装が解いている問題

倍音は音程比なので、A の倍音だけを頼りにすると **A・E・C# を区別できない**。

| 倍音 | 音程比 | 実際にマッチする音名 |
|---|---|---|
| 3次 | 3:1 | E |
| 5次 | 5:1 | C# |

A・E・C# は A dur の三和音そのもので、オーケストラの曲では常時鳴っている。旧実装は
「A のオクターブ × 倍音1〜5次」を無条件の参照集合にしていたため、260829 では演奏中の
C#6(1111Hz。A2=221Hz の5倍音 1105Hz と 9 セントしか違わない)を拾い、合奏3の開始を
本来より 5分24秒うしろに置いて、チューニングと演奏の頭を切り落として配布した。

一方で倍音を単純に捨てることもできない。静かなオーボエ単独音では、部屋鳴りのマイクだと
基音より上位倍音が支配的になる。実測で、本物のチューニングの**出だしの十数秒**は
支配ピークが第3倍音にある。

| 日付 / ブロック | チューニング開始 | 開始直後の支配ピーク | オクターブに落ち着く時刻 |
|---|---|---|---|
| 260802 合奏1 | 00:10:02 | 1317Hz (A4=442 の3倍音 1326Hz) | 00:10:17 |
| 260829 合奏2 | 00:47:13 | 1312-1326Hz (同上) | 00:47:19 |

オクターブだけを見ると、この十数秒がまるごと落ちて開始が遅れる。260802 合奏1 では
15秒、260829 合奏2 では6秒遅れ、そのぶんチューニングの頭を削ることになる。

## 解き方: 「核」はオクターブだけで作り、「頭」だけ倍音を信用する

1. **核(core)**: A の**オクターブのみ**に一致するフレームの1秒あたりの一致率が
   `core_rate_th` 以上、という条件で区間を作る。倍音を使わないので A・E・C# の
   取り違えが起きない。ここがこの検出の選択性を担う。
2. **束ね**: 核どうしが `event_gap` 秒以内なら1つのチューニングとみなす。
   チューニング中に数秒 A が途切れるのは正常な現象である。基準音を吹くオーボエ
   奏者が、リードの状態などによって吹き直すために起きる。
3. **頭出し**: 核の開始から前方向へ、**倍音を含む**参照集合に一致するフレームを
   たどって開始時刻を伸ばす。倍音を根拠に採用してよいのは、**オクターブで裏づけ
   られた核に連続している場合だけ**である。孤立した C# はどの核にも隣接しないので
   採用されない。頭側は音量条件を課さない(本物の出だしは無音閾値を下回るほど
   静かなことがある)。

この設計での実測は、260726・260802・260829 の実チューニング8点をすべて検出し
(誤差 -0.6〜+0.8 秒)、誤検出は 0 件である。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FrameFeatures

EPS = 1e-12

# オーケストラの基準音 A(442Hz 前後で運用される)
A_CENTER_HZ = 442.0
# 参照するオクターブ(A2〜A6 相当)。核の判定はこれだけで行う。
A_OCTAVES = (-2, -1, 0, 1, 2)
# 頭出しでのみ参照する倍音。核に連続する場合に限って信用する。
A_HARMONICS = (1, 2, 3, 4, 5)
# 参照周波数として意味のある帯(features の解析帯域に合わせる)
A_MIN_HZ, A_MAX_HZ = 50.0, 4200.0

# --- 既定値 -----------------------------------------------------------------
# 許容ずれ。基準ピッチは団により 440〜443Hz と幅があり、合わせ切る前は
# さらにばらつくため、狭く取りすぎない。
DEFAULT_TOLERANCE_CENT = 60.0
DEFAULT_TONAL_TH = 0.20         # 純音性の下限
DEFAULT_RATE_WIN = 1.0          # 一致率をならす窓 [s]
DEFAULT_CORE_RATE_TH = 0.5      # 核とみなす一致率の下限
DEFAULT_HEAD_RATE_TH = 0.5      # 頭出しでたどる一致率の下限
DEFAULT_HEAD_MAX_S = 40.0       # 頭出しで遡れる上限 [s]
DEFAULT_EVENT_GAP = 10.0        # この間隔以内の途切れは同一チューニングとみなす
DEFAULT_MIN_DURATION = 12.0     # チューニングとみなす最短の長さ
DEFAULT_MIN_PURITY = 0.50       # 核の区間内で条件を満たすフレームの割合の下限


@dataclass
class TuningEvent:
    start: float            # 最初の A(倍音での頭出しを含む)
    end: float
    duration: float
    purity: float           # 核の区間内の条件充足率。本物ほど高い
    peak_hz: float          # 核の区間内のピーク周波数の中央値
    level_db: float         # 核の区間内の RMS 中央値
    core_start: float = 0.0  # オクターブだけで裏づけられた部分の開始

    @property
    def score(self) -> float:
        """選択に使う指標。純度と長さの積(長く純度が高いものを本物とみなす)。"""
        return self.purity * self.duration

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 2),
            "core_start": round(self.core_start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "purity": round(self.purity, 3),
            "peak_hz": round(self.peak_hz, 2),
            "level_db": round(self.level_db, 2),
            "score": round(self.score, 2),
        }


def _reference_frequencies(harmonics: tuple[int, ...]) -> np.ndarray:
    refs = []
    for o in A_OCTAVES:
        for k in harmonics:
            f = A_CENTER_HZ * (2.0 ** o) * k
            if A_MIN_HZ < f < A_MAX_HZ:
                refs.append(f)
    return np.array(sorted(set(refs)))


def a_reference_frequencies() -> np.ndarray:
    """A のオクターブ × 倍音 からなる参照周波数の集合(頭出し用)。"""
    return _reference_frequencies(A_HARMONICS)


def a_octave_frequencies() -> np.ndarray:
    """A のオクターブのみの参照周波数の集合(核の判定用)。"""
    return _reference_frequencies((1,))


_REFS_HARMONIC = a_reference_frequencies()
_REFS_OCTAVE = a_octave_frequencies()


def _cents_from(hz: np.ndarray, refs: np.ndarray) -> np.ndarray:
    hz = np.maximum(np.asarray(hz, dtype=np.float64), 1.0)
    best = np.full(hz.shape, np.inf)
    for ref in refs:
        best = np.minimum(best, np.abs(1200.0 * np.log2(hz / ref)))
    return best


def cents_from_nearest_a(hz: np.ndarray) -> np.ndarray:
    """各周波数が、A のいずれかの倍音から何セント離れているか(絶対値)。"""
    return _cents_from(hz, _REFS_HARMONIC)


def cents_from_nearest_a_octave(hz: np.ndarray) -> np.ndarray:
    """各周波数が、A のいずれかのオクターブから何セント離れているか(絶対値)。"""
    return _cents_from(hz, _REFS_OCTAVE)


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


def _moving_rate(mask: np.ndarray, width: int) -> np.ndarray:
    """真偽値列を、幅 `width` フレームの移動平均(=一致率)にならす。"""
    if width <= 1:
        return mask.astype(np.float64)
    n = len(mask)
    cum = np.concatenate([[0.0], np.cumsum(mask.astype(np.float64))])
    half = width // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) - half + width, 0, n)
    return (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)


def tuning_frame_mask(
    ff: FrameFeatures,
    silence_db: float,
    *,
    tolerance_cent: float = DEFAULT_TOLERANCE_CENT,
    tonal_th: float = DEFAULT_TONAL_TH,
) -> np.ndarray:
    """フレームごとに「A のオクターブが鳴っている」かの真偽値列(核の判定用)。

    ピッチ安定性とスペクトル変化の条件は課していない。前者はピークが基音と倍音の
    間を行き来するため有害で、後者は本物のチューニングの出だしを落とすためである。
    """
    near_a = cents_from_nearest_a_octave(ff.peak_hz) < tolerance_cent
    tonal = ff.tonal > tonal_th
    active = ff.rms_db > (silence_db + 4.0)
    return near_a & tonal & active


def tuning_head_mask(
    ff: FrameFeatures,
    *,
    tolerance_cent: float = DEFAULT_TOLERANCE_CENT,
    tonal_th: float = DEFAULT_TONAL_TH,
) -> np.ndarray:
    """頭出し用の真偽値列。倍音を含み、音量条件を課さない。

    音量を見ないのは、本物のチューニングの出だしが無音閾値を下回るほど静かな
    ことがあるため(260829 合奏2 の最初の A は -44.2 dBFS、無音閾値 -43.0 dBFS)。
    倍音を含めても危険がないのは、核に連続する部分にしか適用しないからである。
    """
    near_a = cents_from_nearest_a(ff.peak_hz) < tolerance_cent
    return near_a & (ff.tonal > tonal_th)


def detect_tuning_events(
    ff: FrameFeatures,
    silence_db: float,
    *,
    tolerance_cent: float = DEFAULT_TOLERANCE_CENT,
    tonal_th: float = DEFAULT_TONAL_TH,
    rate_win: float = DEFAULT_RATE_WIN,
    core_rate_th: float = DEFAULT_CORE_RATE_TH,
    head_rate_th: float = DEFAULT_HEAD_RATE_TH,
    head_max_s: float = DEFAULT_HEAD_MAX_S,
    event_gap: float = DEFAULT_EVENT_GAP,
    min_duration: float = DEFAULT_MIN_DURATION,
    min_purity: float = DEFAULT_MIN_PURITY,
) -> list[TuningEvent]:
    """チューニング区間を検出する。モジュール冒頭の説明を参照。"""
    fps = ff.fps
    if len(ff.times) == 0:
        return []

    core_mask = tuning_frame_mask(ff, silence_db, tolerance_cent=tolerance_cent,
                                 tonal_th=tonal_th)
    head_mask = tuning_head_mask(ff, tolerance_cent=tolerance_cent, tonal_th=tonal_th)

    win = max(1, int(round(rate_win * fps)))
    core = _moving_rate(core_mask, win) >= core_rate_th
    head = _moving_rate(head_mask, win) >= head_rate_th

    # 近接する核を1回のチューニングとして束ねる(オーボエの吹き直しをまたぐ)。
    gap_frames = int(round(event_gap * fps))
    events: list[list[int]] = []
    for a, b in _runs(core):
        if events and a - events[-1][1] <= gap_frames:
            events[-1][1] = b
        else:
            events.append([a, b])

    head_limit = int(round(head_max_s * fps))
    out: list[TuningEvent] = []
    for a, b in events:
        # 核の手前へ、倍音を含む一致をたどって「最初の A」まで戻る。
        s = a
        floor = max(0, a - head_limit)
        while s > floor and head[s - 1]:
            s -= 1

        dur = (b - s) / fps
        if dur < min_duration:
            continue
        purity = float(np.mean(core_mask[a:b]))
        if purity < min_purity:
            continue
        out.append(TuningEvent(
            start=float(ff.times[s]), end=float(ff.times[b - 1]), duration=dur,
            purity=purity, peak_hz=float(np.median(ff.peak_hz[a:b])),
            level_db=float(np.median(ff.rms_db[a:b])),
            core_start=float(ff.times[a]),
        ))
    return out
