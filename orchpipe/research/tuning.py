"""チューニング検出(研究ブランチ用の入り口)。

このモジュールは以前、研究用に独自の実装を抱えていた。その実装は本番の
`orchpipe/tuning.py` に移植され、さらに本番側で **A・E・C# の取り違え**を直す
改修が入っている(A の倍音だけを頼りにすると 3次倍音が E、5次倍音が C# と一致
してしまう問題)。

研究側に古い実装を残すと、ダイジェストが**直っていないチューニング検出**を
使い続けることになる。実際 260829 では、旧実装が演奏中の C#6 を拾って合奏3の
開始を 5分24秒うしろに置いていた。したがってここは本番の実装をそのまま再輸出し、
実装をひとつに保つ。仕組みの説明は `orchpipe/tuning.py` の冒頭を参照。
"""

from __future__ import annotations

from ..tuning import (  # noqa: F401
    A_CENTER_HZ,
    A_HARMONICS,
    A_MAX_HZ,
    A_MIN_HZ,
    A_OCTAVES,
    DEFAULT_CORE_RATE_TH,
    DEFAULT_EVENT_GAP,
    DEFAULT_HEAD_MAX_S,
    DEFAULT_HEAD_RATE_TH,
    DEFAULT_MIN_DURATION,
    DEFAULT_MIN_PURITY,
    DEFAULT_RATE_WIN,
    DEFAULT_TOLERANCE_CENT,
    DEFAULT_TONAL_TH,
    TuningEvent,
    a_octave_frequencies,
    a_reference_frequencies,
    cents_from_nearest_a,
    cents_from_nearest_a_octave,
    detect_tuning_events,
    tuning_frame_mask,
    tuning_head_mask,
)

__all__ = [
    "A_CENTER_HZ", "A_HARMONICS", "A_MAX_HZ", "A_MIN_HZ", "A_OCTAVES",
    "DEFAULT_CORE_RATE_TH", "DEFAULT_EVENT_GAP", "DEFAULT_HEAD_MAX_S",
    "DEFAULT_HEAD_RATE_TH", "DEFAULT_MIN_DURATION", "DEFAULT_MIN_PURITY",
    "DEFAULT_RATE_WIN", "DEFAULT_TOLERANCE_CENT", "DEFAULT_TONAL_TH",
    "TuningEvent", "a_octave_frequencies", "a_reference_frequencies",
    "cents_from_nearest_a", "cents_from_nearest_a_octave",
    "detect_tuning_events", "tuning_frame_mask", "tuning_head_mask",
]
