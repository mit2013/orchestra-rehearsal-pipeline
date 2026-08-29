"""ステージA: 4値ステート分類(silence / tuning / speech / playing)。

`trimmed/*_final.wav`(合奏ブロック単位)を対象に、0.5 秒ホップでラベル付けする。

判定の優先順位(先に決まったものが勝つ):

    silence  … 既存の適応的 RMS 閾値。ここは main のロジックをそのまま流用。
    tuning   … research/tuning.py(基準A の帯 + 直後の減衰)
    speech   … ASR が採用したセグメント(research/asr.py の judge())
    playing  … 上記のいずれでもない非無音区間

`speech` を最後から2番目に置いているのは、チューニングを発話と取り違えないため。
`playing` を既定値にしているのは、誤って演奏を speech にするとダイジェスト
(ステージC)から演奏が消えてしまうためで、これまでのパイプラインと同じ安全側の倒し方。

main の `features.py` は変更していない。`aggregate_windows` は元から
`win_s` / `hop_s` を引数に取るので、細粒度で呼ぶだけで足りた。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import features as F
from ..util import fmt_time, log
from .asr import AsrSegment, Transcriber, decode_mono16k
from .tuning import TuningEvent, detect_tuning_events

LABELS = ("silence", "tuning", "speech", "playing")

# 細粒度の解析パラメータ。指示書 A-1 の「0.5 秒目安のホップ幅」に合わせる。
FINE_WIN_S = 1.0
FINE_HOP_S = 0.5


@dataclass
class StateSpan:
    start: float
    end: float
    label: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "label": self.label,
            "confidence": round(self.confidence, 3),
        }


def _spans_from_labels(labels: list[str], conf: np.ndarray, times: np.ndarray,
                       hop: float, total: float) -> list[StateSpan]:
    """フレーム列を、同じラベルが続く区間にまとめる。"""
    if not labels:
        return []
    out: list[StateSpan] = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        start = 0.0 if i == 0 else float(times[i]) - hop / 2
        end = total if j >= n else float(times[j]) - hop / 2
        out.append(StateSpan(max(0.0, start), min(total, end), labels[i],
                             float(np.mean(conf[i:j]))))
        i = j
    return out


def classify_block(
    path: Path,
    transcriber: Transcriber | None,
    *,
    min_span: float = 0.5,
    reuse_asr: list[AsrSegment] | None = None,
) -> tuple[list[StateSpan], list[TuningEvent], list[AsrSegment], dict]:
    """1ブロックを4値に分類する。戻り値は (区間, チューニング事象, ASR結果, メタ)。

    `reuse_asr` を渡すと ASR を再実行しない(チューニング検出だけ作り直したい場合に使う)。
    """
    log(f"  特徴量を抽出: {path.name}")
    ff = F.extract_frame_features(path, progress_every=1e9)
    total = float(ff.times[-1])
    wf = F.aggregate_windows(ff, win_s=FINE_WIN_S, hop_s=FINE_HOP_S)
    silence_db = wf.silence_db
    log(f"    {wf.n} 窓 (窓長 {FINE_WIN_S}s / ホップ {FINE_HOP_S}s), 無音閾値 {silence_db:.1f} dBFS")

    # --- tuning ------------------------------------------------------------
    tunings = detect_tuning_events(ff, silence_db)
    log(f"    チューニング候補: {len(tunings)} 件")

    # --- speech(ASR)-------------------------------------------------------
    if reuse_asr is not None:
        asr = reuse_asr
        log(f"    保存済み ASR を再利用: {len(asr)} セグメント")
    else:
        log("    ASR を実行します(VAD で発話候補に絞ってから書き起こし)")
        audio = decode_mono16k(path)
        asr = transcriber.transcribe(audio)
    accepted = [s for s in asr if s.accepted]
    log(f"    ASR セグメント {len(asr)} 件 / 発話として採用 {len(accepted)} 件")

    # --- フレームごとにラベルを決める ---------------------------------------
    t = wf.times
    n = wf.n
    labels = ["playing"] * n
    conf = np.full(n, 0.5)

    # playing / silence の下地
    loud = wf.loudness
    is_silent = loud < silence_db
    for i in range(n):
        if is_silent[i]:
            labels[i] = "silence"
            conf[i] = float(min(1.0, (silence_db - loud[i]) / 10.0 + 0.6))

    # speech(silence より優先。ただし無音のままの箇所は上書きしない)
    for s in accepted:
        sel = (t >= s.start) & (t < s.end)
        for i in np.flatnonzero(sel):
            if labels[i] != "silence":
                labels[i] = "speech"
                conf[i] = 0.7

    # tuning(最優先)
    for e in tunings:
        sel = (t >= e.start) & (t <= e.end)
        for i in np.flatnonzero(sel):
            labels[i] = "tuning"
            conf[i] = e.score

    spans = _spans_from_labels(labels, conf, t, FINE_HOP_S, total)

    # ごく短い区間は前後に吸収する(0.5秒ホップなので単発のちらつきが出やすい)
    cleaned: list[StateSpan] = []
    for sp in spans:
        if cleaned and sp.duration < min_span and sp.label != "tuning":
            cleaned[-1] = StateSpan(cleaned[-1].start, sp.end, cleaned[-1].label,
                                    cleaned[-1].confidence)
        else:
            cleaned.append(sp)
    merged: list[StateSpan] = [cleaned[0]] if cleaned else []
    for sp in cleaned[1:]:
        if sp.label == merged[-1].label:
            merged[-1] = StateSpan(merged[-1].start, sp.end, sp.label,
                                   (merged[-1].confidence + sp.confidence) / 2)
        else:
            merged.append(sp)

    meta = {
        "source": str(path),
        "duration": round(total, 2),
        "win_s": FINE_WIN_S,
        "hop_s": FINE_HOP_S,
        "silence_db": round(silence_db, 2),
        "n_tuning_events": len(tunings),
        "n_asr_segments": len(asr),
        "n_asr_accepted": len(accepted),
    }
    return merged, tunings, asr, meta


def summarize(spans: list[StateSpan]) -> dict[str, dict]:
    """ラベルごとの合計時間・出現回数。"""
    out = {l: {"seconds": 0.0, "count": 0} for l in LABELS}
    for sp in spans:
        out[sp.label]["seconds"] += sp.duration
        out[sp.label]["count"] += 1
    total = sum(v["seconds"] for v in out.values()) or 1.0
    for l in LABELS:
        out[l]["seconds"] = round(out[l]["seconds"], 1)
        out[l]["ratio"] = round(out[l]["seconds"] / total, 4)
    return out


def format_summary(name: str, spans: list[StateSpan], total: float) -> str:
    s = summarize(spans)
    lines = [f"  {name}  (全長 {fmt_time(total)})"]
    for l in LABELS:
        v = s[l]
        lines.append(f"    {l:<8} {v['seconds']:>8.1f} 秒 ({v['ratio']*100:>5.1f}%)  {v['count']:>4} 区間")
    return "\n".join(lines)
