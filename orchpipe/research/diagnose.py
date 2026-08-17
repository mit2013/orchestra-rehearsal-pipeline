"""ステージG-1: 参照区間の機械的な抽出と、候補特徴量の分離性能の実測。

指示書 G-1-1 の「機械的な抽出(人手不要)」を実装する。参照区間は3種類:

- `playing`  10秒以上、高いRMSが持続する区間。ただし発話と重なるものは除く
- `silence`  5秒以上、ほぼ無音の区間
- `speech`   ASR がテキストとして採用した発話区間

`playing` と `silence` は**波形(RMS)だけ**から決めている。旧分類の playing 判定
そのものが評価対象なので、参照データをそれに依存させると循環してしまうため。
`speech` だけは ASR が実際に日本語テキストを書き起こせたものに限っており、
これは「非演奏であることが確実」な参照として最も価値が高い。

分離性能は AUC(順位ベースなので分布の形に依らない)と Cohen's d で測る。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import fmt_time, log
from . import playing_features as pf

# --- 参照区間の抽出条件 -----------------------------------------------------
PLAYING_MIN_S = 10.0     # 高RMSがこれ以上続けば「演奏らしい」参照とする
SILENCE_MIN_S = 5.0
SPEECH_MIN_S = 3.0
# 各区間から解析する最大の長さ。長すぎても情報は増えず時間だけ掛かる。
ANALYZE_MAX_S = 20.0
# 1クラスあたりの上限。多すぎると診断に時間が掛かるだけ。
MAX_PER_CLASS = 60


@dataclass
class RefSegment:
    start: float          # 現在の音源上の時刻
    end: float
    cls: str              # "playing" | "silence" | "speech"
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def collect_references(
    rms_db: np.ndarray,
    times: np.ndarray,
    speech_spans: list[tuple[float, float, str]],
    offset: float = 0.0,
) -> list[RefSegment]:
    """RMS の並びと ASR の発話区間から参照区間を機械的に集める。

    `offset` は「解析に使った特徴量の座標」から「現在の音源の座標」への平行移動。
    """
    # 生の RMS は 62.5fps で激しく上下するので、そのまま帯で切ると「10秒以上
    # 続く区間」がほとんど取れない(実測で静かな有音が 0 件だった)。
    # 1 秒の移動平均にしてから区間を切る。
    fps = len(times) / max(float(times[-1] - times[0]), 1e-6)
    k = max(1, int(round(fps)))
    kern = np.ones(k) / k
    rms_s = np.convolve(rms_db.astype(np.float64), kern, mode="same")

    silence_db = float(np.percentile(rms_s, 5))
    loud_db = float(np.percentile(rms_s, 70))
    quiet_db = float(np.percentile(rms_s, 45))
    log(f"  RMS(1秒平滑) 分布: 5%点 {silence_db:.1f} / 45%点 {quiet_db:.1f} / "
        f"70%点 {loud_db:.1f} dBFS")
    rms_db = rms_s

    speech = [(a, b) for a, b, _t in speech_spans]

    def overlaps_speech(a: float, b: float) -> bool:
        return any(not (b <= s or a >= e) for s, e in speech)

    refs: list[RefSegment] = []

    # --- playing: 高RMSが持続 -------------------------------------------------
    for a, b in _runs(rms_db > loud_db):
        t0, t1 = float(times[a]), float(times[b - 1])
        if t1 - t0 < PLAYING_MIN_S or overlaps_speech(t0, t1):
            continue
        refs.append(RefSegment(t0 + offset, t1 + offset, "playing", "高RMS持続"))

    # --- silence: ほぼ無音 ----------------------------------------------------
    for a, b in _runs(rms_db < silence_db + 6.0):
        t0, t1 = float(times[a]), float(times[b - 1])
        if t1 - t0 < SILENCE_MIN_S:
            continue
        refs.append(RefSegment(t0 + offset, t1 + offset, "silence", "ほぼ無音"))

    # --- quiet_probe: 静かだが無音ではない区間(正解ラベルではない参考データ) --
    # 「弱奏」と「遠くの発言」がここに混在する。参照区間の playing が高RMSばかりに
    # 偏るため、特徴量がこの中間帯でどう振る舞うかを見るための探り。
    for a, b in _runs((rms_db > silence_db + 6.0) & (rms_db < quiet_db)):
        t0, t1 = float(times[a]), float(times[b - 1])
        if t1 - t0 < PLAYING_MIN_S or overlaps_speech(t0, t1):
            continue
        refs.append(RefSegment(t0 + offset, t1 + offset, "quiet_probe", "静かな有音"))

    # --- speech: ASR がテキストを書き起こせたもの ------------------------------
    for a, b, text in speech_spans:
        if b - a < SPEECH_MIN_S:
            continue
        refs.append(RefSegment(a + offset, b + offset, "speech", text[:24]))

    # 長いものを優先して各クラス上限まで残す(短い断片より情報が安定するため)
    out: list[RefSegment] = []
    for cls in ("playing", "silence", "speech", "quiet_probe"):
        got = sorted([r for r in refs if r.cls == cls],
                     key=lambda r: -r.duration)[:MAX_PER_CLASS]
        out += got
    out.sort(key=lambda r: r.start)
    return out


def measure(src: Path, refs: list[RefSegment]) -> list[dict]:
    """各参照区間の特徴量を実測する。"""
    rows = []
    for i, r in enumerate(refs, start=1):
        dur = min(r.duration, ANALYZE_MAX_S)
        x = pf.decode(src, r.start, dur)
        feats = pf.extract(x)
        rows.append({"start": round(r.start, 2), "end": round(r.end, 2),
                     "cls": r.cls, "note": r.note,
                     "analyzed_sec": round(dur, 2), **feats.to_json()})
        if i % 20 == 0:
            log(f"    {i}/{len(refs)} 区間")
    return rows


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U から求める AUC。0.5 が「まったく分離できない」。"""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 同値の順位を平均に均す
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    mean_rank = np.zeros(len(vals))
    np.add.at(mean_rank, inv, ranks)
    mean_rank /= cnt
    ranks = mean_rank[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    s = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / s) if s > 0 else 0.0


def best_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """正答率が最大になる閾値と、そのときの正答率。"""
    cand = np.unique(np.concatenate([pos, neg]))
    best = (0.0, 0.0)
    for t in cand:
        for sign in (1, -1):
            acc = (np.mean(sign * pos >= sign * t) * len(pos)
                   + np.mean(sign * neg < sign * t) * len(neg)) / (len(pos) + len(neg))
            if acc > best[1]:
                best = (float(t) * sign, float(acc))
    return best


FEATURE_KEYS = ["polyphony", "harmonic_ratio", "stability", "onset_rate",
                "rms_db", "flatness", "centroid"]

FEATURE_JA = {
    "polyphony": "多声性(倍音系列の数)",
    "harmonic_ratio": "調和性(倍音で説明できる割合)",
    "stability": "スペクトル安定性",
    "onset_rate": "オンセット密度 [回/秒]",
    "rms_db": "RMS [dBFS](参考)",
    "flatness": "フラットネス(参考)",
    "centroid": "スペクトル重心 [Hz](参考)",
}


def _rms_matched(rows: list[dict], pos_cls: tuple, neg_cls: tuple,
                 tol_db: float = 4.0) -> tuple[list[dict], list[dict]]:
    """RMS が重なる範囲だけを取り出した部分集合。

    **この録音では playing の参照が高RMS、speech が低RMSに偏っており、そのままでは
    どの特徴量も「音量差」を測っているだけになる。**交絡を切り分けるため、両クラスの
    RMS 分布が重なる帯だけで比べ直す。
    """
    pos = [r for r in rows if r["cls"] in pos_cls]
    neg = [r for r in rows if r["cls"] in neg_cls]
    if not pos or not neg:
        return [], []
    lo = max(min(r["rms_db"] for r in pos), min(r["rms_db"] for r in neg)) - tol_db
    hi = min(max(r["rms_db"] for r in pos), max(r["rms_db"] for r in neg)) + tol_db
    return ([r for r in pos if lo <= r["rms_db"] <= hi],
            [r for r in neg if lo <= r["rms_db"] <= hi])


def separation_report(rows: list[dict]) -> dict:
    """演奏 vs 非演奏、および演奏 vs 発言 の分離性能をまとめる。"""
    def vec(cls_list, key):
        return np.array([r[key] for r in rows if r["cls"] in cls_list], dtype=np.float64)

    out: dict = {"n": {c: sum(1 for r in rows if r["cls"] == c)
                       for c in ("playing", "silence", "speech", "quiet_probe")},
                 "features": {}}

    mp, mn = _rms_matched(rows, ("playing",), ("speech", "silence"))
    out["rms_matched"] = {
        "n_playing": len(mp), "n_nonplaying": len(mn),
        "rms_range_db": ([round(min(r["rms_db"] for r in mp + mn), 1),
                          round(max(r["rms_db"] for r in mp + mn), 1)] if mp and mn else None),
    }

    for key in FEATURE_KEYS:
        p = vec(("playing",), key)
        n_all = vec(("silence", "speech"), key)
        n_sp = vec(("speech",), key)
        qp = vec(("quiet_probe",), key)
        thr, acc = best_threshold(p, n_all)
        matched_auc = (round(auc(np.array([r[key] for r in mp], dtype=np.float64),
                                 np.array([r[key] for r in mn], dtype=np.float64)), 4)
                       if mp and mn else None)
        out["features"][key] = {
            "quiet_probe_mean": round(float(qp.mean()), 4) if len(qp) else None,
            "auc_rms_matched": matched_auc,
            "playing_mean": round(float(p.mean()), 4) if len(p) else None,
            "playing_sd": round(float(p.std(ddof=1)), 4) if len(p) > 1 else None,
            "nonplaying_mean": round(float(n_all.mean()), 4) if len(n_all) else None,
            "nonplaying_sd": round(float(n_all.std(ddof=1)), 4) if len(n_all) > 1 else None,
            "speech_mean": round(float(n_sp.mean()), 4) if len(n_sp) else None,
            "auc_vs_all": round(auc(p, n_all), 4),
            "auc_vs_speech": round(auc(p, n_sp), 4),
            "cohens_d": round(cohens_d(p, n_all), 3),
            "best_threshold": round(thr, 4),
            "best_accuracy": round(acc, 4),
        }
    return out


def format_report(rep: dict) -> str:
    lines = []
    n = rep["n"]
    lines.append(f"参照区間: playing {n['playing']} / silence {n['silence']} / "
                 f"speech {n['speech']} / quiet_probe {n['quiet_probe']}(参考)"
                 f"  (計 {sum(n.values())})")
    m = rep["rms_matched"]
    if m["rms_range_db"]:
        lines.append(f"RMS を揃えた部分集合: playing {m['n_playing']} / 非演奏 "
                     f"{m['n_nonplaying']}  ({m['rms_range_db'][0]}〜{m['rms_range_db'][1]} dBFS)")
    else:
        lines.append("RMS を揃えた部分集合: **両クラスの音量分布に重なりがない**"
                     "(= 素の AUC は音量差を測っているだけの可能性が高い)")
    lines.append("")
    lines.append(f"{'特徴量':<30} {'演奏':>9} {'非演奏':>9} {'発言':>9} {'静か参考':>9} "
                 f"{'AUC':>6} {'AUC統制':>8} {'d':>6}")
    lines.append("-" * 100)
    order = sorted(rep["features"].items(),
                   key=lambda kv: -abs(kv[1]["auc_vs_all"] - 0.5))
    for key, f in order:
        q = f"{f['quiet_probe_mean']:>9.3f}" if f["quiet_probe_mean"] is not None else f"{'-':>9}"
        am = (f"{f['auc_rms_matched']:>8.3f}" if f["auc_rms_matched"] is not None
              else f"{'重なりなし':>8}")
        lines.append(
            f"{FEATURE_JA[key]:<30} {f['playing_mean']:>9.3f} {f['nonplaying_mean']:>9.3f} "
            f"{f['speech_mean']:>9.3f} {q} {f['auc_vs_all']:>6.3f} {am} {f['cohens_d']:>6.2f}"
        )
    return "\n".join(lines)


def load_states_and_asr(rdir: Path, key: str) -> tuple[list[tuple[float, float, str]], dict]:
    """ASR が採用した発話区間と、states.json の meta を返す。"""
    asr = json.loads((rdir / f"{key}_asr.json").read_text(encoding="utf-8"))
    spans = [(s["start"], s["end"], s["text"]) for s in asr["segments"] if s["accepted"]]
    meta = json.loads((rdir / f"{key}_states.json").read_text(encoding="utf-8"))["meta"]
    return spans, meta
