"""区間の意味づけ: 特徴量から「合奏」/「不要区間」の境界候補を作る。

features.py が出す粒度非依存の特徴量に、練習録音の構造(音出し → チューニング →
合奏 → 休憩 → …)という知識をかぶせる層。判定は単一閾値の無音検出ではなく、
複数特徴量を頑健化したうえで合成したスコアに対する 2 値クラスタリング
(大津の方法)+ ヒステリシスで行う。

指示書 4章の手がかりの対応:
  音出し・休憩 : リズムの統一感が低い / 強弱のメリハリが乏しい / 強音が途切れず
                 無音区間が少ない  -> pulse_clarity・dyn_range・silence が低い
  合奏         : 指揮者が止めるため無音・弱音が断続的に挟まる / 統一感が高い
                 -> silence_ratio が中程度、pulse_clarity・dyn_range が高い
  チューニング : 単一持続音でスペクトルが単純 -> tuning_score。合奏の直前に必ず
                 現れるので、境界位置のスナップ先として使う。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from .features import FrameFeatures, WindowFeatures, tuning_frames
from .util import fmt_time, log

EPS = 1e-12

# スコア合成の重み。合奏らしさに効く向きを正にしてある。
WEIGHTS = {
    "pulse_clarity": 1.00,   # リズムの統一感
    "dyn_range": 0.90,       # 強弱のメリハリ
    "silence_term": 0.85,    # 適度に無音が挟まる(多すぎる=ただの無人も除外)
    "flux_crest": 0.45,      # アタックの揃い具合
    "flatness": -0.40,       # 雑音的なほど合奏らしくない
}


@dataclass
class Segment:
    index: int
    start: float
    end: float
    label: str
    action: str          # "keep" | "remove"
    confidence: float

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "start": fmt_time(self.start),
            "end": fmt_time(self.end),
            "label": self.label,
            "action": self.action,
            "confidence": round(float(self.confidence), 2),
        }


# ---------------------------------------------------------------------------
# スコア計算
# ---------------------------------------------------------------------------

def _robust_z(x: np.ndarray) -> np.ndarray:
    """中央値とMADによる頑健なzスコア(外れ値に引きずられないため)。"""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < EPS:
        scale = float(np.std(x)) or 1.0
    return (x - med) / scale


def _silence_term(silence_ratio: np.ndarray) -> np.ndarray:
    """無音率を「合奏らしさ」に変換する台形カーブ。

    合奏中は指揮者が止めるので無音がほどよく挟まる。一方で無音率がほぼ 1 の区間は
    単に誰もいない(録りっぱなし)だけなので、そこは合奏らしさを下げる。
    """
    rise = np.clip(silence_ratio / 0.25, 0.0, 1.0)
    fall = np.clip((0.85 - silence_ratio) / 0.20, 0.0, 1.0)
    return rise * fall


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = k if k % 2 == 1 else k + 1
    pad = np.pad(x, k // 2, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, k)
    return np.median(view, axis=1)


def ensemble_score(wf: WindowFeatures, smooth_s: float = 0.0) -> np.ndarray:
    """窓ごとの「合奏らしさ」スコア。高いほど合奏、低いほど音出し/休憩。"""
    comps = {
        "pulse_clarity": _robust_z(wf.pulse_clarity),
        "dyn_range": _robust_z(wf.dyn_range),
        "silence_term": _robust_z(_silence_term(wf.silence_ratio)),
        "flux_crest": _robust_z(wf.flux_crest),
        "flatness": _robust_z(wf.flatness),
    }
    score = sum(WEIGHTS[k] * v for k, v in comps.items())

    # 完全な無音・ほぼ無人の区間は合奏ではありえないので明示的に落とす。
    score = np.where(wf.silence_ratio > 0.92, score - 4.0, score)

    # 既定では平滑化しない。短い変動を無視する役目はビタビの遷移ペナルティが担っており、
    # ここで中央値フィルタを重ねると同じ仕事を二重にやったうえ、境界の時間分解能だけを
    # 失う。実測では 60 秒を超える平滑化で合奏開始の推定が 10 分以上ずれた。
    k = max(1, int(round(smooth_s / wf.hop_s))) if smooth_s > 0 else 1
    return _median_filter(np.asarray(score, dtype=np.float64), k)


def _otsu(x: np.ndarray, bins: int = 256) -> float:
    """1次元の大津の方法。合奏が全体の何割かに依存せず 2 クラスに分ける。"""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < EPS:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    total = w.sum()
    w0 = np.cumsum(w)
    w1 = total - w0
    s = np.cumsum(w * centers)
    stot = s[-1]
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.where(valid, s / np.maximum(w0, EPS), 0.0)
    mu1 = np.where(valid, (stot - s) / np.maximum(w1, EPS), 0.0)
    var_between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return float(centers[int(np.argmax(var_between))])


def _class_centroids(x: np.ndarray) -> tuple[float, float]:
    """スコアを 2 クラスに分けたときの各クラス中心(1次元 k-means)。"""
    c = np.array([np.percentile(x, 10), np.percentile(x, 90)], dtype=np.float64)
    for _ in range(60):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        if not lab.any() or lab.all():
            break
        new = np.array([x[lab == 0].mean(), x[lab == 1].mean()])
        if np.allclose(new, c):
            break
        c = new
    return float(c[0]), float(c[1])


def decision_threshold(score: np.ndarray, bias: float = 0.25) -> float:
    """合奏/非合奏を分ける閾値。

    大津の方法は「合奏が全体の 7 割」のような偏った分布だと合奏側に食い込んだ位置を
    返しがちなので、2 クラスの中心間距離の `bias` 倍だけ低いほうへずらす。
    本物の合奏を削ってしまう誤りは、音出しが少し残る誤りよりはるかに痛いので、
    意図的に「残す側」へ倒している。
    """
    otsu = _otsu(score)
    lo_c, hi_c = _class_centroids(score)
    return otsu - bias * max(hi_c - lo_c, 0.0)


# ---------------------------------------------------------------------------
# 2値化と区間化
# ---------------------------------------------------------------------------

def _viterbi(d: np.ndarray, penalty: float) -> np.ndarray:
    """2状態(合奏/非合奏)のビタビ探索で最適な区間分割を求める。

    単純な閾値+ヒステリシスだと、合奏中の一時的なスコア低下(長い強奏など)で
    区間が細切れになり、そのあとの最小長補正が連鎖的に破綻する。ここでは
    「窓ごとの合奏らしさの合計」と「状態を切り替える回数」のトレードオフを
    全体最適で解くことで、浅い谷は無視しつつ深く長い谷(実際の休憩)だけを
    境界にする。penalty が大きいほど区間は少なく長くなる。
    """
    n = len(d)
    dp0, dp1 = -d[0], d[0]
    bp = np.zeros((n, 2), dtype=np.int8)
    for i in range(1, n):
        sw0 = dp1 - penalty
        keep0 = dp0 >= sw0
        new0 = (dp0 if keep0 else sw0) - d[i]
        bp[i, 0] = 0 if keep0 else 1

        sw1 = dp0 - penalty
        keep1 = dp1 >= sw1
        new1 = (dp1 if keep1 else sw1) + d[i]
        bp[i, 1] = 1 if keep1 else 0

        dp0, dp1 = new0, new1

    state = 1 if dp1 >= dp0 else 0
    mask = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        mask[i] = bool(state)
        state = int(bp[i, state])
    return mask


def _count_keep_runs(mask: np.ndarray) -> int:
    return sum(1 for _, _, v in _runs(mask) if v)


def _fit_penalty(d: np.ndarray, target: int) -> tuple[np.ndarray, float]:
    """合奏区間がちょうど `target` 個になる遷移ペナルティを二分探索する。

    ペナルティを上げるほど区間数は単調に減るので、対数スケールで挟み撃ちにする。
    ぴったり合わない場合は target 以下で最も近いものを返す。
    """
    lo, hi = 0.05, 1.0e5
    best: tuple[np.ndarray, float] | None = None
    for _ in range(64):
        mid = float(np.sqrt(lo * hi))
        mask = _viterbi(d, mid)
        k = _count_keep_runs(mask)
        if k == target:
            return mask, mid
        if best is None or abs(k - target) < abs(_count_keep_runs(best[0]) - target):
            best = (mask, mid)
        if k > target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0001:
            break
    assert best is not None
    return best


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """[(開始index, 終了index(排他), 値), ...] に変換。"""
    out = []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        out.append((i, j, bool(mask[i])))
        i = j
    return out


def _mask_to_bounds(mask: np.ndarray, wf: WindowFeatures, total: float) -> list[tuple[float, float, bool]]:
    """窓マスクを実時間の区間リストに直す(全体を隙間なく覆う)。"""
    segs = []
    for a, b, val in _runs(mask):
        start = 0.0 if a == 0 else float(wf.times[a])
        end = total if b >= len(mask) else float(wf.times[b])
        segs.append((start, end, val))
    return segs


def _enforce_min_durations(
    segs: list[tuple[float, float, bool]], min_keep: float, min_remove: float
) -> list[tuple[float, float, bool]]:
    """短すぎる区間を隣に吸収する。粗い区切りなので細切れは不要。"""
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e, v) in enumerate(segs):
            limit = min_keep if v else min_remove
            if e - s >= limit:
                continue
            # 短い区間は反転して隣とつながる(=隣に吸収される)。
            segs[i] = (s, e, not v)
            merged = [segs[0]]
            for seg in segs[1:]:
                if seg[2] == merged[-1][2]:
                    merged[-1] = (merged[-1][0], seg[1], seg[2])
                else:
                    merged.append(seg)
            segs = merged
            changed = True
            break
    return segs


def _apply_split_hint(
    segs: list[tuple[float, float, bool]], splits: int
) -> list[tuple[float, float, bool]]:
    """「最終的に何分割にしたいか」のヒントで合奏区間の数を合わせる。

    多すぎる場合は、いちばん短い合奏区間を落とすか、いちばん短い合間を吸収するかの
    うち「短いほう」を選んで 1 つずつ減らす。足りない場合は無理に分割せず警告する。
    """
    def keeps(ss):
        return [i for i, s in enumerate(ss) if s[2]]

    def merge_adjacent(ss):
        out = [ss[0]]
        for seg in ss[1:]:
            if seg[2] == out[-1][2]:
                out[-1] = (out[-1][0], seg[1], seg[2])
            else:
                out.append(seg)
        return out

    guard = 0
    while len(keeps(segs)) > splits and guard < 100:
        guard += 1
        k = keeps(segs)
        # 内側の合間(合奏と合奏に挟まれた remove 区間)
        gaps = [
            i for i in range(len(segs))
            if not segs[i][2] and 0 < i < len(segs) - 1 and segs[i - 1][2] and segs[i + 1][2]
        ]
        shortest_keep = min(k, key=lambda i: segs[i][1] - segs[i][0])
        keep_len = segs[shortest_keep][1] - segs[shortest_keep][0]
        shortest_gap = min(gaps, key=lambda i: segs[i][1] - segs[i][0]) if gaps else None
        gap_len = (segs[shortest_gap][1] - segs[shortest_gap][0]) if shortest_gap is not None else float("inf")

        target = shortest_keep if keep_len <= gap_len else shortest_gap
        segs[target] = (segs[target][0], segs[target][1], not segs[target][2])
        segs = merge_adjacent(segs)

    n = len(keeps(segs))
    if n < splits:
        log(f"警告: 合奏候補が {n} 区間しか検出できませんでした(指定 --splits {splits})。"
            "candidates.json を手で分割してください。")
    return segs


# ---------------------------------------------------------------------------
# チューニング事象
# ---------------------------------------------------------------------------

def find_tuning_events(ff: FrameFeatures, min_s: float = 4.0, gap_s: float = 2.0) -> list[tuple[float, float]]:
    """持続する単一音(チューニング)の区間を抽出する。"""
    mask = tuning_frames(ff)
    fps = ff.fps
    events: list[list[float]] = []
    for a, b, val in _runs(mask):
        if not val:
            continue
        t0, t1 = float(ff.times[a]), float(ff.times[b - 1])
        if events and t0 - events[-1][1] <= gap_s:
            events[-1][1] = t1
        else:
            events.append([t0, t1])
    return [(a, b) for a, b in events if b - a >= min_s]


def _apply_guard(
    segs: list[tuple[float, float, bool]], guard: float, total: float
) -> list[tuple[float, float, bool]]:
    """keep 区間を前後に `guard` 秒だけ広げる安全マージン。

    境界推定の残差は、本物の演奏を削る方向と、不要区間を少し含む方向の両方に出る。
    前者は取り返しがつかず後者は聴き飛ばせばよいだけなので、意図的に外側へ広げて
    「削りすぎ」を構造的に起こさないようにする。隣の区間を潰さない範囲でのみ広げる。
    """
    if guard <= 0:
        return segs
    # 隣の不要区間を食い尽くさないよう、各方向の拡張量は隣の長さの一定割合までに抑える。
    # (休憩が guard の 2 倍より短いと、両側から広げて休憩が消えてしまうため)
    max_share = 0.6
    out = list(segs)
    for i, (s, e, v) in enumerate(out):
        if not v:
            continue
        ns, ne = s, e
        if i > 0:
            prev_s = out[i - 1][0]
            ns = s - min(guard, max_share * (s - prev_s))
        if i < len(out) - 1:
            nxt_e = out[i + 1][1]
            ne = e + min(guard, max_share * (nxt_e - e))
        out[i] = (ns, ne, True)
        if i > 0:
            out[i - 1] = (out[i - 1][0], ns, out[i - 1][2])
        if i < len(out) - 1:
            out[i + 1] = (ne, out[i + 1][1], out[i + 1][2])

    # 拡張の結果ごく短くなった区間は残しても意味がないので隣に吸収する。
    cleaned: list[tuple[float, float, bool]] = []
    for s, e, v in out:
        if e - s < 15.0 and cleaned:
            cleaned[-1] = (cleaned[-1][0], e, cleaned[-1][2])
        else:
            cleaned.append((s, e, v))
    merged = [cleaned[0]]
    for seg in cleaned[1:]:
        if seg[2] == merged[-1][2]:
            merged[-1] = (merged[-1][0], seg[1], seg[2])
        else:
            merged.append(seg)
    return merged


def _snap_to_tuning(
    boundary: float, events: list[tuple[float, float]], back_s: float = 120.0, fwd_s: float = 60.0
) -> tuple[float, bool]:
    """合奏開始の境界を、直前のチューニング終了位置に寄せる。

    チューニングは音出し/休憩の最後に必ず現れるので、その終端が合奏の開始に最も近い。
    """
    best = None
    for a, b in events:
        if -back_s <= (b - boundary) <= fwd_s:
            d = abs(b - boundary)
            if best is None or d < best[0]:
                best = (d, b)
    if best is None:
        return boundary, False
    return best[1], True


# ---------------------------------------------------------------------------
# ラベル付け
# ---------------------------------------------------------------------------

def _label_segments(
    segs: list[tuple[float, float, bool]],
    score: np.ndarray,
    wf: WindowFeatures,
    thr: float,
    snapped: set[int],
) -> list[Segment]:
    spread = float(np.percentile(score, 90) - np.percentile(score, 10)) or 1.0
    n_keep = sum(1 for s in segs if s[2])
    keep_i = 0
    remove_positions = [i for i, s in enumerate(segs) if not s[2]]

    out: list[Segment] = []
    for i, (start, end, val) in enumerate(segs):
        sel = (wf.times >= start) & (wf.times < end)
        mean_score = float(np.mean(score[sel])) if np.any(sel) else float(thr)
        conf = 0.5 + 0.42 * min(abs(mean_score - thr) / (spread / 2), 1.0)
        if i in snapped:
            conf = min(0.97, conf + 0.06)

        if val:
            keep_i += 1
            label = f"合奏{keep_i}(推定)" if n_keep > 1 else "合奏(推定)"
        else:
            if i == remove_positions[0] and start <= 1.0:
                label = "音出し(推定)"
            elif i == remove_positions[-1] and end >= segs[-1][1] - 1.0:
                label = "片付け(推定)"
            else:
                label = "休憩(推定)"

        out.append(
            Segment(
                index=i + 1,
                start=start,
                end=end,
                label=label,
                action="keep" if val else "remove",
                confidence=round(min(0.97, max(0.5, conf)), 2),
            )
        )
    return out


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def propose_segments(
    ff: FrameFeatures,
    wf: WindowFeatures,
    total_duration: float,
    splits: int | None = None,
    min_keep_s: float = 480.0,
    min_remove_s: float = 60.0,
    smooth_s: float = 0.0,
    default_penalty: float = 12.0,
    guard_s: float = 120.0,
) -> tuple[list[Segment], dict]:
    """境界候補を提案する。戻り値は (区間リスト, 解析メタ情報)。"""
    score = ensemble_score(wf, smooth_s=smooth_s)
    thr = decision_threshold(score)
    spread = float(np.percentile(score, 90) - np.percentile(score, 10)) or 1.0
    d = (score - thr) / spread

    if splits:
        mask, penalty = _fit_penalty(d, splits)
    else:
        # ヒントがない場合は、数分程度の谷では切り替わらない程度の既定値。
        penalty = default_penalty
        mask = _viterbi(d, penalty)
    log(
        f"合奏らしさスコア: 閾値={thr:.2f} (大津={_otsu(score):.2f}, 分布幅={spread:.2f}) / "
        f"遷移ペナルティ={penalty:.2f} -> 合奏 {_count_keep_runs(mask)} 区間"
    )

    segs = _mask_to_bounds(mask, wf, total_duration)
    segs = _enforce_min_durations(segs, min_keep_s, min_remove_s)
    if splits and sum(1 for s in segs if s[2]) > splits:
        segs = _apply_split_hint(segs, splits)
        segs = _enforce_min_durations(segs, min_keep_s, min_remove_s)

    events = find_tuning_events(ff)
    log(f"チューニング候補: {len(events)} 件 " +
        ", ".join(fmt_time(a) for a, _ in events[:12]) + (" ..." if len(events) > 12 else ""))

    # 合奏の開始境界のみチューニング終端にスナップする(合奏終了側は手がかりが弱い)。
    snapped: set[int] = set()
    for i in range(1, len(segs)):
        if not segs[i][2]:
            continue
        new_b, ok = _snap_to_tuning(segs[i][0], events)
        # 前後の区間が最小長を割らない範囲でのみ動かす。
        if ok and segs[i - 1][0] + 60.0 < new_b < segs[i][1] - 60.0:
            segs[i - 1] = (segs[i - 1][0], new_b, segs[i - 1][2])
            segs[i] = (new_b, segs[i][1], segs[i][2])
            snapped.add(i)

    segs = _apply_guard(segs, guard_s, total_duration)

    result = _label_segments(segs, score, wf, thr, snapped)
    meta = {
        "threshold": thr,
        "otsu": _otsu(score),
        "transition_penalty": penalty,
        "smooth_s": smooth_s,
        "guard_s": guard_s,
        "silence_db": wf.silence_db,
        "win_s": wf.win_s,
        "hop_s": wf.hop_s,
        "weights": WEIGHTS,
        "tuning_events": [
            {"start": fmt_time(a), "end": fmt_time(b), "duration": round(b - a, 1)} for a, b in events
        ],
        "snapped_boundaries": sorted(fmt_time(segs[i][0]) for i in snapped),
    }
    return result, meta


def segments_to_json(segs: Iterable[Segment]) -> list[dict]:
    return [s.to_json() for s in segs]
