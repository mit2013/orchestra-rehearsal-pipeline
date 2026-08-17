"""ステージG-2: 積極的な証拠にもとづく分類の再設計。

旧 `states.py` は「無音・チューニング・発言を確定し、**残りを playing とする**」
消去法だった。そのため証拠が何もない区間まで playing に落ち、ダイジェストの
圧縮率が体感と乖離していた(実測8割、体感3〜5割)。

本モジュールでは4クラスすべてを積極的な証拠で判定し、**どのクラスの証拠も
立たない区間は `unclear` として明示する**。指示書 §2 のとおり、ダイジェストは
`trimmed/` に完全版がある前提の派生物なので、迷った区間は残さず捨てる方向に倒す。
これは本編トリミングでの「多めに残す」安全側バイアスの意図的な逆転である。

各クラスの証拠(G-1 の実測にもとづく):

- `silence`  レベルが無音閾値付近であること
- `speech`   ASR が実際に日本語テキストを書き起こせたこと(最も強い非演奏の証拠)
- `tuning`   最強ピークが基準音Aの倍音に載り続けること(`tuning.py` と同じ考え方)
- `playing`  **調和性**(倍音系列で説明できるエネルギーの割合)が高いこと

G-1 の実測(260802 合奏2、参照区間157件)では調和性が最もよく分離した:

    演奏     n=32  0.229 〜 0.285  (中央 0.263)
    弱奏参考 n= 5  0.215 〜 0.259  (中央 0.248)   ← 音量は発言と同等
    発言     n=60  0.126 〜 0.219  (中央 0.169)
    無音     n=60  0.140 〜 0.193  (中央 0.164)

閾値 0.20 で演奏 32/32・弱奏 5/5 が演奏側、無音 0/60・発言 3/60 のみ誤り。
**弱奏(-45〜-49 dBFS、発言と同じ音量帯)が演奏側に入る**ことが重要で、
この指標が音量ではなく音の構造を見ていることの裏づけになっている。

補助として、スペクトル安定性(発話は音色が目まぐるしく変わる)とオンセット密度
(発話の音節アタックのほうが密)を使う。フラットネスは AUC 0.502 と無価値
だったので使わない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import log
from . import playing_features as pf
from .tuning import a_reference_frequencies

WIN_S = 1.0
HOP_S = 0.5

# --- 演奏の証拠(G-1 の実測から)-------------------------------------------
# 調和性の中心。発言の最大 0.219 と演奏の最小 0.229 のあいだに取る。
H_CENTER = 0.205
# 確信度に変換するときの傾き。0.17 で約0.06、0.24 で約0.94 になる。
H_SLOPE = 80.0
# 補助指標の中心(演奏と発言の中間に置く)
STAB_CENTER = 0.865
STAB_SLOPE = 60.0
ONSET_CENTER = 0.70          # 回/秒。これより密なら発話寄り
ONSET_SLOPE = 3.0

# 補助指標が主指標を動かせる幅。主従を崩さないよう控えめにする。
AUX_WEIGHT = 0.25

# 無音とみなすレベル(分布の5%点からの上乗せ)
SILENCE_MARGIN_DB = 6.0
# チューニングとみなす A 一致率
TUNING_A_RATIO = 0.55
# チューニングとみなす最短の長さ。`tuning.py` の既定と揃えてある。
MIN_TUNING_S = 12.0

# 演奏に挟まれた短い unclear を playing に埋め戻す上限。
#
# 閾値付近の unclear は、その約半分(15.4分中7.3分)が playing に挟まれており、
# 長さの中央値は 1.0 秒しかない。人手で10件を聴いて確認したところ、演奏に
# 挟まれた7件のうち6件は実際に弱音部の演奏だった(残り1件はチューニングの
# 終わりかけ)。これらを落とすと圧縮率は上がるが、ダイジェストが1〜3秒おきに
# 途切れてブツ切りになる。
#
# 一方、silence や speech に隣接する unclear は埋め戻さない。同じ確認で、
# 「演奏は既に止まっており指揮者が指示している」場面が正しく非演奏と判定
# できていたため。片側が silence の場合も同様に埋め戻さない。
MAX_FILL_S = 5.0

LABELS = ("silence", "tuning", "speech", "playing", "unclear")


@dataclass
class Span:
    start: float
    end: float
    label: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2),
                "label": self.label, "confidence": round(float(self.confidence), 3)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def playing_confidence(h: np.ndarray, stab: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """演奏である確信度 0-1。

    主判定は調和性。補助2つは ±`AUX_WEIGHT` の範囲でしか効かせない
    (指示書 G-2 の「最も分離性能が高かった特徴量を主判定に据え、他は補助」)。
    """
    main = _sigmoid(H_SLOPE * (h - H_CENTER))
    aux = 0.5 * (_sigmoid(STAB_SLOPE * (stab - STAB_CENTER))
                 + _sigmoid(-ONSET_SLOPE * (onset - ONSET_CENTER)))
    return np.clip(main * (1.0 - AUX_WEIGHT) + main * aux * AUX_WEIGHT * 2.0, 0.0, 1.0)


def _windows(fs: pf.FrameSeries, win_s: float, hop_s: float):
    """フレーム列を窓ごとの統計に畳む。"""
    fps = fs.fps
    w = max(1, int(round(win_s * fps)))
    hp = max(1, int(round(hop_s * fps)))
    n = len(fs.times)
    starts = np.arange(0, max(n - w + 1, 1), hp)
    times = np.array([float(fs.times[a]) for a in starts])

    def med(arr):
        return np.array([float(np.median(arr[a:a + w])) for a in starts])

    onset = []
    med_f = float(np.median(fs.flux))
    mad_f = float(np.median(np.abs(fs.flux - med_f))) or 1e-6
    th = med_f + 3.0 * 1.4826 * mad_f
    for a in starts:
        seg = fs.flux[a:a + w]
        if len(seg) < 3:
            onset.append(0.0)
            continue
        hits = (seg[1:-1] > th) & (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
        onset.append(float(hits.sum()) / win_s)

    return (times, med(fs.harmonic_ratio), med(fs.stability),
            np.array(onset), med(fs.rms_db), np.array(
                [float(np.mean(fs.a_match[a:a + w])) for a in starts]))


def classify(
    src: Path,
    speech_spans: list[tuple[float, float]],
    win_s: float = WIN_S,
    hop_s: float = HOP_S,
    play_threshold: float = 0.5,
) -> tuple[list[Span], dict]:
    """ブロック全体を分類する。戻り値は (区間リスト, メタ情報)。"""
    log(f"  特徴量を抽出します: {src.name}")
    fs = pf.frame_series(src, a_refs=a_reference_frequencies())
    times, h, stab, onset, rms, amatch = _windows(fs, win_s, hop_s)

    silence_db = float(np.percentile(fs.rms_db, 5)) + SILENCE_MARGIN_DB
    conf = playing_confidence(h, stab, onset)

    is_speech = np.zeros(len(times), dtype=bool)
    for a, b in speech_spans:
        is_speech |= (times + win_s > a) & (times < b)

    labels = np.full(len(times), "unclear", dtype=object)
    scores = np.zeros(len(times))

    # 証拠の強い順に確定させる。playing は最後で、しかも閾値を満たすものだけ。
    sil = rms < silence_db
    labels[sil] = "silence"
    scores[sil] = 1.0 - conf[sil]

    tun = (~sil) & (amatch > TUNING_A_RATIO) & (h > H_CENTER)
    labels[tun] = "tuning"
    scores[tun] = amatch[tun]

    sp = (~sil) & (~tun) & is_speech
    labels[sp] = "speech"
    scores[sp] = 1.0 - conf[sp]

    # チューニングは「持続する単一音」である。合奏中にもたまたま最強ピークが
    # A の倍音に載る窓が散発するので、短い塊は採用しない(実測で 158 区間に
    # 散っていた。本物は冒頭の 1 件だけ)。
    min_tun = int(round(MIN_TUNING_S / hop_s))
    i = 0
    while i < len(labels):
        if labels[i] != "tuning":
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] == "tuning":
            j += 1
        if j - i < min_tun:
            labels[i:j] = "unclear"      # 証拠不足として一旦戻す
        i = j

    rest = (labels == "unclear") | ((~sil) & (~tun) & (~sp))
    play = rest & (conf >= play_threshold)
    labels[play] = "playing"
    scores[play] = conf[play]
    unclear = rest & (conf < play_threshold)
    labels[unclear] = "unclear"
    scores[unclear] = conf[unclear]

    # 窓は win_s ぶんの幅を持つが hop_s ずつ進むので、そのまま区間にすると
    # 重なって二重計上になる。各窓には hop_s ぶんだけ受け持たせ、最後の窓だけ
    # 窓長いっぱいまで伸ばして隙間なく敷き詰める。
    spans: list[Span] = []
    for i, t in enumerate(times):
        a = float(t)
        b = float(t) + (win_s if i == len(times) - 1 else hop_s)
        if spans and labels[i] == spans[-1].label:
            spans[-1].end = b
            spans[-1].confidence = max(spans[-1].confidence, float(scores[i]))
        else:
            if spans:
                a = spans[-1].end
            spans.append(Span(a, b, str(labels[i]), float(scores[i])))

    meta = {
        "source": str(src),
        "duration": float(fs.times[-1]),
        "win_s": win_s, "hop_s": hop_s,
        "silence_db": round(silence_db, 2),
        "play_threshold": play_threshold,
        "h_center": H_CENTER,
        "n_windows": int(len(times)),
        "confidence_percentiles": {
            str(p): round(float(np.percentile(conf, p)), 3) for p in (10, 25, 50, 75, 90)
        },
    }
    return spans, meta


def fill_playing_gaps(spans: list[Span],
                      max_s: float = MAX_FILL_S) -> tuple[list[Span], dict]:
    """演奏に挟まれた短い `unclear` を playing に埋め戻す。

    **両隣がともに playing の場合だけ**が対象。片側が silence や speech の
    unclear は、実際に演奏が止まっている場面であることが人手確認で分かって
    いるので触らない。埋め戻した区間の確信度はそのまま残すので、あとから
    「本来は閾値未満だった」ことが分かる。
    """
    n_filled, sec_filled = 0, 0.0
    for i in range(1, len(spans) - 1):
        s = spans[i]
        if s.label != "unclear" or s.duration > max_s:
            continue
        if spans[i - 1].label == "playing" and spans[i + 1].label == "playing":
            s.label = "playing"
            n_filled += 1
            sec_filled += s.duration

    merged: list[Span] = []
    for s in spans:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
            merged[-1].confidence = max(merged[-1].confidence, s.confidence)
        else:
            merged.append(s)
    return merged, {"n_filled": n_filled, "seconds_filled": round(sec_filled, 1),
                    "max_fill_s": max_s}


def summarize(spans: list[Span], total: float) -> dict:
    out = {}
    for lab in LABELS:
        sec = sum(s.duration for s in spans if s.label == lab)
        out[lab] = {"seconds": round(sec, 1),
                    "count": sum(1 for s in spans if s.label == lab),
                    "ratio": round(sec / total, 4) if total else 0.0}
    return out
