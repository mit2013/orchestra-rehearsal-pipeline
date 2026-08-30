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

# --- 調和性のヒステリシス ---------------------------------------------------
# 調和性は合奏で 0.22〜0.30、打楽器が入ると 0.15〜0.20 に落ちる。しきい値
# 0.205 のすぐ両側なので、ひと続きのフレーズの中で確信度が 1.00 と 0.00 を
# 1〜2秒おきに往復する。260829 後半 00:31:31-00:32:01 は最大 -11.0 dBFS の
# トゥッティだが、判定は 6 回反転していた。
#
# そこで入りと抜けでしきい値を分ける。いったん演奏と認めたら、明確に下回る
# まで演奏のままにする。
HYST_ENTER_H = 0.215
HYST_STAY_H = 0.185

# --- 音量による演奏の上書き -------------------------------------------------
# 打楽器は倍音系列を持たないので調和性では拾えない。ティンパニの膜のモードは
# 1 : 1.5 : 2 : 2.44 : 2.9 と整数比でないため、音程が聞こえていても調和性は
# 0.00〜0.14 にしかならない(260829 3楽章冒頭で実測)。
#
# 指示書の当初案は「ブロック内の窓 RMS の上位10パーセンタイル」だったが、
# 実測すると**ほとんど効かなかった**。ソロのティンパニは -20.0 dBFS で、
# ブロックの 90% 点 -19.9 dBFS に届かない。トゥッティ(-11〜-17 dBFS)が
# 上位を占めるので、ソロは相対的に「小さい音」になってしまう。
#
# 代わりに無音閾値からの相対量で持つ。260829 後半では、この値で拾われる
# 発言窓は 309 窓中 2 窓(0.6%)しかなかった。
LOUD_MARGIN_DB = 14.0

# --- 文脈による橋渡し -------------------------------------------------------
# 弱音のトレモロは、窓ひとつを見る限り本物の無音と区別がつかない。
# 260829 1楽章 00:45:56-00:46:40(人手確認で「44秒間ずっと演奏中」)の実測:
#
#            RMS     調和性   安定性   フラックス
#   ppトレモロ  -50.7   0.151   0.878   0.206
#   本物の無音  -51.5   0.159   0.825   0.217
#
# どの指標でも分離しない。救えるのは文脈だけである。そこで **playing に挟まれ、
# 発言をひとつも含まない**非 playing の連なりを playing に倒す。発言を含む
# 場合に倒さないのは、指揮者が止めて指示している場面をそのまま残すためで、
# これが誤って演奏を作り出さないための主たる歯止めになっている。
#
# 30 秒で効果が飽和した(45/60/90 秒にしても結果が変わらない)ため、最も
# 弱い設定として 30 秒を既定にする。黙って止めた場合に飲み込む量の上限でもある。
BRIDGE_MAX_S = 30.0
# 橋渡しを止めるクラス。発言は「指揮者が止めている」ことの証拠であり、
# チューニングは残さないとダイジェストから外す判断ができなくなる。
BRIDGE_STOP_LABELS = ("speech", "tuning")

# 隙間の中で無音がこれより長く続いていたら橋渡ししない。
#
# 発言を含まないという条件だけでは、指揮者が黙って止めた場合を拾ってしまう。
# 260829 の3つの通し稽古(演奏だと確認済み)にある隙間 126 件を調べると、
# 中の連続無音は最長でも 8 秒だった(5 秒以内が 99%)。一方で通し稽古の外には
# 14〜17 秒続く無音を含む隙間があった。演奏が続いている限り、無音が10秒以上
# 途切れなく続くことはない。
BRIDGE_MAX_SILENCE_RUN_S = 8.0

# 橋渡しの前に落とす playing の断片、および橋渡しに必要な前後の演奏の長さ。
#
# 発言も長い無音も含まない隙間でも、部分練習の日には待ち時間を拾ってしまう。
# 260802 合奏2 で橋渡しが足した箇所を人が聴いたところ、11件中8件が「無駄時間
# (入れてはいけない)」、残り3件も「範囲の終わりかけに演奏が始まる」だけで、
# 「演奏している」は0件だった。
#
# 実測で分かったのは、それらの隙間の**後ろ側の playing が極端に短い**ことである
# (中央 1.5 秒、95%点 3.7 秒)。演奏が再開したのではなく、単発のノイズが
# playing と判定されているだけで、その両側の待ち時間が橋渡しされていた。
# 一方 260829 の通し稽古で救うべき隙間は、後ろ側の playing が中央 22.5 秒あった。
#
# そこで橋渡しの前に短い playing の断片を落とし、さらに前後の演奏が十分に
# 続いていることを条件にする。この2つで 260802 の8件はすべて弾ける。
BRIDGE_MIN_PLAY_S = 3.0     # これより短い playing は断片とみなし、橋渡しの足場にしない
BRIDGE_MIN_FLANK_S = 10.0   # 前後の演奏がともにこの長さ以上でなければ橋渡ししない
BRIDGE_PASSES = 4           # 橋渡しで足場が伸びると次が成立するので繰り返す

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

# ブロック末尾の「音出し」を切り分ける2条件。
#
# 練習が終わったあとも録音は回り続けるので、各自が勝手に音を出す時間が
# ブロックの末尾に張り付く。調和性から見ると本物の演奏と区別がつかず、実際
# 5ブロック中4ブロックで、最後の playing 区間(合計 329.7 秒)が音出しだった。
# 確信度は 1.00 で、主指標では原理的に拾えない。
#
# 人手判定つき10標本(音出し4・演奏6)で分離できたのは次の2つだけだった。
#
#   位置    : 音出しは録音を止めるまで続くので、区間の終わりがブロック末尾に
#             接する。音出しの4件は末尾から 0.1〜7.6 秒、本物の演奏だった
#             1件は 114.3 秒手前で終わっていた
#   音量の起伏: 合奏は全員が同じフレーズを共有するので一緒に鳴り始めて一緒に
#             止み、全体の音量が上下する。音出しは各自が無関係なので統計的に
#             のっぺりする。音出し 1.68〜2.83 に対し演奏 4.31〜8.31 で、
#             両者の間に 1.5 倍の隔たりがあった
#
# 起伏だけで判定すると、長く伸ばした弱奏の和音を誤って落とす危険がある
# (マーラー2番4楽章のような場面)。そこで**両方の条件を満たす場合だけ**
# 落とすことにし、対象も最後の playing 区間ひとつに限る。ブロック中間の
# 楽曲には原理的に触れない。標本が10件と少ないための保守的な設計である。
WARMUP_END_GAP_S = 30.0    # 区間の終わりがブロック末尾からこの秒数以内
WARMUP_DYN_TH = 3.5        # 1秒平均RMSの標準偏差[dB]がこれ未満
WARMUP_MIN_S = 10.0        # これより短い区間は統計が安定しないので判定しない
# 起伏を測る長さ。閾値はこの長さで較正してあるので、変えると閾値も較正し直しになる。
WARMUP_MAX_ANALYZE_S = 60.0

LABELS = ("silence", "tuning", "speech", "playing", "warmup", "unclear")


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


def harmonic_hysteresis(h: np.ndarray,
                        enter: float = HYST_ENTER_H,
                        stay: float = HYST_STAY_H) -> np.ndarray:
    """調和性のヒステリシス。`enter` で演奏に入り、`stay` を割るまで抜けない。

    しきい値ぎわで判定が往復するのを止めるためのもの。冒頭の定数の説明を参照。
    """
    out = np.zeros(len(h), dtype=bool)
    on = False
    for i, v in enumerate(h):
        on = (v >= stay) if on else (v >= enter)
        out[i] = on
    return out


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
    hyst_enter: float = HYST_ENTER_H,
    hyst_stay: float = HYST_STAY_H,
    loud_margin_db: float = LOUD_MARGIN_DB,
) -> tuple[list[Span], dict]:
    """ブロック全体を分類する。戻り値は (区間リスト, メタ情報)。"""
    log(f"  特徴量を抽出します: {src.name}")
    fs = pf.frame_series(src, a_refs=a_reference_frequencies())
    times, h, stab, onset, rms, amatch = _windows(fs, win_s, hop_s)

    silence_db = float(np.percentile(fs.rms_db, 5)) + SILENCE_MARGIN_DB
    conf = playing_confidence(h, stab, onset)

    # 演奏の証拠は3つの経路のいずれかで立つ。
    #   1. 確信度がしきい値以上(従来どおり)
    #   2. 調和性のヒステリシスで演奏の途中にいる
    #   3. 無音閾値を大きく超える音量が出ている(打楽器)
    evidence = ((conf >= play_threshold)
                | harmonic_hysteresis(h, hyst_enter, hyst_stay)
                | (rms >= silence_db + loud_margin_db))

    is_speech = np.zeros(len(times), dtype=bool)
    for a, b in speech_spans:
        is_speech |= (times + win_s > a) & (times < b)

    labels = np.full(len(times), "unclear", dtype=object)
    scores = np.zeros(len(times))

    # 証拠の強い順に確定させる。playing は最後で、しかも証拠が立つものだけ。
    #
    # ただし無音判定は**演奏の証拠に譲る**。マーラー2番4楽章(Urlicht)のような
    # 弱奏は、調和性が 0.20〜0.23 と演奏を示しているのにレベルが無音閾値を割る。
    # 260829 後半のアタッカ以降では窓の 13.1% が silence とされ、そのうち
    # 29/62 は調和性がしきい値を超えていた。順序をそのままにすると、演奏の証拠が
    # 立っていてもレベルだけで打ち消されてしまう。
    sil = (rms < silence_db) & ~(h > H_CENTER)
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
    play = rest & evidence
    labels[play] = "playing"
    scores[play] = conf[play]
    unclear = rest & (~evidence)
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
        "hyst_enter_h": hyst_enter,
        "hyst_stay_h": hyst_stay,
        "loud_margin_db": loud_margin_db,
        "loud_override_db": round(silence_db + loud_margin_db, 2),
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

    merged = _merge_same(spans)
    return merged, {"n_filled": n_filled, "seconds_filled": round(sec_filled, 1),
                    "max_fill_s": max_s}


def _merge_same(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for s in spans:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
            merged[-1].confidence = max(merged[-1].confidence, s.confidence)
        else:
            merged.append(s)
    return merged


def bridge_playing(spans: list[Span],
                   max_s: float = BRIDGE_MAX_S,
                   max_silence_run_s: float = BRIDGE_MAX_SILENCE_RUN_S,
                   min_play_s: float = BRIDGE_MIN_PLAY_S,
                   min_flank_s: float = BRIDGE_MIN_FLANK_S,
                   passes: int = BRIDGE_PASSES,
                   ) -> tuple[list[Span], dict]:
    """演奏に挟まれ、発言をひとつも含まない非 playing の連なりを playing に倒す。

    `fill_playing_gaps` との違いは2つある。silence をまたげること、そして
    unclear と silence が入り混じった連なりをひとまとまりとして扱うことである。

    なぜ必要か。弱音のトレモロは、窓ひとつを見る限り本物の無音と区別がつかない
    (冒頭の `BRIDGE_MAX_S` の説明を参照)。レベルでもスペクトルでも分離しない
    ので、「前後で演奏していて、誰も喋っていない」という文脈だけが手がかりになる。

    なぜ安全か。指揮者が止めれば必ず何か言う。260829 の3つの通し稽古(計 39.8 分)
    では、範囲の中に発言が 1.0 秒しかなく、それも1楽章の末尾だった。発言を含む
    連なりを対象から外すことで、本物の停止はそのまま残る。黙って止めた場合に
    飲み込む量は `max_s` で頭打ちになる。

    `speech` に加えて `tuning` も橋渡しを止める。チューニングは前後を演奏に
    挟まれるうえ発言を伴わないので、これを除かないと playing に塗り潰されて
    しまい、ダイジェストからチューニングを外す判断ができなくなる。
    """
    out = [Span(s.start, s.end, s.label, s.confidence) for s in spans]

    # 単発のノイズを足場にしないよう、短い playing は先に断片として下ろす。
    n_dropped, sec_dropped = 0, 0.0
    for s in out:
        if s.label == "playing" and s.duration < min_play_s:
            s.label = "unclear"
            n_dropped += 1
            sec_dropped += s.duration
    out = _merge_same(out)

    n_bridged, sec_bridged = 0, 0.0
    for _ in range(max(1, passes)):
        changed = False
        i = 0
        while i < len(out):
            if out[i].label != "playing":
                i += 1
                continue
            j = i + 1
            blocked = False
            while j < len(out) and out[j].label != "playing":
                if out[j].label in BRIDGE_STOP_LABELS:
                    blocked = True
                # 無音が長く続くなら演奏は止まっている。冒頭の定数の説明を参照。
                if out[j].label == "silence" and out[j].duration > max_silence_run_s:
                    blocked = True
                j += 1
            if j < len(out) and not blocked:
                gap = out[j].start - out[i].end
                flank = min(out[i].duration, out[j].duration)
                if 0 < gap <= max_s and flank >= min_flank_s:
                    for k in range(i + 1, j):
                        sec_bridged += out[k].duration
                        out[k].label = "playing"
                    n_bridged += 1
                    changed = True
            i = j
        # 橋渡しで足場が伸びると隣の隙間が条件を満たすようになる。
        out = _merge_same(out)
        if not changed:
            break

    return out, {"n_bridged": n_bridged,
                 "seconds_bridged": round(sec_bridged, 1),
                 "n_fragments_dropped": n_dropped,
                 "seconds_fragments_dropped": round(sec_dropped, 1),
                 "bridge_max_s": max_s,
                 "bridge_max_silence_run_s": max_silence_run_s,
                 "bridge_min_play_s": min_play_s,
                 "bridge_min_flank_s": min_flank_s}


def dynamic_range(src: Path, start: float, end: float,
                  max_s: float = WARMUP_MAX_ANALYZE_S) -> float:
    """区間の「音量の起伏」= 1秒平均 RMS の標準偏差[dB]。

    合奏なら全体が同時に鳴り始め同時に止むので値が大きく、各自が無関係に
    音を出しているだけなら小さくなる。調和性が「音が鳴っているか」しか
    見ないのに対し、こちらは「まとまっているか」を見る。

    測るのは区間の**中央**から `max_s` 秒ぶんである。区間の端は静寂や発話への
    遷移を含んでいて、それ自体が大きな音量変化として数えられてしまうため、
    端を含めると音出しでも値が持ち上がる(実測で 2.83 が 3.58 になった)。
    閾値もこの中央での測り方で較正してある。
    """
    dur = min(end - start, max_s)
    if dur < 1.0:
        return float("nan")
    mid = (start + end) / 2.0
    x = pf.decode(src, max(start, mid - dur / 2.0), dur)
    n = (len(x) - pf.FRAME) // pf.HOP + 1
    if n < 2:
        return float("nan")
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, pf.FRAME), strides=(x.strides[0] * pf.HOP, x.strides[0]),
        writeable=False)
    rms = 20.0 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12)
    w = int(round(pf.SR / pf.HOP))          # 1 秒ぶんのフレーム数
    k = len(rms) // w
    if k < 2:
        return float("nan")
    return float(rms[:k * w].reshape(k, w).mean(axis=1).std())


def mark_trailing_warmup(src: Path, spans: list[Span], total: float,
                         end_gap_s: float = WARMUP_END_GAP_S,
                         dyn_th: float = WARMUP_DYN_TH,
                         min_s: float = WARMUP_MIN_S) -> tuple[list[Span], dict]:
    """ブロック末尾に張り付いた最後の `playing` 区間が音出しなら `warmup` にする。

    位置と音量の起伏の**両方**が条件を満たす場合だけ落とす。対象は最後の
    playing 区間ひとつに限るので、ブロック中間の楽曲には影響しない。
    """
    info = {"end_gap_s": end_gap_s, "dyn_th": dyn_th, "min_s": min_s,
            "marked": False, "reason": "playing 区間がありません"}
    idx = [i for i, s in enumerate(spans) if s.label == "playing"]
    if not idx:
        return spans, info

    i = idx[-1]
    s = spans[i]
    gap = total - s.end
    info.update({"span": [round(s.start, 2), round(s.end, 2)],
                 "end_gap": round(gap, 1), "duration": round(s.duration, 1)})

    if gap > end_gap_s:
        info["reason"] = f"末尾から {gap:.1f} 秒手前で終わっており、音出しの位置ではありません"
        return spans, info
    if s.duration < min_s:
        info["reason"] = f"{s.duration:.1f} 秒と短く、起伏の統計が安定しません"
        return spans, info

    dyn = dynamic_range(src, s.start, s.end)
    info["dyn"] = None if dyn != dyn else round(dyn, 2)
    if dyn != dyn:                                  # NaN
        info["reason"] = "音量の起伏を測れませんでした"
        return spans, info
    if dyn >= dyn_th:
        info["reason"] = f"音量の起伏 {dyn:.2f} が閾値 {dyn_th} 以上で、合奏と判断しました"
        return spans, info

    s.label = "warmup"
    info.update({"marked": True, "seconds": round(s.duration, 1),
                 "reason": f"末尾から {gap:.1f} 秒・音量の起伏 {dyn:.2f} で音出しと判断しました"})

    merged: list[Span] = []
    for sp in spans:
        if merged and merged[-1].label == sp.label:
            merged[-1].end = sp.end
            merged[-1].confidence = max(merged[-1].confidence, sp.confidence)
        else:
            merged.append(sp)
    return merged, info


def summarize(spans: list[Span], total: float) -> dict:
    out = {}
    for lab in LABELS:
        sec = sum(s.duration for s in spans if s.label == lab)
        out[lab] = {"seconds": round(sec, 1),
                    "count": sum(1 for s in spans if s.label == lab),
                    "ratio": round(sec / total, 4) if total else 0.0}
    return out
