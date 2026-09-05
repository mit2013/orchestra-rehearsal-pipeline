"""ラウドネス測定と、配布用のダイナミクス処理(コンプレッサ + リミッター)。

## なぜピーク正規化をやめるのか

打楽器が入った 260829 で、ピーク正規化の限界がはっきり出た。`normalize_scope: date`
はその日でいちばん大きいピークを基準に全ブロック共通のゲインを決めるため、合奏3の
ピーク +11.43 dBFS が基準になり、打楽器の鳴らなかった合奏1(ピーク +0.31 dBFS)まで
一緒に 12 dB 近く下げられた。結果、ブロック間で **8.6 LU** の差が残った。

| | 統合ラウドネス | ラウドネスレンジ | トゥルーピーク |
|---|---|---|---|
| 前半(合奏1) | -34.3 LUFS | 27.7 LU | -10.3 dBFS |
| 中盤(合奏2) | -25.7 LUFS | 24.5 LU | -1.1 dBFS |
| 後半(合奏3) | -28.5 LUFS | 24.5 LU | -1.0 dBFS |

**ピークは1サンプルの話で、人が感じる音量とは別物である。** ブロックごとに
統合ラウドネスを目標に合わせれば、この差は原理的に生じない。

## 処理の順序

**ゲイン -> コンプレッサ -> トゥルーピークリミッター** の順に当てる。コンプで
打楽器の立ち上がりを先にならしておけば、リミッターはほとんど動かずに済む。
リミッターを先に置くと、打楽器のピークだけがリミッターに当たって音色が変わる。

リミッターは安全弁であり、常時動作させるものではない。

## 目標ラウドネスの合わせ方

コンプを通すとラウドネスが下がるため、「測って一度ゲインを当てる」だけでは目標に
乗らない。そこで測定を2段に分ける。

1. 素の統合ラウドネス I0 を測り、暫定ゲイン g0 = 目標 - I0 を決める
2. g0 を当ててコンプまで通したときの統合ラウドネス I1 を測り、
   最終ゲイン g1 = g0 + (目標 - I1) とする
3. g1 で本番の書き出しを行う

コンプは 2:1 でしきい値も高いのでほぼ線形に効き、この1回の補正で目標の
±0.5 LU に収まる。測定は `-f null` に捨てるので実時間の 1/280 程度で終わる。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .util import FFMPEG, PipelineError

# --- 既定値 -----------------------------------------------------------------
# 練習録音は音楽配信(-14 LUFS 前後)ほど上げる必要はなく、下げすぎると
# 「小さくて聞こえない」という現状の不満が残る。その中間。
DEFAULT_TARGET_LUFS = -20.0
# トゥルーピークの上限。書き出し(16bit / MP3)でのクリップを防ぐ安全弁。
DEFAULT_TRUE_PEAK_DB = -1.0
# alimiter は**サンプルピーク**で頭打ちにするので、そのままではインターサンプルの山が
# 上限を超える。260829 合奏3 で実測すると、天井 -1.3 dBFS で当てたのにトゥルーピークは
# -0.4 dBFS まで出た(0.9 dB の超過)。そこで **4倍にアップサンプルしてから当て、
# 戻す**。同じ素材で天井 -1.05 dBFS → トゥルーピーク -1.0 dBFS ちょうどに収まる。
LIMITER_OVERSAMPLE = 4
LIMITER_MARGIN_DB = 0.1
# パイプラインの素材は 48kHz。オーバーサンプルの倍率をかけて使う。
DEFAULT_SAMPLE_RATE = 48000

# --- パラレルコンプレッサ(小さい音だけを持ち上げる) -------------------------
# 通常のコンプは**大きい音を下げる**ので、静かなところは録れたままの小ささで
# 出ていく。260829 後半 00:31:32 からの 45 秒では、1秒ごとの実効音量が
# 静音部 -52.7 dBFS / 大音量部 -13.1 dBFS と 39.5 dB 開いていた。スマホの内蔵
# スピーカーでは前者はまず聴こえない。
#
# クラシックの録音では、これをパラレルコンプで解く。信号を2つに分け、片方を
# ピークで大きく潰れるコンプに通して持ち上げ、直の信号に足し戻す。**大きいところは
# ほぼ変わらず、小さいところだけが上がる。** 実測(makeup +17 dB):
#
#   静音部 -52.7 -> -35.3 dBFS (+17.3 dB) / 大音量部 -13.1 -> -12.5 dBFS (+0.7 dB)
#
# Waves の MV2(Low Level +14 dB)を同じ狙いで設定した結果は静 +13.1 / 動 +1.3 で、
# ほぼ同じかむしろ大音量部への影響が大きかった。**この用途にプラグインは要らない。**
PARALLEL_MAKEUP_DB = 17.0      # 0 で無効。大きいほど静音部が持ち上がる。上限として使う
#
# ただし **持ち上げるとノイズフロアも同じだけ上がる。** 260829 の外部マイクは
# 良いので気にならなかったが、S/N の悪い日には空調音が目立つ。しかもノイズフロアは
# ブロックによって大きく違う(260829 の実測、0.4秒窓の下位0.1%):
#
#   合奏1 -51.5 dBFS / 合奏2 -60.9 dBFS / 合奏3 -57.9 dBFS   ― 同じ日で 9.4 dB 差
#
# そこで固定値を上限とし、**仕上がりのノイズフロアが目標ラウドネスから
# これ以上近づかないところまで**に持ち上げ量を自動で抑える。会場やマイクが
# 変わった日に自動で加減が効く。
NOISE_FLOOR_BELOW_TARGET_DB = 14.0   # 目標 -20 LUFS なら天井は -34 dBFS
NOISE_FLOOR_WIN_S = 0.4              # 短い窓のほうが本当の暗騒音に近い
NOISE_FLOOR_PERCENTILE = 0.1
PARALLEL_THRESHOLD_DB = -46.0  # 静音部より下。ここから上を潰して定常に近づける
PARALLEL_RATIO = 20.0
PARALLEL_ATTACK_MS = 20.0
PARALLEL_RELEASE_MS = 400.0
PARALLEL_KNEE_DB = 8.0

# --- コンプレッサ(緩く。「かかっている」と分からない程度) -------------------
DEFAULT_COMP_RATIO = 2.0          # 明確に分からない程度
DEFAULT_COMP_THRESHOLD_OFFSET = 6.0   # しきい値 = 目標ラウドネス + この値 [dB]
DEFAULT_COMP_ATTACK_MS = 100.0    # 打楽器の立ち上がりの質感を残す
DEFAULT_COMP_RELEASE_MS = 1000.0  # フレーズ単位で戻る。ポンピングを避ける
DEFAULT_COMP_KNEE_DB = 6.0        # ソフトニー。しきい値付近の不連続をなくす

_I_RE = re.compile(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
_LRA_RE = re.compile(r"LRA:\s*(-?\d+(?:\.\d+)?)\s*LU")
_TP_RE = re.compile(r"True peak:\s*\n\s*Peak:\s*(-?inf|-?\d+(?:\.\d+)?)", re.MULTILINE)


@dataclass
class Loudness:
    integrated: float       # 統合ラウドネス [LUFS]
    lra: float              # ラウドネスレンジ [LU]
    true_peak: float        # トゥルーピーク [dBFS]

    def to_json(self) -> dict:
        return {
            "integrated_lufs": round(self.integrated, 2),
            "lra_lu": round(self.lra, 2),
            "true_peak_db": round(self.true_peak, 2),
        }

    def describe(self) -> str:
        return (f"{self.integrated:+.1f} LUFS / レンジ {self.lra:.1f} LU / "
                f"TP {self.true_peak:+.1f} dBFS")


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def compressor_filter(
    threshold_db: float,
    ratio: float = DEFAULT_COMP_RATIO,
    attack_ms: float = DEFAULT_COMP_ATTACK_MS,
    release_ms: float = DEFAULT_COMP_RELEASE_MS,
    knee_db: float = DEFAULT_COMP_KNEE_DB,
) -> str:
    """acompressor のフィルタ文字列。しきい値は dBFS で受けて線形に直す。

    acompressor の threshold は線形振幅(0.000976563〜1)しか受け付けない。
    """
    thr = min(max(db_to_linear(threshold_db), 0.000976563), 1.0)
    return (f"acompressor=threshold={thr:.6f}:ratio={ratio:g}:attack={attack_ms:g}"
            f":release={release_ms:g}:knee={knee_db:g}:detection=rms:link=average")


def limiter_filter(
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    oversample: int = LIMITER_OVERSAMPLE,
) -> str:
    """トゥルーピークリミッター。アップサンプルして当ててから戻す。

    天井は `LIMITER_MARGIN_DB` だけ下に置く。`oversample=1` にすると
    素のサンプルピークリミッターになる(比較用)。
    """
    ceiling = db_to_linear(true_peak_db - LIMITER_MARGIN_DB)
    lim = f"alimiter=limit={ceiling:.6f}:attack=5:release=50:level=disabled"
    if oversample <= 1:
        return lim
    return f"aresample={sample_rate * oversample},{lim},aresample={sample_rate}"


def parallel_graph(makeup_db: float = PARALLEL_MAKEUP_DB) -> str:
    """パラレルコンプの部分グラフ。入力を受けて、合成後の続きに繋がる形を返す。

    ラベルを使うので**戻り値には `;` が含まれる**。`-af` の単純フィルタ列では
    使えないため、呼び出し側は `-filter_complex` を使うこと(`measure` は自動で
    切り替える)。
    """
    comp = compressor_filter(PARALLEL_THRESHOLD_DB, PARALLEL_RATIO, PARALLEL_ATTACK_MS,
                             PARALLEL_RELEASE_MS, PARALLEL_KNEE_DB)
    return (f"asplit=2[pc_d][pc_c];"
            f"[pc_c]{comp},volume={makeup_db:.6f}dB[pc_w];"
            f"[pc_d][pc_w]amix=inputs=2:normalize=0")


def master_chain(
    gain_db: float,
    *,
    threshold_db: float,
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
    ratio: float = DEFAULT_COMP_RATIO,
    attack_ms: float = DEFAULT_COMP_ATTACK_MS,
    release_ms: float = DEFAULT_COMP_RELEASE_MS,
    knee_db: float = DEFAULT_COMP_KNEE_DB,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    oversample: int = LIMITER_OVERSAMPLE,
    with_limiter: bool = True,
    parallel_db: float = PARALLEL_MAKEUP_DB,
) -> str:
    """ゲイン -> パラレルコンプ -> コンプレッサ -> リミッター のフィルタ列。

    `parallel_db` が 0 より大きいとパラレルコンプが入り、**戻り値にラベルと `;` が
    含まれる**(`-filter_complex` が要る)。0 にすると従来どおりの単純なフィルタ列。

    ラウドネスを測るだけの下見では `oversample=1` にしてよい。リミッターの
    オーバーサンプルは統合ラウドネスをほとんど動かさない一方、処理時間は3倍になる。
    """
    head = f"volume={gain_db:.6f}dB"
    if parallel_db > 0:
        head = f"{head},{parallel_graph(parallel_db)}"
    parts = [head, compressor_filter(threshold_db, ratio, attack_ms, release_ms, knee_db)]
    if with_limiter:
        parts.append(limiter_filter(true_peak_db, sample_rate, oversample))
    return ",".join(parts)


def noise_floor(
    path: Path,
    win_s: float = NOISE_FLOOR_WIN_S,
    percentile: float = NOISE_FLOOR_PERCENTILE,
    sr: int = 16000,
    start: float | None = None,
    dur: float | None = None,
    pre_filter: str | None = None,
) -> float:
    """暗騒音の水準[dBFS]。短い窓の実効音量の下位パーセンタイルで測る。

    「いちばん静かな瞬間がどれくらいか」を知りたいので、ラウドネス(K特性)ではなく
    素の RMS を使う。窓を短くするほど本当の無音に近づくが、短すぎると波形の谷を
    拾うので 0.4 秒にしてある。

    `start` / `dur` で範囲を、`pre_filter` で前処理を指定できる。1本のプロキシに
    複数ブロックが入っている現場経路では、ブロックごとに切り出したうえで、
    符号化前に当てた固定ゲインを戻してから測る必要がある。
    """
    import numpy as np

    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path)]
    if pre_filter:
        cmd += ["-af", pre_filter]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(f"暗騒音の測定に失敗: {path.name}")
    x = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    n = max(1, int(sr * win_s))
    if len(x) < n * 10:
        return float("-inf")
    frames = np.array([20 * np.log10(np.sqrt((x[i:i + n] ** 2).mean()) + 1e-12)
                       for i in range(0, len(x) - n, n)])
    return float(np.percentile(frames, percentile))


def limit_parallel_makeup(
    floor_db: float,
    gain_db: float,
    max_makeup_db: float,
    ceiling_db: float,
) -> tuple[float, str]:
    """暗騒音が天井を超えないところまで持ち上げ量を抑える。

    パラレルの経路は、しきい値より下ではほぼ素通しに makeup を足した形になるので、
    **仕上がりの暗騒音 ≒ 元の暗騒音 + ゲイン + makeup** とみなせる(260829 の実測で
    makeup +17 に対し下位1%が +17.7 dB 動いた)。
    """
    if max_makeup_db <= 0:
        return 0.0, "パラレルコンプ無効"
    after = floor_db + gain_db
    allowed = ceiling_db - after
    if allowed >= max_makeup_db:
        return max_makeup_db, (f"暗騒音 {floor_db:+.1f} dBFS。上限 {max_makeup_db:+.0f} dB "
                               f"のままで天井 {ceiling_db:+.0f} dBFS に収まる")
    makeup = max(0.0, allowed)
    return makeup, (f"暗騒音 {floor_db:+.1f} dBFS と高いため {max_makeup_db:+.0f} -> "
                    f"{makeup:+.1f} dB に抑制(天井 {ceiling_db:+.0f} dBFS)")


def _run_ebur128(cmd: list[str], what: str) -> Loudness:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(
            f"ラウドネス測定に失敗: {what}\n"
            + proc.stderr.decode("utf-8", "replace").strip()[-800:]
        )
    text = proc.stderr.decode("utf-8", "replace")
    mi, ml, mt = _I_RE.search(text), _LRA_RE.search(text), _TP_RE.search(text)
    if not (mi and ml and mt):
        raise PipelineError(f"ラウドネス値を読み取れませんでした: {what}")
    tp = mt.group(1).lower()
    return Loudness(
        integrated=float(mi.group(1)),
        lra=float(ml.group(1)),
        true_peak=float("-inf") if tp.endswith("inf") else float(tp),
    )


def measure(
    path: Path,
    start: float | None = None,
    dur: float | None = None,
    pre_filter: str | None = None,
) -> Loudness:
    """1本のファイルのラウドネスを測る。`pre_filter` を通した結果も測れる。"""
    cmd = [FFMPEG, "-hide_banner", "-v", "info", "-nostats"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path)]
    tail = "ebur128=peak=true:framelog=quiet"
    chain = f"{pre_filter},{tail}" if pre_filter else tail
    if ";" in chain:
        # パラレルコンプのようにラベルを使うチェーンは -af では扱えない。
        cmd += ["-filter_complex", f"[0:a]{chain}[s]", "-map", "[s]", "-f", "null", "-"]
    else:
        cmd += ["-af", chain, "-f", "null", "-"]
    return _run_ebur128(cmd, str(path))


def measure_complex(
    inputs: list[Path],
    filter_complex: str,
    out_label: str,
    trim: str | None = None,
) -> Loudness:
    """複数入力を filter_complex で合成した結果のラウドネスを測る。

    `trim` を渡すと、合成結果のうちその範囲だけを測る(`atrim` のフィルタ文字列)。
    `-ss`/`-t` は入力単位に効くので filter_complex の途中では使えない。
    """
    cmd = [FFMPEG, "-hide_banner", "-v", "info", "-nostats"]
    for p in inputs:
        cmd += ["-i", str(p)]
    tail = "ebur128=peak=true:framelog=quiet"
    if trim:
        tail = f"{trim},{tail}"
    cmd += [
        "-filter_complex",
        f"{filter_complex};[{out_label}]{tail}[s]",
        "-map", "[s]", "-f", "null", "-",
    ]
    return _run_ebur128(cmd, " + ".join(p.name for p in inputs))


# --- 目標ラウドネスへのゲイン収束 -------------------------------------------
# コンプを通すとラウドネスが下がるので、当てて測り直して補正する。残差が
# これ以下になったら打ち切る。
GAIN_SETTLE_LU = 0.15
# パラレルコンプが入るとマスターの応答が線形から外れ、3回では収束しきらない
# ことがある(実測で残差 +0.5 LU が残った)。測定は実時間の 1/280 程度なので
# 回数を増やす損は小さい。
GAIN_MAX_ITER = 5


def solve_gain(
    measure_after,
    target_lufs: float,
    gain_db: float,
    settle_lu: float = GAIN_SETTLE_LU,
    max_iter: int = GAIN_MAX_ITER,
    on_step=None,
) -> tuple[float, "Loudness"]:
    """マスター通過後のラウドネスが目標に乗るゲインを求める。

    `measure_after(gain_db) -> Loudness` を呼びながら残差を足し込む。母艦の
    ブロック書き出し(`mix.py`)と現場プロキシからの切り出し(`field.py`)で
    **同じ手続きを使うため**にここへ置いてある。分けて書くと設定が食い違う。
    """
    result = None
    for _ in range(max_iter):
        result = measure_after(gain_db)
        resid = target_lufs - result.integrated
        if on_step is not None:
            on_step(gain_db, result, resid)
        if abs(resid) <= settle_lu:
            break
        gain_db += resid
    return gain_db, result
