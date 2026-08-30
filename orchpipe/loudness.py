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
) -> str:
    """ゲイン -> コンプレッサ -> リミッター のフィルタ列。

    ラウドネスを測るだけの下見では `oversample=1` にしてよい。リミッターの
    オーバーサンプルは統合ラウドネスをほとんど動かさない一方、処理時間は3倍になる。
    """
    parts = [f"volume={gain_db:.6f}dB",
             compressor_filter(threshold_db, ratio, attack_ms, release_ms, knee_db)]
    if with_limiter:
        parts.append(limiter_filter(true_peak_db, sample_rate, oversample))
    return ",".join(parts)


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
    chain = "ebur128=peak=true:framelog=quiet"
    if pre_filter:
        chain = f"{pre_filter},{chain}"
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
