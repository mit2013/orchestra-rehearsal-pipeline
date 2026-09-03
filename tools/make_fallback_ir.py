#!/usr/bin/env python3
"""同梱用のインパルス応答 `assets/ir/hall.wav` を作る。

実測 IR(Waves の IR ライブラリなど)はライセンス品なので、このリポジトリには
入れられない。かといって IR が無いとリバーブがまったく動かないので、**合成した
ホール残響**を既定の代替として置いてある。これはどこかの実在ホールを模したもの
ではなく、オーケストラの練習録音に足して不自然にならない程度の響きを、初期反射と
拡散残響から組み立てたものである。

構成は二段。

1. **初期反射** — 13〜53 ms に離散的な反射を6発置く。左右で到達時刻をずらして
   広がりを作る。壁と天井からの一次反射に相当する。
2. **拡散残響** — 帯域ごとに減衰時間の違う雑音。実際のホールは空気吸収と内装の
   吸音で高い音ほど速く減衰するので、そこを再現している。左右は無相関にする。

帯域別の残響時間は下の BANDS を参照。中低域で 2.1 秒、10 kHz 付近で 0.8 秒に
落ちる設計で、Disney Hall(1.47 秒)と Birmingham Symphony Hall(2.00 秒)の
あいだに収まる。

`orchpipe.reverb.prepare_ir` が読み込み時にエネルギーを 1 に正規化し、
プリディレイを付け直すので、ここでは絶対レベルを気にしなくてよい。

    python3 tools/make_fallback_ir.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
DURATION_S = 2.4
SEED = 20260904

# (下限Hz, 上限Hz, 残響時間[秒])。低いほど長く残る。
BANDS = [
    (20, 250, 2.45),
    (250, 2000, 2.10),
    (2000, 8000, 1.45),
    (8000, 20000, 0.80),
]

# (時刻[ms], 振幅, 左右差[ms])。壁・天井からの一次反射。
EARLY = [
    (13.0, 0.62, 1.7),
    (19.0, 0.48, -2.3),
    (27.0, 0.40, 3.1),
    (35.0, 0.31, -1.1),
    (41.0, 0.26, 2.6),
    (53.0, 0.19, -3.4),
]


def band_limited_noise(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """`lo`〜`hi` Hz だけを持つ雑音。FFT でビンを落として作る。"""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n)


def diffuse_tail(n: int, rng: np.random.Generator) -> np.ndarray:
    """帯域ごとに減衰時間の違う拡散残響。左右は無相関にする。"""
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for lo, hi, rt in BANDS:
        env = np.exp(-6.908 * t / rt)          # 60 dB 落ちるまでを rt 秒に
        for ch in range(2):
            out[:, ch] += band_limited_noise(n, lo, hi, rng) * env
    return out


def early_reflections(n: int) -> np.ndarray:
    out = np.zeros((n, 2))
    for ms, amp, spread in EARLY:
        for ch, sign in ((0, -0.5), (1, 0.5)):
            i = int(round((ms + sign * spread) / 1000.0 * SR))
            if 0 <= i < n:
                out[i, ch] += amp
    return out


def build(duration_s: float = DURATION_S, seed: int = SEED) -> np.ndarray:
    n = int(duration_s * SR)
    rng = np.random.default_rng(seed)
    ir = early_reflections(n) + diffuse_tail(n, rng) * 0.5
    # 先頭に直接音を1サンプル置く。prepare_ir はここを基準に前を捨てる。
    ir[0] = [1.0, 1.0]
    return ir / np.abs(ir).max() * 0.98


def rt60(x: np.ndarray) -> float:
    """Schroeder 逆積分から T20 を測り、60 dB に外挿する(検算用)。"""
    y = x.mean(axis=1)
    y = y[int(np.argmax(np.abs(y))):]
    e = np.cumsum(y[::-1] ** 2)[::-1]
    e = 10 * np.log10(np.maximum(e / e[0], 1e-12))
    at = lambda db: (int(np.argmax(e <= db)) / SR if np.any(e <= db) else float("nan"))
    return (at(-25) - at(-5)) * 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="assets/ir/hall.wav")
    ap.add_argument("--seconds", type=float, default=DURATION_S)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    ir = build(args.seconds, args.seed)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), ir, SR, subtype="PCM_24")
    print(f"{dst}  {len(ir)/SR:.2f} 秒 / {SR} Hz / ステレオ / "
          f"{dst.stat().st_size/1024:.0f} KB")
    print(f"残響時間(実測 T20 から外挿): {rt60(ir):.2f} 秒")


if __name__ == "__main__":
    main()
