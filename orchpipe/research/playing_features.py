"""ステージG-1: 「演奏している」ことを積極的に示す特徴量の候補。

既存の4値分類は「無音・チューニング・発言を確定し、残りを playing とする」消去法で
あり、ダイジェストの圧縮率が体感と乖離する主因と判断されている。本モジュールは
その置き換えに向けて、演奏/非演奏を分離できる特徴量を実測するためのもの。

指示書 G-1-2 の4候補をすべて実装している:

- `polyphony`      多声性。同時に鳴っている倍音系列の数
- `harmonic_ratio` 調和性。倍音系列に説明できるエネルギーの割合
- `stability`      スペクトルの短時間安定性。発話は音色が目まぐるしく変わる
- `onset_rate`     オンセット密度 [回/秒]

比較用に低次の特徴量(RMS・フラットネス・重心)も同時に返す。

解析はいずれも 16kHz モノラルで行う。周波数分解能を稼ぐため、`features.py` の
既定(64ms 窓)より長い 128ms 窓を使う。倍音系列を数えるには 16Hz 刻みでは
低音域の基音を分離できないため。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..util import FFMPEG, PipelineError

SR = 16000
FRAME = 2048          # 128 ms
HOP = 512             # 32 ms
F_LO, F_HI = 55.0, 4000.0

# 倍音系列とみなす許容ずれ(比率)。±3% ≒ ±52 cent。
HARMONIC_TOL = 0.03
# 1つの系列として認めるのに必要な倍音の本数(基音を含む)
MIN_HARMONICS = 2
# ピークとして拾う下限(そのフレームの最大値に対する比)
PEAK_FLOOR = 0.06


@dataclass
class PlayingFeatures:
    polyphony: float
    harmonic_ratio: float
    stability: float
    onset_rate: float
    rms_db: float
    flatness: float
    centroid: float

    def to_json(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


def decode(path: Path, start: float, dur: float) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(f"デコードに失敗: {path}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def _spectra(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(フレーム数, ビン数) の振幅スペクトルと周波数軸。"""
    if len(x) < FRAME:
        return np.zeros((0, FRAME // 2 + 1)), np.fft.rfftfreq(FRAME, 1 / SR)
    n = 1 + (len(x) - FRAME) // HOP
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, FRAME), strides=(x.strides[0] * HOP, x.strides[0]), writeable=False)
    win = np.hanning(FRAME).astype(np.float32)
    mag = np.abs(np.fft.rfft(view * win, axis=1)).astype(np.float32)
    return mag, np.fft.rfftfreq(FRAME, 1 / SR)


def _peaks(spec: np.ndarray) -> np.ndarray:
    """局所最大のインデックス(振幅がフレーム最大の PEAK_FLOOR 倍以上)。"""
    if spec.size < 3:
        return np.empty(0, dtype=int)
    hi = spec.max()
    if hi <= 0:
        return np.empty(0, dtype=int)
    idx = np.flatnonzero(
        (spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:]) & (spec[1:-1] > hi * PEAK_FLOOR)
    ) + 1
    return idx


def _harmonic_analysis(spec: np.ndarray, freqs: np.ndarray) -> tuple[int, float]:
    """1フレームの (倍音系列の数, 系列に説明できたエネルギー比)。

    強いピークから順に基音候補とみなし、その整数倍の位置にあるピークを
    その系列のものとして回収する。回収済みのピークは他の系列に使わない。
    ノイズ的な音は倍音関係を作れないので系列が立たず、比率も上がらない。
    """
    idx = _peaks(spec)
    if len(idx) == 0:
        return 0, 0.0
    f = freqs[idx]
    a = spec[idx].astype(np.float64)
    band = (f >= F_LO) & (f <= F_HI)
    f, a, idx = f[band], a[band], idx[band]
    if len(f) == 0:
        return 0, 0.0

    total = float(np.sum(spec[(freqs >= F_LO) & (freqs <= F_HI)].astype(np.float64)))
    if total <= 0:
        return 0, 0.0

    order = np.argsort(-a)
    claimed = np.zeros(len(f), dtype=bool)
    n_series = 0
    explained = 0.0

    for i in order:
        if claimed[i]:
            continue
        f0 = f[i]
        if f0 <= 0:
            continue
        # この基音候補の整数倍に載るピークを集める
        ratio = f / f0
        near = np.abs(ratio - np.round(ratio)) < HARMONIC_TOL
        members = np.flatnonzero(near & (~claimed) & (np.round(ratio) >= 1))
        if len(members) < MIN_HARMONICS:
            continue
        claimed[members] = True
        n_series += 1
        explained += float(np.sum(a[members]))

    return n_series, min(explained / total, 1.0)


@dataclass
class FrameSeries:
    """ブロック全体をフレーム単位で見た候補特徴量。"""

    times: np.ndarray
    harmonic_ratio: np.ndarray
    polyphony: np.ndarray
    stability: np.ndarray      # 直前フレームとのスペクトル類似度
    flux: np.ndarray
    rms_db: np.ndarray
    a_match: np.ndarray        # 最強ピークが A の倍音に一致しているか(0/1)

    @property
    def fps(self) -> float:
        return SR / HOP


def _stream(path: Path, chunk_s: float = 60.0):
    """ffmpeg から 16kHz モノラルを読み、フレーム境界を跨がないように渡す。"""
    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = int(chunk_s * SR) * 4
    tail = np.zeros(0, dtype=np.float32)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                break
            x = np.frombuffer(buf, dtype="<f4")
            cur = np.concatenate([tail, x]) if tail.size else np.ascontiguousarray(x)
            if len(cur) < FRAME:
                tail = cur
                continue
            n = 1 + (len(cur) - FRAME) // HOP
            yield cur[: (n - 1) * HOP + FRAME], n
            tail = cur[n * HOP:].copy()
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        if proc.wait() != 0:
            raise PipelineError(f"デコードに失敗: {path}\n{err.strip()[-400:]}")


def frame_series(path: Path, a_refs: np.ndarray | None = None,
                 a_tol_cent: float = 60.0, progress=None) -> FrameSeries:
    """ブロック全体をストリーミングでフレーム特徴量にする。

    83 分ぶんのスペクトルを一度に持つと数百 MB になるため、チャンクごとに
    スカラーへ畳んでから連結する。
    """
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("h", "p", "s", "f", "r", "a")}
    prev_spec: np.ndarray | None = None
    n_done = 0

    for buf, n in _stream(path):
        mag, freqs = _spectra(buf)
        if mag.shape[0] == 0:
            continue
        band = (freqs >= F_LO) & (freqs <= F_HI)

        h = np.empty(mag.shape[0], dtype=np.float32)
        p = np.empty(mag.shape[0], dtype=np.float32)
        for k in range(mag.shape[0]):
            ns, r = _harmonic_analysis(mag[k], freqs)
            p[k] = ns
            h[k] = r
        acc["h"].append(h)
        acc["p"].append(p)

        logm = np.log(mag + 1e-8)
        logm = logm - logm.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(logm, axis=1) + 1e-12
        ref = prev_spec if prev_spec is not None else logm[0]
        prevm = np.vstack([ref[None, :], logm[:-1]])
        pn = np.linalg.norm(prevm, axis=1) + 1e-12
        acc["s"].append((np.sum(prevm * logm, axis=1) / (pn * nrm)).astype(np.float32))
        prev_spec = logm[-1].copy()

        pm = np.vstack([mag[0][None, :], mag[:-1]])
        acc["f"].append((np.sum(np.maximum(mag - pm, 0.0), axis=1)
                         / (np.sum(mag, axis=1) + 1e-12)).astype(np.float32))

        frames = np.lib.stride_tricks.as_strided(
            buf, shape=(mag.shape[0], FRAME),
            strides=(buf.strides[0] * HOP, buf.strides[0]), writeable=False)
        acc["r"].append((20.0 * np.log10(
            np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12)).astype(np.float32))

        if a_refs is not None:
            mb = mag[:, band]
            pk = freqs[band][np.argmax(mb, axis=1)]
            cents = np.min(np.abs(1200.0 * np.log2(
                np.maximum(pk, 1.0)[:, None] / a_refs[None, :])), axis=1)
            acc["a"].append((cents < a_tol_cent).astype(np.float32))
        else:
            acc["a"].append(np.zeros(mag.shape[0], dtype=np.float32))

        n_done += mag.shape[0]
        if progress:
            progress(n_done * HOP / SR)

    if n_done == 0:
        raise PipelineError(f"解析できるフレームがありません: {path}")

    def cat(k):
        return np.concatenate(acc[k])

    return FrameSeries(
        times=(np.arange(n_done, dtype=np.float64) * HOP + FRAME / 2) / SR,
        harmonic_ratio=cat("h"), polyphony=cat("p"), stability=cat("s"),
        flux=cat("f"), rms_db=cat("r"), a_match=cat("a"),
    )


def extract(x: np.ndarray) -> PlayingFeatures:
    """1区間ぶんの波形から候補特徴量を計算する。"""
    mag, freqs = _spectra(x)
    if mag.shape[0] < 3:
        return PlayingFeatures(0, 0, 0, 0, -120.0, 0, 0)

    # --- 多声性・調和性(フレームごとに求めて中央値)------------------------
    poly, hratio = [], []
    for k in range(mag.shape[0]):
        n, r = _harmonic_analysis(mag[k], freqs)
        poly.append(n)
        hratio.append(r)

    # --- スペクトルの短時間安定性 -------------------------------------------
    # 隣接フレームの対数スペクトル同士のコサイン類似度。発話は音色が動くので下がる。
    logm = np.log(mag + 1e-8)
    logm = logm - logm.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(logm, axis=1) + 1e-12
    cos = np.sum(logm[:-1] * logm[1:], axis=1) / (nrm[:-1] * nrm[1:])
    stability = float(np.median(cos))

    # --- オンセット密度 ------------------------------------------------------
    prev = np.vstack([mag[0][None, :], mag[:-1]])
    flux = np.sum(np.maximum(mag - prev, 0.0), axis=1) / (np.sum(mag, axis=1) + 1e-12)
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) or 1e-6
    th = med + 3.0 * 1.4826 * mad
    hits = (flux[1:-1] > th) & (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:])
    dur = len(x) / SR
    onset_rate = float(np.sum(hits) / dur) if dur > 0 else 0.0

    # --- 比較用の低次特徴量 ---------------------------------------------------
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))
    band = (freqs >= F_LO) & (freqs <= F_HI)
    mb = mag[:, band].astype(np.float64)
    flat = float(np.median(np.exp(np.mean(np.log(mb + 1e-10), axis=1))
                           / (np.mean(mb, axis=1) + 1e-12)))
    cent = float(np.median(np.sum(freqs[band][None, :] * mb, axis=1)
                           / (np.sum(mb, axis=1) + 1e-12)))

    return PlayingFeatures(
        polyphony=float(np.median(poly)),
        harmonic_ratio=float(np.median(hratio)),
        stability=stability,
        onset_rate=onset_rate,
        rms_db=20.0 * np.log10(rms + 1e-20),
        flatness=flat,
        centroid=cent,
    )
