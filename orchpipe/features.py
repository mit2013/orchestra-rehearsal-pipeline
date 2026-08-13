"""汎用の音響特徴量抽出モジュール。

このモジュールは「練習の粗い区切り」に特化していない。フレーム単位の低次特徴量を
ストリーミングで抽出し、任意の窓長・ホップで統計量に集約するだけの層である。
フェーズ1では窓 20 秒 / ホップ 5 秒という粗い粒度で使うが、将来の
「指揮者が止めた単位でのチャプター分割」や「指揮者発言のみカット」では
同じ関数を窓 1〜2 秒で呼べばよい。区間の意味づけ(合奏/音出し/休憩)は
segment.py 側の責務で、ここには持ち込まない。

巨大な32bit float WAV を丸ごとメモリに載せないよう、ffmpeg のパイプから
ブロック単位で読みながらフレーム特徴量だけを蓄積する。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .util import FFMPEG, PipelineError, log

EPS = 1e-12

# 解析用の既定パラメータ。原音の 48kHz/32bit は解析には過剰なので 16kHz モノラルに落とす。
# (書き出しは常に元データから行うので、ここでの間引きは音質に影響しない)
DEFAULT_SR = 16000
DEFAULT_FRAME = 1024      # 64 ms
DEFAULT_HOP = 256         # 16 ms -> 62.5 フレーム/秒


# ---------------------------------------------------------------------------
# フレーム単位の特徴量
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """フレーム(既定 64ms 窓 / 16ms ホップ)ごとの低次特徴量。"""

    sr: int
    frame: int
    hop: int
    times: np.ndarray        # 各フレーム中心の時刻 [s]
    rms_db: np.ndarray       # ラウドネス [dBFS]
    flux: np.ndarray         # スペクトルフラックス(オンセット強度)
    flatness: np.ndarray     # スペクトルフラットネス 0-1(1に近いほど雑音的)
    centroid: np.ndarray     # スペクトル重心 [Hz]
    tonal: np.ndarray        # 最強ピーク近傍のエネルギー集中度 0-1(純音性)
    peak_hz: np.ndarray      # 最強スペクトルピークの周波数 [Hz](放物線補間つき)

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0


def _pcm_stream(path: Path, sr: int, block_frames: int):
    """ffmpeg で 16kHz モノラル float32 にデコードし、ブロックごとに yield する。"""
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = block_frames * 4
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                break
            yield np.frombuffer(buf, dtype="<f4")
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc != 0:
            raise PipelineError(f"デコードに失敗しました: {path}\n{err.strip()}")


def _frame_view(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """コピーなしでフレーム行列 (n_frames, frame) のビューを作る。"""
    if len(x) < frame:
        return np.empty((0, frame), dtype=x.dtype)
    n = 1 + (len(x) - frame) // hop
    return np.lib.stride_tricks.as_strided(
        x, shape=(n, frame), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    )


def extract_frame_features(
    path: Path,
    sr: int = DEFAULT_SR,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    block_seconds: float = 120.0,
    progress_every: float = 600.0,
) -> FrameFeatures:
    """音声ファイル全体をストリーミング処理してフレーム特徴量を返す。"""
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)

    # 純音性・ピッチ推定は楽音の基音が乗る帯域に限定する(超低域のゴロつきと
    # 高域のノイズに引きずられないように)。
    band = (freqs >= 55.0) & (freqs <= 4000.0)
    band_idx = np.flatnonzero(band)
    b0, b1 = int(band_idx[0]), int(band_idx[-1]) + 1
    band_freqs = freqs[b0:b1]
    bin_hz = float(freqs[1])

    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("rms", "flux", "flat", "cent", "tonal", "peak")
    }
    tail = np.zeros(0, dtype=np.float32)
    prev_mag: np.ndarray | None = None
    n_done = 0
    next_report = progress_every

    block_frames = int(block_seconds * sr)
    log(f"特徴量抽出開始: {path.name} ({sr}Hz, 窓{frame/sr*1000:.0f}ms, ホップ{hop/sr*1000:.0f}ms)")

    for chunk in _pcm_stream(path, sr, block_frames):
        buf = np.concatenate([tail, chunk]) if tail.size else np.ascontiguousarray(chunk)
        frames = _frame_view(buf, frame, hop)
        if frames.shape[0] == 0:
            tail = buf
            continue
        tail = buf[frames.shape[0] * hop:].copy()

        # --- 時間領域 ---
        acc["rms"].append(np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + EPS))

        # --- 周波数領域 ---
        mag = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)

        # スペクトルフラックス: 増加分のみ。ブロック境界をまたいで直前フレームを引き継ぐ。
        ref = prev_mag if prev_mag is not None else mag[0]
        prev = np.vstack([ref[None, :], mag[:-1]])
        norm = np.sum(mag, axis=1) + EPS
        acc["flux"].append((np.sum(np.maximum(mag - prev, 0.0), axis=1) / norm).astype(np.float32))
        prev_mag = mag[-1].copy()

        # スペクトルフラットネス(幾何平均/算術平均)と重心。
        logm = np.log(mag + EPS)
        acc["flat"].append(
            (np.exp(np.mean(logm, axis=1)) / (np.mean(mag, axis=1) + EPS)).astype(np.float32)
        )
        acc["cent"].append(
            (np.sum(freqs[None, :] * mag, axis=1) / norm).astype(np.float32)
        )

        # 純音性とピッチ: 帯域内の最強ピークとその近傍±2ビンのエネルギー比。
        # チューニングの基準音のように単一持続音が鳴っていると 1 に近づく。
        pw = mag[:, b0:b1] ** 2
        tot = np.sum(pw, axis=1) + EPS
        pk = np.argmax(pw, axis=1)
        rows = np.arange(pw.shape[0])
        lo = np.maximum(pk - 2, 0)
        hi = np.minimum(pk + 3, pw.shape[1])
        csum = np.concatenate([np.zeros((pw.shape[0], 1), np.float32), np.cumsum(pw, axis=1)], axis=1)
        acc["tonal"].append((csum[rows, hi] - csum[rows, lo]) / tot)

        # 放物線補間でビン幅(約16Hz)より細かいピッチを得る。
        pkc = np.clip(pk, 1, pw.shape[1] - 2)
        a = pw[rows, pkc - 1]
        b = pw[rows, pkc]
        c = pw[rows, pkc + 1]
        denom = a - 2 * b + c
        delta = np.where(np.abs(denom) > EPS, 0.5 * (a - c) / np.where(np.abs(denom) > EPS, denom, 1.0), 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        acc["peak"].append((band_freqs[pkc] + delta * bin_hz).astype(np.float32))

        n_done += frames.shape[0]
        secs = n_done * hop / sr
        if secs >= next_report:
            log(f"  ... {secs/60:.0f} 分処理済み")
            next_report += progress_every

    if n_done == 0:
        raise PipelineError(f"解析できるフレームがありません: {path}")

    def cat(key: str) -> np.ndarray:
        return np.concatenate(acc[key]).astype(np.float32)

    rms = cat("rms")
    ff = FrameFeatures(
        sr=sr,
        frame=frame,
        hop=hop,
        times=(np.arange(len(rms), dtype=np.float64) * hop + frame / 2) / sr,
        rms_db=(20.0 * np.log10(rms + EPS)).astype(np.float32),
        flux=cat("flux"),
        flatness=cat("flat"),
        centroid=cat("cent"),
        tonal=cat("tonal"),
        peak_hz=cat("peak"),
    )
    log(f"特徴量抽出完了: {len(rms)} フレーム ({ff.duration/60:.1f} 分)")
    return ff


# ---------------------------------------------------------------------------
# 窓単位への集約
# ---------------------------------------------------------------------------

@dataclass
class WindowFeatures:
    """任意の窓長・ホップで集約した中次特徴量。

    フェーズ1は 20s/5s、将来の細粒度チャプター分割は 2s/0.5s といった具合に
    同じ構造体を粒度違いで使い回す想定。
    """

    win_s: float
    hop_s: float
    times: np.ndarray          # 各窓の中心時刻 [s]
    starts: np.ndarray
    ends: np.ndarray
    loudness: np.ndarray       # 平均ラウドネス [dBFS]
    silence_ratio: np.ndarray  # 無音(閾値以下)フレームの割合 0-1
    dyn_range: np.ndarray      # ラウドネスの p90-p10 [dB] = 強弱のメリハリ
    loud_std: np.ndarray       # ラウドネスの標準偏差 [dB]
    pulse_clarity: np.ndarray  # オンセット包絡の自己相関ピーク 0-1 = リズムの統一感
    flux_crest: np.ndarray     # オンセット包絡の尖り具合(揃ったアタックで大)
    flatness: np.ndarray       # 平均スペクトルフラットネス
    tonality: np.ndarray       # 平均純音性
    tuning_score: np.ndarray   # 単一持続音らしさ 0-1
    silence_db: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.times)


def _silence_threshold_db(rms_db: np.ndarray) -> float:
    """全体分布から適応的に無音閾値を決める(固定閾値に頼らない)。"""
    floor = float(np.percentile(rms_db, 5))
    active = float(np.percentile(rms_db, 90))
    return float(max(floor + 6.0, active - 26.0))


def _autocorr_peak(env: np.ndarray, fps: float, lo_s: float = 0.24, hi_s: float = 1.6) -> float:
    """オンセット包絡の自己相関ピーク = 拍の明瞭さ。

    全奏者が揃って演奏していると包絡に周期構造が出て 1 に近づく。各自バラバラの
    個人練習では周期構造が消えて 0 に近づく。
    """
    x = env - env.mean()
    n = len(x)
    if n < 16:
        return 0.0
    denom = float(np.dot(x, x))
    if denom < EPS:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n] / denom
    lo = max(1, int(lo_s * fps))
    hi = min(n - 1, int(hi_s * fps))
    if hi <= lo:
        return 0.0
    return float(np.clip(np.max(ac[lo:hi]), 0.0, 1.0))


def _tuning_frames(ff: FrameFeatures, tonal_th: float = 0.32) -> np.ndarray:
    """チューニングらしいフレームの真偽値列。

    「単一の持続音がスペクトル的に単純なまま一定時間続く」ことを、
    純音性が高く・ピッチが安定し・スペクトル変化が小さいフレームとして拾う。
    """
    fps = ff.fps
    win = max(3, int(round(0.5 * fps)))  # 0.5秒でピッチの揺れを見る

    # 半音単位に直すと、周波数によらず同じ尺度で安定性を測れる。
    semitone = 12.0 * np.log2(np.maximum(ff.peak_hz, 1.0) / 440.0)
    pad = np.pad(semitone, (win // 2, win - win // 2 - 1), mode="edge")
    view = _frame_view(np.ascontiguousarray(pad), win, 1)[: len(semitone)]
    spread = np.max(view, axis=1) - np.min(view, axis=1)

    flux_th = float(np.percentile(ff.flux, 40))
    stable = (spread < 0.6) & (ff.tonal > tonal_th) & (ff.flux < max(flux_th, EPS))

    # チューニングは無音ではない。極端に静かなフレームは除く。
    active = ff.rms_db > (_silence_threshold_db(ff.rms_db) + 4.0)
    return stable & active


def tuning_frames(ff: FrameFeatures, tonal_th: float = 0.32) -> np.ndarray:
    """`_tuning_frames` の公開版(区間検出側から利用する)。"""
    return _tuning_frames(ff, tonal_th)


def aggregate_windows(
    ff: FrameFeatures,
    win_s: float = 20.0,
    hop_s: float = 5.0,
) -> WindowFeatures:
    """フレーム特徴量を窓単位の統計量に集約する。"""
    fps = ff.fps
    wf = max(2, int(round(win_s * fps)))
    hf = max(1, int(round(hop_s * fps)))
    n_total = len(ff.times)
    if n_total < wf:
        raise PipelineError("音声が解析窓より短すぎます")
    n_win = 1 + (n_total - wf) // hf

    sil_db = _silence_threshold_db(ff.rms_db)
    is_silent = ff.rms_db < sil_db
    is_tuning = _tuning_frames(ff)

    out = {
        k: np.zeros(n_win, dtype=np.float32)
        for k in (
            "loudness", "silence_ratio", "dyn_range", "loud_std",
            "pulse_clarity", "flux_crest", "flatness", "tonality", "tuning_score",
        )
    }
    starts = np.zeros(n_win)
    ends = np.zeros(n_win)

    for i in range(n_win):
        a = i * hf
        b = a + wf
        db = ff.rms_db[a:b]
        env = ff.flux[a:b]

        starts[i] = ff.times[a] - ff.frame / (2 * ff.sr)
        ends[i] = ff.times[b - 1] + ff.frame / (2 * ff.sr)

        out["loudness"][i] = db.mean()
        out["silence_ratio"][i] = is_silent[a:b].mean()
        p10, p90 = np.percentile(db, [10, 90])
        out["dyn_range"][i] = p90 - p10
        out["loud_std"][i] = db.std()
        out["pulse_clarity"][i] = _autocorr_peak(env, fps)

        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med))) + EPS
        out["flux_crest"][i] = float(np.clip((np.percentile(env, 98) - med) / (mad * 10.0), 0.0, 3.0))

        out["flatness"][i] = ff.flatness[a:b].mean()
        out["tonality"][i] = ff.tonal[a:b].mean()
        out["tuning_score"][i] = is_tuning[a:b].mean()

    return WindowFeatures(
        win_s=win_s,
        hop_s=hop_s,
        times=(starts + ends) / 2,
        starts=starts,
        ends=ends,
        silence_db=sil_db,
        meta={"fps": fps, "n_frames": n_total},
        **out,
    )


# ---------------------------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------------------------

def save_frame_features(path: Path, ff: FrameFeatures) -> None:
    np.savez_compressed(
        path,
        sr=ff.sr, frame=ff.frame, hop=ff.hop,
        times=ff.times, rms_db=ff.rms_db, flux=ff.flux,
        flatness=ff.flatness, centroid=ff.centroid, tonal=ff.tonal, peak_hz=ff.peak_hz,
    )


def load_frame_features(path: Path) -> FrameFeatures:
    z = np.load(path)
    return FrameFeatures(
        sr=int(z["sr"]), frame=int(z["frame"]), hop=int(z["hop"]),
        times=z["times"], rms_db=z["rms_db"], flux=z["flux"],
        flatness=z["flatness"], centroid=z["centroid"], tonal=z["tonal"], peak_hz=z["peak_hz"],
    )
