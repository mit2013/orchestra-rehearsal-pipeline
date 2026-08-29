"""ステージF-2: 状態分類の視覚的な一覧。

対象ブロックを10分ごとに分割し、各区間について
「波形 + スペクトログラム + 予測ラベルの色帯」を1枚の画像にする。
人間がざっと見て、明らかな誤分類の当たりを付けられるようにするのが目的。

スペクトログラムは ffmpeg ではなく numpy の rfft で作る。既存の
`features.extract_frame_features` と同じ 16kHz モノラルのストリームを使うので、
分類に使った信号とまったく同じものを見ていることになる。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..util import FFMPEG, PipelineError, fmt_time, log
from .states import StateSpan
from .states2 import LABELS

# 4クラスの色。ラベル帯・凡例・スペクトログラム上の縦線で共通に使う。
LABEL_COLORS = {
    "silence": "#9e9e9e",
    "tuning": "#8e44ad",
    "speech": "#e74c3c",
    "playing": "#2ecc71",
    # G-14 で追加。ブロック末尾に張り付いた、各自の音出し。
    "warmup": "#9a7fd0",
    # G-2 で追加。どのクラスの証拠も立たなかった区間。
    "unclear": "#f0a020",
}
LABEL_JA = {
    "silence": "無音",
    "tuning": "チューニング",
    "speech": "発言",
    "playing": "演奏",
    "warmup": "音出し",
    "unclear": "証拠不足",
}

CHUNK_S = 600.0        # 1枚あたり10分
SPEC_SR = 16000
SPEC_FRAME = 1024
SPEC_HOP = 512


def _decode(path: Path, start: float, dur: float) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SPEC_SR), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(f"デコードに失敗: {path}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def _spectrogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dB スケールのスペクトログラムと周波数軸を返す。"""
    if len(x) < SPEC_FRAME:
        return np.zeros((SPEC_FRAME // 2 + 1, 1)), np.fft.rfftfreq(SPEC_FRAME, 1 / SPEC_SR)
    n = 1 + (len(x) - SPEC_FRAME) // SPEC_HOP
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, SPEC_FRAME),
        strides=(x.strides[0] * SPEC_HOP, x.strides[0]), writeable=False)
    win = np.hanning(SPEC_FRAME).astype(np.float32)
    mag = np.abs(np.fft.rfft(view * win, axis=1)).T
    db = 20.0 * np.log10(mag + 1e-10)
    return db, np.fft.rfftfreq(SPEC_FRAME, 1 / SPEC_SR)


def _setup_font() -> bool:
    import matplotlib
    from matplotlib import font_manager
    for p in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False


def render_block(
    src: Path,
    spans: list[StateSpan],
    total: float,
    outdir: Path,
    block: str,
    chunk_s: float = CHUNK_S,
) -> list[Path]:
    """10分ごとに1枚ずつ画像を書き出す。"""
    import matplotlib
    matplotlib.use("Agg")
    has_jp = _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_chunks = int(np.ceil(total / chunk_s))

    for i in range(n_chunks):
        t0 = i * chunk_s
        t1 = min(total, t0 + chunk_s)
        x = _decode(src, t0, t1 - t0)
        if len(x) == 0:
            continue
        db, freqs = _spectrogram(x)
        tx = np.arange(len(x)) / SPEC_SR + t0

        fig, (ax_w, ax_s, ax_l) = plt.subplots(
            3, 1, figsize=(24, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 3, 0.7], "hspace": 0.08})

        # --- 波形(min/max 包絡で間引く)---
        step = max(1, len(x) // 4000)
        usable = (len(x) // step) * step
        v = x[:usable].reshape(-1, step)
        tt = tx[:usable].reshape(-1, step)[:, 0]
        ax_w.fill_between(tt, v.min(axis=1), v.max(axis=1), color="#2f5d8a", linewidth=0)
        ax_w.set_ylabel("波形" if has_jp else "wave")
        ax_w.set_xlim(t0, t1)
        ax_w.grid(alpha=0.2)

        # --- スペクトログラム(4kHz まで。楽音と声の帯域を見るのに十分)---
        fmax_idx = int(np.searchsorted(freqs, 4000.0))
        st = np.linspace(t0, t1, db.shape[1])
        ax_s.imshow(db[:fmax_idx], aspect="auto", origin="lower",
                    extent=[t0, t1, 0, freqs[fmax_idx - 1]],
                    cmap="magma", vmin=np.percentile(db, 40), vmax=np.percentile(db, 99.5))
        ax_s.set_ylabel("周波数 [Hz]" if has_jp else "Hz")
        # 基準音 A(442Hz)とその倍音に補助線。チューニングの目視確認用。
        for k in (1, 2, 3):
            f = 442.0 * k
            if f < freqs[fmax_idx - 1]:
                ax_s.axhline(f, color="#00e5ff", lw=0.6, ls=":", alpha=0.55)

        # --- 予測ラベルの色帯 ---
        for sp in spans:
            if sp.end <= t0 or sp.start >= t1:
                continue
            a, b = max(sp.start, t0), min(sp.end, t1)
            c = LABEL_COLORS[sp.label]
            ax_l.axvspan(a, b, color=c, alpha=0.95, linewidth=0)
            # 波形・スペクトログラム側にも薄く重ねる
            ax_w.axvspan(a, b, color=c, alpha=0.16, linewidth=0)
            if sp.label in ("speech", "tuning"):
                ax_s.axvspan(a, b, color=c, alpha=0.13, linewidth=0)
            # 十分な幅があればラベル名を書く
            if b - a >= chunk_s * 0.012:
                ax_l.text((a + b) / 2, 0.5,
                          LABEL_JA[sp.label] if has_jp else sp.label,
                          ha="center", va="center", fontsize=7.5, color="white")
        ax_l.set_ylim(0, 1)
        ax_l.set_yticks([])
        ax_l.set_ylabel("予測" if has_jp else "pred")
        ax_l.set_xlabel("ブロック内の時刻 [分:秒]" if has_jp else "time")

        ticks = np.arange(t0, t1 + 1, 30.0)
        ax_l.set_xticks(ticks)
        ax_l.set_xticklabels([fmt_time(t)[3:] for t in ticks], fontsize=8)

        ax_w.legend(handles=[Patch(facecolor=LABEL_COLORS[l],
                                   label=(LABEL_JA[l] if has_jp else l)) for l in LABELS],
                    loc="upper right", ncol=4, fontsize=9)
        ax_w.set_title(
            f"{block}  {fmt_time(t0)} - {fmt_time(t1)}  "
            f"({i+1}/{n_chunks})   青緑の点線 = A(442Hz)とその倍音"
            if has_jp else f"{block} {fmt_time(t0)}-{fmt_time(t1)}")

        dst = outdir / f"{block}_{i+1:02d}_{fmt_time(t0).replace(':','-')}.png"
        fig.savefig(dst, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(dst)
        log(f"    {dst.name}")
    return written
