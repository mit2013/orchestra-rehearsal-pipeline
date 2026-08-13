"""確認用の成果物: 境界前後のプレビュー音声と、境界マーク入りの波形画像。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FrameFeatures, WindowFeatures
from .segment import Segment
from .util import FFMPEG, fmt_time, log, run, safe_filename

JP_FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def _setup_font() -> bool:
    """日本語ラベルが豆腐にならないよう、あればヒラギノを使う。"""
    import matplotlib
    from matplotlib import font_manager

    for path in JP_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False


# ---------------------------------------------------------------------------
# プレビュー音声
# ---------------------------------------------------------------------------

def write_preview_clips(
    segs: list[Segment],
    sources: dict[str, Path],
    dst_dir: Path,
    pad_s: float = 15.0,
    total: float | None = None,
) -> list[Path]:
    """各境界の前後 ±pad_s を切り出す。

    確認用途なので 16bit WAV(小さく、どのプレイヤーでも鳴る)にする。
    書き出す本編は常に元の 32bit float から作るので、ここでの変換は成果物に影響しない。
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [s.start for s in segs[1:]]  # 先頭 0:00 は境界ではない
    written: list[Path] = []

    for i, b in enumerate(boundaries, start=1):
        start = max(0.0, b - pad_s)
        dur = pad_s * 2
        if total is not None:
            dur = min(dur, max(0.5, total - start))
        stamp = fmt_time(b).replace(":", "-")
        for tag, src in sources.items():
            out = dst_dir / f"b{i:02d}_{stamp}_{tag}.wav"
            if out.exists():
                continue
            run(
                [
                    FFMPEG, "-hide_banner", "-v", "error", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                    "-c:a", "pcm_s16le", str(out),
                ]
            )
            written.append(out)
        log(f"プレビュー b{i:02d}: {fmt_time(b)} 付近 (±{pad_s:.0f}s)")
    return written


# ---------------------------------------------------------------------------
# 波形画像
# ---------------------------------------------------------------------------

def _envelope(times: np.ndarray, values: np.ndarray, n_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画素数まで間引く。各画素の最小/最大を残して波形の密度感を保つ。"""
    n = len(values)
    step = max(1, n // n_px)
    usable = (n // step) * step
    v = values[:usable].reshape(-1, step)
    t = times[:usable].reshape(-1, step)[:, 0]
    return t, v.min(axis=1), v.max(axis=1)


def write_waveform_png(
    ff: FrameFeatures,
    wf: WindowFeatures,
    score: np.ndarray,
    segs: list[Segment],
    dst: Path,
    threshold: float,
    title: str = "",
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    has_jp = _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t, lo, hi = _envelope(ff.times, ff.rms_db, 4000)
    total = float(ff.times[-1])

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(22, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12},
    )

    ax0.fill_between(t / 60, lo, hi, color="#2f5d8a", linewidth=0)
    ax0.set_ylim(max(-90, float(np.percentile(ff.rms_db, 0.5)) - 3), 3)
    ax0.set_ylabel("RMS [dBFS]")
    ax0.axhline(wf.silence_db, color="#888", ls=":", lw=1)

    for s in segs:
        color = "#2ecc71" if s.action == "keep" else "#e74c3c"
        ax0.axvspan(s.start / 60, s.end / 60, color=color, alpha=0.13, linewidth=0)
        ax1.axvspan(s.start / 60, s.end / 60, color=color, alpha=0.13, linewidth=0)
        mid = (s.start + s.end) / 2 / 60
        label = s.label if has_jp else f"#{s.index} {s.action}"
        ax0.text(
            mid, 1.5, f"{s.index}. {label}\n{fmt_time(s.start)}–{fmt_time(s.end)}",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        )

    for s in segs[1:]:
        for ax in (ax0, ax1):
            ax.axvline(s.start / 60, color="#111", lw=1.4, ls="--")

    ax1.plot(wf.times / 60, score, color="#8e44ad", lw=1.0)
    ax1.axhline(threshold, color="#111", ls="-", lw=1.0)
    ax1.set_ylabel("合奏らしさ" if has_jp else "ensemble score")
    ax1.set_xlabel("時間 [分]" if has_jp else "time [min]")
    ax1.set_xlim(0, total / 60)
    ax1.grid(alpha=0.25)
    ax0.grid(alpha=0.25)

    ticks = np.arange(0, total / 60 + 1, 10)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([fmt_time(x * 60)[:5] for x in ticks], fontsize=8)

    ax0.legend(
        handles=[
            Patch(facecolor="#2ecc71", alpha=0.3, label="keep (残す)" if has_jp else "keep"),
            Patch(facecolor="#e74c3c", alpha=0.3, label="remove (削る)" if has_jp else "remove"),
        ],
        loc="lower right", fontsize=9,
    )
    ax0.set_title(title or dst.stem)

    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"波形画像を書き出しました: {dst}")
    return dst
