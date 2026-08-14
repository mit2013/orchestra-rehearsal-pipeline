"""トラック単位のピーク正規化。

`apply` が出力した `trimmed/` 配下の各ブロックについて、ext/int それぞれ独立に
ピークを -1 dBFS に合わせる。

基準値の算出範囲と適用範囲を分けているのが要点:

- **算出**: guard(既定120秒)で外側に広げた部分には音出しや休憩が混入している
  可能性があり、そこが最大ピークだと本編が不当に小さくなる。したがって前後の
  guard 相当を除いた中央部分だけからピークを測る。
- **適用**: ゲイン自体はブロック全体(guard 部分を含む)に一律で掛ける。

ダイナミクスを変える処理(コンプ・リミッタ)は使わず、一律ゲインのみ。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .util import FFMPEG, PipelineError, log, probe_audio, run

TARGET_DB = -1.0
NORM_SUFFIX = "_norm"

_PEAK_RE = re.compile(r"Peak level dB:\s*(-?inf|-?\d+(?:\.\d+)?)", re.IGNORECASE)


def measure_peak_db(path: Path, start: float | None = None, dur: float | None = None) -> float:
    """指定範囲のピークを dBFS で返す。32bit float なので 0 dBFS 超も正しく返る。"""
    cmd = [FFMPEG, "-hide_banner", "-v", "info"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-af", "astats=measure_perchannel=none", "-f", "null", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(
            f"ピーク測定に失敗: {path}\n{proc.stderr.decode('utf-8', 'replace').strip()[-800:]}"
        )
    text = proc.stderr.decode("utf-8", "replace")
    m = _PEAK_RE.search(text)
    if not m:
        raise PipelineError(f"ピーク値を読み取れませんでした: {path}")
    val = m.group(1).lower()
    if val.endswith("inf"):
        return float("-inf")
    return float(val)


def block_files(trimmed: Path, groups: list[str]) -> dict[tuple[str, str], Path]:
    """`{NN}_{label}_{group}.wav` を {(ブロック名, 系統): パス} で返す。

    `_norm` / `_final` の付いたものは対象外。
    """
    out: dict[tuple[str, str], Path] = {}
    if not trimmed.is_dir():
        return out
    pattern = re.compile(r"^(?P<block>.+)_(?P<group>" + "|".join(map(re.escape, groups)) + r")\.wav$")
    for p in sorted(trimmed.iterdir()):
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            out[(m.group("block"), m.group("group"))] = p
    return out


def norm_path(src: Path) -> Path:
    return src.with_name(src.stem + NORM_SUFFIX + src.suffix)


def _reference_window(dur: float, ref_margin: float) -> tuple[float | None, float | None, str]:
    """基準ピークを測る範囲。中央部分が短すぎるときは全体から測る。"""
    if dur > 2 * ref_margin + 30.0:
        span = dur - 2 * ref_margin
        return ref_margin, span, f"中央 {span/60:.1f}分 (前後 {ref_margin:.0f}秒を除外)"
    return None, None, f"全体 {dur/60:.1f}分 (短いためマージン除外なし)"


def run_normalize(
    outdir: Path,
    groups: list[str],
    target_db: float = TARGET_DB,
    ref_margin: float = 120.0,
    scope: str = "date",
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    trimmed = outdir / "trimmed"
    files = block_files(trimmed, groups)
    if not files:
        raise PipelineError(
            f"{trimmed} に正規化対象がありません。先に `apply` を実行してください。"
        )

    log(
        f"正規化開始: {len(files)} ファイル / 目標 {target_db:+.1f} dBFS / "
        f"基準除外マージン {ref_margin:.0f}秒 / 基準範囲 {scope}"
    )

    # --- 1. まず全ブロックの基準ピークを測る -------------------------------
    # scope="date" では系統内の最大ピークが必要なので、書き出す・書き出さないに
    # かかわらず全ブロックを測っておかないと基準が決まらない。
    peaks: dict[tuple[str, str], float] = {}
    windows: dict[tuple[str, str], str] = {}
    for key, src in sorted(files.items()):
        dur = probe_audio(src)["duration"]
        start, span, desc = _reference_window(dur, ref_margin)
        peaks[key] = measure_peak_db(src, start, span)
        windows[key] = desc

    # --- 2. ゲインを決める --------------------------------------------------
    gains: dict[tuple[str, str], float] = {}
    if scope == "date":
        # 系統ごとに、全ブロックの本編ピークの最大値を基準にする。同じゲインを
        # その系統の全ブロックに掛けるので、ブロック間の音量差が元のまま残る。
        for group in groups:
            voiced = [(b, p) for (b, g), p in peaks.items() if g == group and p != float("-inf")]
            if not voiced:
                log(f"  [{group}] 有音のブロックがありません。この系統はスキップします。")
                continue
            ref_block, ref_peak = max(voiced, key=lambda x: x[1])
            gain = target_db - ref_peak
            for b, g in peaks:
                if g == group:
                    gains[(b, g)] = gain
            others = ", ".join(
                f"{b}={p:+.2f}" for b, p in sorted(voiced) if b != ref_block
            )
            log(
                f"  [{group}] 基準ピーク {ref_peak:+.2f} dBFS ← {ref_block} "
                f"({len(voiced)} ブロック中の最大{'; 他: ' + others if others else ''}) "
                f"-> 系統共通ゲイン {gain:+.2f} dB"
            )
    else:
        for key, p in peaks.items():
            if p != float("-inf"):
                gains[key] = target_db - p

    # --- 3. 適用 ------------------------------------------------------------
    out: dict[tuple[str, str], Path] = {}
    for (block, group), src in sorted(files.items()):
        key = (block, group)
        dst = norm_path(src)
        out[key] = dst
        if key not in gains:
            log(f"  スキップ(無音): {src.name}")
            continue
        if dst.exists() and not force:
            log(f"  スキップ(既存): {dst.name}")
            continue

        gain = gains[key]
        peak = peaks[key]
        run(
            [
                FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                "-i", str(src),
                "-af", f"volume={gain:.6f}dB",
                "-c:a", "pcm_f32le", "-rf64", "auto",
                str(dst),
            ],
            desc=(
                f"  {src.name} -> {dst.name}  基準={windows[key]}  "
                f"自ブロックのピーク {peak:+.2f} dBFS, 適用ゲイン {gain:+.2f} dB"
            ),
        )

        # 適用範囲は全体なので、除外した guard 部分が 0 dBFS を超えることがありうる。
        # 32bit float なのでファイル上は壊れないが、後段の書き出しで問題になるため知らせる。
        full_peak = measure_peak_db(dst)
        if full_peak > 0.0:
            log(
                f"    注意: {dst.name} の全体ピークは {full_peak:+.2f} dBFS です"
                "(基準から除外した音出し/休憩部分が本編より大きいため)。"
                "32bit float なので値は保持されますが、最終書き出し時は要確認。"
            )

    return out
