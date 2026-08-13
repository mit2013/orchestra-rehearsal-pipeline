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


def run_normalize(
    outdir: Path,
    groups: list[str],
    target_db: float = TARGET_DB,
    ref_margin: float = 120.0,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    trimmed = outdir / "trimmed"
    files = block_files(trimmed, groups)
    if not files:
        raise PipelineError(
            f"{trimmed} に正規化対象がありません。先に `apply` を実行してください。"
        )

    log(f"正規化開始: {len(files)} ファイル / 目標 {target_db:+.1f} dBFS / 基準除外マージン {ref_margin:.0f}秒")
    out: dict[tuple[str, str], Path] = {}

    for (block, group), src in sorted(files.items()):
        dst = norm_path(src)
        out[(block, group)] = dst
        if dst.exists() and not force:
            log(f"  スキップ(既存): {dst.name}")
            continue

        dur = probe_audio(src)["duration"]
        # 中央部分が短すぎると基準が不安定なので、その場合は全体から測る。
        if dur > 2 * ref_margin + 30.0:
            start, span = ref_margin, dur - 2 * ref_margin
            scope = f"中央 {span/60:.1f}分 (前後 {ref_margin:.0f}秒を除外)"
        else:
            start, span = None, None
            scope = f"全体 {dur/60:.1f}分 (短いためマージン除外なし)"

        peak = measure_peak_db(src, start, span)
        if peak == float("-inf"):
            log(f"  スキップ(無音): {src.name}")
            continue

        gain = target_db - peak
        run(
            [
                FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                "-i", str(src),
                "-af", f"volume={gain:.6f}dB",
                "-c:a", "pcm_f32le", "-rf64", "auto",
                str(dst),
            ],
            desc=f"  {src.name} -> {dst.name}  基準={scope}  ピーク {peak:+.2f} dBFS, ゲイン {gain:+.2f} dB",
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
