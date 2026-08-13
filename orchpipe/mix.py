"""外部マイク・内蔵マイクのミックス。

`source` の値で挙動が変わる:

- `ext_only` / `int_only`: 対応する正規化済みファイルをそのまま採用する。
  合成しないのでピーク超過は起こりえない。
- `mix`: ext_norm と int_norm を指定比率で加算合成し、合成後のピークが安全閾値を
  超えていたら**信号全体を一律の線形ゲインで下げる**。コンプレッサーやリミッタの
  ようなダイナミクスを変える処理は使わない。

安全処理について(実装上の事実):

比率の合計が 1.0 以下(6:4、8:2 など)の凸結合であれば、三角不等式より
|w1·a + w2·b| ≤ w1·|a| + w2·|b| ≤ max(peak_a, peak_b) が常に成り立つため、
合成ピークが入力のどちらのピークをも上回ることは数学的に起こりえない。
実際に超過しうるのは比率の合計が 1.0 を超える設定(例 ext=1.0, int=0.8)である。
比率は固定しない仕様なのでその設定は取りうる。したがってこの安全処理は必要だが、
発動条件は「位相の重なり」ではなく「比率の合計が 1 を超えること」である。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import SessionConfig
from .normalize import measure_peak_db, norm_path
from .util import FFMPEG, PipelineError, log, run

SAFE_PEAK_DB = -1.0
FINAL_SUFFIX = "_final"


def final_path(trimmed: Path, block: str) -> Path:
    return trimmed / f"{block}{FINAL_SUFFIX}.wav"


def _mix_filter(w_ext: float, w_int: float, gain_db: float | None = None) -> str:
    chain = (
        f"[0:a]volume={w_ext:.6f}[a];"
        f"[1:a]volume={w_int:.6f}[b];"
        f"[a][b]amix=inputs=2:duration=longest:normalize=0[m]"
    )
    if gain_db is not None:
        chain += f";[m]volume={gain_db:.6f}dB[out]"
        return chain
    return chain


def run_mix(
    outdir: Path,
    cfg: SessionConfig,
    groups: list[str] | None = None,
    blocks: dict[tuple[str, str], Path] | None = None,
    safe_peak_db: float = SAFE_PEAK_DB,
    force: bool = False,
) -> list[Path]:
    trimmed = outdir / "trimmed"
    if blocks is None:
        from .normalize import block_files

        raw = block_files(trimmed, list(groups or ("ext", "int")))
        blocks = {k: norm_path(v) for k, v in raw.items()}

    names = sorted({b for (b, _g) in blocks})
    if not names:
        raise PipelineError(
            f"{trimmed} にミックス対象がありません。先に `normalize` を実行してください。"
        )

    log(f"ミックス開始: {len(names)} ブロック / source={cfg.source}")
    written: list[Path] = []

    for block in names:
        dst = final_path(trimmed, block)
        if dst.exists() and not force:
            log(f"  スキップ(既存): {dst.name}")
            written.append(dst)
            continue

        if cfg.source in ("ext_only", "int_only"):
            group = "ext" if cfg.source == "ext_only" else "int"
            src = blocks.get((block, group))
            if src is None or not src.exists():
                raise PipelineError(
                    f"{block}: source={cfg.source} に必要な {group} の正規化済みファイルがありません"
                    f"({src if src else '未検出'})"
                )
            shutil.copyfile(src, dst)
            log(f"  {dst.name}  <- {src.name} ({cfg.source}、合成なし)")
            written.append(dst)
            continue

        # --- source == "mix" -------------------------------------------------
        ext_src = blocks.get((block, "ext"))
        int_src = blocks.get((block, "int"))
        for tag, p in (("ext", ext_src), ("int", int_src)):
            if p is None or not p.exists():
                raise PipelineError(
                    f"{block}: source=mix に必要な {tag} の正規化済みファイルがありません"
                )

        w_ext = float(cfg.mix_ratio["ext"])
        w_int = float(cfg.mix_ratio["int"])

        # 1パス目: 書き出さずに合成後のピークだけを測る。
        peak = _measure_mix_peak(ext_src, int_src, w_ext, w_int)
        if peak > safe_peak_db:
            gain = safe_peak_db - peak
            log(
                f"  {block}: 合成ピーク {peak:+.2f} dBFS が安全閾値 {safe_peak_db:+.1f} dBFS を超過 "
                f"-> 全体を {gain:+.2f} dB スケールダウン"
            )
        else:
            gain = None
            log(f"  {block}: 合成ピーク {peak:+.2f} dBFS(閾値 {safe_peak_db:+.1f} dBFS 以内、スケール調整なし)")

        # 2パス目: 必要な補正ゲインを織り込んで書き出す。
        out_label = "out" if gain is not None else "m"
        run(
            [
                FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                "-i", str(ext_src), "-i", str(int_src),
                "-filter_complex", _mix_filter(w_ext, w_int, gain),
                "-map", f"[{out_label}]",
                "-c:a", "pcm_f32le", "-rf64", "auto",
                str(dst),
            ],
            desc=f"  {dst.name}  ext×{w_ext:g} + int×{w_int:g}",
        )
        final_peak = measure_peak_db(dst)
        log(f"    書き出し後のピーク: {final_peak:+.2f} dBFS")
        if final_peak > safe_peak_db + 0.05:
            raise PipelineError(
                f"{dst.name}: スケールダウン後もピークが {final_peak:+.2f} dBFS で "
                f"閾値 {safe_peak_db:+.1f} dBFS を超えています"
            )
        written.append(dst)

    return written


def _measure_mix_peak(ext_src: Path, int_src: Path, w_ext: float, w_int: float) -> float:
    """合成結果を書き出さずにピークだけ測る(-f null で捨てる)。"""
    import re
    import subprocess

    from .normalize import _PEAK_RE

    cmd = [
        FFMPEG, "-hide_banner", "-v", "info",
        "-i", str(ext_src), "-i", str(int_src),
        "-filter_complex", _mix_filter(w_ext, w_int) + ";[m]astats=measure_perchannel=none[s]",
        "-map", "[s]", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(
            "合成ピークの測定に失敗しました\n"
            + proc.stderr.decode("utf-8", "replace").strip()[-800:]
        )
    m = _PEAK_RE.search(proc.stderr.decode("utf-8", "replace"))
    if not m:
        raise PipelineError("合成ピーク値を読み取れませんでした")
    val = m.group(1).lower()
    return float("-inf") if val.endswith("inf") else float(val)
