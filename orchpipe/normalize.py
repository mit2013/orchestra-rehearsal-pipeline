"""トラック単位のラウドネス正規化。

`apply` が出力した `trimmed/` 配下の各ブロックについて、ext/int それぞれ独立に
**統合ラウドネス**を目標値(既定 -20 LUFS)に合わせる。ピーク正規化ではない理由は
`loudness.py` の冒頭を参照。

基準値の算出範囲と適用範囲を分けているのは従来どおり:

- **算出**: guard(既定120秒)で外側に広げた部分には音出しや休憩の話し声が混入して
  いる。EBU R128 のゲーティングでも人の声は落ちないので、前後の guard 相当を除いた
  中央部分だけから測る。
- **適用**: ゲイン自体はブロック全体(guard 部分を含む)に一律で掛ける。

**この段階ではダイナミクスを変えない。** コンプレッサとリミッターは、実際に配布する
信号ができあがる `mix` の段階で1回だけ通す(`loudness.py` の「処理の順序」を参照)。
したがってここの出力 `_norm.wav` は 0 dBFS を超えうる。32bit float なので保持される。

`session_config.json` の `normalize_scope` は**使わない**。ブロックごとに目標
ラウドネスへ合わせるので、日付全体で基準をそろえる必要がなくなった。キー自体は
既存の設定ファイルとの互換のために残してあるが、値は無視される。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .loudness import DEFAULT_TARGET_LUFS, Loudness, measure
from .util import FFMPEG, PipelineError, log, probe_audio, run, write_json

NORM_SUFFIX = "_norm"

_PEAK_RE = re.compile(r"Peak level dB:\s*(-?inf|-?\d+(?:\.\d+)?)", re.IGNORECASE)


def measure_peak_db(path: Path, start: float | None = None, dur: float | None = None) -> float:
    """指定範囲のピークを dBFS で返す。32bit float なので 0 dBFS 超も正しく返る。

    正規化の基準には使わなくなったが、素材が 0 dBFS を超えているかの確認と
    `mix.py` の安全処理で使うため残してある。
    """
    cmd = [FFMPEG, "-hide_banner", "-v", "info", "-nostats"]
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


def norm_block_files(trimmed: Path, groups: list[str]) -> dict[tuple[str, str], Path]:
    """正規化済みファイル `{NN}_{label}_{group}_norm.wav` から直接ブロックを拾う。

    `block_files` は正規化**前**の `_ext.wav` を見るが、あれは `_norm.wav` を
    作ったあとは要らない中間ファイルで、容量を空けるために消されることがある。
    `mix` の実際の入力は `_norm.wav` のほうなので、前段が消えていても動くように
    こちらから引けるようにしておく。
    """
    out: dict[tuple[str, str], Path] = {}
    if not trimmed.is_dir():
        return out
    pattern = re.compile(r"^(?P<block>.+)_(?P<group>" + "|".join(map(re.escape, groups))
                         + r")" + re.escape(NORM_SUFFIX) + r"\.wav$")
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
    """基準ラウドネスを測る範囲。中央部分が短すぎるときは全体から測る。"""
    if dur > 2 * ref_margin + 30.0:
        span = dur - 2 * ref_margin
        return ref_margin, span, f"中央 {span/60:.1f}分 (前後 {ref_margin:.0f}秒を除外)"
    return None, None, f"全体 {dur/60:.1f}分 (短いためマージン除外なし)"


def run_normalize(
    outdir: Path,
    groups: list[str],
    target_lufs: float = DEFAULT_TARGET_LUFS,
    ref_margin: float = 120.0,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    trimmed = outdir / "trimmed"
    files = block_files(trimmed, groups)
    if not files:
        raise PipelineError(
            f"{trimmed} に正規化対象がありません。先に `apply` を実行してください。"
        )

    log(
        f"正規化開始: {len(files)} ファイル / 目標 {target_lufs:+.1f} LUFS / "
        f"基準除外マージン {ref_margin:.0f}秒"
    )

    # --- 1. 基準ラウドネスを測る -------------------------------------------
    stats: dict[tuple[str, str], Loudness] = {}
    windows: dict[tuple[str, str], str] = {}
    for key, src in sorted(files.items()):
        dur = probe_audio(src)["duration"]
        start, span, desc = _reference_window(dur, ref_margin)
        stats[key] = measure(src, start, span)
        windows[key] = desc
        log(f"  測定 {src.name}: {stats[key].describe()}  基準={desc}")

    # --- 2. 適用 ------------------------------------------------------------
    out: dict[tuple[str, str], Path] = {}
    record: dict[str, dict] = {}
    for (block, group), src in sorted(files.items()):
        key = (block, group)
        dst = norm_path(src)
        out[key] = dst
        st = stats[key]
        if st.integrated == float("-inf"):
            log(f"  スキップ(無音): {src.name}")
            continue

        gain = target_lufs - st.integrated
        record[f"{block}_{group}"] = {
            "source": src.name, "window": windows[key],
            "before": st.to_json(), "gain_db": round(gain, 2),
        }

        if dst.exists() and not force:
            log(f"  スキップ(既存): {dst.name}")
            continue

        run(
            [
                FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                "-i", str(src),
                "-af", f"volume={gain:.6f}dB",
                "-c:a", "pcm_f32le", "-rf64", "auto",
                str(dst),
            ],
            desc=(f"  {src.name} -> {dst.name}  "
                  f"{st.integrated:+.1f} LUFS -> {target_lufs:+.1f} LUFS "
                  f"(ゲイン {gain:+.2f} dB)"),
        )

        # ここではまだリミッターを通していないので 0 dBFS 超がありうる。
        # 32bit float なので値は壊れないが、mix でコンプ・リミッターを通すまでは
        # そのまま書き出してはいけない。
        full_peak = measure_peak_db(dst)
        record[f"{block}_{group}"]["peak_after_gain_db"] = round(full_peak, 2)
        if full_peak > 0.0:
            log(f"    {dst.name} のピークは {full_peak:+.2f} dBFS です"
                "(mix のリミッターで -1 dBTP に収めます)")

    write_json(outdir / "loudness.json",
               {"target_lufs": target_lufs, "ref_margin_s": ref_margin, "blocks": record})
    log(f"測定結果を保存しました: {(outdir / 'loudness.json').name}")
    return out
