"""外部マイク・内蔵マイクのミックスと、配布用のマスタリング。

`source` の値で合成の仕方が変わる:

- `ext_only` / `int_only`: 対応する正規化済みファイルをそのまま素材にする
- `mix`: ext_norm と int_norm を指定比率で加算合成する

合成のあと、**すべての source で共通に**ダイナミクス処理を1回だけ通す。

    ゲイン -> コンプレッサ(2:1、緩く) -> トゥルーピークリミッター(-1 dBTP)

ここが配布する音そのものになる段階なので、コンプとリミッターはここに置く。
`normalize` の段階でブロックごとに掛けてしまうと、`source=mix` のときに ext と int を
別々に潰してから足すことになり、二重に効く。処理順序と、コンプによるラウドネスの
低下をどう補正するかは `loudness.py` の冒頭を参照。

旧実装にあった「合成ピークが安全閾値を超えたら全体をスケールダウンする」処理は
廃止した。リミッターが天井を保証するので不要であり、一律スケールダウンは
ブロック間の音量差を戻してしまう。
"""

from __future__ import annotations

from pathlib import Path

from . import reverb as reverb_mod
from .config import SessionConfig
from .loudness import (
    LIMITER_OVERSAMPLE,
    DEFAULT_COMP_ATTACK_MS,
    DEFAULT_COMP_KNEE_DB,
    DEFAULT_COMP_RATIO,
    DEFAULT_COMP_RELEASE_MS,
    DEFAULT_COMP_THRESHOLD_OFFSET,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TRUE_PEAK_DB,
    GAIN_MAX_ITER,
    PARALLEL_MAKEUP_DB,
    PARALLEL_THRESHOLD_DB,
    GAIN_SETTLE_LU,
    master_chain,
    measure,
    measure_complex,
    solve_gain,
)
from .normalize import _reference_window, norm_path
from .util import FFMPEG, PipelineError, log, probe_audio, read_json, run, write_json

FINAL_SUFFIX = "_final"
# 統合ラウドネスがこれ以上ずれていたら異常とみなす(仕様は ±1 LU)。
LOUDNESS_TOLERANCE_LU = 1.0


def final_path(trimmed: Path, block: str) -> Path:
    return trimmed / f"{block}{FINAL_SUFFIX}.wav"


def _trim_filter(start: float | None, dur: float | None) -> str | None:
    """測定範囲を切り出すフィルタ。`-ss`/`-t` は入力単位なので filter_complex では使えない。"""
    if start is None and dur is None:
        return None
    end = None if (start is None or dur is None) else start + dur
    parts = []
    if start is not None:
        parts.append(f"start={start:.3f}")
    if end is not None:
        parts.append(f"end={end:.3f}")
    return "atrim=" + ":".join(parts) + ",asetpts=N/SR/TB"


def _premix_filter(cfg: SessionConfig, n_inputs: int) -> str:
    """合成部分の filter_complex。出力ラベルは [m]。"""
    if n_inputs == 1:
        return "[0:a]anull[m]"
    w_ext = float(cfg.mix_ratio["ext"])
    w_int = float(cfg.mix_ratio["int"])
    return (
        f"[0:a]volume={w_ext:.6f}[a];"
        f"[1:a]volume={w_int:.6f}[b];"
        f"[a][b]amix=inputs=2:duration=longest:normalize=0[m]"
    )


def _block_inputs(
    block: str, cfg: SessionConfig, blocks: dict[tuple[str, str], Path]
) -> tuple[list[Path], str]:
    """このブロックの入力ファイルと、その説明文。"""
    if cfg.source in ("ext_only", "int_only"):
        group = "ext" if cfg.source == "ext_only" else "int"
        src = blocks.get((block, group))
        if src is None or not src.exists():
            raise PipelineError(
                f"{block}: source={cfg.source} に必要な {group} の正規化済みファイルが"
                f"ありません({src if src else '未検出'})"
            )
        return [src], f"{src.name} ({cfg.source}、合成なし)"

    ext_src = blocks.get((block, "ext"))
    int_src = blocks.get((block, "int"))
    for tag, p in (("ext", ext_src), ("int", int_src)):
        if p is None or not p.exists():
            raise PipelineError(
                f"{block}: source=mix に必要な {tag} の正規化済みファイルがありません"
            )
    w_ext = float(cfg.mix_ratio["ext"])
    w_int = float(cfg.mix_ratio["int"])
    return [ext_src, int_src], f"ext×{w_ext:g} + int×{w_int:g}"


def run_mix(
    outdir: Path,
    cfg: SessionConfig,
    groups: list[str] | None = None,
    blocks: dict[tuple[str, str], Path] | None = None,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
    ref_margin: float = 120.0,
    comp_ratio: float = DEFAULT_COMP_RATIO,
    comp_threshold_offset: float = DEFAULT_COMP_THRESHOLD_OFFSET,
    comp_attack_ms: float = DEFAULT_COMP_ATTACK_MS,
    comp_release_ms: float = DEFAULT_COMP_RELEASE_MS,
    comp_knee_db: float = DEFAULT_COMP_KNEE_DB,
    parallel_db: float = PARALLEL_MAKEUP_DB,
    reverb_mix: float = reverb_mod.DEFAULT_MIX,
    reverb_ir: Path | None = None,
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

    threshold_db = target_lufs + comp_threshold_offset
    log(f"ミックス開始: {len(names)} ブロック / source={cfg.source}")
    log(f"  目標 {target_lufs:+.1f} LUFS / 天井 {true_peak_db:+.1f} dBTP / "
        f"コンプ {comp_ratio:g}:1 しきい値 {threshold_db:+.1f} dBFS "
        f"アタック {comp_attack_ms:g}ms リリース {comp_release_ms:g}ms "
        f"ニー {comp_knee_db:g}dB")

    def chain(gain_db: float, sample_rate: int, *, oversample: int = LIMITER_OVERSAMPLE,
              with_limiter: bool = True) -> str:
        return master_chain(
            gain_db, threshold_db=threshold_db, true_peak_db=true_peak_db,
            ratio=comp_ratio, attack_ms=comp_attack_ms, release_ms=comp_release_ms,
            knee_db=comp_knee_db, sample_rate=sample_rate, oversample=oversample,
            with_limiter=with_limiter, parallel_db=parallel_db,
        )

    if parallel_db > 0:
        log(f"  パラレルコンプ: makeup {parallel_db:+.0f} dB "
            f"(しきい値 {PARALLEL_THRESHOLD_DB:+.0f} dBFS / 小さい音だけを持ち上げる)")
    ir_prepared = None
    if reverb_mix > 0:
        ir_prepared = reverb_mod.prepare_ir(reverb_ir or reverb_mod.default_ir(), trimmed)
        log(f"  残響: {ir_prepared.name} を {reverb_mix*100:.0f}% で混ぜます")

    written: list[Path] = []
    record: dict[str, dict] = {}

    for block in names:
        dst = final_path(trimmed, block)
        if dst.exists() and not force:
            log(f"  スキップ(既存): {dst.name}")
            written.append(dst)
            continue

        inputs, desc = _block_inputs(block, cfg, blocks)
        premix = _premix_filter(cfg, len(inputs))
        temps: list[Path] = []

        # 残響はマスターチェーンの手前に置く。あとにすると、足した残響のぶん
        # ラウドネスとトゥルーピークがずれて、solve_gain の保証が崩れる。
        if ir_prepared is not None:
            if len(inputs) > 1:
                pre = trimmed / f"{block}_premix.wav"
                cmd = [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y"]
                for src in inputs:
                    cmd += ["-i", str(src)]
                cmd += ["-filter_complex", premix, "-map", "[m]",
                        "-c:a", "pcm_f32le", "-rf64", "auto", str(pre)]
                run(cmd, desc="    合成中...")
                temps.append(pre)
            else:
                pre = inputs[0]
            verb = trimmed / f"{block}_verb.wav"
            reverb_mod.apply_reverb(pre, verb, ir_prepared, reverb_mix)
            temps.append(verb)
            inputs, premix = [verb], _premix_filter(cfg, 1)
            desc += f" + 残響 {ir_prepared.stem} {reverb_mix*100:.0f}%"

        # 音出し・休憩の話し声が混じる guard 部分は測定から外す(normalize と同じ考え方)。
        # ゲインの適用はブロック全体に対して行う。
        info = probe_audio(inputs[0])
        dur, sr = info["duration"], info["sample_rate"]
        start, span, window_desc = _reference_window(dur, ref_margin)
        trim = _trim_filter(start, span)

        log(f"  {dst.name}  <- {desc}")

        # 1. 素の統合ラウドネス -> 暫定ゲイン
        pre = measure_complex(inputs, premix, "m", trim=trim)
        gain = target_lufs - pre.integrated
        log(f"    合成後 {pre.describe()}  基準={window_desc} -> 暫定ゲイン {gain:+.2f} dB")

        # 2. コンプを通すとラウドネスが下がるので、実際に通して測り直して補正する。
        #    下見なのでリミッターのオーバーサンプルは省く(統合ラウドネスは変わらない)。
        def after_master(g: float):
            return measure_complex(
                inputs, f"{premix};[m]{chain(g, sr, oversample=1)}[out]", "out", trim=trim)

        def report(g: float, res, resid: float) -> None:
            log(f"    マスター通過後 {res.integrated:+.1f} LUFS (残差 {resid:+.2f} LU)")
            if abs(resid) > GAIN_SETTLE_LU:
                log(f"    ゲインを {g + resid:+.2f} dB に補正")

        gain, after_comp = solve_gain(after_master, target_lufs, gain, on_step=report)

        # 3. 書き出し
        cmd = [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y"]
        for src in inputs:
            cmd += ["-i", str(src)]
        cmd += [
            "-filter_complex", f"{premix};[m]{chain(gain, sr)}[out]",
            "-map", "[out]",
            "-c:a", "pcm_f32le", "-rf64", "auto",
            str(dst),
        ]
        run(cmd, desc="    書き出し中...")

        # 4. 検証。リミッターの動作量は「リミッター直前のトゥルーピーク」で見る。
        after = measure(dst, start, span)
        whole = measure(dst)
        pre_limit = measure_complex(
            inputs, f"{premix};[m]{chain(gain, sr, with_limiter=False)}[out]", "out",
            trim=trim)
        gr = max(0.0, pre_limit.true_peak - true_peak_db)
        log(f"    結果 {after.describe()} / 全体では {whole.describe()}")
        log(f"    リミッター直前のTP {pre_limit.true_peak:+.1f} dBFS "
            f"-> 最大ゲインリダクション {gr:.1f} dB")

        if abs(after.integrated - target_lufs) > LOUDNESS_TOLERANCE_LU:
            raise PipelineError(
                f"{dst.name}: 統合ラウドネスが {after.integrated:+.1f} LUFS で "
                f"目標 {target_lufs:+.1f} LUFS から {LOUDNESS_TOLERANCE_LU:.1f} LU 以上ずれています"
            )
        if whole.true_peak > true_peak_db + 0.05:
            raise PipelineError(
                f"{dst.name}: トゥルーピークが {whole.true_peak:+.2f} dBFS で "
                f"天井 {true_peak_db:+.1f} dBTP を超えています"
            )

        for t in temps:
            t.unlink(missing_ok=True)

        record[block] = {
            "source": desc,
            "window": window_desc,
            "premix": pre.to_json(),
            "gain_db": round(gain, 2),
            "after": after.to_json(),
            "after_whole": whole.to_json(),
            "pre_limiter_true_peak_db": round(pre_limit.true_peak, 2),
            "limiter_max_gr_db": round(gr, 2),
        }
        written.append(dst)

    if record:
        path = outdir / "loudness.json"
        data = read_json(path) if path.exists() else {}
        data.setdefault("master", {}).update(record)
        data["master_settings"] = {
            "target_lufs": target_lufs,
            "true_peak_db": true_peak_db,
            "comp_ratio": comp_ratio,
            "comp_threshold_db": round(threshold_db, 2),
            "comp_attack_ms": comp_attack_ms,
            "comp_release_ms": comp_release_ms,
            "comp_knee_db": comp_knee_db,
            "parallel_makeup_db": parallel_db,
            "reverb_mix": reverb_mix,
            "reverb_ir": ir_prepared.name if ir_prepared else None,
        }
        write_json(path, data)
        log(f"マスタリングの測定結果を保存しました: {path.name}")

    return written
