"""チャンネル結合とTAKE連結。

- 外部マイク: 各TAKEで Tr1 -> L(FL), Tr2 -> R(FR) を明示マッピングしてステレオ化し、
  TAKE番号順に連結する。
- 内蔵マイク: TrMic をそのままTAKE番号順に連結する。

品質劣化を避けるため出力は常に pcm_f32le(入力と同一)。3時間素材ではWAVの4GB制限を
超えうるので `-rf64 auto` を指定し、必要になった場合のみRF64ヘッダで書き出す。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .ingest import Take
from .util import FFMPEG, PipelineError, log, probe_audio, run

# Tr1 = 外部マイク CH1 = L、Tr2 = 外部マイク CH2 = R(指示書 2章)。
# amerge は入力順に依存して曖昧なので、join の map で物理的な対応を明示する。
JOIN_FILTER = "join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR"


def _concat_filter(labels: list[str]) -> str:
    return "".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=0:a=1[out]"


def build_ext_cmd(takes: list[Take], dst: Path) -> list[str]:
    """Tr1/Tr2 をステレオ化して連結する ffmpeg コマンドを組み立てる。"""
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y"]
    for t in takes:
        cmd += ["-i", str(t.tr1), "-i", str(t.tr2)]

    chains, labels = [], []
    for i, _ in enumerate(takes):
        a, b = 2 * i, 2 * i + 1
        lbl = f"s{i}"
        chains.append(f"[{a}:a][{b}:a]{JOIN_FILTER}[{lbl}]")
        labels.append(lbl)

    if len(labels) == 1:
        graph = ";".join(chains)
        out_label = labels[0]
    else:
        graph = ";".join(chains + [_concat_filter(labels)])
        out_label = "out"

    cmd += [
        "-filter_complex", graph,
        "-map", f"[{out_label}]",
        "-c:a", "pcm_f32le",
        "-rf64", "auto",
        str(dst),
    ]
    return cmd


def build_int_cmd(takes: list[Take], dst: Path) -> list[str]:
    """TrMic(ステレオ)をそのまま連結する ffmpeg コマンド。"""
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y"]
    for t in takes:
        cmd += ["-i", str(t.trmic)]

    if len(takes) == 1:
        cmd += ["-map", "0:a"]
    else:
        cmd += [
            "-filter_complex", _concat_filter([f"{i}:a" for i in range(len(takes))]),
            "-map", "[out]",
        ]
    cmd += ["-c:a", "pcm_f32le", "-rf64", "auto", str(dst)]
    return cmd


def _decode_chunk(path: Path, start: float, dur: float, channel: int | None, sr: int) -> np.ndarray:
    """検証用に短い区間だけを float32 で取り出す(全体は絶対に読み込まない)。"""
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", str(path)]
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(
            f"検証用デコードに失敗: {path}\n{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return np.frombuffer(proc.stdout, dtype="<f4")


def verify_channel_mapping(takes: list[Take], merged_ext: Path, sr: int) -> None:
    """結合後の L が Tr1、R が Tr2 由来であることを実データで確認する。

    最初のTAKEの中ほどを切り出し、結合ファイルの L/R と元の Tr1/Tr2 を直接比較する。
    L==Tr1 かつ R==Tr2 でなければ、マッピングが崩れているので処理を止める。
    """
    t0 = takes[0]
    start = min(60.0, max(0.0, t0.duration / 2))
    dur = 5.0

    ref1 = _decode_chunk(t0.tr1, start, dur, None, sr)
    ref2 = _decode_chunk(t0.tr2, start, dur, None, sr)
    got_l = _decode_chunk(merged_ext, start, dur, 0, sr)
    got_r = _decode_chunk(merged_ext, start, dur, 1, sr)

    n = min(len(ref1), len(ref2), len(got_l), len(got_r))
    if n == 0:
        raise PipelineError("チャンネル検証用のサンプルを取得できませんでした")
    ref1, ref2, got_l, got_r = ref1[:n], ref2[:n], got_l[:n], got_r[:n]

    d_ll = float(np.max(np.abs(got_l - ref1)))
    d_rr = float(np.max(np.abs(got_r - ref2)))
    d_lr = float(np.max(np.abs(got_l - ref2)))

    log(f"チャンネル検証: |L-Tr1|={d_ll:.2e}  |R-Tr2|={d_rr:.2e}  |L-Tr2|={d_lr:.2e}")
    tol = 1e-6
    if d_ll > tol or d_rr > tol:
        raise PipelineError(
            "チャンネルマッピング検証に失敗しました。"
            f"L と Tr1 の最大差 {d_ll:.3e}、R と Tr2 の最大差 {d_rr:.3e}(許容 {tol:.0e})"
        )
    log("チャンネル検証: OK (Tr1 -> L, Tr2 -> R)")


def _check_duration(path: Path, expected: float, what: str) -> None:
    pr = probe_audio(path)
    if abs(pr["duration"] - expected) > 1.0:
        raise PipelineError(
            f"{what}: 結合後の長さ {pr['duration']:.2f}s が期待値 {expected:.2f}s と一致しません"
        )
    log(
        f"{what}: {pr['duration']/60:.1f}分  {pr['channels']}ch  "
        f"{pr['sample_rate']}Hz  {pr['size']/2**30:.2f}GiB"
    )


def run_merge(takes: list[Take], outdir: Path, force: bool = False) -> tuple[Path, Path]:
    ext = outdir / "raw_merged_ext.wav"
    inn = outdir / "raw_merged_int.wav"
    total = sum(t.duration for t in takes)
    sr = takes[0].sample_rate

    if ext.exists() and not force:
        log(f"スキップ(既存): {ext.name} — 作り直すには --force")
    else:
        run(build_ext_cmd(takes, ext), desc=f"外部マイク結合 (Tr1->L, Tr2->R) x {len(takes)} TAKE ...")
    _check_duration(ext, total, "raw_merged_ext.wav")
    verify_channel_mapping(takes, ext, sr)

    if inn.exists() and not force:
        log(f"スキップ(既存): {inn.name} — 作り直すには --force")
    else:
        run(build_int_cmd(takes, inn), desc=f"内蔵マイク結合 (TrMic) x {len(takes)} TAKE ...")
    _check_duration(inn, total, "raw_merged_int.wav")

    return ext, inn
