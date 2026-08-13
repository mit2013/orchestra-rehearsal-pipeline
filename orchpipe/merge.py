"""チャンネル結合とTAKE連結。

系統(channel group)ごとに1本の長尺ステレオWAVを作る。

- 2トラックで1系統を成す場合(M4の外部マイク Tr1/Tr2): 2本のモノラルを左右に
  明示マッピングしてステレオ化し、TAKE番号順に連結する。
  左右の割り当ては session_config.json の `ext_lr_map` に従う。
- 1トラックで1系統の場合(M4の内蔵マイク TrMic): そのままTAKE番号順に連結する。

品質劣化を避けるため出力は常に pcm_f32le(入力と同一)。3時間素材ではWAVの4GB制限を
超えうるので `-rf64 auto` を指定し、必要になった場合のみRF64ヘッダで書き出す。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import SessionConfig
from .ingest import Take
from .recorder_profiles import get_profile
from .util import FFMPEG, PipelineError, log, probe_audio, run

# 2本のモノラルをステレオに組む際の割り当て。amerge は入力順に依存して曖昧なので、
# join の map で「どの入力のどのチャンネルが左右どちらになるか」を明示する。
#   normal : Tr1 -> L(FL), Tr2 -> R(FR)   … 正しい配線
#   swapped: Tr1 -> R(FR), Tr2 -> L(FL)   … 配線ミスの補正
JOIN_MAPS = {
    "normal": "join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR",
    "swapped": "join=inputs=2:channel_layout=stereo:map=0.0-FR|1.0-FL",
}


def _concat_filter(labels: list[str]) -> str:
    return "".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=0:a=1[out]"


def build_group_cmd(
    takes: list[Take], tracks: list[str], dst: Path, lr_map: str = "normal"
) -> list[str]:
    """1系統ぶんの結合コマンドを組み立てる。"""
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y"]
    for t in takes:
        for track in tracks:
            cmd += ["-i", str(t.files[track])]

    n_in = len(tracks)
    chains, labels = [], []
    for i, _ in enumerate(takes):
        if n_in == 2:
            a, b = 2 * i, 2 * i + 1
            lbl = f"s{i}"
            chains.append(f"[{a}:a][{b}:a]{JOIN_MAPS[lr_map]}[{lbl}]")
            labels.append(lbl)
        else:
            labels.append(f"{i}:a")

    if len(labels) == 1:
        if chains:  # 単一TAKE・2トラック: ステレオ化だけ行う
            cmd += ["-filter_complex", ";".join(chains), "-map", f"[{labels[0]}]"]
        else:       # 単一TAKE・1トラック: そのまま
            cmd += ["-map", labels[0]]
    else:
        graph = ";".join(chains + [_concat_filter(labels)])
        cmd += ["-filter_complex", graph, "-map", "[out]"]

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


def verify_channel_mapping(
    takes: list[Take], merged: Path, tracks: list[str], sr: int, lr_map: str = "normal"
) -> None:
    """結合後の L/R が、設定どおりの元トラック由来であることを実データで確認する。

    最初のTAKEの中ほどを切り出し、結合ファイルの L/R と元の2トラックを直接比較する。
    normal なら L==Tr1 かつ R==Tr2、swapped なら L==Tr2 かつ R==Tr1 でなければ止める。
    """
    t0 = takes[0]
    start = min(60.0, max(0.0, t0.duration / 2))
    dur = 5.0
    a_name, b_name = tracks[0], tracks[1]

    ref_a = _decode_chunk(t0.files[a_name], start, dur, None, sr)
    ref_b = _decode_chunk(t0.files[b_name], start, dur, None, sr)
    got_l = _decode_chunk(merged, start, dur, 0, sr)
    got_r = _decode_chunk(merged, start, dur, 1, sr)

    n = min(len(ref_a), len(ref_b), len(got_l), len(got_r))
    if n == 0:
        raise PipelineError("チャンネル検証用のサンプルを取得できませんでした")
    ref_a, ref_b, got_l, got_r = ref_a[:n], ref_b[:n], got_l[:n], got_r[:n]

    # 期待する対応(L に来るべきトラック, R に来るべきトラック)
    want_l, want_r = (ref_a, ref_b) if lr_map == "normal" else (ref_b, ref_a)
    nl, nr = (a_name, b_name) if lr_map == "normal" else (b_name, a_name)

    d_l = float(np.max(np.abs(got_l - want_l)))
    d_r = float(np.max(np.abs(got_r - want_r)))
    d_cross = float(np.max(np.abs(got_l - want_r)))

    log(f"チャンネル検証 ({lr_map}): |L-{nl}|={d_l:.2e}  |R-{nr}|={d_r:.2e}  |L-{nr}|={d_cross:.2e}")
    tol = 1e-6
    if d_l > tol or d_r > tol:
        raise PipelineError(
            f"チャンネルマッピング検証に失敗しました(ext_lr_map={lr_map})。"
            f"L と {nl} の最大差 {d_l:.3e}、R と {nr} の最大差 {d_r:.3e}(許容 {tol:.0e})"
        )
    log(f"チャンネル検証: OK ({nl} -> L, {nr} -> R)")


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


def merged_path(outdir: Path, group: str) -> Path:
    return outdir / f"raw_merged_{group}.wav"


def run_merge(
    takes: list[Take],
    outdir: Path,
    cfg: SessionConfig | None = None,
    channel_groups: dict[str, list[str]] | None = None,
    force: bool = False,
    only_groups: list[str] | None = None,
) -> dict[str, Path]:
    # 系統定義は ingest.json 由来のものを受け取る。渡されなければ既定機種の定義。
    if channel_groups is None:
        channel_groups = get_profile("zoom-m4").channel_groups
    cfg = cfg or SessionConfig()
    total = sum(t.duration for t in takes)
    sr = takes[0].sample_rate
    out: dict[str, Path] = {}

    for group, tracks in channel_groups.items():
        dst = merged_path(outdir, group)
        out[group] = dst
        if only_groups is not None and group not in only_groups:
            log(f"スキップ(対象外): {dst.name}")
            continue

        # 左右の割り当てが関わるのは、2本のモノラルを組む系統だけ。
        lr_map = cfg.ext_lr_map if len(tracks) == 2 else "normal"

        if dst.exists() and not force:
            log(f"スキップ(既存): {dst.name} — 作り直すには --force")
        else:
            desc = f"{group} 系統を結合 ({'+'.join(tracks)}"
            if len(tracks) == 2:
                desc += f", ext_lr_map={lr_map}"
            desc += f") x {len(takes)} TAKE ..."
            run(build_group_cmd(takes, tracks, dst, lr_map), desc=desc)

        _check_duration(dst, total, dst.name)
        if len(tracks) == 2:
            verify_channel_mapping(takes, dst, tracks, sr, lr_map)

    return out
