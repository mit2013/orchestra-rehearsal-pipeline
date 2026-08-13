"""確定済みJSONにもとづく実トリミング。

`action: "keep"` の区間だけを、外部マイク/内蔵マイクの2系統それぞれから切り出す。
再エンコードによる劣化を避けるため出力は入力と同じ pcm_f32le。
"""

from __future__ import annotations

import re
from pathlib import Path

from .util import (
    FFMPEG,
    PipelineError,
    fmt_time,
    log,
    parse_time,
    probe_audio,
    run,
    safe_filename,
)

_SUFFIX_RE = re.compile(r"[((]推定[))]\s*$")


def _clean_label(label: str) -> str:
    return safe_filename(_SUFFIX_RE.sub("", label).strip()) or "segment"


def load_confirmed(path: Path, total_duration: float) -> list[dict]:
    """ユーザーが編集したJSONを読み、検証して keep 区間だけ返す。"""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise PipelineError(f"{path.name} はオブジェクトの配列である必要があります")

    keeps: list[dict] = []
    for i, item in enumerate(data, start=1):
        for key in ("start", "end", "action"):
            if key not in item:
                raise PipelineError(f"{path.name} の {i} 番目に '{key}' がありません")
        action = str(item["action"]).strip().lower()
        if action not in ("keep", "remove"):
            raise PipelineError(
                f"{path.name} の {i} 番目: action は keep か remove です(実際: {item['action']!r})"
            )
        start = parse_time(item["start"])
        end = parse_time(item["end"])
        if end <= start:
            raise PipelineError(
                f"{path.name} の {i} 番目: end({item['end']}) が start({item['start']}) 以下です"
            )
        if start < -0.001 or end > total_duration + 1.0:
            raise PipelineError(
                f"{path.name} の {i} 番目: 区間 {item['start']}–{item['end']} が "
                f"素材の長さ {fmt_time(total_duration)} をはみ出しています"
            )
        if action == "keep":
            keeps.append(
                {
                    "start": start,
                    "end": min(end, total_duration),
                    "label": str(item.get("label") or f"segment{i}"),
                }
            )

    if not keeps:
        raise PipelineError(f"{path.name} に action:\"keep\" の区間が1つもありません")

    keeps.sort(key=lambda k: k["start"])
    for a, b in zip(keeps, keeps[1:]):
        if b["start"] < a["end"] - 0.001:
            log(f"警告: keep 区間が重なっています ({fmt_time(a['start'])}–{fmt_time(a['end'])} と "
                f"{fmt_time(b['start'])}–{fmt_time(b['end'])})。指定どおりに書き出します。")
    return keeps


def run_apply(
    confirmed: Path,
    sources: dict[str, Path],
    dst_dir: Path,
    total_duration: float,
    force: bool = False,
) -> list[Path]:
    keeps = load_confirmed(confirmed, total_duration)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    log(f"トリミング開始: keep {len(keeps)} 区間 x {len(sources)} 系統")
    for n, seg in enumerate(keeps, start=1):
        dur = seg["end"] - seg["start"]
        name = f"{n:02d}_{_clean_label(seg['label'])}"
        for tag, src in sources.items():
            out = dst_dir / f"{name}_{tag}.wav"
            if out.exists() and not force:
                log(f"スキップ(既存): {out.name}")
                written.append(out)
                continue
            run(
                [
                    FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                    "-ss", f"{seg['start']:.3f}",
                    "-i", str(src),
                    "-t", f"{dur:.3f}",
                    "-c:a", "pcm_f32le",
                    "-rf64", "auto",
                    str(out),
                ],
                desc=f"  {out.name}  {fmt_time(seg['start'])}–{fmt_time(seg['end'])} ({dur/60:.1f}分)",
            )
            written.append(out)

    for p in written:
        pr = probe_audio(p)
        log(f"出力: {p.name}  {pr['duration']/60:.1f}分  {pr['channels']}ch  {pr['size']/2**20:.0f}MiB")
    return written
