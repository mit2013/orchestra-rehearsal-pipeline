"""取り込み: TAKEフォルダの走査と整合性検証。

各TAKEに Tr1/Tr2/TrMic が揃っているか、サンプルレート・ビット深度・チャンネル数が
想定どおりかを検証する。不整合があれば PipelineError で処理を止める。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .util import PipelineError, log, probe_audio, write_json

TAKE_DIR_RE = re.compile(r"^(?P<date>\d{6})_(?P<num>\d{3})\.TAKE$")


@dataclass
class Take:
    date: str
    number: int
    dir: Path
    tr1: Path
    tr2: Path
    trmic: Path
    sample_rate: int
    duration: float

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("dir", "tr1", "tr2", "trmic"):
            d[k] = str(d[k])
        return d


def find_take_dirs(root: Path, date: str) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = TAKE_DIR_RE.match(p.name)
        if m and m.group("date") == date:
            found.append((int(m.group("num")), p))
    found.sort(key=lambda t: t[0])
    return found


def scan(root: Path, date: str) -> list[Take]:
    """TAKEを番号順(=時系列順)に走査・検証して返す。"""
    take_dirs = find_take_dirs(root, date)
    if not take_dirs:
        raise PipelineError(f"{root} に {date}_XXX.TAKE フォルダが見つかりません")

    numbers = [n for n, _ in take_dirs]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        log(f"警告: TAKE番号が連続していません {numbers}(欠損の可能性あり)")

    takes: list[Take] = []
    ref: dict | None = None

    for number, d in take_dirs:
        stem = f"{date}_{number:03d}"
        paths = {
            "Tr1": d / f"{stem}_Tr1.WAV",
            "Tr2": d / f"{stem}_Tr2.WAV",
            "TrMic": d / f"{stem}_TrMic.WAV",
        }
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            raise PipelineError(f"{d.name}: {', '.join(missing)} が見つかりません")

        probes = {k: probe_audio(p) for k, p in paths.items()}

        # チャンネル数: 外部マイクはモノラル×2、内蔵マイクはステレオ。
        for key, want in (("Tr1", 1), ("Tr2", 1), ("TrMic", 2)):
            got = probes[key]["channels"]
            if got != want:
                raise PipelineError(
                    f"{d.name}/{probes[key]['name']}: チャンネル数が想定と異なります "
                    f"(期待 {want}ch, 実際 {got}ch)"
                )

        # フォーマットはTAKE間・トラック間で一致していること。
        for key, pr in probes.items():
            sig = (pr["codec_name"], pr["sample_rate"], pr["bits_per_sample"])
            if ref is None:
                ref = {"key": f"{d.name}/{pr['name']}", "sig": sig}
            elif sig != ref["sig"]:
                raise PipelineError(
                    f"フォーマット不一致: {ref['key']} = {ref['sig']} / "
                    f"{d.name}/{pr['name']} = {sig}"
                )

        # Tr1 と Tr2 は同一マイクペアなので長さが一致していなければならない。
        d1, d2, dm = probes["Tr1"]["duration"], probes["Tr2"]["duration"], probes["TrMic"]["duration"]
        if abs(d1 - d2) > 0.05:
            raise PipelineError(
                f"{d.name}: Tr1({d1:.3f}s) と Tr2({d2:.3f}s) の長さが一致しません"
            )
        if abs(d1 - dm) > 1.0:
            log(f"警告: {d.name}: 外部マイク {d1:.1f}s と内蔵マイク {dm:.1f}s の長さが {abs(d1-dm):.2f}s ずれています")

        takes.append(
            Take(
                date=date,
                number=number,
                dir=d,
                tr1=paths["Tr1"],
                tr2=paths["Tr2"],
                trmic=paths["TrMic"],
                sample_rate=probes["Tr1"]["sample_rate"],
                duration=d1,
            )
        )
        log(
            f"TAKE {number:03d}: OK  {d1/60:.1f}分  "
            f"{probes['Tr1']['sample_rate']}Hz/{probes['Tr1']['bits_per_sample']}bit "
            f"{probes['Tr1']['codec_name']}"
        )

    return takes


def run_ingest(root: Path, date: str, outdir: Path) -> list[Take]:
    takes = scan(root, date)
    total = sum(t.duration for t in takes)

    # TAKE境界の絶対時刻(結合後のタイムライン上)を記録しておくと後段で追跡しやすい。
    offsets, acc = [], 0.0
    for t in takes:
        offsets.append({"number": t.number, "start": acc, "duration": t.duration})
        acc += t.duration

    manifest = {
        "date": date,
        "root": str(root),
        "n_takes": len(takes),
        "sample_rate": takes[0].sample_rate,
        "total_duration": total,
        "take_offsets": offsets,
        "takes": [t.as_dict() for t in takes],
    }
    write_json(outdir / "ingest.json", manifest)
    log(f"取り込み完了: {len(takes)} TAKE / 合計 {total/3600:.2f} 時間 -> {outdir/'ingest.json'}")
    return takes
