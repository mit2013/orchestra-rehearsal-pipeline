"""取り込み: TAKEフォルダの走査と整合性検証。

ファイルの探索そのものは `recorder_profiles` に委譲し、ここでは機種によらない
検証(必要なトラックが揃っているか、フォーマットがTAKE間・トラック間で一致するか、
同じ系統のトラックの長さが揃っているか)だけを行う。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .recorder_profiles import RecorderProfile, TakeInfo, get_profile
from .util import PipelineError, log, probe_audio, write_json


@dataclass
class Take:
    """検証済みの1 TAKE。`files` がトラック名 -> 実ファイルの対応。"""

    date: str
    number: int
    dir: Path | None
    files: dict[str, Path]
    sample_rate: int
    duration: float

    # ZOOM M4 向けの短縮アクセサ(既存コードとの互換のために残している)
    @property
    def tr1(self) -> Path:
        return self.files["Tr1"]

    @property
    def tr2(self) -> Path:
        return self.files["Tr2"]

    @property
    def trmic(self) -> Path:
        return self.files["TrMic"]

    def as_dict(self) -> dict:
        d = {
            "date": self.date,
            "number": self.number,
            "dir": str(self.dir) if self.dir is not None else None,
            "files": {k: str(v) for k, v in self.files.items()},
            "sample_rate": self.sample_rate,
            "duration": self.duration,
        }
        # 旧フォーマットの ingest.json しか読めない環境のために従来キーも残す。
        for legacy, track in (("tr1", "Tr1"), ("tr2", "Tr2"), ("trmic", "TrMic")):
            if track in self.files:
                d[legacy] = str(self.files[track])
        return d

    @classmethod
    def from_dict(cls, t: dict) -> "Take":
        if "files" in t:
            files = {k: Path(v) for k, v in t["files"].items()}
        else:  # 旧フォーマット
            files = {
                track: Path(t[legacy])
                for legacy, track in (("tr1", "Tr1"), ("tr2", "Tr2"), ("trmic", "TrMic"))
                if legacy in t
            }
        return cls(
            date=t["date"],
            number=t["number"],
            dir=Path(t["dir"]) if t.get("dir") else None,
            files=files,
            sample_rate=t["sample_rate"],
            duration=t["duration"],
        )


def _expected_channels(tracks: list[str]) -> int:
    """系統内のトラック1本あたりの想定チャンネル数。

    複数トラックで1系統を成す場合(M4の外部マイク Tr1+Tr2)は各トラックがモノラル。
    1トラックで1系統なら、そのファイル自体がステレオ(M4の内蔵マイク TrMic)。
    """
    return 1 if len(tracks) > 1 else 2


def scan(root: Path, date: str, profile: RecorderProfile | None = None) -> list[Take]:
    """TAKEを番号順(=時系列順)に走査・検証して返す。"""
    profile = profile or get_profile("zoom-m4")
    # 「何を探したか」は機種ごとに違うので、見つからないときのメッセージは
    # プロファイル側が出す。ここは念のための一般的な保険にとどめる。
    discovered: list[TakeInfo] = profile.discover(root, date)
    if not discovered:
        raise PipelineError(f"{root} に日付 {date} の TAKE が見つかりません(機種: {profile.name})")

    # 起点は「見つかった最小の番号」であって 001 ではない。同じ日に複数のオケの練習が
    # 入る場合や、失敗したデータを除外した場合、TAKE番号が 001 から始まらないことが
    # 正常にありうるため。ここで検出したいのは「連番の途中が欠けていること」だけで、
    # 先頭が 001 でないこと自体は異常ではない。
    numbers = [t.number for t in discovered]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        log(f"警告: TAKE番号が連続していません {numbers}(欠損の可能性あり)")

    takes: list[Take] = []
    ref: dict | None = None

    for ti in discovered:
        label = ti.dir.name if ti.dir is not None else f"{date}_{ti.number:03d}"
        missing = [t for t, p in ti.files.items() if not p.exists()]
        if missing:
            raise PipelineError(f"{label}: {', '.join(missing)} が見つかりません")

        probes = {t: probe_audio(p) for t, p in ti.files.items()}

        # 系統ごとの想定チャンネル数を検証する。
        for group, tracks in profile.channel_groups.items():
            want = _expected_channels(tracks)
            for t in tracks:
                got = probes[t]["channels"]
                if got != want:
                    raise PipelineError(
                        f"{label}/{probes[t]['name']}: チャンネル数が想定と異なります "
                        f"(期待 {want}ch, 実際 {got}ch)"
                    )

        # フォーマットはTAKE間・トラック間で一致していること。
        for t, pr in probes.items():
            sig = (pr["codec_name"], pr["sample_rate"], pr["bits_per_sample"])
            if ref is None:
                ref = {"key": f"{label}/{pr['name']}", "sig": sig}
            elif sig != ref["sig"]:
                raise PipelineError(
                    f"フォーマット不一致: {ref['key']} = {ref['sig']} / "
                    f"{label}/{pr['name']} = {sig}"
                )

        # 同じ系統に属するトラックは同時収録なので長さが一致していなければならない。
        for group, tracks in profile.channel_groups.items():
            for a, b in zip(tracks, tracks[1:]):
                da, db = probes[a]["duration"], probes[b]["duration"]
                if abs(da - db) > 0.05:
                    raise PipelineError(
                        f"{label}: {a}({da:.3f}s) と {b}({db:.3f}s) の長さが一致しません"
                    )

        # 系統をまたぐズレは止めるほどではないが、気付けるように警告する。
        durations = {g: probes[tracks[0]]["duration"] for g, tracks in profile.channel_groups.items()}
        base_group = next(iter(profile.channel_groups))
        base = durations[base_group]
        for g, d in durations.items():
            if g != base_group and abs(base - d) > 1.0:
                log(f"警告: {label}: {base_group} {base:.1f}s と {g} {d:.1f}s の長さが {abs(base-d):.2f}s ずれています")

        first_track = profile.track_names[0]
        takes.append(
            Take(
                date=date,
                number=ti.number,
                dir=ti.dir,
                files=ti.files,
                sample_rate=probes[first_track]["sample_rate"],
                duration=base,
            )
        )
        log(
            f"TAKE {ti.number:03d}: OK  {base/60:.1f}分  "
            f"{probes[first_track]['sample_rate']}Hz/{probes[first_track]['bits_per_sample']}bit "
            f"{probes[first_track]['codec_name']}"
        )

    return takes


def run_ingest(
    root: Path, date: str, outdir: Path, profile: RecorderProfile | None = None
) -> list[Take]:
    profile = profile or get_profile("zoom-m4")
    takes = scan(root, date, profile)
    total = sum(t.duration for t in takes)

    # TAKE境界の絶対時刻(結合後のタイムライン上)を記録しておくと後段で追跡しやすい。
    offsets, acc = [], 0.0
    for t in takes:
        offsets.append({"number": t.number, "start": acc, "duration": t.duration})
        acc += t.duration

    manifest = {
        "date": date,
        "root": str(root),
        "recorder": profile.name,
        "channel_groups": profile.channel_groups,
        "n_takes": len(takes),
        "sample_rate": takes[0].sample_rate,
        "total_duration": total,
        "take_offsets": offsets,
        "takes": [t.as_dict() for t in takes],
    }
    write_json(outdir / "ingest.json", manifest)
    log(f"取り込み完了: {len(takes)} TAKE / 合計 {total/3600:.2f} 時間 -> {outdir/'ingest.json'}")
    return takes
