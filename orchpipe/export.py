"""曲目単位エクスポートとMP3タグ埋め込み。

`mix` が作った `trimmed/*_final.wav` を、配布用の WAV(32bit float)と
MP3(320kbps、ID3タグ付き)として `output/{date}/export/` に書き出す。

1合奏ブロック = 1トラックとして扱う。`trimmed/` 配下の既存ファイルは読み取り専用で、
このモジュールは一切変更・削除しない。

`variant`(CLI では `--variant`)を渡すと、ファイル名の末尾と ID3 のタイトルに
その名前が入る(`260829_前半_ラウドネス調整版.mp3`)。**既に配った音源を差し替えず、
作り直したものを別版として並べて置く**ための仕組みである。名前が違えば Box も
Drive も新規ファイルとして扱うので、団員が既に持っているファイルはそのまま残る。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .apply import load_confirmed
from .config import SessionConfig
from .util import FFMPEG, PipelineError, log, probe_audio, read_json, run

MP3_BITRATE = "320k"
FINAL_SUFFIX = "_final.wav"

# ブロック数に応じたトラックタイトル。休憩1回なら前後半、2回なら3分割の呼び方をする。
TITLE_RULES: dict[int, list[str]] = {
    2: ["前半", "後半"],
    3: ["前半", "中盤", "後半"],
}


def block_titles(n: int) -> list[str]:
    """ブロック数からトラックタイトルを決める。"""
    if n <= 0:
        raise PipelineError("ブロック数が 0 です")
    titles = TITLE_RULES.get(n)
    if titles is not None:
        return list(titles)
    return [f"コマ{i}" for i in range(1, n + 1)]


def year_from_date(date: str) -> str:
    """`260802` -> `2026`。レコーダの日付は西暦下2桁始まり。"""
    if not re.fullmatch(r"\d{6}", date):
        raise PipelineError(f"日付は6桁の数字である必要があります: {date!r}")
    return f"20{date[:2]}"


@dataclass
class Track:
    number: int
    title: str
    src: Path
    wav: Path
    mp3: Path


def find_final_files(trimmed: Path) -> list[Path]:
    """`{NN}_{ラベル}_final.wav` を先頭の連番順に返す。"""
    if not trimmed.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in sorted(trimmed.iterdir()):
        if not p.is_file() or not p.name.endswith(FINAL_SUFFIX):
            continue
        m = re.match(r"^(\d+)_", p.name)
        found.append((int(m.group(1)) if m else 1 << 30, p))
    found.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in found]


def find_wav_files(outdir: Path, allow_empty: bool = False) -> list[Path]:
    """`export/` に書き出し済みの WAV を名前順で返す(Drive へのアップロード入力)。

    `trimmed/*_final.wav` から名前を組み直さないのは、`--variant` を付けたときに
    別版が同じ名前になってしまい、**既に配った WAV を上書きする**ためである。
    export の WAV は `_final.wav` の実体コピーなので中身は同一である。
    """
    export = outdir / "export"
    if not export.is_dir():
        raise PipelineError(f"{export} がありません。先に `export` を実行してください。")
    files = sorted(p for p in export.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not files and not allow_empty:
        raise PipelineError(f"{export} に WAV がありません。先に `export` を実行してください。")
    return files


def find_mp3_files(outdir: Path) -> list[Path]:
    """`export/` に書き出し済みの MP3 を名前順で返す(Box / Drive 共通の入力)。"""
    export = outdir / "export"
    if not export.is_dir():
        raise PipelineError(f"{export} がありません。先に `export` を実行してください。")
    files = sorted(p for p in export.iterdir() if p.is_file() and p.suffix.lower() == ".mp3")
    if not files:
        raise PipelineError(f"{export} に MP3 がありません。先に `export` を実行してください。")
    return files


def plan_tracks(outdir: Path, date: str, variant: str = "") -> list[Track]:
    """出力するトラックの一覧を組み立てる(まだ書き出さない)。"""
    trimmed = outdir / "trimmed"
    finals = find_final_files(trimmed)
    if not finals:
        raise PipelineError(
            f"{trimmed} に *_final.wav がありません。"
            "先に `normalize` と `mix` を実行してください。"
        )

    # ブロック数は confirmed.json の keep 区間数で決める(指示書 1章)。
    confirmed = outdir / "confirmed.json"
    if confirmed.exists():
        # 尺の妥当性は apply 時に検証済みなので、ここでは上限を設けず区間数だけ見る。
        keeps = load_confirmed(confirmed, float("inf"))
        if len(keeps) != len(finals):
            raise PipelineError(
                f"confirmed.json の keep 区間数 ({len(keeps)}) と "
                f"*_final.wav の本数 ({len(finals)}) が一致しません。"
                "`mix` をやり直してください。"
            )

    titles = block_titles(len(finals))
    export_dir = outdir / "export"
    tail = f"_{variant}" if variant else ""
    return [
        Track(
            number=i,
            title=title,
            src=src,
            wav=export_dir / f"{date}_{title}{tail}.wav",
            mp3=export_dir / f"{date}_{title}{tail}.mp3",
        )
        for i, (src, title) in enumerate(zip(finals, titles), start=1)
    ]


def write_tags(path: Path, *, album: str, title: str, artist: str,
               album_artist: str, track: int, total: int, year: str) -> None:
    """MP3 に ID3 タグを書き込む(指示書 4章のフレーム)。"""
    from mutagen.id3 import ID3, TALB, TDRC, TIT2, TPE1, TPE2, TRCK
    from mutagen.id3 import ID3NoHeaderError

    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("TALB"); tags.delall("TIT2"); tags.delall("TPE1")
    tags.delall("TPE2"); tags.delall("TRCK"); tags.delall("TDRC")
    tags.add(TALB(encoding=3, text=album))
    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=artist))
    tags.add(TPE2(encoding=3, text=album_artist))
    tags.add(TRCK(encoding=3, text=f"{track}/{total}"))
    tags.add(TDRC(encoding=3, text=year))
    tags.save(path)


def run_export(
    outdir: Path,
    date: str,
    cfg: SessionConfig,
    force: bool = False,
    variant: str = "",
) -> list[Track]:
    tracks = plan_tracks(outdir, date, variant)
    export_dir = outdir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    year = year_from_date(date)
    total = len(tracks)

    log(
        f"エクスポート開始: {total} トラック / アルバム={date} / "
        f"団体={cfg.orchestra!r} / 年={year}"
        + (f" / 版={variant}" if variant else "")
    )

    for t in tracks:
        log(f"  [{t.number}/{total}] {t.title}  <- {t.src.name}")

        # WAV: 32bit float のまま。再エンコードせず実体コピーする。
        if t.wav.exists() and not force:
            log(f"    スキップ(既存): {t.wav.name}")
        else:
            shutil.copyfile(t.src, t.wav)
            log(f"    {t.wav.name}  (コピー、無変換)")

        # MP3: 320kbps 固定。
        if t.mp3.exists() and not force:
            log(f"    スキップ(既存): {t.mp3.name}")
        else:
            run(
                [
                    FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
                    "-i", str(t.src),
                    "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
                    "-map_metadata", "-1",
                    str(t.mp3),
                ],
                desc=f"    {t.mp3.name}  (libmp3lame {MP3_BITRATE})",
            )

        write_tags(
            t.mp3,
            album=date,
            title=f"{t.title}({variant})" if variant else t.title,
            artist=cfg.orchestra,
            album_artist=cfg.orchestra,
            track=t.number,
            total=total,
            year=year,
        )
        pr = probe_audio(t.mp3)
        log(
            f"    タグ書き込み完了 / MP3 {pr['duration']/60:.1f}分 "
            f"{pr['size']/2**20:.0f}MiB"
        )

    return tracks


def read_tags(path: Path) -> dict[str, str]:
    """検証用にタグを読み戻す。"""
    from mutagen.id3 import ID3

    tags = ID3(path)
    out: dict[str, str] = {}
    for frame in ("TALB", "TIT2", "TPE1", "TPE2", "TRCK", "TDRC", "TYER"):
        if frame in tags:
            out[frame] = str(tags[frame].text[0])
    return out
