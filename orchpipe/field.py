"""現場前処理 ― iPhone で作る「配布用プロキシ」と、そこからの切り出し。

## 何のためのものか

練習が終わってから団員が聴けるまでの時間を詰める。素材は 3 時間で約 8GB
(32bit float)あり、モバイル回線では送れない。そこで **iPhone 上で 320kbps の
MP3 を 1 本だけ作り、それを母艦へ送る**。母艦は届いた MP3 だけで解析・境界提案・
ブロック切り出しまで済ませ、Box への配布と通知を帰宅前に終える。32bit float の
原本から作る WAV(Drive 用)は帰宅後でよい。

## プロキシに何を載せるか ― 指示書からの変更点

`orchestra_recording_pipeline_field_preprocess.md` §3-2 は「iPhone 側で
**ゲイン確定済み**の配布用マスターを作り、母艦は `-c copy` で切るだけ」と書いている。
**この設計は採れない。** 指示書を書いた時点の母艦は日単位のピーク正規化
(`normalize_scope: date`)だったが、`feature/audio-quality` でブロックごとの
ラウドネス正規化に変わったためである。260829 の実測:

| ブロック | 素の統合ラウドネス | 実際に当てたゲイン |
|---|---|---|
| 合奏1 | -22.3 LUFS | +4.0 dB |
| 合奏2 | -13.4 LUFS | -5.7 dB |
| 合奏3 | -15.9 LUFS | -2.5 dB |

境界が決まる前に 1 本ぶんの単一ゲインを確定させると、この **9.7 dB の差がそのまま
残る**。ブロック間で 8.6 LU ずれていたのが元の不満であり、そこへ戻ることになる。

そこでプロキシには **クリップを避けるための固定ゲインだけ**を載せる。ラウドネス
正規化・コンプ・リミッターは、境界が決まったあとに母艦がブロックごとに当てる。
その代償として MP3 の符号化が 1 世代増える(`-c copy` では切れない)。この 1 世代の
差が実用上問題にならないことは検証すること(指示書 §4-1)。

## 固定ゲインを 1 パスで決められる理由

素材のピークは 0 dBFS を超える(260829 で +11.43 dBFS)。32bit float なので保持
されているが、MP3 に符号化する前に下げないと潰れる。現場で 2 パス目を回すと
microSD からの読み出しがもう一度発生して律速するため、**測ってから決めるのではなく
固定値を当て、同じパスの中で `volumedetect` で結果を確認する**。ピークが 0 dBFS に
達していたらログに警告が出るので、その日は原本から作り直せばよい。

固定ゲインは可逆である。母艦は `-PROXY_GAIN_DB` を戻してからラウドネスを測るので、
プロキシ経由でも原本経由でも同じ目標に合う。
"""

from __future__ import annotations

import re
from pathlib import Path

from .loudness import (
    DEFAULT_COMP_ATTACK_MS,
    DEFAULT_COMP_KNEE_DB,
    DEFAULT_COMP_RATIO,
    DEFAULT_COMP_RELEASE_MS,
    DEFAULT_COMP_THRESHOLD_OFFSET,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TRUE_PEAK_DB,
    LIMITER_OVERSAMPLE,
    NOISE_FLOOR_BELOW_TARGET_DB,
    PARALLEL_THRESHOLD_DB,
    limit_parallel_makeup,
    master_chain,
    measure,
    noise_report,
    solve_gain,
)
from .merge import JOIN_MAPS, SWAP_STEREO
from .util import FFMPEG, PipelineError, fmt_time, log, probe_audio, run

# --- プロキシ ---------------------------------------------------------------
# 符号化前に当てる固定ゲイン。260829 の最大ピーク +11.43 dBFS に 2.6 dB の余裕を
# 見た値。日によって上振れしても volumedetect が気づく。
PROXY_GAIN_DB = -14.0
PROXY_BITRATE = "320k"
PROXY_SUFFIX = "_proxy.mp3"

_MAXVOL_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def proxy_path(outdir: Path, group: str = "ext") -> Path:
    return outdir / f"raw_merged_{group}{PROXY_SUFFIX}"


def proxy_filter(
    n_takes: int,
    n_tracks: int,
    lr_map: str = "normal",
    gain_db: float = PROXY_GAIN_DB,
) -> str:
    """TAKE を連結し、ステレオに組み、固定ゲインを当てる filter_complex。

    **`merge.build_group_cmd` と同じ組み方でなければならない。** 左右の割り当ては
    `merge.JOIN_MAPS` をそのまま使う。ここが食い違うと、現場経由と原本経由で
    L/R が入れ替わる。
    """
    chains: list[str] = []
    labels: list[str] = []
    for i in range(n_takes):
        if n_tracks == 2:
            a, b = 2 * i, 2 * i + 1
            lbl = f"s{i}"
            chains.append(f"[{a}:a][{b}:a]{JOIN_MAPS[lr_map]}[{lbl}]")
            labels.append(lbl)
        elif lr_map == "swapped":
            lbl = f"s{i}"
            chains.append(f"[{i}:a]{SWAP_STEREO}[{lbl}]")
            labels.append(lbl)
        else:
            labels.append(f"{i}:a")

    gain = f"volume={gain_db:.6f}dB,volumedetect[out]"
    if len(labels) == 1:
        return ";".join(chains + [f"[{labels[0]}]{gain}"])
    cat = "".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=0:a=1[cat]"
    return ";".join(chains + [cat, f"[cat]{gain}"])


def build_proxy_cmd(
    inputs: list[Path],
    n_takes: int,
    n_tracks: int,
    dst: Path,
    lr_map: str = "normal",
    gain_db: float = PROXY_GAIN_DB,
    bitrate: str = PROXY_BITRATE,
) -> list[str]:
    """プロキシを作る ffmpeg コマンド。現場(a-Shell)でも母艦でも同じものを使う。

    `inputs` は TAKE ごとにトラック順に並べたファイル(merge と同じ並び)。
    """
    cmd = [FFMPEG, "-hide_banner", "-v", "info", "-stats", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex", proxy_filter(n_takes, n_tracks, lr_map, gain_db),
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", bitrate,
        "-map_metadata", "-1",
        str(dst),
    ]
    return cmd


def build_proxy(
    inputs: list[Path],
    n_takes: int,
    n_tracks: int,
    dst: Path,
    lr_map: str = "normal",
    gain_db: float = PROXY_GAIN_DB,
    bitrate: str = PROXY_BITRATE,
) -> dict:
    """プロキシを作り、符号化直前のピークを確認する。"""
    import subprocess

    cmd = build_proxy_cmd(inputs, n_takes, n_tracks, dst, lr_map, gain_db, bitrate)
    log(f"プロキシを作成: {dst.name}  <- {n_takes} TAKE x {n_tracks} トラック "
        f"(ext_lr_map={lr_map}, ゲイン {gain_db:+.1f} dB, {bitrate})")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise PipelineError(f"プロキシの作成に失敗しました\n{text.strip()[-800:]}")

    m = _MAXVOL_RE.search(text)
    peak = float(m.group(1)) if m else None
    if peak is not None:
        if peak >= -0.1:
            log(f"  警告: 符号化直前のピークが {peak:+.2f} dBFS です。"
                f"ゲイン {gain_db:+.1f} dB では足りていません(潰れている可能性)")
        else:
            log(f"  符号化直前のピーク {peak:+.2f} dBFS(天井まで {-peak:.1f} dB)")
    size_mb = dst.stat().st_size / 2**20
    dur = probe_audio(dst)["duration"]
    log(f"  {fmt_time(dur)} / {size_mb:.0f} MiB")
    return {"path": str(dst), "gain_db": gain_db, "bitrate": bitrate,
            "pre_encode_peak_db": peak, "size_mb": round(size_mb, 1),
            "duration": round(dur, 3)}


# --- プロキシからのブロック切り出し ------------------------------------------

def _clean_label(label: str) -> str:
    from .apply import _clean_label as clean

    return clean(label)


def run_field_export(
    proxy: Path,
    confirmed: Path,
    outdir: Path,
    cfg,
    date: str,
    variant: str = "",
    proxy_gain_db: float = PROXY_GAIN_DB,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
    ref_margin: float = 120.0,
    comp_ratio: float = DEFAULT_COMP_RATIO,
    comp_threshold_offset: float = DEFAULT_COMP_THRESHOLD_OFFSET,
    comp_attack_ms: float = DEFAULT_COMP_ATTACK_MS,
    comp_release_ms: float = DEFAULT_COMP_RELEASE_MS,
    comp_knee_db: float = DEFAULT_COMP_KNEE_DB,
    parallel_db: float = 0.0,
    noise_ceiling_db: float | None = None,
    bitrate: str = PROXY_BITRATE,
    force: bool = False,
) -> list[dict]:
    """プロキシ 1 本から、確定境界に従って配布用 MP3 を書き出す。

    ブロックごとに `mix.run_mix` と同じマスターチェーン(ゲイン -> パラレルコンプ
    -> コンプ -> トゥルーピークリミッター)を通す。ゲインの求め方も
    `loudness.solve_gain` を共有しているので、原本から作った WAV と同じ目標
    ラウドネスに乗る。

    **パラレルコンプの持ち上げ量も `mix` と同じく暗騒音から抑える。** 以前は
    `master_chain` を持ち上げ量を渡さずに呼んでいたため、既定の +17 dB が
    黙って使われ、`limit_parallel_makeup` を一度も通らなかった。260905 は
    空調の大きい会場で、速報版の指揮者の声の場面で「サー」が目立った。
    暗騒音はブロックごとに 9.4 dB 違うことがあるので、日ごとではなく
    ブロックごとに測って決める。

    暗騒音はプロキシのゲインを戻してから、そのブロックの範囲だけを測る。
    プロキシは 1 本に全ブロックが入っているので、ファイル全体を測ると
    休憩や片付けの区間を巻き込んでしまう。
    """
    from .apply import load_confirmed
    from .export import block_titles, write_tags, year_from_date
    from .normalize import _reference_window

    info = probe_audio(proxy)
    total, sr = info["duration"], info["sample_rate"]
    keeps = load_confirmed(confirmed, total)
    if not keeps:
        raise PipelineError(f"{confirmed.name} に keep 区間がありません")

    titles = block_titles(len(keeps))
    export_dir = outdir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tail = f"_{variant}" if variant else ""
    year = year_from_date(date)
    threshold_db = target_lufs + comp_threshold_offset
    undo = f"volume={-proxy_gain_db:.6f}dB"

    log(f"現場プロキシから書き出し: {len(keeps)} ブロック / {proxy.name}")
    log(f"  プロキシのゲイン {proxy_gain_db:+.1f} dB を戻してから測定します")
    log(f"  目標 {target_lufs:+.1f} LUFS / 天井 {true_peak_db:+.1f} dBTP / "
        f"コンプ {comp_ratio:g}:1 しきい値 {threshold_db:+.1f} dBFS")
    ceiling = (noise_ceiling_db if noise_ceiling_db is not None
               else target_lufs - NOISE_FLOOR_BELOW_TARGET_DB)
    if parallel_db > 0:
        log(f"  パラレルコンプ: makeup 上限 {parallel_db:+.0f} dB "
            f"(しきい値 {PARALLEL_THRESHOLD_DB:+.0f} dBFS / 小さい音だけを持ち上げる)")
        log(f"    暗騒音の天井 {ceiling:+.0f} dBFS を超えないところまで"
            f"ブロックごとに抑えます")

    def chain(gain_db: float, *, oversample: int = LIMITER_OVERSAMPLE,
              with_limiter: bool = True, makeup_db: float = 0.0) -> str:
        return master_chain(
            gain_db, threshold_db=threshold_db, true_peak_db=true_peak_db,
            ratio=comp_ratio, attack_ms=comp_attack_ms, release_ms=comp_release_ms,
            knee_db=comp_knee_db, sample_rate=sr, oversample=oversample,
            with_limiter=with_limiter, parallel_db=makeup_db,
        )

    rows: list[dict] = []
    for i, (keep, title) in enumerate(zip(keeps, titles), start=1):
        start, end = keep["start"], keep["end"]
        dur = end - start
        dst = export_dir / f"{date}_{title}{tail}.mp3"
        label = _clean_label(keep.get("label", title))
        log(f"  [{i}/{len(keeps)}] {dst.name}  {fmt_time(start)}-{fmt_time(end)} "
            f"({fmt_time(dur)}, {label})")
        if dst.exists() and not force:
            log("    スキップ(既存)")
            continue

        # 測定範囲はブロックの内側。頭と尻の音出し・雑談を外すのは mix と同じ。
        off, span, window_desc = _reference_window(dur, ref_margin)
        m_start, m_dur = start + off, span

        pre = measure(proxy, m_start, m_dur, pre_filter=undo)
        gain = target_lufs - pre.integrated
        log(f"    素の値 {pre.describe()}  基準={window_desc} -> 暫定ゲイン {gain:+.2f} dB")

        # 暗騒音を測る(mix と同じ)。パラレルコンプを使わなくても毎回測って見せる。
        noise = noise_report(proxy, start=start, dur=dur, pre_filter=undo)
        log(f"    {noise.describe()}")
        for line in noise.hint():
            log(f"      {line}")

        makeup = parallel_db
        if parallel_db > 0:
            makeup, why = limit_parallel_makeup(noise.floor_db, gain, parallel_db, ceiling)
            log(f"    {why}")

        def after_master(g: float):
            return measure(proxy, m_start, m_dur,
                           pre_filter=f"{undo},{chain(g, oversample=1, makeup_db=makeup)}")

        def report(g: float, res, resid: float) -> None:
            log(f"    マスター通過後 {res.integrated:+.1f} LUFS (残差 {resid:+.2f} LU)")
            if abs(resid) > 0.15:
                log(f"    ゲインを {g + resid:+.2f} dB に補正")

        gain, _after_comp = solve_gain(after_master, target_lufs, gain, on_step=report)

        # パラレルコンプが入るとラベル付きのグラフになるので、`-af` では扱えない。
        filt = f"{undo},{chain(gain, makeup_db=makeup)}"
        graph = (["-filter_complex", f"[0:a]{filt}[out]", "-map", "[out]"]
                 if ";" in filt else ["-af", filt])
        run(
            [FFMPEG, "-hide_banner", "-v", "error", "-stats", "-y",
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(proxy),
             *graph,
             "-c:a", "libmp3lame", "-b:a", bitrate,
             "-map_metadata", "-1", str(dst)],
            desc="    書き出し中...",
        )
        write_tags(
            dst, album=date,
            title=f"{title}({variant})" if variant else title,
            artist=cfg.orchestra, album_artist=cfg.orchestra,
            track=i, total=len(keeps), year=year,
        )
        after = measure(dst, off, span)
        whole = measure(dst)
        log(f"    結果 {after.describe()} / 全体では {whole.describe()}")
        rows.append({
            "block": title, "start": round(start, 3), "end": round(end, 3),
            "path": str(dst), "gain_db": round(gain, 2),
            "noise": noise.to_json(),
            "noise_floor_db": (None if noise.floor_db == float("-inf")
                               else round(noise.floor_db, 2)),
            "parallel_makeup_db": round(makeup, 2),
            "window": window_desc, "before": pre.to_json(),
            "after": after.to_json(), "after_whole": whole.to_json(),
            "size_mb": round(dst.stat().st_size / 2**20, 1),
        })
    return rows


# --- 母艦がプロキシを受け取る ------------------------------------------------

def receive_proxy(
    src: Path,
    outdir: Path,
    date: str,
    group: str = "ext",
    tracks: tuple[str, ...] = ("Tr1", "Tr2"),
    recorder: str = "zoom-m4",
    move: bool = False,
) -> dict:
    """届いたプロキシを `output/{date}/` に置き、下流が動く `ingest.json` を書く。

    現場経路では母艦に TAKE フォルダがまだ無い。`propose` は `ingest.json` から
    全長と系統定義を読むので、**プロキシ 1 本を 1 TAKE と見なした最小の
    マニフェスト**をここで作る。帰宅後に本物の `ingest` を走らせると上書きされ、
    そちらが正になる。
    """
    import shutil

    from .util import write_json

    if not src.exists():
        raise PipelineError(f"{src} がありません")
    outdir.mkdir(parents=True, exist_ok=True)
    dst = proxy_path(outdir, group)
    if src.resolve() != dst.resolve():
        (shutil.move if move else shutil.copyfile)(str(src), str(dst))
    info = probe_audio(dst)
    total, sr = info["duration"], info["sample_rate"]

    manifest = {
        "date": date,
        "root": str(outdir.parent.parent),
        "recorder": recorder,
        "channel_groups": {group: list(tracks)},
        "n_takes": 1,
        "sample_rate": sr,
        "total_duration": total,
        "take_offsets": [{"number": 1, "start": 0.0, "duration": total}],
        "takes": [{
            "date": date, "number": 1, "dir": None,
            "files": {t: str(dst) for t in tracks},
            "sample_rate": sr, "duration": total,
        }],
        "field_proxy": {"source": str(src), "path": str(dst),
                        "gain_db": PROXY_GAIN_DB},
    }
    write_json(outdir / "ingest.json", manifest)
    log(f"プロキシを受け取りました: {dst.name}  {fmt_time(total)} / {sr} Hz / "
        f"{dst.stat().st_size / 2**20:.0f} MiB")
    log(f"  ingest.json を書きました(1 TAKE 相当、系統 {group}={'+'.join(tracks)})")
    return manifest


# --- 現場(a-Shell)で流すスクリプト ------------------------------------------

# 機種ごとに変わるのは「何をつなぐか」と「何をコピーするか」だけ。
# ここが実際の操作と食い違うと現場で迷うので、プロファイル名から引く。
RECORDER_NOTES = {
    "zoom-m4": (
        "ZOOM M4 を File Transfer モードにし、USB-C で iPhone につなぐ\n"
        "#      (M4 は必ず**電池で駆動**すること。USB からの給電では足りない)",
        "{date}_*.TAKE をフォルダごと",
    ),
    "zoom-f3": (
        "ZOOM F3 を USB-C で iPhone につなぎ、カードリーダとして認識させる",
        "{date}_*.WAV を(F3 はフォルダを作らないのでファイルを直接)",
    ),
    "single-file": ("音源のあるところへ移動する", "{date} の音源を"),
}


FIELD_SCRIPT_TEMPLATE = """#!/bin/sh
# {date} の練習録音を配布用プロキシ 1 本にまとめる(iPhone / a-Shell 用)。
#
#   1. {connect}
#   2. microSD の {copy} iPhone にコピーする
#   3. このスクリプトのある場所で `sh {name}` を実行する
#
# 出力は {out} 一本({bitrate} MP3)。これだけを母艦へ送れば、境界提案から
# 配布まで進む。32bit float の原本は消さずに持ち帰ること
# (Drive 用の WAV は帰宅後に原本から作る)。
#
# a-Shell のシェルは素朴なので、変数・行継続・set -e・&& を使っていない。
# ffmpeg の行が長いのはそのためである。

{inputs_comment}
ffmpeg -hide_banner -y{inputs} -filter_complex "{filt}" -map "[out]" -c:a libmp3lame -b:a {bitrate} -map_metadata -1 "{out}"

ls -lh "{out}"

# ログの max_volume が -0.1 dB 以上なら、固定ゲイン {gain:+.1f} dB では足りていない。
# その日は帰宅後に原本から作り直すこと。
"""


def field_script(
    date: str,
    files: list[str],
    n_takes: int,
    n_tracks: int,
    out: str | None = None,
    lr_map: str = "normal",
    gain_db: float = PROXY_GAIN_DB,
    bitrate: str = PROXY_BITRATE,
    name: str = "field_master.sh",
    recorder: str = "zoom-m4",
) -> str:
    """現場で流す sh スクリプトの中身。母艦の実装と同じフィルタ列を埋め込む。

    `files` は TAKE ごとにトラック順に並べた相対パス(merge と同じ並び)。
    """
    inputs = "".join(f' -i "{f}"' for f in files)
    inputs_comment = "# 入力: " + ", ".join(files)
    connect, copy = RECORDER_NOTES.get(recorder, RECORDER_NOTES["zoom-m4"])
    return FIELD_SCRIPT_TEMPLATE.format(
        date=date, name=name, inputs=inputs, inputs_comment=inputs_comment,
        connect=connect.format(date=date), copy=copy.format(date=date),
        out=out or f"{date}_proxy.mp3",
        filt=proxy_filter(n_takes, n_tracks, lr_map, gain_db),
        bitrate=bitrate, gain=gain_db,
    )


# --- 届くのを待つ ------------------------------------------------------------
# 転送方式は iCloud Drive でも scp でも「所定のフォルダにファイルが現れる」点は
# 同じなので、フォルダを見張る形にしておけばどちらでも乗る。方式が決まるまでの
# あいだも、これだけで現場経路は通る。
WATCH_POLL_S = 5.0
WATCH_STABLE_S = 15.0

# 受け口は Google Drive にした。iPhone の Drive アプリからこのフォルダへ
# アップロードすれば、母艦が API で拾う。iCloud Drive やローカルフォルダを
# 使いたい場合は --via dir に切り替える(下の `ensure_inbox` 以降)。
#
# Drive を選んだ理由は二つある。配布経路で既に使っていて資格情報が生きていること、
# そして iCloud Drive と違って「Mac 側の同期が有効かどうか」に依存しないこと。
DRIVE_INBOX_NAME = "inbox"

ICLOUD_ROOT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
INBOX_NAME = "orchestra-recording-pipeline/inbox"


def default_inbox() -> Path:
    return ICLOUD_ROOT / "orchestra-recording-pipeline" / "inbox"


def ensure_inbox(path: Path | None = None) -> Path:
    """ローカル受け口のフォルダを用意する。iCloud Drive が無効なら理由を添えて止める。"""
    dst = path or default_inbox()
    if path is None and not ICLOUD_ROOT.is_dir():
        raise PipelineError(
            f"iCloud Drive が見つかりません({ICLOUD_ROOT})。\n"
            "  Mac の システム設定 → Apple アカウント → iCloud → iCloud Drive を"
            "オンにしてください。\n"
            "  Google Drive を使う場合は --via drive、別の場所なら --dir を指定してください。"
        )
    dst.mkdir(parents=True, exist_ok=True)
    return dst


# --- 受け口: Google Drive ----------------------------------------------------

def ensure_drive_inbox(client) -> dict:
    """Drive 側の受け口フォルダを用意して返す。

    配布に使っているルートフォルダ(`orchestra-recording-pipeline`)の直下に
    `inbox` を作る。日付フォルダと並ぶので、iPhone からも迷わず選べる。
    """
    from .gdrive_client import ROOT_FOLDER_NAME
    from .gdrive_upload import ensure_root_folder

    parent, _, _ = ensure_root_folder(client)
    folder, created = client.ensure_folder(DRIVE_INBOX_NAME, parent["id"])
    if created:
        log(f"Drive に受け口フォルダを作りました: "
            f"{ROOT_FOLDER_NAME}/{DRIVE_INBOX_NAME}")
    return folder


def _drive_matches(client, folder_id: str, pattern: str, since: float | None) -> list[dict]:
    """受け口フォルダの中から、パターンに合う音声ファイルを新しい順に返す。"""
    import datetime
    import fnmatch

    out = []
    for f in client.list_children(folder_id, fields_extra="modifiedTime"):
        if f.get("mimeType", "").endswith("folder"):
            continue
        if not fnmatch.fnmatch(f["name"], pattern):
            continue
        mtime = None
        raw = f.get("modifiedTime")
        if raw:
            try:
                mtime = datetime.datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                mtime = None
        if since is not None and mtime is not None and mtime < since:
            continue
        f["_mtime"] = mtime or 0.0
        out.append(f)
    return sorted(out, key=lambda x: x["_mtime"], reverse=True)


def wait_for_proxy_drive(
    client,
    folder_id: str,
    dst_dir: Path,
    pattern: str = "*.mp3",
    since: float | None = None,
    poll_s: float = None,
    stable_s: float = None,
    timeout_s: float | None = None,
) -> Path:
    """Drive の受け口にプロキシが現れるのを待ち、落として返す。

    ローカルフォルダ版と同じく、**サイズが `stable_s` 秒変わらないこと**を
    確認してから落とす。Drive のアップロードは完了時にしか一覧へ出ないのが
    普通だが、大きいファイルで途中の状態が見えることがあるための保険である。
    """
    import time

    poll_s = WATCH_POLL_S if poll_s is None else poll_s
    stable_s = WATCH_STABLE_S if stable_s is None else stable_s
    started = time.time()
    since = started if since is None else since
    log(f"Drive の受け口を見張ります(pattern={pattern} / {poll_s:.0f} 秒ごと / "
        f"{stable_s:.0f} 秒サイズが変わらなければ受け取り)")

    seen: dict[str, tuple[int, float]] = {}
    while True:
        for f in _drive_matches(client, folder_id, pattern, since):
            size = int(f.get("size") or 0)
            prev = seen.get(f["id"])
            if prev is None:
                log(f"  見つけました: {f['name']}  {size / 2**20:.0f} MiB"
                    f"(アップロード完了を待ちます)")
                seen[f["id"]] = (size, time.time())
            elif size != prev[0]:
                seen[f["id"]] = (size, time.time())
            elif time.time() - prev[1] >= stable_s:
                dst = dst_dir / f["name"]
                log(f"  受け取り: {f['name']}  {size / 2**20:.0f} MiB → {dst}")
                last = [-1]

                def _progress(frac: float) -> None:
                    pct = int(frac * 100)
                    if pct >= last[0] + 10:
                        last[0] = pct
                        log(f"    ダウンロード {pct}%")

                client.download(f["id"], dst, on_progress=_progress)
                return dst
        if timeout_s is not None and time.time() - started > timeout_s:
            raise PipelineError(
                f"{timeout_s:.0f} 秒待ちましたが Drive の受け口に "
                f"{pattern} が現れませんでした"
            )
        time.sleep(poll_s)


def wait_for_proxy(
    watch_dir: Path,
    pattern: str = "*.mp3",
    since: float | None = None,
    poll_s: float = WATCH_POLL_S,
    stable_s: float = WATCH_STABLE_S,
    timeout_s: float | None = None,
) -> Path:
    """`watch_dir` に現れるプロキシを待つ。書き込み完了まで待ってから返す。

    転送中のファイルを掴まないよう、**サイズが `stable_s` 秒変わらないこと**を
    確認してから返す。`since` より新しいものだけを対象にする(既定は呼び出し時刻)。
    """
    import time

    if not watch_dir.is_dir():
        raise PipelineError(f"{watch_dir} がありません")
    started = time.time()
    since = started if since is None else since
    log(f"{watch_dir} を見張ります(pattern={pattern} / {poll_s:.0f} 秒ごと / "
        f"{stable_s:.0f} 秒サイズが変わらなければ受け取り)")

    seen: dict[Path, tuple[int, float]] = {}
    while True:
        for p in sorted(watch_dir.glob(pattern)):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime < since:
                continue
            prev = seen.get(p)
            if prev is None:
                log(f"  見つけました: {p.name}  {st.st_size / 2**20:.0f} MiB(書き込み完了を待ちます)")
                seen[p] = (st.st_size, time.time())
            elif st.st_size != prev[0]:
                seen[p] = (st.st_size, time.time())
            elif time.time() - prev[1] >= stable_s:
                log(f"  受け取り: {p.name}  {st.st_size / 2**20:.0f} MiB")
                return p
        if timeout_s is not None and time.time() - started > timeout_s:
            raise PipelineError(
                f"{timeout_s:.0f} 秒待ちましたが {watch_dir} に {pattern} が現れませんでした"
            )
        time.sleep(poll_s)
