"""Google Drive へのアップロードと共有リンク発行。

フォルダ構成は3階層:

    練習録音/{date}/WAV/   動画担当向け。trimmed/*_final.wav を直接アップロード
    練習録音/{date}/MP3/   団員の一部へ個別共有する用。export/*.mp3 をアップロード

共有リンクは2つ設定する({date} フォルダと {date}/MP3 フォルダ)。どちらも
「リンクを知っている全員が閲覧者」で、Box と違い**ダウンロード禁止は付けない**。
親から子への継承に頼らず、フォルダごとに明示的に設定する。

Box のようなパスワード保護は行わない(Drive の共有機能にその概念がないため)。
"""

from __future__ import annotations

import time
from pathlib import Path

from .export import block_titles, find_final_files, find_mp3_files
from .gdrive_client import FOLDER_MIME, ROOT_FOLDER_NAME, DriveClient
from .util import PipelineError, log

DRIVE_ROOT = "root"
WAV_FOLDER = "WAV"
MP3_FOLDER = "MP3"

# 拡張子から、3階層構成でのあるべき置き場所を決める(移行処理で使う)。
SUBFOLDER_BY_EXT = {".wav": WAV_FOLDER, ".mp3": MP3_FOLDER}


def plan_wav(outdir: Path, date: str) -> list[tuple[Path, str]]:
    """(ローカルの _final.wav, Drive 上の表示名)。タイトル採番はフェーズ2と同じ。"""
    finals = find_final_files(outdir / "trimmed")
    if not finals:
        raise PipelineError(
            f"{outdir/'trimmed'} に *_final.wav がありません。"
            "先に `normalize` と `mix` を実行してください。"
        )
    titles = block_titles(len(finals))
    return [(src, f"{date}_{title}.wav") for src, title in zip(finals, titles)]


def plan_mp3(outdir: Path) -> list[tuple[Path, str]]:
    """MP3 は export/ のファイル名をそのまま使う(既に {date}_{タイトル}.mp3 形式)。"""
    return [(p, p.name) for p in find_mp3_files(outdir)]


def migrate_flat_files(
    client: DriveClient, date_folder_id: str, subfolders: dict[str, str]
) -> list[tuple[str, str]]:
    """3階層化以前に `{date}/` 直下へ上げたファイルを、WAV/ や MP3/ へ移動する。

    再アップロードせず親を付け替えるだけなので、重複は生じない。移動先に同名が
    既にある場合は移動せず警告する(取り違えて上書きしないため)。
    """
    moved: list[tuple[str, str]] = []
    for child in client.list_children(date_folder_id):
        if child.get("mimeType") == FOLDER_MIME:
            continue
        ext = Path(child["name"]).suffix.lower()
        target_name = SUBFOLDER_BY_EXT.get(ext)
        if target_name is None:
            log(f"  移行対象外(未知の拡張子): {child['name']}")
            continue
        target_id = subfolders[target_name]
        if client.find_child(child["name"], target_id, folder=False):
            log(
                f"  警告: {child['name']} は既に {target_name}/ にも存在します。"
                "取り違えを避けるため移動しません。Drive 上で手動確認してください。"
            )
            continue
        client.move_file(child["id"], target_id, date_folder_id)
        log(f"  移行: {child['name']}  {date_folder_id} 直下 -> {target_name}/")
        moved.append((child["name"], target_name))
    return moved


def _upload_group(
    client: DriveClient, plan: list[tuple[Path, str]], folder_id: str, label: str
) -> list[dict]:
    out = []
    for i, (src, drive_name) in enumerate(plan, start=1):
        size = src.stat().st_size
        log(f"  [{label} {i}/{len(plan)}] {drive_name}  ({size/2**20:.0f} MiB)  <- {src.name}")
        started = time.time()
        last = [0.0]

        def on_progress(frac: float, _last=last, _size=size, _started=started):
            if frac - _last[0] >= 0.2 or frac >= 1.0:
                _last[0] = frac
                el = time.time() - _started
                rate = (_size * frac / 2**20) / el if el > 0 else 0
                log(f"      {frac*100:.0f}%  ({rate:.0f} MiB/s)")

        info = client.upload(src, drive_name, folder_id, on_progress)
        log(f"      完了: id={info['id']}"
            + ("(既存を新バージョンで置き換え)" if info.get("_replaced") else "(新規)"))
        out.append(info)
    return out


def _share_state(client: DriveClient, folder_id: str) -> dict:
    client.share_anyone_reader(folder_id)
    st = client.get_share_state(folder_id)
    anyone = [p for p in st.get("permissions", []) if p.get("type") == "anyone"]
    return {
        "folder_id": folder_id,
        "url": st.get("webViewLink"),
        "anyone_permission": anyone[0] if anyone else None,
        "copy_requires_writer_permission": st.get("copyRequiresWriterPermission"),
        "can_download": st.get("capabilities", {}).get("canDownload"),
    }


def run_gdrive_upload(
    root: Path,
    date: str,
    outdir: Path,
    auth_timeout: float | None = None,
) -> dict:
    wav_plan = plan_wav(outdir, date)
    mp3_plan = plan_mp3(outdir)
    total = sum(p.stat().st_size for p, _ in wav_plan + mp3_plan)
    log(
        f"Google Drive アップロード開始: WAV {len(wav_plan)} / MP3 {len(mp3_plan)} / "
        f"合計 {total/2**30:.2f} GiB"
    )

    client = DriveClient.connect(root, auth_timeout)

    # --- フォルダ(3階層。いずれも同名があれば再利用)-----------------------
    parent, c_root = client.ensure_folder(ROOT_FOLDER_NAME, DRIVE_ROOT)
    log(f"フォルダ「{ROOT_FOLDER_NAME}」{'を作成' if c_root else 'を再利用'} (id={parent['id']})")
    date_folder, c_date = client.ensure_folder(date, parent["id"])
    log(f"フォルダ「{date}」{'を作成' if c_date else 'を再利用'} (id={date_folder['id']})")

    subfolders: dict[str, str] = {}
    created_sub: dict[str, bool] = {}
    for name in (WAV_FOLDER, MP3_FOLDER):
        f, created = client.ensure_folder(name, date_folder["id"])
        subfolders[name] = f["id"]
        created_sub[name] = created
        log(f"フォルダ「{date}/{name}」{'を作成' if created else 'を再利用'} (id={f['id']})")

    # --- 移行(アップロードより先に行う)------------------------------------
    # 先に移動しておけば、この後のアップロードは移動済みファイルを「同名あり」と
    # 見つけて新バージョンとして扱うので、重複が生じない。
    log("3階層化以前のフラット配置がないか確認します")
    moved = migrate_flat_files(client, date_folder["id"], subfolders)
    log(f"  移行したファイル: {len(moved)} 件")

    # --- アップロード -------------------------------------------------------
    _upload_group(client, wav_plan, subfolders[WAV_FOLDER], "WAV")
    _upload_group(client, mp3_plan, subfolders[MP3_FOLDER], "MP3")

    # --- 共有(2つのフォルダに個別設定)-------------------------------------
    log("共有を設定します(リンクを知っている全員が閲覧者 / ダウンロード制限なし)")
    share_date = _share_state(client, date_folder["id"])
    share_mp3 = _share_state(client, subfolders[MP3_FOLDER])

    def count(folder_id: str) -> int:
        return sum(
            1 for c in client.list_children(folder_id) if c.get("mimeType") != FOLDER_MIME
        )

    return {
        "root_folder_id": parent["id"],
        "date_folder_id": date_folder["id"],
        "wav_folder_id": subfolders[WAV_FOLDER],
        "mp3_folder_id": subfolders[MP3_FOLDER],
        "created": {"root": c_root, "date": c_date, **created_sub},
        "migrated": moved,
        "n_wav": len(wav_plan),
        "n_mp3": len(mp3_plan),
        "n_in_wav": count(subfolders[WAV_FOLDER]),
        "n_in_mp3": count(subfolders[MP3_FOLDER]),
        "n_loose_in_date": count(date_folder["id"]),
        "share_date": share_date,
        "share_mp3": share_mp3,
        "wav_names": [n for _, n in wav_plan],
        "mp3_names": [n for _, n in mp3_plan],
    }
