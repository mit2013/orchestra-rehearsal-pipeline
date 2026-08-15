"""Google Drive へのアップロードと共有リンク発行。

動画担当への素材受け渡しなので、Box(団員向け MP3)とは性格が異なる:

- 対象は WAV(`trimmed/*_final.wav`)。**ローカルにコピーを作らず直接アップロードする**。
- 共有は「リンクを知っている全員が閲覧者」。**ダウンロード禁止は付けない**
  (取得して編集に使うため)。

フォルダは ルート/練習録音/{date} の2階層。どちらも同名があれば再利用する。
"""

from __future__ import annotations

import time
from pathlib import Path

from .export import block_titles, find_final_files
from .gdrive_client import ROOT_FOLDER_NAME, DriveClient
from .util import PipelineError, log

DRIVE_ROOT = "root"


def plan_uploads(outdir: Path, date: str) -> list[tuple[Path, str]]:
    """(ローカルの _final.wav, Drive 上の表示名) の一覧。タイトル採番はフェーズ2と同じ。"""
    trimmed = outdir / "trimmed"
    finals = find_final_files(trimmed)
    if not finals:
        raise PipelineError(
            f"{trimmed} に *_final.wav がありません。"
            "先に `normalize` と `mix` を実行してください。"
        )
    titles = block_titles(len(finals))
    return [(src, f"{date}_{title}.wav") for src, title in zip(finals, titles)]


def run_gdrive_upload(
    root: Path,
    date: str,
    outdir: Path,
    auth_timeout: float | None = None,
) -> dict:
    plan = plan_uploads(outdir, date)
    total_bytes = sum(p.stat().st_size for p, _ in plan)
    log(
        f"Google Drive アップロード開始: {len(plan)} ファイル / "
        f"合計 {total_bytes/2**30:.2f} GiB"
    )

    client = DriveClient.connect(root, auth_timeout)

    parent, created_root = client.ensure_folder(ROOT_FOLDER_NAME, DRIVE_ROOT)
    log(f"フォルダ「{ROOT_FOLDER_NAME}」{'を作成' if created_root else 'を再利用'} (id={parent['id']})")

    folder, created_date = client.ensure_folder(date, parent["id"])
    folder_id = folder["id"]
    log(f"フォルダ「{date}」{'を作成' if created_date else 'を再利用'} (id={folder_id})")

    uploaded = []
    for i, (src, drive_name) in enumerate(plan, start=1):
        size = src.stat().st_size
        log(f"  [{i}/{len(plan)}] {drive_name}  ({size/2**30:.2f} GiB)  <- {src.name}")
        started = time.time()
        last = [0.0]

        def on_progress(frac: float, _last=last, _size=size, _started=started):
            if frac - _last[0] >= 0.1 or frac >= 1.0:
                _last[0] = frac
                el = time.time() - _started
                rate = (_size * frac / 2**20) / el if el > 0 else 0
                log(f"      {frac*100:.0f}%  ({rate:.0f} MiB/s)")

        info = client.upload(src, drive_name, folder_id, on_progress)
        log(f"      完了: id={info['id']}"
            + ("(既存を新バージョンで置き換え)" if info.get("_replaced") else "(新規)"))
        uploaded.append(info)

    # --- 共有(フォルダ単位) ---------------------------------------------
    log("共有を設定します(リンクを知っている全員が閲覧者 / ダウンロード制限なし)")
    client.share_anyone_reader(folder_id)

    state = client.get_share_state(folder_id)
    children = client.list_children(folder_id)
    n_files = sum(1 for c in children if c.get("mimeType") != "application/vnd.google-apps.folder")
    anyone = [p for p in state.get("permissions", []) if p.get("type") == "anyone"]

    return {
        "root_folder_id": parent["id"],
        "root_created": created_root,
        "folder_id": folder_id,
        "folder_created": created_date,
        "url": state.get("webViewLink"),
        "n_uploaded": len(plan),
        "n_in_folder": n_files,
        "anyone_permission": anyone[0] if anyone else None,
        "copy_requires_writer_permission": state.get("copyRequiresWriterPermission"),
        "capabilities": state.get("capabilities", {}),
        "names": [name for _, name in plan],
    }
