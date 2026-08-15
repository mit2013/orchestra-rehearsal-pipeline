"""Box へのアップロードと共有リンク発行。

`output/{date}/export/*.mp3` を Box の日付フォルダへ上げ、フォルダ単位で
パスワード保護・ダウンロード不可の共有リンクを設定する。WAV は対象外。

再実行しても重複しないよう、同名フォルダは再利用する。同名ファイルについては
Box が持つ `sha1` とローカルの SHA-1 を比べ、内容が同じならアップロードを
行わない。異なる場合のみ新しいバージョンとして上げる。
"""

from __future__ import annotations

import re
from pathlib import Path

from .box_client import BoxClient
from .config import SessionConfig
from .export import find_mp3_files
from .util import PipelineError, file_digest, log

PASSWORD_SUFFIX = "{password_suffix}"
ROOT_FOLDER_ID = "0"


def build_password(concert_date: str) -> str:
    """パスワードは `{concert_date}{password_suffix}`(例: {concert_date}{password_suffix})。"""
    if not concert_date:
        raise PipelineError(
            "concert_date が空です。パスワードを生成できないため中止します。\n"
            "  session_config.json(またはプロジェクト直下の pipeline_defaults.json)の "
            "concert_date に演奏会本番日を yyyymmdd 形式で設定してください。"
        )
    if not re.fullmatch(r"\d{8}", concert_date):
        raise PipelineError(
            f"concert_date は yyyymmdd 形式の8桁である必要があります(実際: {concert_date!r})"
        )
    return f"{concert_date}{PASSWORD_SUFFIX}"


def resolve_parent_folder_id(raw: str) -> str:
    """親フォルダID。空ならルート("0")。Box のフォルダIDは数字のみ。"""
    value = (raw or "").strip()
    if not value:
        return ROOT_FOLDER_ID
    if not value.isdigit():
        raise PipelineError(
            f"box_parent_folder_id は Box のフォルダID(数字)である必要があります"
            f"(実際: {value!r})。\n"
            "  フォルダ名ではなくIDです。Box でそのフォルダを開いたときの URL 末尾の数字"
            "(例: https://app.box.com/folder/123456789 なら 123456789)を指定してください。\n"
            "  ルート直下に置く場合は空文字にしてください。"
        )
    return value


def mp3_files(outdir: Path) -> list[Path]:
    """アップロード対象の MP3。Drive 側と同じ実装を使う。"""
    return find_mp3_files(outdir)


def run_box_upload(
    root: Path,
    date: str,
    outdir: Path,
    cfg: SessionConfig,
    auth_timeout: float = 300.0,
) -> dict:
    # パスワードと親フォルダは、通信を始める前に検証しておく。
    password = build_password(cfg.concert_date)
    parent_id = resolve_parent_folder_id(cfg.box_parent_folder_id)
    files = mp3_files(outdir)

    log(f"Box アップロード開始: {len(files)} ファイル / フォルダ名={date} / 親={parent_id}")
    client = BoxClient.connect(root, auth_timeout)

    folder, created = client.ensure_folder(parent_id, date)
    folder_id = folder["id"]
    log(f"フォルダ {'を作成' if created else 'を再利用'}: {folder['name']} (id={folder_id})")

    existing = {it["name"]: it for it in client.list_folder_items(folder_id) if it["type"] == "file"}

    uploaded, skipped = [], []
    for i, p in enumerate(files, start=1):
        size_mb = p.stat().st_size / 2**20
        prev = existing.get(p.name)

        # 内容が同じなら送らない。Box が持つ sha1 とローカルの SHA-1 を比べる。
        if prev and prev.get("sha1"):
            digest = file_digest(p, "sha1")
            if digest == prev["sha1"]:
                log(f"  [{i}/{len(files)}] スキップ(内容同一): {p.name}  sha1={digest[:12]}…")
                skipped.append(p.name)
                continue
            log(f"  [{i}/{len(files)}] {p.name}  内容が変化 "
                f"(ローカル {digest[:12]}… / Box {prev['sha1'][:12]}…)")

        mode = f"新バージョン (既存 id={prev['id']})" if prev else "新規"
        log(f"  [{i}/{len(files)}] {p.name}  {size_mb:.0f} MiB  {mode}")
        info = client.upload(p, folder_id, prev["id"] if prev else None)
        log(f"      完了: id={info['id']}")
        uploaded.append(p.name)

    # --- 共有リンク(フォルダ単位) ---------------------------------------
    log("共有リンクを設定します(パスワード保護 / ダウンロード不可 / リンクを知っている人のみ)")
    client.set_folder_shared_link(folder_id, password, can_download=False, access="open")

    # 設定が本当に反映されたかを読み戻して確認する。
    got = client.get_folder_shared_link(folder_id)
    link = got.get("shared_link") or {}
    items = client.list_folder_items(folder_id)
    n_files = sum(1 for it in items if it["type"] == "file")

    return {
        "folder_id": folder_id,
        "folder_name": folder["name"],
        "created": created,
        "url": link.get("url"),
        "password_enabled": link.get("is_password_enabled"),
        "can_download": (link.get("permissions") or {}).get("can_download"),
        "access": link.get("access"),
        "effective_access": link.get("effective_access"),
        "n_uploaded": len(uploaded),
        "n_skipped": len(skipped),
        "uploaded": uploaded,
        "skipped": skipped,
        "n_target": len(files),
        "n_in_folder": n_files,
        "password": password,
        "shared_link_raw": link,
    }
