"""Google Drive へのアップロードと共有リンク発行。

フォルダ構成は4階層(オーケストラ単位で分離):

    orchestra-recording-pipeline/   自動化専用のルート(旧名: 練習録音)
      {orchestra}/                  session_config の orchestra から自動生成
        {date}/
          WAV/   動画担当向け。trimmed/*_final.wav を直接アップロード
          MP3/   団員へ個別共有する用。export/*.mp3 をアップロード

共有リンクは2つ設定する({date} フォルダと {date}/MP3 フォルダ)。どちらも
「リンクを知っている全員が閲覧者」で、Box と違い**ダウンロード禁止は付けない**。
親から子への継承に頼らず、フォルダごとに明示的に設定する。

帯域とストレージを無駄にしないよう、Drive 上の `md5Checksum` とローカルの MD5 を
比較し、内容が同じならアップロードそのものを行わない(新バージョンも作らない)。
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from .config import SessionConfig
from .export import block_titles, find_final_files, find_mp3_files
from .gdrive_client import (
    FOLDER_MIME,
    LEGACY_ROOT_NAME,
    ROOT_FOLDER_NAME,
    DriveClient,
)
from .util import PipelineError, log

DRIVE_ROOT = "root"
WAV_FOLDER = "WAV"
MP3_FOLDER = "MP3"

# 拡張子から、あるべき置き場所を決める(フラット配置の移行で使う)。
SUBFOLDER_BY_EXT = {".wav": WAV_FOLDER, ".mp3": MP3_FOLDER}

DATE_DIR_RE = re.compile(r"^\d{6}$")


def orchestra_folder_name(orchestra: str) -> str:
    """団体名からフォルダ名を作る。表記揺れを防ぐため必ずここで自動生成する。"""
    name = (orchestra or "").strip()
    if not name:
        raise PipelineError(
            "orchestra が空です。session_config.json の orchestra を設定してください。"
        )
    return name.replace(" ", "_")


def local_md5(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


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
    """MP3 は export/ のファイル名をそのまま使う。"""
    return [(p, p.name) for p in find_mp3_files(outdir)]


# ---------------------------------------------------------------------------
# 移行
# ---------------------------------------------------------------------------

def ensure_root_folder(client: DriveClient) -> tuple[dict, bool, bool]:
    """自動化用ルートを用意する。戻り値は (フォルダ, 新規作成か, リネームしたか)。

    旧名「練習録音」のフォルダが残っていればリネームして引き継ぐ。リネームなので
    フォルダIDは変わらず、既に配布済みの共有リンクもそのまま生きる。
    """
    found = client.find_child(ROOT_FOLDER_NAME, DRIVE_ROOT, folder=True)
    if found:
        return found, False, False

    legacy = client.find_child(LEGACY_ROOT_NAME, DRIVE_ROOT, folder=True)
    if legacy:
        client.rename_file(legacy["id"], ROOT_FOLDER_NAME)
        log(f"旧ルート「{LEGACY_ROOT_NAME}」を「{ROOT_FOLDER_NAME}」にリネームしました "
            f"(id={legacy['id']} は不変)")
        legacy["name"] = ROOT_FOLDER_NAME
        return legacy, False, True

    created, _ = client.ensure_folder(ROOT_FOLDER_NAME, DRIVE_ROOT)
    return created, True, False


def migrate_dates_into_orchestra(
    client: DriveClient, root_id: str, orchestra_id: str
) -> list[str]:
    """ルート直下に直接ある `{date}` フォルダを `{orchestra}/` の下へ移動する。

    フォルダごと親を1回付け替えるだけなので、中の WAV/MP3 とファイルはすべて
    付いてくる。ファイルIDも共有リンクも変わらない。
    """
    moved = []
    for child in client.list_children(root_id):
        if child.get("mimeType") != FOLDER_MIME:
            continue
        if not DATE_DIR_RE.fullmatch(child["name"]):
            continue
        client.move_file(child["id"], orchestra_id, root_id)
        log(f"  移行: {child['name']}/ をフォルダごと {ROOT_FOLDER_NAME}/ 直下から "
            f"{{orchestra}}/ へ移動 (id={child['id']} は不変)")
        moved.append(child["name"])
    return moved


def migrate_flat_files(
    client: DriveClient, date_folder_id: str, subfolders: dict[str, str]
) -> list[tuple[str, str]]:
    """`{date}/` 直下に直接置かれたファイルを WAV/ か MP3/ へ移動する。"""
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
            log(f"  警告: {child['name']} は既に {target_name}/ にも存在します。移動しません。")
            continue
        client.move_file(child["id"], target_id, date_folder_id)
        log(f"  移行: {child['name']} -> {target_name}/")
        moved.append((child["name"], target_name))
    return moved


# ---------------------------------------------------------------------------
# アップロード
# ---------------------------------------------------------------------------

def _upload_group(
    client: DriveClient, plan: list[tuple[Path, str]], folder_id: str, label: str
) -> dict:
    uploaded, skipped = [], []
    for i, (src, drive_name) in enumerate(plan, start=1):
        size = src.stat().st_size
        existing = client.find_child(drive_name, folder_id, folder=False)

        # 内容が同じなら送らない。Drive の md5Checksum とローカルの MD5 を比べる。
        if existing and existing.get("md5Checksum"):
            digest = local_md5(src)
            if digest == existing["md5Checksum"]:
                log(f"  [{label} {i}/{len(plan)}] スキップ(内容同一): {drive_name}  "
                    f"md5={digest[:12]}…")
                skipped.append(drive_name)
                continue
            log(f"  [{label} {i}/{len(plan)}] {drive_name}  内容が変化 "
                f"(ローカル {digest[:12]}… / Drive {existing['md5Checksum'][:12]}…)")

        log(f"  [{label} {i}/{len(plan)}] {drive_name}  ({size/2**20:.0f} MiB)  <- {src.name}")
        started = time.time()
        last = [0.0]

        def on_progress(frac: float, _last=last, _size=size, _started=started):
            if frac - _last[0] >= 0.2 or frac >= 1.0:
                _last[0] = frac
                el = time.time() - _started
                rate = (_size * frac / 2**20) / el if el > 0 else 0
                log(f"      {frac*100:.0f}%  ({rate:.0f} MiB/s)")

        info = client.upload(src, drive_name, folder_id, on_progress, existing=existing)
        log(f"      完了: id={info['id']}"
            + ("(既存を新バージョンで置き換え)" if info.get("_replaced") else "(新規)"))
        uploaded.append(drive_name)
    return {"uploaded": uploaded, "skipped": skipped}


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
    cfg: SessionConfig,
    auth_timeout: float | None = None,
) -> dict:
    orch_name = orchestra_folder_name(cfg.orchestra)
    wav_plan = plan_wav(outdir, date)
    mp3_plan = plan_mp3(outdir)
    total = sum(p.stat().st_size for p, _ in wav_plan + mp3_plan)
    log(
        f"Google Drive アップロード開始: WAV {len(wav_plan)} / MP3 {len(mp3_plan)} / "
        f"最大 {total/2**30:.2f} GiB / 団体フォルダ={orch_name}"
    )

    client = DriveClient.connect(root, auth_timeout)

    # --- フォルダ(4階層。いずれも同名があれば再利用)-----------------------
    root_folder, c_root, renamed = ensure_root_folder(client)
    log(f"ルート「{ROOT_FOLDER_NAME}」"
        f"{'を作成' if c_root else ('をリネームで引き継ぎ' if renamed else 'を再利用')} "
        f"(id={root_folder['id']})")

    orch_folder, c_orch = client.ensure_folder(orch_name, root_folder["id"])
    log(f"フォルダ「{orch_name}」{'を作成' if c_orch else 'を再利用'} (id={orch_folder['id']})")

    # 旧構成: ルート直下に {date} が直接ある。フォルダごと団体フォルダへ移す。
    log("旧構成(ルート直下の日付フォルダ)がないか確認します")
    moved_dates = migrate_dates_into_orchestra(client, root_folder["id"], orch_folder["id"])
    log(f"  移行した日付フォルダ: {len(moved_dates)} 件")

    date_folder, c_date = client.ensure_folder(date, orch_folder["id"])
    log(f"フォルダ「{date}」{'を作成' if c_date else 'を再利用'} (id={date_folder['id']})")

    subfolders: dict[str, str] = {}
    created_sub: dict[str, bool] = {}
    for name in (WAV_FOLDER, MP3_FOLDER):
        f, created = client.ensure_folder(name, date_folder["id"])
        subfolders[name] = f["id"]
        created_sub[name] = created
        log(f"フォルダ「{date}/{name}」{'を作成' if created else 'を再利用'} (id={f['id']})")

    log("日付フォルダ直下のフラット配置がないか確認します")
    moved_files = migrate_flat_files(client, date_folder["id"], subfolders)
    log(f"  移行したファイル: {len(moved_files)} 件")

    # --- アップロード(内容同一ならスキップ)---------------------------------
    wav_res = _upload_group(client, wav_plan, subfolders[WAV_FOLDER], "WAV")
    mp3_res = _upload_group(client, mp3_plan, subfolders[MP3_FOLDER], "MP3")

    # --- 共有(2つのフォルダに個別設定)-------------------------------------
    log("共有を設定します(リンクを知っている全員が閲覧者 / ダウンロード制限なし)")
    share_date = _share_state(client, date_folder["id"])
    share_mp3 = _share_state(client, subfolders[MP3_FOLDER])

    def count(folder_id: str) -> int:
        return sum(1 for c in client.list_children(folder_id) if c.get("mimeType") != FOLDER_MIME)

    return {
        "orchestra_folder": orch_name,
        "root_folder_id": root_folder["id"],
        "orchestra_folder_id": orch_folder["id"],
        "date_folder_id": date_folder["id"],
        "wav_folder_id": subfolders[WAV_FOLDER],
        "mp3_folder_id": subfolders[MP3_FOLDER],
        "created": {"root": c_root, "orchestra": c_orch, "date": c_date, **created_sub},
        "root_renamed": renamed,
        "migrated_dates": moved_dates,
        "migrated_files": moved_files,
        "wav": wav_res,
        "mp3": mp3_res,
        "n_wav": len(wav_plan),
        "n_mp3": len(mp3_plan),
        "n_in_wav": count(subfolders[WAV_FOLDER]),
        "n_in_mp3": count(subfolders[MP3_FOLDER]),
        "n_loose_in_date": count(date_folder["id"]),
        "share_date": share_date,
        "share_mp3": share_mp3,
    }
