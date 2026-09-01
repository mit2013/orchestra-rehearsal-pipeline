"""通知文言の生成と、LINE 経由の自分宛 push 通知。

生成する文言は3種類:

  ① 動画担当(動画担当)向け  … Google Drive の {date} フォルダ(WAV+MP3)
  ② 団員グループ向け          … Box の共有リンク + パスワード
  ③ (任意)ダウンロード可版  … Google Drive の {date}/MP3 フォルダ

**リンクをローカルにキャッシュしない。** 実行のたびに Box / Drive 双方の API へ
問い合わせて最新のリンクを取得する。パスワードだけは Box API が返さないので
`{concert_date}{password_suffix}` の式からその場で計算する。

LINE 通知は付加的な機能なので、失敗しても標準出力と messages.txt の生成は
必ず完了させる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .box_client import BoxClient
from .box_upload import build_password, resolve_parent_folder_id
from .config import SessionConfig
from .gdrive_client import ROOT_FOLDER_NAME, DriveClient
from .gdrive_upload import DRIVE_ROOT, MP3_FOLDER, orchestra_folder_name
from .util import PipelineError, log, read_env

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
MESSAGES_NAME = "messages.txt"

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/{}"


@dataclass
class Links:
    box_url: str
    box_password: str
    drive_date_url: str
    drive_mp3_url: str


# ---------------------------------------------------------------------------
# リンクの取得(毎回 API から。キャッシュしない)
# ---------------------------------------------------------------------------

def fetch_box_link(root: Path, date: str, cfg: SessionConfig) -> tuple[str, str]:
    """Box の {date} フォルダの共有リンクを取得し、パスワードはローカル計算する。"""
    password = build_password(cfg.concert_date, root)
    parent_id = resolve_parent_folder_id(cfg.box_parent_folder_id)
    client = BoxClient.connect(root)
    folder = client.find_folder(parent_id, date)
    if folder is None:
        raise PipelineError(
            f"Box に {date} フォルダが見つかりません(親 id={parent_id})。"
            "先に `box-upload` を実行してください。"
        )
    link = (client.get_folder_shared_link(folder["id"]).get("shared_link") or {})
    url = link.get("url")
    if not url:
        raise PipelineError(
            f"Box の {date} フォルダに共有リンクが設定されていません。"
            "先に `box-upload` を実行してください。"
        )
    return url, password


def fetch_drive_links(root: Path, date: str, cfg: SessionConfig) -> tuple[str, str]:
    """Drive の {date} と {date}/MP3 のフォルダURLを取得する。"""
    orch = orchestra_folder_name(cfg.orchestra)
    client = DriveClient.connect(root)

    root_folder = client.find_child(ROOT_FOLDER_NAME, DRIVE_ROOT, folder=True)
    if root_folder is None:
        raise PipelineError(
            f"Google Drive に「{ROOT_FOLDER_NAME}」フォルダが見つかりません。"
            "先に `gdrive-upload` を実行してください。"
        )
    orch_folder = client.find_child(orch, root_folder["id"], folder=True)
    if orch_folder is None:
        raise PipelineError(f"Google Drive に「{orch}」フォルダが見つかりません。")
    date_folder = client.find_child(date, orch_folder["id"], folder=True)
    if date_folder is None:
        raise PipelineError(
            f"Google Drive に {orch}/{date} フォルダが見つかりません。"
            "先に `gdrive-upload` を実行してください。"
        )
    mp3_folder = client.find_child(MP3_FOLDER, date_folder["id"], folder=True)
    if mp3_folder is None:
        raise PipelineError(f"Google Drive に {orch}/{date}/{MP3_FOLDER} フォルダが見つかりません。")

    return (
        DRIVE_FOLDER_URL.format(date_folder["id"]),
        DRIVE_FOLDER_URL.format(mp3_folder["id"]),
    )


# ---------------------------------------------------------------------------
# 文言
# ---------------------------------------------------------------------------

def build_messages(date: str, links: Links) -> dict[str, str]:
    shirakawa = (
        f"{date}の練習録音です。\n"
        f"{links.drive_date_url}\n"
        f"\n"
        f"WAVとMP3が入っています。"
    )
    members = (
        f"{date}の練習録音をアップロードしました。\n"
        f"{links.box_url}\n"
        f"パスワード: {links.box_password}\n"
        f"\n"
        f"※ストリーミング再生のみで、ダウンロードはできません。"
    )
    downloadable = (
        f"{date}の練習録音(ダウンロード可能版)です。\n"
        f"{links.drive_mp3_url}"
    )
    return {"shirakawa": shirakawa, "members": members, "downloadable": downloadable}


def render(date: str, msgs: dict[str, str]) -> str:
    """①②③をラベル付きで1つのテキストにまとめる。"""
    sep = "=" * 68
    return "\n".join([
        sep,
        f"① 動画担当(動画担当)向け個別メッセージ  [{date}]",
        sep,
        msgs["shirakawa"],
        "",
        sep,
        f"② 団員グループ向けメッセージ  [{date}]",
        sep,
        msgs["members"],
        "",
        sep,
        "③【任意・通常は送らない】ダウンロード可能版の個別共有用",
        "   ※必要な団員にだけ個別送信する想定のものです",
        sep,
        msgs["downloadable"],
        "",
    ])


# ---------------------------------------------------------------------------
# LINE(自分宛 push)
# ---------------------------------------------------------------------------

def push_line(root: Path, text: str) -> tuple[bool, str]:
    """自分宛に LINE で push する。戻り値は (成功したか, 説明)。

    付加的な機能なので、例外は投げず結果を返すだけにする。
    """
    import requests

    env = read_env(root)
    token = env.get("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = env.get("LINE_MY_USER_ID")
    missing = [k for k, v in (("LINE_CHANNEL_ACCESS_TOKEN", token), ("LINE_MY_USER_ID", user_id))
               if not v]
    if missing:
        return False, f".env に {', '.join(missing)} がないため送信しません"

    try:
        resp = requests.post(
            LINE_PUSH_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"to": user_id, "messages": [{"type": "text", "text": text}]},
            timeout=30,
        )
    except Exception as e:  # ネットワーク断など
        return False, f"送信中にエラー: {type(e).__name__}: {e}"

    if resp.status_code == 200:
        return True, "送信しました"
    return False, f"HTTP {resp.status_code}: {resp.text[:300]}"


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def run_notify(root: Path, date: str, outdir: Path, cfg: SessionConfig, send_line: bool = True) -> dict:
    log("Box と Google Drive から最新の共有リンクを取得します(キャッシュしません)")
    box_url, password = fetch_box_link(root, date, cfg)
    log(f"  Box   : {box_url}")
    drive_date_url, drive_mp3_url = fetch_drive_links(root, date, cfg)
    log(f"  Drive : {drive_date_url}")
    log(f"  Drive/MP3: {drive_mp3_url}")

    links = Links(box_url, password, drive_date_url, drive_mp3_url)
    msgs = build_messages(date, links)
    body = render(date, msgs)

    dst = outdir / MESSAGES_NAME
    dst.write_text(body, encoding="utf-8")
    log(f"{MESSAGES_NAME} を保存しました: {dst}")

    line_ok, line_note = (False, "送信しない設定です")
    if send_line:
        line_text = f"{date}の練習録音、パイプライン完了しました。\n\n{body}"
        line_ok, line_note = push_line(root, line_text)
        log(f"LINE 通知: {'成功' if line_ok else '失敗'} — {line_note}")

    return {
        "messages": msgs,
        "body": body,
        "path": dst,
        "links": links,
        "line_ok": line_ok,
        "line_note": line_note,
    }
