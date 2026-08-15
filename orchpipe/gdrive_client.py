"""Google Drive API クライアント(OAuth 2.0 + フォルダ/アップロード/共有)。

Box との違いに注意:

- デスクトップアプリ用クライアントなので Redirect URI の事前登録が不要。
  `InstalledAppFlow.run_local_server(port=0)` が空きポートを自動で選ぶ。
- **リフレッシュトークンは通常ローテーションしない**(Box は毎回入れ替わる)。
  ただしアクセストークンを更新したら都度保存する。
- ダウンロード禁止は設定しない。動画担当が素材として取得する必要があるため。

スコープは `drive.file`(このアプリが作成したファイルのみ)。既存の手動作成フォルダは
見えないので、「練習録音」フォルダもこのアプリが作ったものだけが再利用対象になる。
"""

from __future__ import annotations

import http.client
import os
import socket
import ssl
import stat
import time
from pathlib import Path

from .util import PipelineError, log, require_env

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKENS_NAME = ".google_tokens.json"
ENV_NAME = ".env"

FOLDER_MIME = "application/vnd.google-apps.folder"
# 自動化専用であることが名前から分かるようにしている。
ROOT_FOLDER_NAME = "orchestra-recording-pipeline"
# 3階層時代のルート名。見つかったらリネームして引き継ぐ(ID は変わらない)。
LEGACY_ROOT_NAME = "練習録音"

# 1.8GB 級の WAV を上げるので、必ず再開可能アップロードを使う。
# 回線が切れたときに捨てる量が減るよう、チャンクは控えめにする。
UPLOAD_CHUNK = 16 * 1024 * 1024
MAX_UPLOAD_RETRIES = 8

# 数分かかる転送では一時的な切断が普通に起こる。ここに挙げたものは再開して続行する。
RETRIABLE_ERRORS = (
    BrokenPipeError,
    ConnectionError,
    socket.timeout,
    ssl.SSLError,
    http.client.HTTPException,
    OSError,
)


def load_env(root: Path) -> tuple[str, str]:
    """`.env` から GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET を読む。"""
    a, b = require_env(root, ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"))
    return a, b


def _save_credentials(path: Path, creds) -> None:
    path.write_text(creds.to_json() + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def get_credentials(root: Path, auth_timeout: float | None = None):
    """保存済みトークンを使う。無効なら更新、それも無理なら初回認証を行う。"""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_id, client_secret = load_env(root)
    token_path = root / TOKENS_NAME
    creds = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as e:
            log(f"警告: {TOKENS_NAME} を読めません({e})。再認証します。")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        log("アクセストークンを更新します。")
        try:
            creds.refresh(Request())
            # Google のリフレッシュトークンは通常そのままだが、
            # アクセストークンが変わったので保存し直す。
            _save_credentials(token_path, creds)
            return creds
        except Exception as e:
            log(f"警告: トークン更新に失敗しました({e})。再認証します。")

    # --- 初回(または再)認証 ---------------------------------------------
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    log("Google の認証が必要です。ブラウザで許可してください。")
    creds = flow.run_local_server(
        port=0,
        open_browser=True,
        authorization_prompt_message="ブラウザで次の URL を開いて許可してください:\n{url}",
        success_message="認証が完了しました。ターミナルに戻ってください。",
        timeout_seconds=auth_timeout,
    )
    _save_credentials(token_path, creds)
    log(f"トークンを保存しました: {TOKENS_NAME}")
    return creds


class DriveClient:
    def __init__(self, service):
        self.service = service

    @classmethod
    def connect(cls, root: Path, auth_timeout: float | None = None) -> "DriveClient":
        from googleapiclient.discovery import build

        creds = get_credentials(root, auth_timeout)
        return cls(build("drive", "v3", credentials=creds, cache_discovery=False))

    # -- フォルダ ----------------------------------------------------------

    def find_child(self, name: str, parent_id: str, folder: bool = True) -> dict | None:
        """親の直下から名前で探す。drive.file スコープなので自アプリ作成分のみ見える。"""
        q = [
            f"name = '{name}'",
            f"'{parent_id}' in parents",
            "trashed = false",
        ]
        if folder:
            q.append(f"mimeType = '{FOLDER_MIME}'")
        else:
            q.append(f"mimeType != '{FOLDER_MIME}'")
        resp = (
            self.service.files()
            .list(
                q=" and ".join(q),
                fields="files(id,name,mimeType,md5Checksum,size)",
                pageSize=10,
            )
            .execute()
        )
        files = resp.get("files", [])
        return files[0] if files else None

    def ensure_folder(self, name: str, parent_id: str) -> tuple[dict, bool]:
        """同名フォルダがあれば再利用、無ければ作成。戻り値は (フォルダ, 新規作成か)。"""
        found = self.find_child(name, parent_id, folder=True)
        if found:
            return found, False
        created = (
            self.service.files()
            .create(
                body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                fields="id,name,mimeType",
            )
            .execute()
        )
        return created, True

    def list_children(self, parent_id: str) -> list[dict]:
        out, token = [], None
        while True:
            resp = (
                self.service.files()
                .list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id,name,mimeType,size)",
                    pageSize=200,
                    pageToken=token,
                )
                .execute()
            )
            out.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return out

    def rename_file(self, file_id: str, new_name: str) -> dict:
        """名前だけ変更する。ID は変わらないので共有リンクもそのまま生きる。"""
        return (
            self.service.files()
            .update(fileId=file_id, body={"name": new_name}, fields="id,name")
            .execute()
        )

    def move_file(self, file_id: str, add_parent: str, remove_parent: str) -> dict:
        """親フォルダを付け替える(実体は移動。再アップロードしない)。"""
        return (
            self.service.files()
            .update(
                fileId=file_id,
                addParents=add_parent,
                removeParents=remove_parent,
                fields="id,name,parents",
            )
            .execute()
        )

    # -- アップロード ------------------------------------------------------

    def upload(
        self, path: Path, drive_name: str, parent_id: str, on_progress=None, existing=None
    ) -> dict:
        """再開可能アップロード。同名ファイルがあれば新しいバージョンとして上書きする。"""
        from googleapiclient.http import MediaFileUpload

        if existing is None:
            existing = self.find_child(drive_name, parent_id, folder=False)
        media = MediaFileUpload(
            str(path), mimetype="audio/wav", chunksize=UPLOAD_CHUNK, resumable=True
        )
        if existing:
            request = self.service.files().update(
                fileId=existing["id"], media_body=media, fields="id,name,size,version"
            )
        else:
            request = self.service.files().create(
                body={"name": drive_name, "parents": [parent_id]},
                media_body=media,
                fields="id,name,size,version",
            )

        # 数GBを数分かけて送るので、途中の一時的な切断は普通に起こる。
        # 再開可能アップロードなので、失敗したチャンクから送り直せばよい。
        response = None
        attempts = 0
        while response is None:
            try:
                status, response = request.next_chunk(num_retries=3)
                attempts = 0
            except RETRIABLE_ERRORS as e:
                attempts += 1
                if attempts > MAX_UPLOAD_RETRIES:
                    raise PipelineError(
                        f"{path.name} のアップロードが {MAX_UPLOAD_RETRIES} 回連続で失敗しました: "
                        f"{type(e).__name__}: {e}"
                    ) from e
                wait = min(60.0, 2.0 ** attempts)
                done = getattr(request, "resumable_progress", 0)
                log(
                    f"      通信エラー ({type(e).__name__})。{wait:.0f} 秒後に "
                    f"{done/2**20:.0f} MiB 地点から再開します "
                    f"(リトライ {attempts}/{MAX_UPLOAD_RETRIES})"
                )
                time.sleep(wait)
                continue
            if status and on_progress:
                on_progress(status.progress())
        response["_replaced"] = bool(existing)
        return response

    # -- 共有 --------------------------------------------------------------

    def share_anyone_reader(self, file_id: str) -> dict:
        """「リンクを知っている全員が閲覧者」。ダウンロード制限は付けない。"""
        existing = (
            self.service.permissions()
            .list(fileId=file_id, fields="permissions(id,type,role)")
            .execute()
            .get("permissions", [])
        )
        for p in existing:
            if p.get("type") == "anyone":
                return p
        return (
            self.service.permissions()
            .create(fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id,type,role")
            .execute()
        )

    def get_share_state(self, file_id: str) -> dict:
        meta = (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id,name,webViewLink,copyRequiresWriterPermission,"
                "capabilities(canDownload,canCopy)",
            )
            .execute()
        )
        perms = (
            self.service.permissions()
            .list(fileId=file_id, fields="permissions(id,type,role,allowFileDiscovery)")
            .execute()
            .get("permissions", [])
        )
        meta["permissions"] = perms
        return meta
