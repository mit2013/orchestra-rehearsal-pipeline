"""Box API クライアント(OAuth 2.0 + フォルダ/アップロード/共有リンク)。

公式ドキュメントで確認した仕様に沿って実装している:

- 認可     : GET  https://account.box.com/api/oauth2/authorize
- トークン : POST https://api.box.com/oauth2/token  (application/x-www-form-urlencoded)
- アップロード(50MBまで): POST https://upload.box.com/api/2.0/files/content
- アップロード(20MB以上): チャンク分割セッション。今回のMP3は100MB超なので常にこちら
- 共有リンク: PUT /2.0/folders/{id}  body に shared_link{access,password,permissions{can_download}}

**リフレッシュトークンはローテーションする**(1回使うと無効化され、新しい値が返る)。
リフレッシュのたびに保存し直さないと、しばらくして認証が切れて再許可が必要になる。
有効期限は60日。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from .util import PipelineError, log

AUTH_URL = "https://account.box.com/api/oauth2/authorize"
TOKEN_URL = "https://api.box.com/oauth2/token"
API_BASE = "https://api.box.com/2.0"
UPLOAD_BASE = "https://upload.box.com/api/2.0"

REDIRECT_URI = "http://localhost:8888/callback"
CALLBACK_PORT = 8888

TOKENS_NAME = ".box_tokens.json"
ENV_NAME = ".env"

# 50MB を超える単純アップロードは不可。今回のMP3は必ずチャンク分割になる。
SIMPLE_UPLOAD_LIMIT = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# 資格情報とトークンの永続化
# ---------------------------------------------------------------------------

def load_env(root: Path) -> tuple[str, str]:
    """`.env` から BOX_CLIENT_ID / BOX_CLIENT_SECRET を読む。"""
    p = root / ENV_NAME
    if not p.exists():
        raise PipelineError(
            f"{p} がありません。BOX_CLIENT_ID と BOX_CLIENT_SECRET を記載してください。"
        )
    values: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("BOX_CLIENT_ID", "BOX_CLIENT_SECRET") if not values.get(k)]
    if missing:
        raise PipelineError(f"{ENV_NAME} に {', '.join(missing)} がありません")
    return values["BOX_CLIENT_ID"], values["BOX_CLIENT_SECRET"]


@dataclass
class Tokens:
    access_token: str
    refresh_token: str
    expires_at: float  # epoch 秒

    @property
    def expired(self) -> bool:
        # 期限ぎりぎりで使うと途中で切れるので 60 秒の余裕を見る。
        return time.time() >= self.expires_at - 60


class TokenStore:
    """`.box_tokens.json` の読み書き。資格情報なのでパーミッションを 600 にする。"""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Tokens | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        try:
            return Tokens(data["access_token"], data["refresh_token"], float(data["expires_at"]))
        except (KeyError, TypeError, ValueError):
            log(f"警告: {self.path.name} を解釈できません。再認証します。")
            return None

    def save(self, t: Tokens) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "access_token": t.access_token,
                    "refresh_token": t.refresh_token,
                    "expires_at": t.expires_at,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# 初回認証(ブラウザ + ローカルコールバックサーバ)
# ---------------------------------------------------------------------------

class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None
    state: str = ""

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        q = urllib.parse.parse_qs(parsed.query)
        if q.get("state", [""])[0] != _CallbackHandler.state:
            _CallbackHandler.error = "state が一致しません(CSRF の疑い)"
        elif "code" in q:
            _CallbackHandler.code = q["code"][0]
        else:
            _CallbackHandler.error = q.get("error_description", q.get("error", ["不明"]))[0]

        body = (
            "<html><head><meta charset='utf-8'><title>Box 認証</title></head><body>"
            "<h2>認証が完了しました。ターミナルに戻ってください。</h2>"
            if _CallbackHandler.code
            else f"<html><head><meta charset='utf-8'></head><body><h2>認証に失敗しました: "
            f"{_CallbackHandler.error}</h2>"
        ) + "</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # サーバのアクセスログは抑制
        pass


def authorize_interactive(client_id: str, client_secret: str, timeout: float = 300.0) -> Tokens:
    """ブラウザで許可を取り、認可コードをトークンに交換する。

    認可コードの有効期限は数十秒しかないので、ブラウザを開く前にサーバを起動しておく。
    """
    _CallbackHandler.code = None
    _CallbackHandler.error = None
    _CallbackHandler.state = base64.urlsafe_b64encode(os.urandom(18)).decode().rstrip("=")

    try:
        server = HTTPServer(("localhost", CALLBACK_PORT), _CallbackHandler)
    except OSError as e:
        raise PipelineError(
            f"localhost:{CALLBACK_PORT} を開けません({e})。"
            "同じポートを使う別のプロセスを止めてから再実行してください。"
        ) from None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "state": _CallbackHandler.state,
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    # パイプ経由だと stdout がブロックバッファされ、待機中に URL が見えない。
    # 認可コードの寿命が短くユーザーをすぐ動かす必要があるので、必ず即時に流す。
    import sys as _sys

    banner = (
        "\n" + "=" * 70 + "\n"
        "Box の認証が必要です。ブラウザで許可してください。\n"
        "(ブラウザが自動で開かない場合は以下の URL を開いてください)\n\n"
        f"{url}\n" + "=" * 70 + "\n"
    )
    print(banner, flush=True)
    _sys.stderr.write(banner)
    _sys.stderr.flush()
    log(f"localhost:{CALLBACK_PORT} でコールバックを待機中(最大 {timeout:.0f} 秒)...")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _CallbackHandler.code or _CallbackHandler.error:
            break
        time.sleep(0.3)
    server.shutdown()

    if _CallbackHandler.error:
        raise PipelineError(f"Box の認可に失敗しました: {_CallbackHandler.error}")
    if not _CallbackHandler.code:
        raise PipelineError(
            f"{timeout:.0f} 秒以内に認可が完了しませんでした。もう一度実行してください。"
        )

    log("認可コードを取得しました。アクセストークンに交換します。")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": _CallbackHandler.code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise PipelineError(f"トークン交換に失敗しました ({resp.status_code}): {resp.text[:400]}")
    d = resp.json()
    return Tokens(d["access_token"], d["refresh_token"], time.time() + float(d["expires_in"]))


# ---------------------------------------------------------------------------
# API クライアント
# ---------------------------------------------------------------------------

class BoxClient:
    def __init__(self, client_id: str, client_secret: str, store: TokenStore, tokens: Tokens):
        self.client_id = client_id
        self.client_secret = client_secret
        self.store = store
        self.tokens = tokens

    @classmethod
    def connect(cls, root: Path, auth_timeout: float = 300.0) -> "BoxClient":
        client_id, client_secret = load_env(root)
        store = TokenStore(root / TOKENS_NAME)
        tokens = store.load()
        if tokens is None:
            log("保存済みトークンがありません。初回認証を行います。")
            tokens = authorize_interactive(client_id, client_secret, auth_timeout)
            store.save(tokens)
            log(f"トークンを保存しました: {store.path.name}")
        client = cls(client_id, client_secret, store, tokens)
        if tokens.expired:
            client.refresh()
        return client

    # -- 認証 --------------------------------------------------------------

    def refresh(self) -> None:
        """アクセストークンを更新する。**新しいリフレッシュトークンを必ず保存する。**"""
        log("アクセストークンを更新します。")
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.tokens.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise PipelineError(
                f"トークンの更新に失敗しました ({resp.status_code})。"
                f"{TOKENS_NAME} を削除して再認証してください。\n{resp.text[:400]}"
            )
        d = resp.json()
        # Box はリフレッシュのたびに新しいリフレッシュトークンを返す(ローテーション)。
        # 受け取った値を必ず保存し直す。
        self.tokens = Tokens(
            d["access_token"],
            d.get("refresh_token", self.tokens.refresh_token),
            time.time() + float(d["expires_in"]),
        )
        self.store.save(self.tokens)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens.access_token}"}

    def request(self, method: str, url: str, *, retry_auth: bool = True, **kw) -> requests.Response:
        if self.tokens.expired:
            self.refresh()
        headers = {**self._headers(), **kw.pop("headers", {})}
        resp = requests.request(method, url, headers=headers, timeout=kw.pop("timeout", 120), **kw)
        if resp.status_code == 401 and retry_auth:
            self.refresh()
            return self.request(method, url, retry_auth=False, **kw)
        return resp

    @staticmethod
    def _check(resp: requests.Response, what: str, ok=(200, 201)) -> dict:
        if resp.status_code not in ok:
            raise PipelineError(f"{what} に失敗しました ({resp.status_code}): {resp.text[:500]}")
        return resp.json() if resp.content else {}

    # -- フォルダ ----------------------------------------------------------

    def list_folder_items(self, folder_id: str) -> list[dict]:
        items, offset = [], 0
        while True:
            resp = self.request(
                "GET",
                f"{API_BASE}/folders/{folder_id}/items",
                params={"limit": 1000, "offset": offset, "fields": "id,name,type,size,sha1"},
            )
            d = self._check(resp, f"フォルダ {folder_id} の一覧取得", ok=(200,))
            items.extend(d.get("entries", []))
            offset += len(d.get("entries", []))
            if offset >= int(d.get("total_count", 0)) or not d.get("entries"):
                break
        return items

    def find_folder(self, parent_id: str, name: str) -> dict | None:
        for it in self.list_folder_items(parent_id):
            if it["type"] == "folder" and it["name"] == name:
                return it
        return None

    def ensure_folder(self, parent_id: str, name: str) -> tuple[dict, bool]:
        """同名フォルダがあれば再利用し、無ければ作る。戻り値は (フォルダ, 新規作成か)。"""
        found = self.find_folder(parent_id, name)
        if found:
            return found, False
        resp = self.request(
            "POST", f"{API_BASE}/folders",
            json={"name": name, "parent": {"id": parent_id}},
        )
        return self._check(resp, f"フォルダ {name} の作成"), True

    # -- アップロード(チャンク分割)----------------------------------------

    def upload_chunked(self, path: Path, folder_id: str, existing_file_id: str | None) -> dict:
        """20MB 超のファイルをチャンク分割してアップロードする。

        `existing_file_id` を渡すと、そのファイルの新しいバージョンとして上げる。
        """
        size = path.stat().st_size
        if existing_file_id:
            url = f"{UPLOAD_BASE}/files/{existing_file_id}/upload_sessions"
            body = {"file_size": size, "file_name": path.name}
        else:
            url = f"{UPLOAD_BASE}/files/upload_sessions"
            body = {"folder_id": folder_id, "file_size": size, "file_name": path.name}

        session = self._check(self.request("POST", url, json=body), "アップロードセッションの作成")
        part_size = int(session["part_size"])
        total_parts = int(session["total_parts"])
        endpoints = session["session_endpoints"]

        parts: list[dict] = []
        whole = hashlib.sha1()
        with path.open("rb") as f:
            offset = 0
            for i in range(total_parts):
                chunk = f.read(part_size)
                if not chunk:
                    break
                whole.update(chunk)
                digest = base64.b64encode(hashlib.sha1(chunk).digest()).decode()
                end = offset + len(chunk) - 1
                resp = self.request(
                    "PUT", endpoints["upload_part"],
                    headers={
                        "digest": f"sha={digest}",
                        "content-range": f"bytes {offset}-{end}/{size}",
                        "content-type": "application/octet-stream",
                    },
                    data=chunk,
                    timeout=300,
                )
                part = self._check(resp, f"パート {i+1}/{total_parts} のアップロード", ok=(200, 201))
                parts.append(part["part"])
                offset += len(chunk)
                if (i + 1) % 5 == 0 or i + 1 == total_parts:
                    log(f"      {i+1}/{total_parts} パート ({offset/2**20:.0f}/{size/2**20:.0f} MiB)")

        whole_digest = base64.b64encode(whole.digest()).decode()
        for attempt in range(30):
            resp = self.request(
                "POST", endpoints["commit"],
                headers={"digest": f"sha={whole_digest}"},
                json={"parts": parts},
                timeout=300,
            )
            if resp.status_code in (200, 201):
                d = resp.json()
                return d["entries"][0] if "entries" in d else d
            if resp.status_code == 202:
                wait = float(resp.headers.get("Retry-After", 5))
                log(f"      commit 処理中。{wait:.0f} 秒待機して再試行します。")
                time.sleep(wait)
                continue
            raise PipelineError(f"commit に失敗しました ({resp.status_code}): {resp.text[:500]}")
        raise PipelineError("commit が完了しませんでした(再試行上限に到達)")

    def upload(self, path: Path, folder_id: str, existing_file_id: str | None = None) -> dict:
        return self.upload_chunked(path, folder_id, existing_file_id)

    # -- 共有リンク --------------------------------------------------------

    def set_folder_shared_link(
        self, folder_id: str, password: str, can_download: bool = False, access: str = "open"
    ) -> dict:
        resp = self.request(
            "PUT", f"{API_BASE}/folders/{folder_id}",
            params={"fields": "shared_link"},
            json={
                "shared_link": {
                    "access": access,
                    "password": password,
                    "permissions": {"can_download": can_download},
                }
            },
        )
        return self._check(resp, "共有リンクの設定", ok=(200,)).get("shared_link", {})

    def get_folder_shared_link(self, folder_id: str) -> dict:
        resp = self.request(
            "GET", f"{API_BASE}/folders/{folder_id}", params={"fields": "shared_link,name"}
        )
        return self._check(resp, "共有リンクの取得", ok=(200,))
