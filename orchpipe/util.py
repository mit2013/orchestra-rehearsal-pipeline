"""共通ユーティリティ: ログ、時刻フォーマット、ffmpeg/ffprobe 実行ラッパ。"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

_T0 = time.time()


def log(msg: str) -> None:
    """経過時間つき進捗ログ。3時間素材は処理が長いので必ず経過を出す。"""
    el = time.time() - _T0
    print(f"[{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}", file=sys.stderr, flush=True)


class PipelineError(RuntimeError):
    """検証失敗など、処理を止めるべきエラー。"""


# --------------------------------------------------------------------------
# 時刻フォーマット
# --------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    """秒 -> "HH:MM:SS"(候補JSON用)。"""
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_time(value: Any) -> float:
    """"HH:MM:SS" / "MM:SS" / 秒数(int/float/文字列) -> 秒。"""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise PipelineError("時刻が空です")
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) > 3:
        raise PipelineError(f"時刻の形式が不正です: {value!r}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


# --------------------------------------------------------------------------
# 外部コマンド
# --------------------------------------------------------------------------

def run(cmd: Sequence[str], *, desc: str = "") -> None:
    """コマンドを実行し、失敗したら stderr つきで PipelineError。"""
    if desc:
        log(desc)
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-25:]
        raise PipelineError(
            "コマンド失敗:\n  " + shlex.join(cmd) + "\n" + "\n".join("  " + t for t in tail)
        )


def ffprobe_json(path: Path) -> dict:
    """1本の音声ファイルのストリーム/フォーマット情報を取得。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise PipelineError(
            f"ffprobe に失敗しました: {path}\n{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return json.loads(proc.stdout.decode("utf-8"))


def probe_audio(path: Path) -> dict:
    """必要なメタデータだけを平たい dict にして返す。"""
    info = ffprobe_json(path)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if len(streams) != 1:
        raise PipelineError(f"音声ストリームが1本ではありません({len(streams)}本): {path}")
    st = streams[0]
    duration = st.get("duration") or info.get("format", {}).get("duration")
    if duration is None:
        raise PipelineError(f"再生時間を取得できません: {path}")
    return {
        "path": str(path),
        "name": path.name,
        "codec_name": st.get("codec_name"),
        "sample_fmt": st.get("sample_fmt"),
        "sample_rate": int(st["sample_rate"]),
        "channels": int(st["channels"]),
        "bits_per_sample": int(st.get("bits_per_sample") or 0),
        "duration": float(duration),
        "size": int(info.get("format", {}).get("size") or 0),
    }


def file_digest(path: Path, algo: str, chunk: int = 4 * 1024 * 1024) -> str:
    """ファイル全体のハッシュを16進で返す。

    アップロード先に既にある同名ファイルと内容が同じかを判定するために使う。
    数GBのファイルを扱うので、必ず分割して読む。
    """
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# ファイル/パス
# --------------------------------------------------------------------------

_BAD_CHARS = str.maketrans({c: "_" for c in '/\\:*?"<>|\n\r\t'})


def safe_filename(text: str) -> str:
    """ラベルをファイル名に使える形にする(日本語はそのまま残す)。"""
    out = text.translate(_BAD_CHARS).strip().strip(".")
    out = "_".join(out.split())
    return out or "segment"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise PipelineError(f"ファイルが見つかりません: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


ENV_NAME = ".env"


def read_env(root: Path) -> dict[str, str]:
    """`.env` を KEY=VALUE として読む。無ければ空 dict。"""
    p = root / ENV_NAME
    if not p.exists():
        return {}
    values: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def require_env(root: Path, keys: Sequence[str]) -> list[str]:
    """必須キーを取り出す。欠けていれば PipelineError。"""
    values = read_env(root)
    if not (root / ENV_NAME).exists():
        raise PipelineError(
            f"{root / ENV_NAME} がありません。{', '.join(keys)} を記載してください。"
        )
    missing = [k for k in keys if not values.get(k)]
    if missing:
        raise PipelineError(f"{ENV_NAME} に {', '.join(missing)} がありません")
    return [values[k] for k in keys]


def out_dir(root: Path, date: str) -> Path:
    d = root / "output" / date
    d.mkdir(parents=True, exist_ok=True)
    return d
