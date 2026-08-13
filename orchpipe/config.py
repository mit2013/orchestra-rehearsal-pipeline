"""セッション設定 (`output/{date}/session_config.json`) の読み書き。

`confirmed.json` と同じ思想で、`ingest` 時に既定値で生成し、以降はユーザーが
手で編集する。パイプラインは勝手に上書きしない。

    {
      "recorder": "zoom-m4",
      "ext_lr_map": "normal",
      "source": "ext_only",
      "mix_ratio": {"ext": 0.6, "int": 0.4}
    }

- `ext_lr_map`: "normal" (Tr1=L, Tr2=R) / "swapped" (Tr1=R, Tr2=L)
                配線ミスは録音セッション単位で起きるので日付ごとに上書きできる。
- `source`    : "ext_only" / "int_only" / "mix"(既定は ext_only)
- `mix_ratio` : source が "mix" のときだけ使う。比率は今後試す前提で固定しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .util import PipelineError, log, read_json, write_json

LR_MAPS = ("normal", "swapped")
SOURCES = ("ext_only", "int_only", "mix")

CONFIG_NAME = "session_config.json"


@dataclass
class SessionConfig:
    recorder: str = "zoom-m4"
    ext_lr_map: str = "normal"
    source: str = "ext_only"
    mix_ratio: dict[str, float] = field(default_factory=lambda: {"ext": 0.6, "int": 0.4})

    def to_json(self) -> dict:
        return {
            "recorder": self.recorder,
            "ext_lr_map": self.ext_lr_map,
            "source": self.source,
            "mix_ratio": {k: float(v) for k, v in self.mix_ratio.items()},
        }

    def validate(self) -> None:
        if self.ext_lr_map not in LR_MAPS:
            raise PipelineError(
                f"{CONFIG_NAME}: ext_lr_map は {' / '.join(LR_MAPS)} のいずれかです"
                f"(実際: {self.ext_lr_map!r})"
            )
        if self.source not in SOURCES:
            raise PipelineError(
                f"{CONFIG_NAME}: source は {' / '.join(SOURCES)} のいずれかです"
                f"(実際: {self.source!r})"
            )
        for key in ("ext", "int"):
            if key not in self.mix_ratio:
                raise PipelineError(f"{CONFIG_NAME}: mix_ratio に '{key}' がありません")
            try:
                v = float(self.mix_ratio[key])
            except (TypeError, ValueError):
                raise PipelineError(
                    f"{CONFIG_NAME}: mix_ratio.{key} は数値である必要があります"
                    f"(実際: {self.mix_ratio[key]!r})"
                ) from None
            if v < 0:
                raise PipelineError(f"{CONFIG_NAME}: mix_ratio.{key} は 0 以上である必要があります")
        if self.source == "mix" and sum(float(v) for v in self.mix_ratio.values()) <= 0:
            raise PipelineError(f"{CONFIG_NAME}: source=mix なのに mix_ratio がすべて 0 です")

    @property
    def swap_ext_lr(self) -> bool:
        return self.ext_lr_map == "swapped"


def config_path(outdir: Path) -> Path:
    return outdir / CONFIG_NAME


def load(outdir: Path) -> SessionConfig:
    """設定を読む。無ければ既定値(ファイルは作らない)。"""
    p = config_path(outdir)
    if not p.exists():
        cfg = SessionConfig()
        cfg.validate()
        return cfg
    data = read_json(p)
    if not isinstance(data, dict):
        raise PipelineError(f"{CONFIG_NAME} はオブジェクトである必要があります")
    default = SessionConfig()
    cfg = SessionConfig(
        recorder=data.get("recorder", default.recorder),
        ext_lr_map=data.get("ext_lr_map", default.ext_lr_map),
        source=data.get("source", default.source),
        mix_ratio=data.get("mix_ratio", default.mix_ratio),
    )
    cfg.validate()
    return cfg


def ensure(outdir: Path, recorder: str, force: bool = False) -> SessionConfig:
    """`ingest` 時に既定値で生成する。既存ファイルは上書きしない。"""
    p = config_path(outdir)
    if p.exists() and not force:
        cfg = load(outdir)
        log(f"既存の {CONFIG_NAME} を使用: source={cfg.source}, ext_lr_map={cfg.ext_lr_map}")
        return cfg
    cfg = SessionConfig(recorder=recorder)
    cfg.validate()
    write_json(p, cfg.to_json())
    log(f"{CONFIG_NAME} を既定値で作成しました: {p}")
    return cfg
