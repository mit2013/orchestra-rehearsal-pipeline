"""セッション設定 (`output/{date}/session_config.json`) の読み書き。

`confirmed.json` と同じ思想で、`ingest` 時に既定値で生成し、以降はユーザーが
手で編集する。パイプラインは勝手に上書きしない。

    {
      "ext_lr_map": "normal",
      "normalize_scope": "date",
      "source": "ext_only",
      "mix_ratio": {"ext": 0.6, "int": 0.4},
      "orchestra": "Windrose Sinfonie Orchester",
      "concert_date": "{concert_date}",
      "box_parent_folder_id": ""
    }

レコーダ機種はここでは持たない。取り込み時に決まる情報であり、`ingest.json` の
`recorder` / `channel_groups` が唯一の情報源である。二重に持つと食い違いうるため、
このファイルは「取り込み後にユーザーが調整する設定」だけを持つ。

- `ext_lr_map`: "normal" (Tr1=L, Tr2=R) / "swapped" (Tr1=R, Tr2=L)
                配線ミスは録音セッション単位で起きるので日付ごとに上書きできる。
- `normalize_scope`: **使われない。** 正規化がピーク基準からラウドネス基準に
                変わり、ブロックごとに目標ラウドネスへ合わせるようになったため、
                日付全体で基準をそろえるという考え方自体がなくなった。既存の
                `session_config.json` をそのまま読めるようにキーだけ残してある。
- `source`    : "ext_only" / "int_only" / "mix"(既定は ext_only)
- `mix_ratio` : source が "mix" のときだけ使う。比率は今後試す前提で固定しない。
- `orchestra` : MP3タグに埋め込む団体名。`ingest` 時にプロジェクト直下の
                `pipeline_defaults.json` からコピーする。既に値があれば上書きしない。
                ある時期は同じ団体の練習が続く運用なので、既定値を変えたいときは
                `pipeline_defaults.json` を書き換えれば以降の新規 `ingest` に反映される。
                特定の日付だけ別団体にしたい場合はその日付の値を直接編集する。
- `concert_date` / `box_parent_folder_id`:
                Box アップロード用。`orchestra` と同じく `pipeline_defaults.json` から
                `ingest` 時にコピーし、既に値があれば上書きしない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .util import PipelineError, log, read_json, write_json

LR_MAPS = ("normal", "swapped")
SOURCES = ("ext_only", "int_only", "mix")
NORMALIZE_SCOPES = ("date", "block")

DEFAULTS_NAME = "pipeline_defaults.json"
FALLBACK_ORCHESTRA = "Windrose Sinfonie Orchester"

# pipeline_defaults.json から session_config.json へ引き継ぐキーと、その既定値。
INHERITED_DEFAULTS = {"orchestra": FALLBACK_ORCHESTRA, "concert_date": "", "box_parent_folder_id": ""}

# マスタリングのうち「耳でしか良否が決まらない」処理の既定値。
#
# **組み込みの既定はすべて 0(無効)である。** これらは会場と機材で適・不適が変わり、
# ソフトは自分の出来を判定できない。残響が合うかどうかも、静音部をどれだけ持ち上げて
# よいかも、その部屋を聴いた人にしか分からない。知らない環境に強い加工を既定で当てると、
# 使う人は「何かおかしいが原因が分からない」という状態になる。効果を数値で検証できる
# 処理(ラウドネス、トゥルーピーク、コンプ)とは扱いを分ける。
#
# 自分の会場で値が決まったら `pipeline_defaults.json` の `mastering` に書く。そこが
# 各自の環境の置き場で、リポジトリには入らない(`.gitignore` 済み)。この録音環境では
# 残響 0.15 / パラレルコンプ +17 dB を採用しているが、それはこの環境での結論であって
# 既定値ではない。
MASTERING_KEY = "mastering"
MASTERING_DEFAULTS = {"reverb_mix": 0.0, "parallel_db": 0.0, "denoise_db": 0.0}

CONFIG_NAME = "session_config.json"


@dataclass
class SessionConfig:
    ext_lr_map: str = "normal"
    normalize_scope: str = "date"
    source: str = "ext_only"
    mix_ratio: dict[str, float] = field(default_factory=lambda: {"ext": 0.6, "int": 0.4})
    orchestra: str = FALLBACK_ORCHESTRA
    concert_date: str = ""
    box_parent_folder_id: str = ""

    def to_json(self) -> dict:
        return {
            "ext_lr_map": self.ext_lr_map,
            "normalize_scope": self.normalize_scope,
            "source": self.source,
            "mix_ratio": {k: float(v) for k, v in self.mix_ratio.items()},
            "orchestra": self.orchestra,
            "concert_date": self.concert_date,
            "box_parent_folder_id": self.box_parent_folder_id,
        }

    def validate(self) -> None:
        if self.ext_lr_map not in LR_MAPS:
            raise PipelineError(
                f"{CONFIG_NAME}: ext_lr_map は {' / '.join(LR_MAPS)} のいずれかです"
                f"(実際: {self.ext_lr_map!r})"
            )
        if self.normalize_scope not in NORMALIZE_SCOPES:
            raise PipelineError(
                f"{CONFIG_NAME}: normalize_scope は {' / '.join(NORMALIZE_SCOPES)} のいずれかです"
                f"(実際: {self.normalize_scope!r})"
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
        if not str(self.orchestra).strip():
            raise PipelineError(f"{CONFIG_NAME}: orchestra が空です")
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
    # 旧フォーマットに残っている "recorder" キーは無視する(ingest.json が情報源)。
    cfg = SessionConfig(
        ext_lr_map=data.get("ext_lr_map", default.ext_lr_map),
        normalize_scope=data.get("normalize_scope", default.normalize_scope),
        source=data.get("source", default.source),
        mix_ratio=data.get("mix_ratio", default.mix_ratio),
        orchestra=data.get("orchestra", default.orchestra),
        concert_date=str(data.get("concert_date", default.concert_date)),
        box_parent_folder_id=str(data.get("box_parent_folder_id", default.box_parent_folder_id)),
    )
    cfg.validate()
    return cfg


def load_project_defaults(root: Path) -> dict:
    """プロジェクト直下の `pipeline_defaults.json` を読む。無ければ組み込みの既定値。"""
    p = root / DEFAULTS_NAME
    if not p.exists():
        return dict(INHERITED_DEFAULTS)
    data = read_json(p)
    if not isinstance(data, dict):
        raise PipelineError(f"{DEFAULTS_NAME} はオブジェクトである必要があります")
    return {k: str(data.get(k, v)) for k, v in INHERITED_DEFAULTS.items()}


def load_mastering_defaults(root: Path) -> dict[str, float]:
    """`pipeline_defaults.json` の `mastering` を読む。無いキーは 0(無効)。

    日付ごとではなくプロジェクト単位の設定である。会場が変わらないかぎり値も
    変わらないため、`session_config.json` には持たせていない。特定の日だけ変えたい
    ときは `--reverb-mix` などで上書きする。
    """
    p = root / DEFAULTS_NAME
    got = {}
    if p.exists():
        data = read_json(p)
        if not isinstance(data, dict):
            raise PipelineError(f"{DEFAULTS_NAME} はオブジェクトである必要があります")
        got = data.get(MASTERING_KEY) or {}
        if not isinstance(got, dict):
            raise PipelineError(f"{DEFAULTS_NAME} の {MASTERING_KEY} はオブジェクトである必要があります")
    out = {}
    for k, v in MASTERING_DEFAULTS.items():
        try:
            out[k] = float(got.get(k, v))
        except (TypeError, ValueError):
            raise PipelineError(
                f"{DEFAULTS_NAME} の {MASTERING_KEY}.{k} は数値である必要があります: {got.get(k)!r}"
            ) from None
    return out


def ensure(outdir: Path, root: Path, force: bool = False) -> SessionConfig:
    """`ingest` 時に既定値で生成する。既存ファイルの既存の値は上書きしない。

    既存ファイルに `orchestra` が無い場合だけは、`pipeline_defaults.json` の値を
    書き足す(「初回のみコピーする」という指示のため。値があれば触らない)。
    """
    p = config_path(outdir)
    defaults = load_project_defaults(root)

    if p.exists() and not force:
        raw = read_json(p)
        cfg = load(outdir)
        missing = [k for k in INHERITED_DEFAULTS if k not in raw]
        if missing:
            for k in missing:
                setattr(cfg, k, defaults[k])
            cfg.validate()
            write_json(p, cfg.to_json())
            log(f"既存の {CONFIG_NAME} に " +
                ", ".join(f"{k}={defaults[k]!r}" for k in missing) + " を追記しました")
        else:
            log(f"既存の {CONFIG_NAME} を使用: source={cfg.source}, "
                f"ext_lr_map={cfg.ext_lr_map}, orchestra={cfg.orchestra!r}, "
                f"concert_date={cfg.concert_date!r}")
        return cfg

    cfg = SessionConfig(**defaults)
    cfg.validate()
    write_json(p, cfg.to_json())
    log(f"{CONFIG_NAME} を既定値で作成しました (orchestra={cfg.orchestra!r}, concert_date={cfg.concert_date!r}): {p}")
    return cfg
