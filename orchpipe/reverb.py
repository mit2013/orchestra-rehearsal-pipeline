"""ホールの残響を足す(畳み込みリバーブ)。

## なぜ足すのか

練習会場の響きはコンサートホールより短く、通しで聴くと平板に感じる。実在ホールの
インパルス応答を少量だけ混ぜると、同じ演奏が「ホールで聴いている」ように寄る。
音量特性は変わらないので、`loudness.py` のラウドネス正規化やトゥルーピークの
保証には影響しない(順序としてリバーブが先、マスターチェーンが後)。

## なぜ ffmpeg ではなく pedalboard なのか

ffmpeg の `afir` はこの環境で wet 側が出力されない(`dry=0` にすると完全な無音、
`dry` を変えると出力全体がその倍率で動く)。24bit の IR では `Invalid argument` で
落ちるという別の問題もあり、当てにできなかった。`pedalboard.Convolution` は
素直に動く。pedalboard は pip で入る純粋なライブラリで、**プラグイン本体は要らない**。

## インパルス応答

既定は Waves Complete IR Library に入っている**実在ホールの実測データ**
(`Birmingham Symphony Hall`、残響 2.00 秒)。世界有数のオーケストラ用ホールで、
測った減衰も実際の公称値(1.85〜2.4 秒)と合っている。

このライブラリは `.wir` という拡張子だが、**中身は RIFF WAVE のヘッダ識別子を
2箇所差し替えただけ**である(`RIFF`→`wvIR`、`WAVE`→`ver1`、加えて
bits/sample に 23 という無効な値が入っている)。`read_wir` で元に戻せば
標準の 32bit float WAV として読める。

ファイル名の末尾は収録条件を表す。`m`=モノ / `x`=XYステレオ(2ch) /
`s`=トゥルーステレオ(4ch)、`o`=無指向 / `c`=単一指向、末尾の数字はマイク位置。
ここでは `x` の 2ch を使う。

畳み込みの前に IR へ二つ手を入れている。

- **直接音より前を捨て、先頭に 30 ms の無音を足す。** 実測 IR は直接音が 1 ms
  付近から始まるので、そのまま混ぜると残響が音にべったり張り付いて輪郭が鈍る。
  以前使っていた Waves IRLive の `Hall 2` は 37 ms のプリディレイを持っていて、
  「言われれば分かる程度で輪郭は保たれる」という評価はその状態に対するものだった。
  そこを揃えている。
- **エネルギーを 1 に正規化する。** ホールごとにピークが 0.2〜4476 とばらばらなので、
  そのままでは `--reverb-mix` の意味がホームごとに変わってしまう。

`assets/ir/hall.wav` を置くとそちらが優先される。Waves を消す予定があるなら、
そこへ書き出しておくこと(`pipeline.py` の `--reverb-ir` でも直接指定できる)。
なおこのライブラリは Waves のライセンス品なので、**公開リポジトリには同梱できない**。
"""

from __future__ import annotations

from pathlib import Path

from .util import FFMPEG, PipelineError, log, probe_audio, run

# 混ぜる量。0 で無効。0.15 は「言われれば分かる」程度で、輪郭は保たれる。
DEFAULT_MIX = 0.15
# 一度に処理するフレーム数。77 分のブロックを丸ごと読むと 1.8GB になるので刻む。
BLOCK_FRAMES = 1 << 20

# 直接音の前に置く余裕。実測 IR は直接音から始まるので、ここで作ってやる。
PREDELAY_MS = 30.0

WAVES_IR_DIR = Path("/Applications/Waves/Data/IR1Impulses V2/IR-Live Impulses/Halls")
LEGACY_IR_NAME = "Hall 2.wav"
REPO_IR = Path(__file__).resolve().parent.parent / "assets" / "ir" / "hall.wav"

# Waves Complete IR Library(別途ダウンロードするもの。Waves 本体には含まれない)
WIR_LIBRARY = (Path.home() / "Documents" / "Waves" / "Waves_Complete_IR_Library"
               / "Sampled Acoustics V2")
DEFAULT_HALL = "Concert Halls/Birmingham Symphony Hall"


def hall_ir(name: str = DEFAULT_HALL) -> Path | None:
    """ホール名から XY ステレオ(2ch)の `.wir` を1本選ぶ。

    無指向(`o`)を優先し、無ければ単一指向(`c`)。同じホールでもマイク位置が
    複数あるので、名前順で最初のものを採る。
    """
    d = WIR_LIBRARY / name
    if not d.is_dir():
        return None
    xs = sorted(f for f in d.glob("*.wir") if f.name.rsplit("_", 1)[-1].startswith("x"))
    if not xs:
        return None
    omni = [f for f in xs if "o" in f.name.rsplit("_", 1)[-1][:2]]
    return (omni or xs)[0]


def default_ir() -> Path:
    """使うインパルス応答。リポジトリ内に置いてあればそちらを優先する。"""
    if REPO_IR.exists():
        return REPO_IR
    hall = hall_ir()
    if hall is not None:
        return hall
    return WAVES_IR_DIR / LEGACY_IR_NAME


def read_wir(path: Path):
    """Waves の `.wir` を numpy 配列で読む。ヘッダを標準の RIFF WAVE に戻すだけ。"""
    import io
    import struct

    import soundfile as sf

    b = bytearray(path.read_bytes())
    if b[0:4] == b"wvIR":
        b[0:4] = b"RIFF"
        b[8:12] = b"WAVE"
        # bits/sample に 23 が入っている。fmt=3(float)/ブロック長4 なので 32 が正。
        struct.pack_into("<H", b, 0x22, 32)
    elif b[0:4] != b"RIFF":
        raise PipelineError(f"{path.name} は WAV でも .wir でもありません")
    return sf.read(io.BytesIO(bytes(b)), always_2d=True)


def prepare_ir(src: Path, workdir: Path, predelay_ms: float = PREDELAY_MS) -> Path:
    """IR を畳み込みに使える形に整える。

    32bit float の WAV に揃えたうえで、直接音より前を捨て、`predelay_ms` の
    無音を先頭に足し、エネルギーを 1 に正規化する。詳細はモジュールの説明を参照。
    """
    import numpy as np
    import soundfile as sf

    if not src.exists():
        raise PipelineError(
            f"インパルス応答が見つかりません: {src}\n"
            f"  Waves の IR ライブラリを入れるか、{REPO_IR} に WAV を置いてください。"
        )
    tag = f"{src.stem.replace(' ', '_')}_pd{int(predelay_ms)}"
    dst = workdir / f"ir_{tag}_f32.wav"
    if dst.exists():
        return dst
    workdir.mkdir(parents=True, exist_ok=True)

    x, sr = read_wir(src)
    x = x.astype("float64")

    # 直接音の手前を落とす。全チャンネルの和で位置を決める(chごとにずらさない)。
    pk = int(np.argmax(np.abs(x).sum(axis=1)))
    x = x[pk:]

    pad = int(round(predelay_ms / 1000.0 * sr))
    if pad > 0:
        x = np.vstack([np.zeros((pad, x.shape[1])), x])

    energy = float(np.sqrt((x ** 2).sum()))
    if energy <= 0:
        raise PipelineError(f"{src.name} は無音です")
    x = x / energy

    sf.write(str(dst), x.astype("float32"), sr, subtype="FLOAT")
    log(f"    IR を整えました: {src.name} -> {dst.name} "
        f"({x.shape[1]}ch / {sr} Hz / {len(x)/sr:.2f} 秒 / "
        f"プリディレイ {predelay_ms:.0f} ms)")
    return dst


def apply_reverb(src: Path, dst: Path, ir: Path, mix: float = DEFAULT_MIX,
                 block: int = BLOCK_FRAMES) -> dict:
    """`src` に残響を足して `dst` に書く。長い音源でも刻んで処理する。"""
    import numpy as np
    import pedalboard
    from pedalboard.io import AudioFile

    info = probe_audio(src)
    ir_info = probe_audio(ir)
    log(f"  残響を付加: {dst.name}  <- {src.name} + {ir.name} "
        f"(混合 {mix*100:.0f}% / IR {ir_info['duration']:.2f} 秒)")

    board = pedalboard.Pedalboard([pedalboard.Convolution(str(ir), float(mix))])
    tail = int(ir_info["duration"] * info["sample_rate"]) + 1

    with AudioFile(str(src)) as f:
        sr, ch, total = f.samplerate, f.num_channels, f.frames
        written = 0
        with AudioFile(str(dst), "w", sr, ch) as out:
            while f.tell() < f.frames:
                y = board(f.read(block), sr, reset=False)
                out.write(y); written += y.shape[1]
            # 畳み込みに遅れがある実装のために、無音を流して残りを吐き出させる。
            # ただし**元と同じ長さで打ち切る**。ブロックの尺が変わると、確定境界や
            # ダイジェストの区間と食い違うためである。末尾は音出しなので実害はない。
            while written < total:
                y = board(np.zeros((ch, min(tail, total - written + tail)), dtype="float32"),
                          sr, reset=False)
                take = min(y.shape[1], total - written)
                if take <= 0:
                    break
                out.write(y[:, :take]); written += take
    return {"ir": str(ir), "mix": mix, "ir_seconds": round(ir_info["duration"], 3),
            "frames": written}
