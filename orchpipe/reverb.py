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

既定は Waves IRLive に同梱の実在ホール(`Hall 2`、残響 1.81 秒、立ち上がり 37 ms)。
4種を聴き比べたうえで人が選んだもの(他は Hall 3 が 1.72 秒、Hall 4 が 1.96 秒、
Hall 5 が 3.29 秒)。
これは普通の 48kHz ステレオ WAV なので、そのまま読める。`assets/ir/hall.wav` を
置くとそちらが優先される。Waves を消す予定があるなら、そこへコピーしておくこと。
"""

from __future__ import annotations

from pathlib import Path

from .util import FFMPEG, PipelineError, log, probe_audio, run

# 混ぜる量。0 で無効。0.15 は「言われれば分かる」程度で、輪郭は保たれる。
DEFAULT_MIX = 0.15
# 一度に処理するフレーム数。77 分のブロックを丸ごと読むと 1.8GB になるので刻む。
BLOCK_FRAMES = 1 << 20

WAVES_IR_DIR = Path("/Applications/Waves/Data/IR1Impulses V2/IR-Live Impulses/Halls")
DEFAULT_IR_NAME = "Hall 2.wav"
REPO_IR = Path(__file__).resolve().parent.parent / "assets" / "ir" / "hall.wav"


def default_ir() -> Path:
    """使うインパルス応答。リポジトリ内に置いてあればそちらを優先する。"""
    if REPO_IR.exists():
        return REPO_IR
    return WAVES_IR_DIR / DEFAULT_IR_NAME


def prepare_ir(src: Path, workdir: Path) -> Path:
    """IR を 32bit float の WAV に揃える(24bit のままだと扱えない実装がある)。"""
    if not src.exists():
        raise PipelineError(
            f"インパルス応答が見つかりません: {src}\n"
            f"  Waves の IRLive を入れるか、{REPO_IR} に WAV を置いてください。"
        )
    info = probe_audio(src)
    dst = workdir / f"ir_{src.stem.replace(' ', '_')}_f32.wav"
    if dst.exists():
        return dst
    workdir.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-hide_banner", "-v", "error", "-y", "-i", str(src),
         "-c:a", "pcm_f32le", "-ar", str(info["sample_rate"]), str(dst)],
        desc=f"    IR を変換: {src.name}")
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
