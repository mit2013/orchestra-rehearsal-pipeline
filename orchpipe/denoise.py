"""実測したノイズ形状によるスペクトル減算で、空調などの定常音を抜く。

## なぜ要るのか

会場によっては、天井埋め込み式のエアコンがフルパワーで回っていることがある。
260905(管セクション練習・狭い部屋・低い天井)がそれで、指揮者が話している場面で
「サー」という音がはっきり聞こえた。パラレルコンプで静音部を持ち上げると、
**もともと聞こえていた暗騒音も一緒に持ち上がる**ので、聴き手には「ここから
音をいじったな」と分かってしまう。持ち上げ量を下げれば目立たなくなるが、
今度は指揮者の声が聞き取れない。

そこで、持ち上げる前に**引く**。ユーザーの評価は明確だった。

> 除去なしは管が小さい音で伸ばしている時にかなり気になるので、できればある程度
> 除去したい。-12dBが良さそうだけど「無音」の時だけ、「人工的シュワシュワ」が
> ちょっと気になるかな、程度。演奏への影響込みで -12 が良いです。

## なぜ ffmpeg の `afftdn` ではないのか

試したが、実測で最大 0.6 dB しか減らなかった。理由は二つある。

- `residual_floor` の既定が -38 dB で、暗騒音が -36 dBFS にいるとほぼ頭打ちになる
- `nt=white` は白色雑音を仮定するが、この空調ノイズは白くない。エネルギーの
  **64% が 160〜500 Hz** に寄っていて、100 Hz 以下は 7.9% しかない

つまり形の合わないモデルを当てていた。素材から実際の形を測って引くほうが素直で、
実装も 100 行に収まる。

## 何をしているか

1. **ノイズ形状を測る。** 別途「無音だけ」を録っているわけではないので、
   フレームのエネルギーが下位 10% のものを静かなフレームとみなし、その平均
   パワースペクトルを取る。合奏ブロックには必ず休符と間があるので実用になる。
2. **各フレームから引く。** 過剰減算係数 α=2.0 を掛けて引く。ちょうど 1 倍だと
   引き残しが揺らいで耳につくので、少し多めに引くのが定石である。
3. **引きすぎを止める。** ゲインに `10^(-reduce_db/20)` の下限を置く。これが
   `reduce_db` の実体で、**どれだけ引くかの上限**を決めている。下限を置かないと
   ノイズが完全に消える代わりに、残った成分が単独の正弦波のように鳴る
   (musical noise、ユーザーの言う「シュワシュワ」)。
4. **ゲインを均す。** 周波数方向 3 ビン・時間方向 3 フレームで移動平均する。
   ゲインが隣と大きく違うほど musical noise が出るため。

## -12 dB で何が起きるか

260905 の3ブロックで、無音区間の帯域別の変化を測ったもの(dB)。

    帯域          前半    中盤    後半
    20-100 Hz     -7.5    -7.8    -7.9
    100-160 Hz    -7.6    -7.5    -7.7
    160-250 Hz    -6.8    -6.8    -6.6
    4-8 kHz       -4.2    -4.3    -3.6
    8-15 kHz      -2.0    -2.7    -2.3
    無音区間全体  -7.3    -7.4    -7.5

エネルギーの主たる 160〜500 Hz で 7 dB 前後、高域では 2〜4 dB。低域ほどよく
引けているのは、そこにノイズが集中していて信号との比が悪いからである。

## 処理の順序

**残響のあと、マスターチェーンの前**に置く。残響を先に掛けるのは、残響がノイズも
一緒に伸ばしてしまうと引きにくくなるため……ではなく、逆である。残響は元の音に
対して掛けるべきもので、ノイズを引いたあとの音に掛けると、消し残しだけが
リバーブで伸びて目立つ。マスターチェーンより前なのは `reverb.py` と同じ理由で、
あとに置くとラウドネスとトゥルーピークの保証が崩れる。

## メモリ

STFT を丸ごと配列にすると 57 分のブロックで 5 GB を超えるので、重なり保存法で
一定長ずつ処理する。時間方向の平滑化があるため、チャンクの前後に 3 フレームの
のりしろを取って捨てている。この刻み方をしても出力は一括処理と一致する
(20 チャンクで照合済み)。

それでも `apply_denoise` はファイルを丸ごと読むので、63 分ステレオで 5 GB 前後を
使う。ブロック単位で呼ぶ前提なので、これ以上は詰めていない。
"""

from __future__ import annotations

from pathlib import Path

from .util import PipelineError, log, probe_audio

# 既定の減衰量。260905 でユーザーが「演奏への影響込みで -12 が良い」と決めたもの。
DEFAULT_REDUCE_DB = 12.0

NFFT, HOP = 2048, 512
# 過剰減算係数。1.0 だと引き残しが揺らいで耳につく。
ALPHA = 2.0
# ゲインの平滑化幅(周波数ビン / 時間フレーム)。musical noise を抑える。
SMOOTH_F, SMOOTH_T = 3, 3
# 1回に扱うフレーム数(約 44 秒)と、その前後に取るのりしろ。
CHUNK_FRAMES = 4096
PAD = SMOOTH_T
# ノイズ形状を測るときに「静か」とみなすフレームの割合 [%]。
QUIET_PERCENTILE = 10.0


def _window():
    import numpy as np

    # 端が重複しないハン窓。重なり保存法で足したときに平坦になる。
    return np.hanning(NFFT + 1)[:-1]


def _frames(x, lo: int, hi: int):
    """フレーム `lo`〜`hi` を切り出して窓を掛け、実数FFTを返す。"""
    import numpy as np

    idx = np.arange(NFFT)[None, :] + HOP * np.arange(lo, hi)[:, None]
    return np.fft.rfft(x[idx] * _window(), axis=1)


def _gain(power, noise):
    """パワースペクトルからスペクトル減算のゲインを作る。"""
    import numpy as np

    return np.sqrt(
        np.maximum(power - ALPHA * noise[None, :], 0.0) / np.maximum(power, 1e-20)
    )


def _smooth(gain, floor: float):
    """ゲインに下限を掛けたうえで、周波数方向・時間方向に均す。"""
    import numpy as np

    gain = np.maximum(gain, floor)
    k = np.ones(SMOOTH_F) / SMOOTH_F
    gain = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, gain)
    k = np.ones(SMOOTH_T) / SMOOTH_T
    return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, gain)


def noise_profile(x, pct: float = QUIET_PERCENTILE):
    """静かなフレームの平均パワースペクトル。

    2 パスで走査し、スペクトルを溜め込まない。1 パス目でフレームごとの
    エネルギーだけを集めてしきい値を決め、2 パス目でしきい値以下のものを平均する。
    """
    import numpy as np

    nfr = 1 + (len(x) - NFFT) // HOP
    step = 4096

    def each():
        for i in range(0, nfr, step):
            yield np.abs(_frames(x, i, min(nfr, i + step))) ** 2

    energy = np.concatenate([p.sum(1) for p in each()])
    thr = np.percentile(energy, pct)

    total = np.zeros(NFFT // 2 + 1)
    n = 0
    for p in each():
        sel = p[p.sum(1) <= thr]
        if len(sel):
            total += sel.sum(0)
            n += len(sel)
    return total / max(n, 1)


def denoise(x, noise, reduce_db: float):
    """1チャンネルぶんの波形からノイズを引く。重なり保存法で刻んで処理する。"""
    import numpy as np

    w = _window()
    floor = 10 ** (-reduce_db / 20.0)
    nfr = 1 + (len(x) - NFFT) // HOP
    y = np.zeros(len(x) + NFFT)
    norm = np.zeros(len(x) + NFFT)

    i = 0
    while i < nfr:
        lo = max(0, i - PAD)
        hi = min(nfr, i + CHUNK_FRAMES + PAD)
        spec = _frames(x, lo, hi)
        g = _smooth(_gain(np.abs(spec) ** 2, noise), floor)
        fr = np.fft.irfft(spec * g, NFFT, axis=1) * w
        # のりしろぶんは捨て、本体だけを重ね合わせる
        for j in range(i - lo, min(nfr, i + CHUNK_FRAMES) - lo):
            o = (lo + j) * HOP
            y[o:o + NFFT] += fr[j]
            norm[o:o + NFFT] += w ** 2
        i += CHUNK_FRAMES

    return (y / np.maximum(norm, 1e-8))[:len(x)]


def apply_denoise(src: Path, dst: Path, reduce_db: float = DEFAULT_REDUCE_DB) -> dict:
    """`src` からノイズを引いて `dst` に書く。チャンネルごとに形を測り直す。

    左右でマイクの位置が違えばノイズの入り方も違うので、プロファイルは
    チャンネルごとに取る。
    """
    import numpy as np
    import soundfile as sf

    if reduce_db <= 0:
        raise PipelineError(f"ノイズ除去量は正の値にしてください: {reduce_db}")

    info = probe_audio(src)
    log(f"  ノイズ除去: {dst.name}  <- {src.name} "
        f"(実測ノイズ形状で -{reduce_db:.0f} dB まで引く)")

    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if len(x) < NFFT:
        raise PipelineError(f"{src.name} が短すぎます({len(x)} サンプル)")

    out = np.empty_like(x)
    for c in range(x.shape[1]):
        ch = x[:, c].astype(np.float64)
        out[:, c] = denoise(ch, noise_profile(ch), reduce_db).astype(np.float32)
        del ch
    del x

    sf.write(str(dst), out, sr, subtype="FLOAT")
    return {"reduce_db": reduce_db, "channels": out.shape[1],
            "sample_rate": sr, "duration": round(info["duration"], 3)}
