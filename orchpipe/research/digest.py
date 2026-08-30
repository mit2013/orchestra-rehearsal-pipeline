"""ステージC: ダイジェスト版の作成。

ステージAで `playing` と判定された区間だけを、前後にマージンを付けて抜き出し、
時系列順に結合する。結合点にはクロスフェードを入れてクリックノイズを避ける。

マージンで隣接区間と重なった場合は統合する。指示書 C-3 の
「迷ったら: マージンで演奏を削りすぎるより多めに残す」に従い、
マージンは前後 3 秒を既定とし、統合の閾値も緩めにしてある。
"""

from __future__ import annotations

from pathlib import Path

from .. import features as F
from ..tuning import detect_tuning_events
from ..util import FFMPEG, PipelineError, log, run
from .states import StateSpan

MARGIN_S = 3.0
CROSSFADE_S = 0.075   # 75ms。指示書の 50〜100ms の中間
MIN_KEEP_S = 1.0      # これより短い断片は繋いでも聴き取れないので捨てる

# 予備拍リードタイム。マージンとは目的が異なる独立したパラメータである。
#
# - マージン(`margin`)は**防御的**な値で、「本物の演奏の出だしを検出し損ねている
#   ぶんを取り戻す」ためのもの。検出の遅れという誤差に対する保険であり、誤差が
#   小さければ本来は不要になる性質のもの。
# - 予備拍リードタイム(`lead`)は**音楽的**な値で、演奏の始まりが正確に分かって
#   いても、その手前を必ず残す。指揮者の振り上げ・奏者のブレス・予備拍がここに
#   入るので、これが無いと出だしが唐突に聞こえる。
#
# 目的が違うので合成せず、重ね合わせる。最終的な先頭は
# 「マージンで確定した位置」と「演奏開始 - リードタイム」の early い方になる。
# リードタイムは speech へのクランプを受けない。予備拍の直前に指揮者の合図が
# 入るのはむしろ自然であり、そこを避ける理由がないため。
LEAD_S = 3.0

# 余韻テイル。予備拍リードタイムの終わり側の対応物だが、根拠は別である。
#
# 予備拍が「演奏が始まる前の準備動作を残す」ためのものなのに対し、余韻は
# 「鳴り終わった音が消えるまでを残す」ためのもの。和音の解決、弓を離す音、
# 残響がここに入る。これが無いと、終止形が鳴り切る前にダイジェストが次の
# 範囲へ飛んでしまい、演奏が途中で断ち切られたように聞こえる。
#
# 実測では、マージン 4.0 秒だけの状態で 495 の演奏区間のうち 35 区間が
# 「終わり側の余裕ゼロ」だった。先頭側は予備拍によってゼロが 0 区間まで
# 解消していたので、切れ方の非対称はこの余韻の不在から来ていた。
# 1.0 秒でも 35 区間すべてが解消するが、和音の解決や残響を残すには短いため
# 既定を 1.5 秒とした(5時間半のうち増加は 52 秒、発言の混入は 0.68% -> 1.01%)。
#
# 予備拍と同様、speech へのクランプは受けない。演奏の直後に指揮者が話し始める
# のはむしろ普通であり、そこを避けると余韻がまったく残らなくなるため。
TAIL_S = 1.5


def playing_ranges(spans: list[StateSpan], total: float,
                   margin: float = MARGIN_S,
                   clamp_to_speech: bool = True,
                   lead: float = 0.0,
                   tail: float = 0.0) -> list[tuple[float, float]]:
    """`playing` 区間にマージンを付け、重なりを統合した範囲リスト。

    **マージンは speech 側には広げない。** 当初は前後一律に広げていたが、
    実データでは speech 区間の 76〜98% が 6 秒未満だったため、前後 3 秒の
    マージンが発言をまたいで橋渡ししてしまい、ダイジェストに発言が
    14〜58% 残っていた。

    そこで2段構えにする:

    1. 隣接区間が `speech` なら、その側のマージンを 0 にする
    2. `clamp_to_speech`(既定 True)なら、さらに「最も近い speech 区間の境界」
       までしかマージンを伸ばさない。playing → silence(2秒)→ speech のように
       間に短い silence を挟む場合、1 だけでは 3 秒のマージンが silence を
       越えて speech に届いてしまうため。
    """
    idx = [i for i, s in enumerate(spans) if s.label == "playing"]
    if not idx:
        return []

    speech = [(s.start, s.end) for s in spans if s.label == "speech"]

    raw: list[tuple[float, float]] = []
    for i in idx:
        s = spans[i]
        left = 0.0 if (i > 0 and spans[i - 1].label == "speech") else margin
        right = 0.0 if (i + 1 < len(spans) and spans[i + 1].label == "speech") else margin
        a = max(0.0, s.start - left)
        b = min(total, s.end + right)

        if clamp_to_speech and speech:
            # 直前の speech の終端より前には戻らない
            before = [e for _st, e in speech if e <= s.start]
            if before:
                a = max(a, max(before))
            # 直後の speech の始端より先には出ない
            after = [st for st, _e in speech if st >= s.end]
            if after:
                b = min(b, min(after))

        # 予備拍リードタイムを重ねる。speech クランプの後に適用するので、
        # 直前の指示や合図があっても手前を確保できる。
        if lead > 0.0:
            a = min(a, max(0.0, s.start - lead))

        # 余韻テイルも同じく speech クランプの後に重ねる。
        if tail > 0.0:
            b = max(b, min(total, s.end + tail))

        if b > a:
            raw.append((a, b))

    if not raw:
        return []
    raw.sort()
    merged = [list(raw[0])]
    for a, b in raw[1:]:
        if a <= merged[-1][1]:          # 重なる/接する → 統合
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_KEEP_S]


# --- ブロック冒頭のチューニングの除外 ---------------------------------------
#
# ダイジェストにチューニングは要らない。分類器の `tuning` クラスに頼ると
# 取りこぼす(260829 では 合奏1 のチューニングがほぼ playing に分類され、
# 約100秒がそのままダイジェストの冒頭に入っていた)。
#
# 一方で、境界を決める段階でチューニングの位置は既に正確に分かっている。
# ブロックの開始はチューニング開始の 5 秒前に置いてあるからである
# (`segment.TUNING_PRE_ROLL_S`)。そこで分類の結果によらず、
# **ブロック先頭からチューニング終了まで**を範囲から落とす。
HEAD_SKIP_S = 90.0        # チューニングを検出できなかったときに落とす長さ
HEAD_SEARCH_S = 300.0     # ブロック先頭からこの範囲にあるものを冒頭のチューニングとみなす


def head_tuning_end(src: Path, search_s: float = HEAD_SEARCH_S,
                    fallback_s: float = HEAD_SKIP_S) -> tuple[float, str]:
    """ブロック冒頭のチューニングが終わる時刻を返す。戻り値は (時刻, 説明)。

    検出できなかった場合は安全側に倒して `fallback_s` を返す。
    """
    ff = F.extract_frame_features(src, progress_every=1e9)
    wf = F.aggregate_windows(ff)
    events = detect_tuning_events(ff, wf.silence_db)
    head = [e for e in events if e.start <= search_s]
    if not head:
        return fallback_s, f"チューニングを検出できず、既定の {fallback_s:.0f} 秒を除外"
    e = min(head, key=lambda x: x.start)
    return e.end, (f"チューニング {e.start:.1f}-{e.end:.1f} 秒"
                   f"(純度 {e.purity:.2f} / {e.duration:.0f}秒)を除外")


def drop_head(ranges: list[tuple[float, float]], until: float,
              min_keep: float = MIN_KEEP_S) -> list[tuple[float, float]]:
    """`until` より前を範囲から落とす。またいでいる範囲は頭を詰める。"""
    out: list[tuple[float, float]] = []
    for a, b in ranges:
        if b <= until:
            continue
        a = max(a, until)
        if b - a >= min_keep:
            out.append((a, b))
    return out


def build_digest(src: Path, ranges: list[tuple[float, float]], dst: Path,
                 crossfade: float = CROSSFADE_S) -> dict:
    """範囲を切り出してクロスフェード結合する。出力は 32bit float。"""
    if not ranges:
        raise PipelineError(f"{src.name}: playing 区間がありません")

    dst.parent.mkdir(parents=True, exist_ok=True)
    total_kept = sum(b - a for a, b in ranges)

    # 1本しかないなら素直に切り出すだけ(クロスフェード不要)
    if len(ranges) == 1:
        a, b = ranges[0]
        run([FFMPEG, "-hide_banner", "-v", "error", "-y",
             "-ss", f"{a:.3f}", "-t", f"{b-a:.3f}", "-i", str(src),
             "-c:a", "pcm_f32le", "-rf64", "auto", str(dst)],
            desc=f"    {dst.name}(1区間)")
        return {"n_ranges": 1, "kept_seconds": total_kept}

    # acrossfade は2入力ずつなので、フィルタグラフで数珠つなぎにする。
    # 各区間は crossfade ぶん重ねて繋ぐため、実尺は (n-1)*crossfade だけ短くなる。
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-y"]
    for a, b in ranges:
        cmd += ["-ss", f"{a:.3f}", "-t", f"{b-a:.3f}", "-i", str(src)]

    chains = []
    prev = "0:a"
    for i in range(1, len(ranges)):
        out = f"x{i}"
        chains.append(f"[{prev}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out}]")
        prev = out
    cmd += ["-filter_complex", ";".join(chains), "-map", f"[{prev}]",
            "-c:a", "pcm_f32le", "-rf64", "auto", str(dst)]
    run(cmd, desc=f"    {dst.name}({len(ranges)}区間をクロスフェード結合)")
    return {"n_ranges": len(ranges), "kept_seconds": total_kept,
            "crossfade_loss": (len(ranges) - 1) * crossfade}
