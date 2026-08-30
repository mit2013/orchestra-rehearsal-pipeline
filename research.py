#!/usr/bin/env python3
"""研究ブランチ用のエントリポイント(音源分析の高度化・試作)。

    python research.py states  --date 260802   # ステージA: 4値ステート分類 + ASR
    python research.py onset   --date 260802   # ステージB: 合奏開始オンセット検出
    python research.py digest  --date 260802   # ステージC: ダイジェスト版
    python research.py transcript --date 260802 # ステージD: 書き起こし整形
    python research.py sections --date 260802  # ステージE: 楽章切れ目候補
    python research.py all     --date 260802   # A→C→D→E を通しで

main の `pipeline.py` は一切変更していない。既存モジュールは import して
再利用するだけで、変更もしていない(`features.aggregate_windows` は元から
win_s/hop_s を引数に取るため、細粒度で呼ぶだけで足りた)。

出力はすべて `output/{date}/research/` 配下。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from orchpipe.export import find_final_files
from orchpipe.research import digest as digest_mod
from orchpipe.research import onset as onset_mod
from orchpipe.research import asr as asr_mod
from orchpipe.research import diagnose as diag_mod
from orchpipe.research import review as review_mod
from orchpipe.research import sections as sections_mod
from orchpipe.research import states2 as states2_mod
from orchpipe.research import visualize as viz_mod
from orchpipe.research.asr import AsrSegment, Transcriber, merge_utterances
from orchpipe.research.diagnose import load_states_and_asr
from orchpipe.research.states import (
    StateSpan,
    classify_block,
    format_summary,
    summarize,
)
from orchpipe.util import PipelineError, fmt_time, log, out_dir, probe_audio, write_json

ROOT_DEFAULT = Path(__file__).resolve().parent


def research_dir(outdir: Path) -> Path:
    d = outdir / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def block_key(p: Path) -> str:
    """`01_合奏1_final.wav` -> `01_合奏1`"""
    return p.name[: -len("_final.wav")]


def load_states(rdir: Path, key: str) -> tuple[list[StateSpan], dict]:
    data = json.loads((rdir / f"{key}_states.json").read_text(encoding="utf-8"))
    spans = [StateSpan(s["start"], s["end"], s["label"], s["confidence"])
             for s in data["spans"]]
    return spans, data["meta"]


def load_asr(rdir: Path, key: str) -> list[AsrSegment]:
    data = json.loads((rdir / f"{key}_asr.json").read_text(encoding="utf-8"))
    return [AsrSegment(s["start"], s["end"], s["text"], s["no_speech_prob"],
                       s["avg_logprob"], s["compression_ratio"],
                       s["accepted"], s.get("reject_reason", ""))
            for s in data["segments"]]


# ---------------------------------------------------------------------------
# ステージA
# ---------------------------------------------------------------------------

def cmd_states(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = find_final_files(outdir / "trimmed")
    if not finals:
        raise PipelineError(f"{outdir/'trimmed'} に *_final.wav がありません")
    if args.only:
        finals = [p for p in finals if args.only in p.name]
        if not finals:
            raise PipelineError(f"--only {args.only!r} に一致するブロックがありません")
        log(f"対象ブロックを限定: {', '.join(p.name for p in finals)}")

    tr = None if args.reuse_asr else Transcriber(args.model, threads=args.threads)
    started = time.time()
    for i, p in enumerate(finals, start=1):
        key = block_key(p)
        log(f"[{i}/{len(finals)}] {key}")
        t0 = time.time()
        prev = load_asr(rdir, key) if args.reuse_asr else None
        spans, tunings, asr, meta = classify_block(p, tr, reuse_asr=prev)
        meta["elapsed_sec"] = round(time.time() - t0, 1)
        meta["model"] = args.model

        write_json(rdir / f"{key}_states.json",
                   {"meta": meta, "summary": summarize(spans),
                    "spans": [s.to_json() for s in spans]})
        write_json(rdir / f"{key}_asr.json",
                   {"meta": {"model": args.model, "source": str(p)},
                    "segments": [s.to_json() for s in asr]})
        write_json(rdir / f"{key}_tuning.json",
                   {"events": [e.to_json() for e in tunings]})
        print(format_summary(key, spans, meta["duration"]))
        log(f"    所要 {meta['elapsed_sec']:.0f} 秒")
    log(f"ステージA 完了: 合計 {time.time()-started:.0f} 秒")


# ---------------------------------------------------------------------------
# ステージC
# ---------------------------------------------------------------------------

def cmd_digest(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = find_final_files(outdir / "trimmed")
    rows = []
    for p in finals:
        key = block_key(p)
        spans, meta = load_states(rdir, key)
        total = meta["duration"]
        ranges = digest_mod.playing_ranges(spans, total, margin=args.margin)
        dst = rdir / f"{key}_digest.wav"
        log(f"  {key}: playing {len([s for s in spans if s.label=='playing'])} 区間 "
            f"-> マージン統合後 {len(ranges)} 範囲")
        info = digest_mod.build_digest(p, ranges, dst, crossfade=args.crossfade)
        kept = info["kept_seconds"] - info.get("crossfade_loss", 0.0)
        rows.append({"block": key, "original_sec": round(total, 1),
                     "digest_sec": round(kept, 1),
                     "ratio": round(kept / total, 4),
                     "n_ranges": info["n_ranges"], "path": str(dst)})
        log(f"    {fmt_time(total)} -> {fmt_time(kept)}  (圧縮率 {kept/total*100:.1f}%)")
    write_json(rdir / "digest_summary.json", {"blocks": rows})


# ---------------------------------------------------------------------------
# ステージD
# ---------------------------------------------------------------------------

def cmd_transcript(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = find_final_files(outdir / "trimmed")
    for p in finals:
        key = block_key(p)
        segs = load_asr(rdir, key)
        acc = [s for s in segs if s.accepted]
        utts = merge_utterances(acc, max_gap=args.merge_gap)
        # json は元の細かいタイムスタンプを保持したまま、連結後の並びを併記する。
        write_json(rdir / f"{key}_transcript.json",
                   {"meta": {"block": key, "n_total": len(segs), "n_accepted": len(acc),
                             "n_utterances": len(utts), "merge_gap_sec": args.merge_gap},
                    "segments": [{"start": round(s.start, 2), "end": round(s.end, 2),
                                  "text": s.text} for s in acc],
                    "utterances": [u.to_json() for u in utts]})
        # txt は読みやすさ優先。連結後のまとまりを1行にする。
        lines = [f"# {args.date} {key} 書き起こし",
                 f"# {len(utts)} 発言(採用 {len(acc)} / 全 {len(segs)} セグメントを "
                 f"間隔 {args.merge_gap} 秒以内で連結)", ""]
        for u in utts:
            lines.append(f"[{fmt_time(u.start)} - {fmt_time(u.end)}] {u.text}")
        (rdir / f"{key}_transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"  {key}: {len(utts)} 発言(採用 {len(acc)} / 全 {len(segs)} セグメント)")


# ---------------------------------------------------------------------------
# ステージE
# ---------------------------------------------------------------------------

def cmd_sections(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = find_final_files(outdir / "trimmed")
    for p in finals:
        key = block_key(p)
        segs = load_asr(rdir, key)
        utts = merge_utterances([s for s in segs if s.accepted], max_gap=args.merge_gap)
        cands = sections_mod.dedupe(sections_mod.find_candidates(utts))
        write_json(rdir / f"{key}_section_candidates.json",
                   {"meta": {"block": key, "n_candidates": len(cands)},
                    "candidates": [c.to_json() for c in cands]})
        log(f"  {key}: 候補 {len(cands)} 件")
        for c in cands:
            print(f"    [{fmt_time(c.time)}] {c.kind:<14} 「{c.matched}」  … {c.text[:44]}")


# ---------------------------------------------------------------------------
# ステージF-2: 視覚的な一覧
# ---------------------------------------------------------------------------

def cmd_visualize(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = find_final_files(outdir / "trimmed")
    if args.block:
        finals = [p for p in finals if args.block in p.name]
        if not finals:
            raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")
    all_written = []
    for p in finals:
        key = block_key(p)
        spans, meta = load_states(rdir, key)
        dst = rdir / "eval" / key / "overview"
        log(f"  {key}: 全長 {fmt_time(meta['duration'])} を {args.chunk/60:.0f} 分ごとに描画")
        all_written += viz_mod.render_block(p, spans, meta["duration"], dst, key,
                                            chunk_s=args.chunk)
    print()
    print(f"=== ステージF-2: 視覚的な一覧 ({args.date}) ===")
    print(f"  {len(all_written)} 枚を書き出しました")
    if all_written:
        print(f"  出力先: {all_written[0].parent}")


# ---------------------------------------------------------------------------
# ステージG-1: 演奏を特徴づける音響的な違いの実測
# ---------------------------------------------------------------------------

def cmd_diagnose(args) -> None:
    import numpy as np

    from orchpipe import features as feat

    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = [p for p in find_final_files(outdir / "trimmed") if args.block in p.name]
    if not finals:
        raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")
    src = finals[0]
    key = block_key(src)

    speech_spans, meta = load_states_and_asr(rdir, key)

    # 既存の解析は再生成前の音源に対するものであることがある。尺の差を
    # そのまま平行移動量として扱う(先頭に足された分だけ全体がずれるため)。
    cur = probe_audio(src)["duration"]
    offset = args.offset if args.offset is not None else round(cur - meta["duration"], 2)
    if abs(offset) > 0.01:
        log(f"既存の解析は {meta['duration']:.1f} 秒、現在の音源は {cur:.1f} 秒。"
            f"参照区間を {offset:+.2f} 秒ずらして現在の音源に合わせます")

    cache = rdir / f"{key}_ff_fine.npz"
    if cache.exists() and not args.force:
        log(f"特徴量キャッシュを再利用: {cache.name}")
        ff = feat.load_frame_features(cache)
    else:
        ff = feat.extract_frame_features(src)
        feat.save_frame_features(cache, ff)

    # 参照区間の抽出は「解析に使った座標」で行い、最後に offset で現在の音源へ移す。
    # RMS 列そのものは現在の音源から取っているので、ここでは offset を 0 にして、
    # ASR 由来の区間だけを offset ぶんずらす。
    shifted = [(a + offset, b + offset, t) for a, b, t in speech_spans]
    refs = diag_mod.collect_references(ff.rms_db, ff.times, shifted, offset=0.0)
    log(f"参照区間: {len(refs)} 件")

    rows = diag_mod.measure(src, refs)
    rep = diag_mod.separation_report(rows)
    write_json(rdir / f"{key}_g1_reference_features.json",
               {"meta": {"block": key, "source": str(src), "offset_sec": offset,
                         "n_refs": len(refs)},
                "summary": rep, "segments": rows})

    print()
    print(f"=== ステージG-1: 分離性能 ({args.date} {key}) ===")
    print(diag_mod.format_report(rep))
    print()
    print(f"  詳細: {rdir / f'{key}_g1_reference_features.json'}")


# ---------------------------------------------------------------------------
# ステージG-2: 積極的な証拠にもとづく分類
# ---------------------------------------------------------------------------

def cmd_states2(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = [p for p in find_final_files(outdir / "trimmed") if args.block in p.name]
    if not finals:
        raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")

    for src in finals:
        key = block_key(src)
        segs = load_asr(rdir, key)
        speech = [(s.start, s.end) for s in segs if s.accepted]
        log(f"{key}: ASR の発話区間 {len(speech)} 件を証拠として使います")

        spans, meta = states2_mod.classify(
            src, speech, win_s=args.win, hop_s=args.hop,
            play_threshold=args.play_threshold,
            hyst_enter=args.hyst_enter, hyst_stay=args.hyst_stay,
            loud_margin_db=args.loud_margin)
        spans, fill = states2_mod.fill_playing_gaps(spans, max_s=args.fill_gap)
        meta["gap_fill"] = fill
        log(f"  演奏に挟まれた短い unclear を埋め戻し: {fill['n_filled']} 件 / "
            f"{fill['seconds_filled']:.0f} 秒(上限 {fill['max_fill_s']:.0f} 秒)")
        spans, br = states2_mod.bridge_playing(
            spans, max_s=args.bridge, max_silence_run_s=args.bridge_max_silence)
        meta["bridge"] = br
        log(f"  発言を含まない非 playing を文脈で橋渡し: {br['n_bridged']} 件 / "
            f"{br['seconds_bridged']:.0f} 秒(上限 {br['bridge_max_s']:.0f} 秒)")
        spans, warm = states2_mod.mark_trailing_warmup(
            src, spans, meta["duration"],
            end_gap_s=args.warmup_gap, dyn_th=args.warmup_dyn)
        meta["trailing_warmup"] = warm
        log(f"  末尾の音出し判定: {'除去 ' + str(warm['seconds']) + ' 秒' if warm['marked'] else '該当なし'}"
            f" — {warm['reason']}")
        summary = states2_mod.summarize(spans, meta["duration"])
        write_json(rdir / f"{key}_states2.json",
                   {"meta": meta, "summary": summary,
                    "spans": [s.to_json() for s in spans]})

        print()
        print(f"=== ステージG-2: 新しい分類 ({args.date} {key}) ===")
        print(f"  全長 {fmt_time(meta['duration'])} / 窓 {meta['win_s']}s / "
              f"演奏の閾値 {meta['play_threshold']}")
        for lab in states2_mod.LABELS:
            s = summary[lab]
            print(f"    {lab:<9} {s['seconds']/60:7.1f}分  {s['ratio']*100:5.1f}%  "
                  f"({s['count']} 区間)")


# ---------------------------------------------------------------------------
# ステージG-3: 新しい分類でのダイジェストと、旧版との比較
# ---------------------------------------------------------------------------

def load_states2(rdir: Path, key: str):
    data = json.loads((rdir / f"{key}_states2.json").read_text(encoding="utf-8"))
    spans = [states2_mod.Span(s["start"], s["end"], s["label"], s["confidence"])
             for s in data["spans"]]
    return spans, data["meta"]


def cmd_digest2(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = [p for p in find_final_files(outdir / "trimmed") if args.block in p.name]
    if not finals:
        raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")

    rows = []
    for p in finals:
        key = block_key(p)
        spans, meta = load_states2(rdir, key)
        total = meta["duration"]
        ranges = digest_mod.playing_ranges(spans, total, margin=args.margin,
                                           lead=args.lead, tail=args.tail)
        dst = rdir / f"{key}_digest2.wav"
        log(f"  {key}: playing {len([s for s in spans if s.label=='playing'])} 区間 "
            f"-> マージン統合後 {len(ranges)} 範囲")
        # チューニングの除外は最後に当てる。分類の結果を問わず範囲から落とす。
        if args.head_skip >= 0:
            until, why = digest_mod.head_tuning_end(p, fallback_s=args.head_skip)
            before = len(ranges)
            ranges = digest_mod.drop_head(ranges, until)
            log(f"    冒頭を除外: {why} -> 範囲 {before} から {len(ranges)}")
        info = digest_mod.build_digest(p, ranges, dst, crossfade=args.crossfade)
        kept = info["kept_seconds"] - info.get("crossfade_loss", 0.0)
        rows.append({"block": key, "original_sec": round(total, 1),
                     "digest_sec": round(kept, 1), "ratio": round(kept / total, 4),
                     "n_ranges": info["n_ranges"], "path": str(dst)})
        log(f"    {fmt_time(total)} -> {fmt_time(kept)}  (圧縮率 {kept/total*100:.1f}%)")
    write_json(rdir / "digest2_summary.json", {"blocks": rows})


def cmd_compare(args) -> None:
    """旧分類と新分類を同じ形式の画像で並べて出す(指示書 G-3)。"""
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = [p for p in find_final_files(outdir / "trimmed") if args.block in p.name]
    if not finals:
        raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")

    written = []
    for p in finals:
        key = block_key(p)
        old_spans, old_meta = load_states(rdir, key)
        new_spans, new_meta = load_states2(rdir, key)
        total = new_meta["duration"]
        for tag, spans in (("old", old_spans), ("new", new_spans)):
            dst = rdir / "eval" / key / f"overview_{tag}"
            log(f"  {key} [{tag}]: {fmt_time(total)} を {args.chunk/60:.0f} 分ごとに描画")
            written += viz_mod.render_block(p, spans, total, dst, f"{key}_{tag}",
                                            chunk_s=args.chunk)
    print()
    print(f"=== ステージG-3: 新旧の比較画像 ({args.date}) ===")
    print(f"  {len(written)} 枚")
    for d in sorted({w.parent for w in written}):
        print(f"    {d}")


def cmd_review(args) -> None:
    """採用範囲の境界について、レビュー用クリップと記入シートを出す。"""
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    finals = [p for p in find_final_files(outdir / "trimmed") if args.block in p.name]
    if not finals:
        raise PipelineError(f"--block {args.block!r} に一致するブロックがありません")

    for p in finals:
        key = block_key(p)
        spans, meta = load_states2(rdir, key)
        total = meta["duration"]
        ranges = digest_mod.playing_ranges(spans, total, margin=args.margin,
                                           lead=args.lead, tail=args.tail)
        dst = rdir / "eval" / key / "boundaries"
        clips = review_mod.build(p, key, ranges, spans, dst,
                                 crossfade=args.crossfade,
                                 pre_s=args.pre, post_s=args.post, total=total)
        review_mod.write_sheet(clips, dst.parent / f"{key}_boundary_review.csv")
        print()
        print(f"=== 境界レビュー ({args.date} {key}) ===")
        print(f"  範囲 {len(ranges)} / クリップ {len(clips)} 本 "
              f"(マージン {args.margin}s + 予備拍 {args.lead}s + 余韻 {args.tail}s)")
        print(f"  クリップ: {dst}")
        print(f"  シート  : {dst.parent / f'{key}_boundary_review.csv'}")


# ---------------------------------------------------------------------------
# ステージB
# ---------------------------------------------------------------------------

# B-0 で人間が確定させた正解時刻(結合ファイルのタイムライン、秒)。
# 未入力の間は None。入力され次第ここを埋める。
# 単位は秒。人間が聴き取ったチューニング区間の「先頭」。
#   260726 は管セク練習で、オーボエのA→各管楽器が合わせる単一フェーズ(約20秒)。
#   260802 はフルオケで、管楽器チューニング→一旦停止→弦楽器チューニングの
#   2段階構成(約100秒)。静かなオーボエ単独音から始まるため難易度が高い。
GROUND_TRUTH: dict[str, dict[str, float | None]] = {
    "260802": {"合奏1": 601.0, "合奏2": 6043.0},          # 00:10:01 / 01:40:43
    "260726": {"合奏1": 517.0, "合奏2": 4471.0, "合奏3": 8689.0},  # 00:08:37 / 01:14:31 / 02:24:49
}

# B-0 の概算時刻(参考値。正解が無い間の探索アンカーとして使う)
ANCHORS: dict[str, list[tuple[str, float]]] = {
    "260802": [("合奏1", 522.0), ("合奏2", 6114.0)],
    "260726": [("合奏1", 496.0), ("合奏2", 4468.0), ("合奏3", 8712.0)],
}


def cmd_onset(args) -> None:
    outdir = out_dir(args.root, args.date)
    rdir = research_dir(outdir)
    merged = outdir / "raw_merged_ext.wav"
    if not merged.exists():
        raise PipelineError(
            f"{merged} がありません。`pipeline.py merge --date {args.date} --groups ext` を先に実行してください。"
        )
    anchors = ANCHORS[args.date]
    refs = {k: v for k, v in GROUND_TRUTH.get(args.date, {}).items() if v is not None}
    results = onset_mod.detect_onsets(merged, anchors, references=refs)

    write_json(rdir / "onset_results.json",
               {"meta": {"date": args.date, "has_ground_truth": bool(refs)},
                "results": [r.to_json() for r in results]})
    print()
    print(f"=== ステージB: 合奏開始オンセット ({args.date}) ===")
    for r in results:
        det = "検出なし" if r.detected is None else fmt_time(r.detected)
        ref = "未取得" if r.reference is None else fmt_time(r.reference)
        err = "—" if r.error is None else f"{r.error:+.1f}秒"
        mark = "" if r.error is None else ("  ✓ ±3秒以内" if abs(r.error) <= 3.0 else "  ★ 基準未達")
        print(f"  {r.block:<8} 検出 {det:<10} 正解 {ref:<10} 誤差 {err:<8}{mark}")


def cmd_all(args) -> None:
    cmd_states(args)
    cmd_digest(args)
    cmd_transcript(args)
    cmd_sections(args)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--date", required=True)
        sp.add_argument("--root", type=Path, default=ROOT_DEFAULT)
        return sp

    def asr_opts(sp):
        sp.add_argument("--model", default="medium", help="faster-whisper のモデルサイズ")
        sp.add_argument("--threads", type=int, default=8)
        sp.add_argument("--only", default=None, help="ブロック名の一部で対象を限定")
        sp.add_argument("--reuse-asr", action="store_true",
                        help="保存済みの *_asr.json を再利用し、ASR を再実行しない")
        return sp

    def text_opts(sp):
        sp.add_argument("--merge-gap", type=float, default=asr_mod.MERGE_GAP_S,
                        help="この間隔(秒)以内で隣接する発話セグメントを連結する")
        return sp

    def digest_opts(sp):
        sp.add_argument("--margin", type=float, default=digest_mod.MARGIN_S)
        sp.add_argument("--crossfade", type=float, default=digest_mod.CROSSFADE_S)
        return sp

    asr_opts(common(sub.add_parser("states"))).set_defaults(func=cmd_states)
    common(sub.add_parser("onset")).set_defaults(func=cmd_onset)

    sp = common(sub.add_parser("visualize", help="ステージF-2: 波形+スペクトログラム+予測ラベルの一覧"))
    sp.add_argument("--block", default=None, help="ブロック名の一部で対象を限定")
    sp.add_argument("--chunk", type=float, default=viz_mod.CHUNK_S, help="1枚あたりの秒数")
    sp.set_defaults(func=cmd_visualize)
    sp = common(sub.add_parser("diagnose", help="ステージG-1: 演奏/非演奏の分離性能を実測"))
    sp.add_argument("--block", default="合奏2", help="ブロック名の一部で対象を限定")
    sp.add_argument("--offset", type=float, default=None,
                    help="既存解析から現在の音源への平行移動[秒](既定は尺の差から自動)")
    sp.add_argument("--force", action="store_true", help="特徴量キャッシュを作り直す")
    sp.set_defaults(func=cmd_diagnose)
    sp = common(sub.add_parser("states2", help="ステージG-2: 積極的な証拠にもとづく分類"))
    sp.add_argument("--block", default="合奏2", help="ブロック名の一部で対象を限定")
    sp.add_argument("--win", type=float, default=states2_mod.WIN_S)
    sp.add_argument("--hop", type=float, default=states2_mod.HOP_S)
    sp.add_argument("--play-threshold", type=float, default=0.5,
                    help="この確信度以上を playing とする(高いほど厳しく捨てる)")
    sp.add_argument("--fill-gap", type=float, default=states2_mod.MAX_FILL_S,
                    help="演奏に挟まれたこの長さ以下の unclear を playing に埋め戻す")
    sp.add_argument("--bridge", type=float, default=states2_mod.BRIDGE_MAX_S,
                    help="演奏に挟まれ発言を含まない非 playing を、この長さまで橋渡しする"
                         "(0 で無効)")
    sp.add_argument("--bridge-max-silence", type=float,
                    default=states2_mod.BRIDGE_MAX_SILENCE_RUN_S,
                    help="隙間の中で無音がこれより長く続いていたら橋渡ししない[秒]")
    sp.add_argument("--hyst-enter", type=float, default=states2_mod.HYST_ENTER_H,
                    help="調和性がこの値以上で演奏に入る")
    sp.add_argument("--hyst-stay", type=float, default=states2_mod.HYST_STAY_H,
                    help="調和性がこの値を割るまで演奏のままにする")
    sp.add_argument("--loud-margin", type=float, default=states2_mod.LOUD_MARGIN_DB,
                    help="無音閾値からこれだけ上のレベルは調和性を問わず playing とする")
    sp.add_argument("--warmup-gap", type=float, default=states2_mod.WARMUP_END_GAP_S,
                    help="最後の playing 区間がブロック末尾からこの秒数以内で終わるとき音出しを疑う")
    sp.add_argument("--warmup-dyn", type=float, default=states2_mod.WARMUP_DYN_TH,
                    help="音量の起伏[dB]がこれ未満なら音出しと判断する(0 で無効)")
    sp.set_defaults(func=cmd_states2)
    sp = digest_opts(common(sub.add_parser("digest2", help="ステージG-3: 新分類でのダイジェスト")))
    sp.add_argument("--block", default="合奏2", help="ブロック名の一部で対象を限定")
    sp.add_argument("--head-skip", type=float, default=digest_mod.HEAD_SKIP_S,
                    help="冒頭のチューニングを検出できなかったときに落とす長さ[秒]。"
                         "負の値で冒頭除外そのものを無効にする")
    sp.add_argument("--lead", type=float, default=digest_mod.LEAD_S,
                    help="予備拍リードタイム[秒]。演奏開始の手前を常にこれだけ残す")
    sp.add_argument("--tail", type=float, default=digest_mod.TAIL_S,
                    help="余韻テイル[秒]。演奏終了の後ろを常にこれだけ残す")
    sp.set_defaults(func=cmd_digest2)

    sp = digest_opts(common(sub.add_parser("review", help="採用範囲の境界のレビュー用クリップとCSV")))
    sp.add_argument("--block", default="合奏2", help="ブロック名の一部で対象を限定")
    sp.add_argument("--lead", type=float, default=digest_mod.LEAD_S)
    sp.add_argument("--tail", type=float, default=digest_mod.TAIL_S)
    sp.add_argument("--pre", type=float, default=review_mod.PRE_S,
                    help="カット位置より手前をクリップに含める秒数")
    sp.add_argument("--post", type=float, default=review_mod.POST_S)
    sp.set_defaults(func=cmd_review)

    sp = common(sub.add_parser("compare", help="ステージG-3: 旧新の分類を並べた画像"))
    sp.add_argument("--block", default="合奏2", help="ブロック名の一部で対象を限定")
    sp.add_argument("--chunk", type=float, default=viz_mod.CHUNK_S)
    sp.set_defaults(func=cmd_compare)

    digest_opts(common(sub.add_parser("digest"))).set_defaults(func=cmd_digest)
    text_opts(common(sub.add_parser("transcript"))).set_defaults(func=cmd_transcript)
    text_opts(common(sub.add_parser("sections"))).set_defaults(func=cmd_sections)
    text_opts(digest_opts(asr_opts(common(sub.add_parser("all"))))).set_defaults(func=cmd_all)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except PipelineError as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
