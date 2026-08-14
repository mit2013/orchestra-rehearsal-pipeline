#!/usr/bin/env python3
"""アマオケ練習録音 自動編集パイプライン(フェーズ1)

    python pipeline.py ingest  --date 260802
    python pipeline.py merge   --date 260802
    python pipeline.py propose --date 260802 --splits 2
    # ユーザーが preview_clips/ と waveform.png を確認して confirmed.json を編集
    python pipeline.py apply   --date 260802
    # session_config.json を確認・編集 (source, mix_ratio, ext_lr_map)
    python pipeline.py normalize --date 260802
    python pipeline.py mix       --date 260802

    python pipeline.py all     --date 260802 --splits 2   # ingest+merge+propose を通しで

スコープは 取り込み → チャンネル結合 → TAKE連結 → 不要区間の候補提案 → 確定後のトリミング
→ 正規化 → ミックス。曲目単位エクスポート・MP3タグ・アップロード・通知は次フェーズ。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orchpipe import apply as apply_mod
from orchpipe import config as config_mod
from orchpipe import features as feat
from orchpipe import ingest as ingest_mod
from orchpipe import merge as merge_mod
from orchpipe import mix as mix_mod
from orchpipe import normalize as norm_mod
from orchpipe import preview as preview_mod
from orchpipe import segment as seg_mod
from orchpipe.recorder_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from orchpipe.util import PipelineError, fmt_time, log, out_dir, read_json, write_json

ROOT_DEFAULT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------

def _session_info(root: Path, date: str, outdir: Path) -> tuple[str, dict[str, list[str]]]:
    """`ingest.json` から機種と系統定義を読む。

    `--recorder` は `ingest` でしか受け付けない。下流は取り込み時に記録された
    `recorder` / `channel_groups` に従うので、系統名をコード側に埋め込む必要がない。
    """
    manifest = outdir / "ingest.json"
    if not manifest.exists():
        log(f"{manifest.name} がないため、既定プロファイル({DEFAULT_PROFILE})で取り込みます")
        ingest_mod.run_ingest(root, date, outdir, get_profile(DEFAULT_PROFILE))
    data = read_json(manifest)
    recorder = data.get("recorder", DEFAULT_PROFILE)
    # channel_groups を持たない旧フォーマットは、機種名からプロファイル定義で補う。
    groups = data.get("channel_groups") or get_profile(recorder).channel_groups
    return recorder, {g: list(tracks) for g, tracks in groups.items()}


def _check_groups(spec: str | None, groups) -> list[str] | None:
    """`--groups` の指定を検証して返す。未指定なら None(=全系統)。"""
    if not spec:
        return None
    want = [g.strip() for g in spec.split(",") if g.strip()]
    unknown = [g for g in want if g not in groups]
    if unknown:
        raise PipelineError(
            f"未知の系統です: {', '.join(unknown)}(利用可能: {', '.join(groups)})"
        )
    return want


def _load_takes(root: Path, date: str, outdir: Path) -> list[ingest_mod.Take]:
    """ingest.json があれば再走査を省く(再開しやすくするため)。"""
    manifest = outdir / "ingest.json"
    if manifest.exists():
        data = read_json(manifest)
        return [ingest_mod.Take.from_dict(t) for t in data["takes"]]
    return ingest_mod.run_ingest(root, date, outdir, get_profile(DEFAULT_PROFILE))


def _merged_paths(outdir: Path, groups) -> dict[str, Path]:
    return {g: merge_mod.merged_path(outdir, g) for g in groups}


def _require_merged(outdir: Path, groups) -> dict[str, Path]:
    paths = _merged_paths(outdir, groups)
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise PipelineError(
            f"{', '.join(missing)} がありません。先に `merge` を実行してください。"
        )
    return paths


# ---------------------------------------------------------------------------
# サブコマンド
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> None:
    outdir = out_dir(args.root, args.date)
    profile = get_profile(args.recorder or DEFAULT_PROFILE)
    ingest_mod.run_ingest(args.root, args.date, outdir, profile)
    config_mod.ensure(outdir)


def cmd_merge(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    cfg = config_mod.load(outdir)
    takes = _load_takes(args.root, args.date, outdir)
    only = _check_groups(args.groups, groups)
    merge_mod.run_merge(takes, outdir, cfg, groups, force=args.force, only_groups=only)


def cmd_propose(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    takes = _load_takes(args.root, args.date, outdir)
    total = sum(t.duration for t in takes)
    if args.source not in groups:
        raise PipelineError(
            f"--source は {', '.join(groups)} のいずれかです(実際: {args.source!r}、"
            f"機種 {recorder})"
        )
    sources = _require_merged(outdir, groups)
    analysis_src = sources[args.source]

    cache = outdir / f"features_{args.source}.npz"
    if cache.exists() and not args.force:
        log(f"特徴量キャッシュを再利用: {cache.name}(作り直すには --force)")
        ff = feat.load_frame_features(cache)
    else:
        ff = feat.extract_frame_features(analysis_src)
        feat.save_frame_features(cache, ff)
        log(f"特徴量を保存: {cache.name}")

    wf = feat.aggregate_windows(ff, win_s=args.win, hop_s=args.hop)
    log(f"窓集約: {wf.n} 窓 (窓長 {wf.win_s:.0f}s / ホップ {wf.hop_s:.0f}s), "
        f"無音閾値 {wf.silence_db:.1f} dBFS")

    segs, meta = seg_mod.propose_segments(
        ff, wf, total,
        splits=args.splits,
        min_keep_s=args.min_keep * 60,
        min_remove_s=args.min_remove * 60,
        smooth_s=args.smooth,
        default_penalty=args.penalty,
        guard_s=args.guard,
    )

    print()
    print(f"=== 境界候補 ({args.date}) 全長 {fmt_time(total)} ===")
    for s in segs:
        mark = "残す" if s.action == "keep" else "削る"
        print(f"  {s.index:2d}. {s.start and fmt_time(s.start) or '00:00:00'} - {fmt_time(s.end)}  "
              f"[{mark}] {s.label}  (信頼度 {s.confidence:.2f}, {(s.end-s.start)/60:.1f}分)")
    print()

    cand = outdir / "candidates.json"
    write_json(cand, seg_mod.segments_to_json(segs))
    write_json(outdir / "analysis.json", {"date": args.date, "source": args.source,
                                          "total_duration": total, **meta})
    log(f"候補を書き出しました: {cand}")

    confirmed = outdir / "confirmed.json"
    if not confirmed.exists() or args.force:
        write_json(confirmed, seg_mod.segments_to_json(segs))
        log(f"編集用のひな型を用意しました: {confirmed}")
    else:
        log(f"既存の {confirmed.name} は上書きしませんでした(--force で再生成)")

    if not args.no_previews:
        preview_mod.write_preview_clips(
            segs, sources, outdir / "preview_clips", pad_s=args.pad, total=total
        )
        score = seg_mod.ensemble_score(wf, smooth_s=args.smooth)
        preview_mod.write_waveform_png(
            ff, wf, score, segs, outdir / "waveform.png",
            threshold=meta["threshold"],
            title=f"{args.date}  境界候補 ({args.source})  全長 {fmt_time(total)}",
        )

    print("次の手順: preview_clips/ を聴き、waveform.png を見て confirmed.json を修正 -> "
          f"python pipeline.py apply --date {args.date}")


def cmd_apply(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    takes = _load_takes(args.root, args.date, outdir)
    total = sum(t.duration for t in takes)
    sources = _require_merged(outdir, groups)
    want = _check_groups(args.groups, groups)
    if want:
        sources = {g: p for g, p in sources.items() if g in want}
        log(f"対象系統を限定: {', '.join(sources)}")
    confirmed = Path(args.input) if args.input else outdir / "confirmed.json"
    if not confirmed.exists():
        raise PipelineError(f"{confirmed} がありません。先に `propose` を実行してください。")
    apply_mod.run_apply(confirmed, sources, outdir / "trimmed", total, force=args.force)


def cmd_normalize(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    norm_mod.run_normalize(
        outdir, list(groups),
        target_db=args.target,
        ref_margin=args.ref_margin,
        force=args.force,
    )


def cmd_mix(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    cfg = config_mod.load(outdir)
    log(f"session_config: source={cfg.source}, mix_ratio={cfg.mix_ratio}, ext_lr_map={cfg.ext_lr_map}")
    written = mix_mod.run_mix(
        outdir, cfg, list(groups), safe_peak_db=args.safe_peak, force=args.force
    )
    print()
    print(f"=== 最終ファイル ({args.date}) ===")
    for p in written:
        print(f"  {p.name}")


def cmd_all(args) -> None:
    cmd_ingest(args)
    cmd_merge(args)
    cmd_propose(args)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--date", required=True, help="TAKEフォルダの日付部分 (例: 260802)")
        sp.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="TAKEフォルダを置いた親ディレクトリ")
        sp.add_argument("--force", action="store_true", help="既存の中間ファイルを作り直す")
        return sp

    def with_recorder(sp):
        """`--recorder` は取り込み時にしか意味を持たない。

        下流の各段は output/{date}/ingest.json に記録された recorder / channel_groups に
        従うので、機種を指定し直す必要がない(指定できると食い違いの原因になる)。
        """
        sp.add_argument("--recorder", choices=sorted(PROFILES), default=DEFAULT_PROFILE,
                        help=f"レコーダ機種 (既定: {DEFAULT_PROFILE})")
        return sp

    with_recorder(common(sub.add_parser("ingest", help="TAKEを走査・検証する"))).set_defaults(func=cmd_ingest)

    sp = common(sub.add_parser("merge", help="チャンネル結合とTAKE連結"))
    sp.add_argument("--groups", default=None, help="対象系統をカンマ区切りで限定 (例: ext)")
    sp.set_defaults(func=cmd_merge)

    sp = common(sub.add_parser("propose", help="不要区間の候補を提案する"))
    sp.add_argument("--splits", type=int, default=None, help="最終的に何分割(=合奏いくつ)にしたいかのヒント")
    sp.add_argument("--source", default="ext",
                    help="解析に使う系統 (ingest.json の channel_groups から選ぶ)")
    sp.add_argument("--win", type=float, default=20.0, help="解析窓長 [秒]")
    sp.add_argument("--hop", type=float, default=5.0, help="解析ホップ [秒]")
    sp.add_argument("--min-keep", type=float, default=8.0, help="合奏区間の最小長 [分]")
    sp.add_argument("--min-remove", type=float, default=1.0, help="不要区間の最小長 [分]")
    sp.add_argument("--pad", type=float, default=15.0, help="プレビューの前後幅 [秒]")
    sp.add_argument("--smooth", type=float, default=0.0, help="スコアの平滑化長 [秒]。0=無効(推奨)")
    sp.add_argument("--guard", type=float, default=120.0, help="keep区間を外側へ広げる安全マージン [秒]")
    sp.add_argument("--penalty", type=float, default=12.0, help="--splits 未指定時の区間切り替えペナルティ")
    sp.add_argument("--no-previews", action="store_true", help="プレビュー音声と波形画像を作らない")
    sp.set_defaults(func=cmd_propose)

    sp = common(sub.add_parser("apply", help="確定JSONにもとづきトリミング"))
    sp.add_argument("--input", default=None, help="確定JSON (既定: output/{date}/confirmed.json)")
    sp.add_argument("--groups", default=None, help="対象系統をカンマ区切りで限定 (例: ext)")
    sp.set_defaults(func=cmd_apply)

    sp = common(sub.add_parser("normalize", help="trimmed/ の各ブロックをピーク正規化"))
    sp.add_argument("--target", type=float, default=norm_mod.TARGET_DB, help="目標ピーク [dBFS]")
    sp.add_argument("--ref-margin", type=float, default=120.0,
                    help="基準ピークの算出から除外する前後の長さ [秒] (guard 相当)")
    sp.set_defaults(func=cmd_normalize)

    sp = common(sub.add_parser("mix", help="正規化済み ext/int から最終ファイルを作る"))
    sp.add_argument("--safe-peak", type=float, default=mix_mod.SAFE_PEAK_DB,
                    help="合成後に超えてはならないピーク [dBFS]")
    sp.set_defaults(func=cmd_mix)

    sp = with_recorder(common(sub.add_parser("all", help="ingest + merge + propose を通しで実行")))
    sp.add_argument("--splits", type=int, default=None)
    sp.add_argument("--source", default="ext")
    sp.add_argument("--win", type=float, default=20.0)
    sp.add_argument("--hop", type=float, default=5.0)
    sp.add_argument("--min-keep", type=float, default=8.0)
    sp.add_argument("--min-remove", type=float, default=1.0)
    sp.add_argument("--pad", type=float, default=15.0)
    sp.add_argument("--smooth", type=float, default=0.0)
    sp.add_argument("--guard", type=float, default=120.0)
    sp.add_argument("--penalty", type=float, default=12.0)
    sp.add_argument("--no-previews", action="store_true")
    sp.set_defaults(func=cmd_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except PipelineError as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1
    except NotImplementedError as e:
        print(f"\n未実装: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました(中間ファイルは残っているので再実行で続きから)", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
