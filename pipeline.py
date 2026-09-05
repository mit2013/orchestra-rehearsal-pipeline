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
    python pipeline.py export    --date 260802
    python pipeline.py box-upload --date 260802
    python pipeline.py gdrive-upload --date 260802
    python pipeline.py notify    --date 260802

    python pipeline.py all     --date 260802 --splits 2   # ingest+merge+propose を通しで

スコープは 取り込み → チャンネル結合 → TAKE連結 → 不要区間の候補提案 → 確定後のトリミング
→ 正規化 → ミックス → 曲目単位エクスポート(WAV/MP3・タグ埋め込み)。
通知文言の生成と LINE 通知まで。ダイジェスト版は次フェーズ。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orchpipe import apply as apply_mod
from orchpipe import config as config_mod
from orchpipe import denoise as denoise_mod
from orchpipe import box_upload as box_mod
from orchpipe import export as export_mod
from orchpipe import field as field_mod
from orchpipe import review_page as review_mod
from orchpipe import reverb as reverb_mod
from orchpipe import gdrive_upload as gdrive_mod
from orchpipe import features as feat
from orchpipe import ingest as ingest_mod
from orchpipe import merge as merge_mod
from orchpipe import mix as mix_mod
from orchpipe import notify as notify_mod
from orchpipe import loudness as loud_mod
from orchpipe import normalize as norm_mod
from orchpipe import preview as preview_mod
from orchpipe import segment as seg_mod
from orchpipe.recorder_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from orchpipe.util import (PipelineError, fmt_time, log, out_dir, parse_time,
                           read_json, write_json)

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
    """解析対象になる結合済みファイル。

    現場経路では原本を結合した WAV が母艦に無く、iPhone が作った 320kbps の
    プロキシ 1 本しか届いていない。WAV が無くプロキシがあればそちらを使う
    (`propose` は ffmpeg でデコードして特徴量を取るだけなので、入力が MP3 でも
    そのまま動く)。
    """
    out: dict[str, Path] = {}
    for g in groups:
        wav = merge_mod.merged_path(outdir, g)
        if not wav.exists():
            proxy = field_mod.proxy_path(outdir, g)
            if proxy.exists():
                out[g] = proxy
                continue
        out[g] = wav
    return out


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
    config_mod.ensure(outdir, args.root)


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
        tuning_first=args.tuning_first,
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
    proxies = [p.name for p in sources.values() if p.suffix.lower() == ".mp3"]
    if proxies:
        raise PipelineError(
            f"{', '.join(proxies)} は現場プロキシ(MP3)です。`apply` は原本の WAV を"
            "切り出す段なので使えません。現場経路では `field-export` を使ってください"
        )
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
        target_lufs=args.target_lufs,
        ref_margin=args.ref_margin,
        force=args.force,
    )


def cmd_mix(args) -> None:
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    cfg = config_mod.load(outdir)
    log(f"session_config: source={cfg.source}, mix_ratio={cfg.mix_ratio}, ext_lr_map={cfg.ext_lr_map}")
    written = mix_mod.run_mix(
        outdir, cfg, list(groups),
        target_lufs=args.target_lufs,
        true_peak_db=args.true_peak,
        ref_margin=args.ref_margin,
        comp_ratio=args.comp_ratio,
        comp_threshold_offset=args.comp_threshold_offset,
        comp_attack_ms=args.comp_attack,
        comp_release_ms=args.comp_release,
        comp_knee_db=args.comp_knee,
        parallel_db=args.parallel,
        noise_ceiling_db=args.noise_ceiling,
        reverb_mix=args.reverb_mix,
        reverb_ir=Path(args.reverb_ir) if args.reverb_ir else None,
        denoise_db=args.denoise,
        force=args.force,
    )
    print()
    print(f"=== 最終ファイル ({args.date}) ===")
    for p in written:
        print(f"  {p.name}")


def cmd_export(args) -> None:
    outdir = out_dir(args.root, args.date)
    cfg = config_mod.load(outdir)
    tracks = export_mod.run_export(outdir, args.date, cfg, force=args.force,
                                   variant=args.variant)
    print()
    print(f"=== エクスポート ({args.date}) ===")
    for t in tracks:
        print(f"  {t.number}. {t.title:<6} {t.wav.name}  /  {t.mp3.name}")
    print(f"\n  出力先: {outdir / 'export'}")


def cmd_field_script(args) -> None:
    """現場(iPhone / a-Shell)で流すスクリプトを書き出す。"""
    outdir = out_dir(args.root, args.date)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = config_mod.load(outdir) if (outdir / "session_config.json").exists() else config_mod.SessionConfig()
    profile = get_profile(args.recorder or DEFAULT_PROFILE)
    manifest = outdir / "ingest.json"
    if manifest.exists():
        # 実際の TAKE 名を使う。iPhone にコピーしたときの構成をそのまま想定する。
        # 系統定義も ingest.json 側を正とする(F3 は録音モードで系統が変わるため、
        # プロファイルの既定値をそのまま信じてはいけない)。
        groups = read_json(manifest).get("channel_groups") or profile.channel_groups
        tracks = list(groups[args.group])
        loaded = _load_takes(args.root, args.date, outdir)
        takes = len(loaded)
        files = []
        for t in loaded:
            for tr in tracks:
                name = Path(t.files[tr]).name
                # M4 は TAKE フォルダごとコピーする。F3 はフォルダを掘らない。
                files.append(f"{Path(t.dir).name}/{name}" if t.dir else name)
    else:
        # 原本がまだ母艦に無い段階では、命名規則からパスを組み立てる。
        # 機種ごとに違うのでプロファイルに任せる(M4 は TAKE フォルダ、F3 は直下)。
        tracks = list(profile.channel_groups[args.group])
        takes = args.takes
        files = [profile.relative_files(args.date, i)[t]
                 for i in range(1, takes + 1) for t in tracks]
    text = field_mod.field_script(
        args.date, files, takes, len(tracks),
        out=f"{args.date}_proxy.mp3", lr_map=cfg.ext_lr_map,
        gain_db=args.gain, bitrate=args.bitrate, name=args.name,
        recorder=profile.name,
    )
    dst = outdir / args.name
    dst.write_text(text, encoding="utf-8")
    print(text)
    print(f"  保存先: {dst}")


def cmd_field_proxy(args) -> None:
    """母艦側でプロキシを作る(検証用、および iPhone が使えないときの代替)。"""
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    cfg = config_mod.load(outdir)
    takes = _load_takes(args.root, args.date, outdir)
    tracks = list(groups[args.group])
    inputs = [t.files[tr] for t in takes for tr in tracks]
    lr_map = cfg.ext_lr_map if len(tracks) == 2 else "normal"
    dst = Path(args.output) if args.output else field_mod.proxy_path(outdir, args.group)
    if dst.exists() and not args.force:
        raise PipelineError(f"{dst.name} が既にあります(作り直すには --force)")
    info = field_mod.build_proxy(inputs, len(takes), len(tracks), dst,
                                 lr_map=lr_map, gain_db=args.gain, bitrate=args.bitrate)
    print()
    print(f"=== 現場プロキシ ({args.date}) ===")
    print(f"  {dst}")
    print(f"  {info['size_mb']:.0f} MiB / ゲイン {info['gain_db']:+.1f} dB / {info['bitrate']}")
    print(f"  符号化直前のピーク: {info['pre_encode_peak_db']:+.2f} dBFS")


def cmd_field_receive(args) -> None:
    """届いたプロキシを output/{date}/ に置き、下流が動く ingest.json を書く。"""
    outdir = out_dir(args.root, args.date)
    field_mod.receive_proxy(Path(args.input), outdir, args.date,
                            group=args.group, move=args.move)
    config_mod.ensure(outdir, args.root)


def _review_blocks(outdir: Path, path: Path) -> list[dict]:
    """レビュー対象の keep 区間。名前は export と同じ規則で付ける。"""
    from orchpipe.apply import load_confirmed
    from orchpipe.export import block_titles

    keeps = load_confirmed(path, float("inf"))
    titles = block_titles(len(keeps))
    return [{"start": k["start"], "end": k["end"], "name": t,
             "label": k.get("label", "")}
            for k, t in zip(keeps, titles)]


def cmd_review_page(args) -> None:
    """ブロックの頭と尻を聴いて境界を決めるページを組み立てる。"""
    outdir = out_dir(args.root, args.date)
    recorder, groups = _session_info(args.root, args.date, outdir)
    cfg = config_mod.load(outdir)
    src = Path(args.source) if args.source else _require_merged(outdir, groups)[args.group]
    given = Path(args.input) if args.input else None
    if given is None:
        for name in ("confirmed.json", "candidates.json"):
            if (outdir / name).exists():
                given = outdir / name
                break
    if given is None or not given.exists():
        raise PipelineError(
            f"{outdir} に confirmed.json も candidates.json もありません。"
            "先に `propose` を実行してください。"
        )
    log(f"境界の入力: {given.name} / 音源: {src.name}")
    blocks = _review_blocks(outdir, given)
    dst = review_mod.build(
        outdir, args.date, src, blocks, orchestra=cfg.orchestra,
        pre_s=args.pre, post_s=args.post, bitrate=args.bitrate, force=args.force,
    )
    print()
    print(f"=== 境界レビューのページ ({args.date}) ===")
    for b in blocks:
        print(f"  {b['name']:<6} {fmt_time(b['start'])} – {fmt_time(b['end'])}  "
              f"({fmt_time(b['end'] - b['start'])})")
    print(f"  {dst}")
    print("  Artifact として公開すれば iPhone で判定できます。")


def cmd_review_apply(args) -> None:
    """レビューページに保存された判定を confirmed.json に反映する。"""
    outdir = out_dir(args.root, args.date)
    state = read_json(Path(args.input))
    items = state.get("state", state).get("items", {})
    if not items:
        raise PipelineError(f"{args.input} に items がありません")

    src = outdir / "confirmed.json"
    if not src.exists():
        raise PipelineError(f"{src} がありません")
    data = read_json(src)
    keeps = [d for d in data if str(d.get("action", "")).lower() == "keep"]

    moved = []
    for i, k in enumerate(keeps, start=1):
        for kind, key in (("head", "start"), ("tail", "end")):
            it = items.get(f"b{i}-{kind}") or {}
            shift = float(it.get("shift") or 0)
            if not shift:
                continue
            before = parse_time(k[key])
            after = max(0.0, before + shift)
            k[key] = fmt_time(after)
            moved.append((i, kind, before, after, shift))

    if not moved:
        print("動かす境界はありませんでした。confirmed.json は変更していません。")
        return

    # remove 区間の端を、隣り合う keep に合わせて詰め直す。keep どうしが隣り合う
    # ときは動かさない(どちらの移動も意図されたものなので上書きしてはいけない)。
    for i, cur in enumerate(data):
        if str(cur.get("action", "")).lower() != "remove":
            continue
        if i > 0 and str(data[i - 1].get("action", "")).lower() == "keep":
            cur["start"] = data[i - 1]["end"]
        if i + 1 < len(data) and str(data[i + 1].get("action", "")).lower() == "keep":
            cur["end"] = data[i + 1]["start"]

    if not args.force:
        backup = src.with_suffix(".json.review_bak")
        backup.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"元の境界を控えました: {backup.name}")
    write_json(src, data)
    print()
    print(f"=== 境界を更新しました ({args.date}) ===")
    for i, kind, before, after, shift in moved:
        print(f"  ブロック{i} の{'頭' if kind == 'head' else '尻'}  "
              f"{fmt_time(before)} -> {fmt_time(after)}  ({shift:+.0f} 秒)")


def cmd_field_watch(args) -> None:
    """受け口にプロキシが届くのを待ち、境界レビューの手前まで進める。"""
    outdir = out_dir(args.root, args.date)
    via = "dir" if args.dir else args.via

    if via == "drive":
        from orchpipe.gdrive_client import DriveClient

        client = DriveClient.connect(args.root)
        folder = field_mod.ensure_drive_inbox(client)
        log(f"受け口: Google Drive の orchestra-recording-pipeline/"
            f"{field_mod.DRIVE_INBOX_NAME}(id={folder['id']})")
        outdir.mkdir(parents=True, exist_ok=True)
        src = field_mod.wait_for_proxy_drive(
            client, folder["id"], outdir, pattern=args.pattern,
            poll_s=args.poll, stable_s=args.stable, timeout_s=args.timeout,
        )
        # 落としてきたものを動かす。コピーを二重に残さない。
        move = True
    else:
        watch_dir = field_mod.ensure_inbox(Path(args.dir) if args.dir else None)
        log(f"受け口: {watch_dir}")
        src = field_mod.wait_for_proxy(
            watch_dir, pattern=args.pattern, poll_s=args.poll,
            stable_s=args.stable, timeout_s=args.timeout,
        )
        move = args.move

    field_mod.receive_proxy(src, outdir, args.date, group=args.group, move=move)
    config_mod.ensure(outdir, args.root)
    if args.receive_only:
        print(f"\n受け取りました: {field_mod.proxy_path(outdir, args.group)}")
        return

    # 以降は既存のサブコマンドと同じ処理を、同じ引数の既定値で呼ぶ。
    sub = build_parser()
    prop = sub.parse_args(["propose", "--date", args.date, "--root", str(args.root),
                           "--source", args.group, "--no-previews"]
                          + (["--splits", str(args.splits)] if args.splits else []))
    cmd_propose(prop)
    rev = sub.parse_args(["review-page", "--date", args.date, "--root", str(args.root),
                          "--group", args.group])
    cmd_review_page(rev)
    print()
    print("次: review_page.html を Artifact として公開し、iPhone で境界を確定する")
    print(f"    そのあと  pipeline.py review-apply --date {args.date} --input <保存されたJSON>")
    print(f"    続けて    pipeline.py field-export --date {args.date}")


def cmd_field_export(args) -> None:
    """プロキシ 1 本から確定境界でブロックを切り出し、配布用 MP3 にする。"""
    outdir = out_dir(args.root, args.date)
    cfg = config_mod.load(outdir)
    proxy = Path(args.input) if args.input else field_mod.proxy_path(outdir, args.group)
    if not proxy.exists():
        raise PipelineError(f"{proxy} がありません。先に `field-receive` を実行してください。")
    confirmed = Path(args.confirmed) if args.confirmed else outdir / "confirmed.json"
    rows = field_mod.run_field_export(
        proxy, confirmed, outdir, cfg, args.date,
        variant=args.variant, proxy_gain_db=args.gain,
        target_lufs=args.target_lufs, true_peak_db=args.true_peak,
        ref_margin=args.ref_margin, bitrate=args.bitrate, force=args.force,
    )
    write_json(outdir / "field_export.json", {"proxy": str(proxy), "blocks": rows})
    print()
    print(f"=== 現場経路の書き出し ({args.date}) ===")
    for r in rows:
        a = r["after"]
        print(f"  {Path(r['path']).name}  {a['integrated_lufs']:+.1f} LUFS / "
              f"レンジ {a['lra_lu']:.1f} LU / TP {a['true_peak_db']:+.1f} dBFS / "
              f"{r['size_mb']:.0f} MiB")


def cmd_box_upload(args) -> None:
    outdir = out_dir(args.root, args.date)
    cfg = config_mod.load(outdir)
    r = box_mod.run_box_upload(args.root, args.date, outdir, cfg, auth_timeout=args.auth_timeout)
    print()
    print(f"=== Box アップロード完了 ({args.date}) ===")
    print(f"  フォルダ      : {r['folder_name']} (id={r['folder_id']}, "
          f"{'新規作成' if r['created'] else '既存を再利用'})")
    print(f"  対象          : {r['n_target']} ファイル "
          f"(アップロード {r['n_uploaded']} / スキップ {r['n_skipped']}) "
          f"/ フォルダ内の総ファイル数 {r['n_in_folder']}")
    for nm in r["skipped"]:
        print(f"      スキップ(内容同一): {nm}")
    for nm in r["uploaded"]:
        print(f"      アップロード: {nm}")
    print(f"  パスワード保護: {r['password_enabled']}")
    print(f"  ダウンロード可: {r['can_download']}  (False であること)")
    print(f"  アクセス範囲  : {r['access']}")
    print()
    print(f"  共有リンク: {r['url']}")
    print(f"  パスワード: {r['password']}")


def cmd_gdrive_upload(args) -> None:
    outdir = out_dir(args.root, args.date)
    cfg = config_mod.load(outdir)
    r = gdrive_mod.run_gdrive_upload(
        args.root, args.date, outdir, cfg, auth_timeout=args.auth_timeout
    )
    c = r["created"]
    print()
    print(f"=== Google Drive アップロード完了 ({args.date}) ===")
    print(f"  フォルダ構成  : orchestra-recording-pipeline/{r['orchestra_folder']}/"
          f"{args.date}/{{WAV,MP3}}")
    labels = (("orchestra-recording-pipeline", "root"), (r["orchestra_folder"], "orchestra"),
              (args.date, "date"), ("WAV", "WAV"), ("MP3", "MP3"))
    for label, key in labels:
        state = "新規作成" if c[key] else "既存を再利用"
        if key == "root" and r["root_renamed"]:
            state = "旧名からリネームして引き継ぎ(ID不変)"
        print(f"      {label:<32} {state}")
    if r["migrated_dates"]:
        print(f"  移行(日付フォルダ): {', '.join(r['migrated_dates'])} をフォルダごと団体配下へ移動")
    else:
        print("  移行(日付フォルダ): 対象なし")
    if r["migrated_files"]:
        print(f"  移行(ファイル)  : {len(r['migrated_files'])} 件")
    for label, res, n, in_n in (("WAV", r["wav"], r["n_wav"], r["n_in_wav"]),
                                ("MP3", r["mp3"], r["n_mp3"], r["n_in_mp3"])):
        print(f"  {label}: 対象 {n} 本 / フォルダ内 {in_n} 本 "
              f"(アップロード {len(res['uploaded'])} / スキップ {len(res['skipped'])})")
        for nm in res["skipped"]:
            print(f"      スキップ(内容同一): {nm}")
        for nm in res["uploaded"]:
            print(f"      アップロード: {nm}")
    print(f"  日付フォルダ直下の残ファイル: {r['n_loose_in_date']} 件 (0 であること)")
    print()
    for label, s in (("日付フォルダ(WAV+MP3、動画担当向け)", r["share_date"]),
                     ("MP3フォルダ(団員個別共有向け)", r["share_mp3"])):
        p = s["anyone_permission"] or {}
        print(f"  【{label}】")
        print(f"    共有権限        : type={p.get('type')} role={p.get('role')}")
        print(f"    ダウンロード制限: copyRequiresWriterPermission="
              f"{s['copy_requires_writer_permission']} / canDownload={s['can_download']}")
        print(f"    共有リンク: {s['url']}")
        print()


def cmd_notify(args) -> None:
    outdir = out_dir(args.root, args.date)
    cfg = config_mod.load(outdir)
    r = notify_mod.run_notify(args.root, args.date, outdir, cfg, send_line=not args.no_line)
    print()
    print(r["body"])
    print(f"  保存先: {r['path']}")
    print(f"  LINE 通知: {'送信しました' if r['line_ok'] else '未送信 — ' + r['line_note']}")


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
    sp.add_argument("--no-tuning-first", dest="tuning_first", action="store_false",
                    help="チューニング起点をやめ、合奏らしさスコアからの区切りだけで決める")
    sp.set_defaults(func=cmd_propose)

    sp = common(sub.add_parser("apply", help="確定JSONにもとづきトリミング"))
    sp.add_argument("--input", default=None, help="確定JSON (既定: output/{date}/confirmed.json)")
    sp.add_argument("--groups", default=None, help="対象系統をカンマ区切りで限定 (例: ext)")
    sp.set_defaults(func=cmd_apply)

    sp = common(sub.add_parser("normalize", help="trimmed/ の各ブロックをラウドネス正規化"))
    sp.add_argument("--target-lufs", type=float, default=loud_mod.DEFAULT_TARGET_LUFS,
                    help="目標の統合ラウドネス [LUFS]")
    sp.add_argument("--ref-margin", type=float, default=120.0,
                    help="基準ラウドネスの算出から除外する前後の長さ [秒] (guard 相当)")
    sp.set_defaults(func=cmd_normalize)

    sp = common(sub.add_parser("mix", help="正規化済み ext/int から最終ファイルを作る"))
    sp.add_argument("--target-lufs", type=float, default=loud_mod.DEFAULT_TARGET_LUFS,
                    help="目標の統合ラウドネス [LUFS]")
    sp.add_argument("--true-peak", type=float, default=loud_mod.DEFAULT_TRUE_PEAK_DB,
                    help="トゥルーピークの上限 [dBTP]")
    sp.add_argument("--ref-margin", type=float, default=120.0,
                    help="ラウドネス測定から除外する前後の長さ [秒] (guard 相当)")
    sp.add_argument("--comp-ratio", type=float, default=loud_mod.DEFAULT_COMP_RATIO,
                    help="コンプレッサのレシオ")
    sp.add_argument("--comp-threshold-offset", type=float,
                    default=loud_mod.DEFAULT_COMP_THRESHOLD_OFFSET,
                    help="コンプのしきい値を目標ラウドネスから何 dB 上に置くか")
    sp.add_argument("--comp-attack", type=float, default=loud_mod.DEFAULT_COMP_ATTACK_MS,
                    help="コンプのアタック [ms]")
    sp.add_argument("--comp-release", type=float, default=loud_mod.DEFAULT_COMP_RELEASE_MS,
                    help="コンプのリリース [ms]")
    sp.add_argument("--comp-knee", type=float, default=loud_mod.DEFAULT_COMP_KNEE_DB,
                    help="コンプのニー幅 [dB]")
    sp.add_argument("--parallel", type=float, default=loud_mod.PARALLEL_MAKEUP_DB,
                    help="パラレルコンプの makeup [dB]。小さい音だけを持ち上げる。0 で無効")
    sp.add_argument("--noise-ceiling", type=float, default=None,
                    help="仕上がりの暗騒音の上限 [dBFS]。既定は目標ラウドネス "
                         f"-{loud_mod.NOISE_FLOOR_BELOW_TARGET_DB:g} dB")
    sp.add_argument("--reverb-mix", type=float, default=reverb_mod.DEFAULT_MIX,
                    help="ホール残響を混ぜる割合 (0〜1)。0 で無効")
    sp.add_argument("--reverb-ir", default=None,
                    help="インパルス応答(既定: Birmingham Symphony Hall。.wir も可)")
    sp.add_argument("--denoise", type=float, default=0.0,
                    help="空調などの定常音を実測形状で引く量 [dB]。0 で無効。"
                         f"入れるなら {denoise_mod.DEFAULT_REDUCE_DB:g} 前後")
    sp.set_defaults(func=cmd_mix)

    sp = common(sub.add_parser("export", help="曲目単位のWAV/MP3書き出しとタグ埋め込み"))
    sp.add_argument("--variant", default="",
                    help="版名。ファイル名末尾とID3タイトルに入る(例 ラウドネス調整版)。"
                         "既配布分を差し替えず別版として並べたいときに使う")
    sp.set_defaults(func=cmd_export)

    sp = with_recorder(common(sub.add_parser(
        "field-script", help="現場(iPhone/a-Shell)で流すプロキシ作成スクリプトを出す")))
    sp.add_argument("--group", default="ext", help="対象系統 (既定: ext)")
    sp.add_argument("--takes", type=int, default=2, help="その日の TAKE 数")
    sp.add_argument("--gain", type=float, default=field_mod.PROXY_GAIN_DB,
                    help="符号化前に当てる固定ゲイン [dB]")
    sp.add_argument("--bitrate", default=field_mod.PROXY_BITRATE)
    sp.add_argument("--name", default="field_master.sh")
    sp.set_defaults(func=cmd_field_script)

    sp = common(sub.add_parser("field-proxy", help="母艦側でプロキシを作る(検証・代替用)"))
    sp.add_argument("--group", default="ext", help="対象系統 (既定: ext)")
    sp.add_argument("--output", default=None, help="出力先 (既定: output/{date}/raw_merged_ext_proxy.mp3)")
    sp.add_argument("--gain", type=float, default=field_mod.PROXY_GAIN_DB)
    sp.add_argument("--bitrate", default=field_mod.PROXY_BITRATE)
    sp.set_defaults(func=cmd_field_proxy)

    sp = common(sub.add_parser("review-page", help="ブロックの頭と尻を聴いて境界を決めるページを作る"))
    sp.add_argument("--input", default=None,
                    help="境界の入力 (既定: confirmed.json、無ければ candidates.json)")
    sp.add_argument("--source", default=None, help="音源 (既定: 結合済み WAV かプロキシ)")
    sp.add_argument("--group", default="ext")
    sp.add_argument("--pre", type=float, default=review_mod.PRE_S,
                    help="境界の手前に含める長さ [秒]")
    sp.add_argument("--post", type=float, default=review_mod.POST_S,
                    help="境界の後ろに含める長さ [秒]")
    sp.add_argument("--bitrate", default=review_mod.CLIP_BITRATE)
    sp.set_defaults(func=cmd_review_page)

    sp = common(sub.add_parser("review-apply", help="レビューページの判定を confirmed.json に反映"))
    sp.add_argument("--input", required=True, help="ページから取り出した JSON")
    sp.set_defaults(func=cmd_review_apply)

    sp = common(sub.add_parser("field-receive", help="届いたプロキシを取り込み ingest.json を書く"))
    sp.add_argument("--input", required=True, help="受け取ったプロキシ MP3")
    sp.add_argument("--group", default="ext")
    sp.add_argument("--move", action="store_true", help="コピーではなく移動する")
    sp.set_defaults(func=cmd_field_receive)

    sp = common(sub.add_parser("field-watch", help="プロキシが届くのを待ち、境界レビューの手前まで進める"))
    sp.add_argument("--via", choices=("drive", "dir"), default="drive",
                    help="受け口の種類(既定: drive = Google Drive の "
                         "orchestra-recording-pipeline/inbox)")
    sp.add_argument("--dir", default=None,
                    help="ローカルのフォルダを見張る(指定すると --via dir になる。"
                         "省略時の既定は iCloud Drive の "
                         "orchestra-recording-pipeline/inbox)")
    sp.add_argument("--pattern", default="*.mp3")
    sp.add_argument("--group", default="ext")
    sp.add_argument("--splits", type=int, default=None, help="propose に渡す分割数のヒント")
    sp.add_argument("--poll", type=float, default=field_mod.WATCH_POLL_S)
    sp.add_argument("--stable", type=float, default=field_mod.WATCH_STABLE_S,
                    help="サイズがこの秒数変わらなければ書き込み完了とみなす")
    sp.add_argument("--timeout", type=float, default=None, help="待つ上限 [秒]")
    sp.add_argument("--move", action="store_true")
    sp.add_argument("--receive-only", action="store_true",
                    help="受け取るだけで propose / review-page は走らせない")
    sp.set_defaults(func=cmd_field_watch)

    sp = common(sub.add_parser("field-export", help="プロキシから配布用 MP3 を切り出す"))
    sp.add_argument("--input", default=None, help="プロキシ MP3 (既定: output/{date}/raw_merged_ext_proxy.mp3)")
    sp.add_argument("--confirmed", default=None, help="確定JSON (既定: output/{date}/confirmed.json)")
    sp.add_argument("--group", default="ext")
    sp.add_argument("--variant", default="", help="版名。ファイル名末尾とID3タイトルに入る")
    sp.add_argument("--gain", type=float, default=field_mod.PROXY_GAIN_DB,
                    help="プロキシに当てた固定ゲイン [dB]。測定前に打ち消す")
    sp.add_argument("--target-lufs", type=float, default=loud_mod.DEFAULT_TARGET_LUFS)
    sp.add_argument("--true-peak", type=float, default=loud_mod.DEFAULT_TRUE_PEAK_DB)
    sp.add_argument("--ref-margin", type=float, default=120.0)
    sp.add_argument("--bitrate", default=field_mod.PROXY_BITRATE)
    sp.set_defaults(func=cmd_field_export)

    sp = common(sub.add_parser("box-upload", help="MP3 を Box にアップロードし共有リンクを発行"))
    sp.add_argument("--auth-timeout", type=float, default=300.0,
                    help="初回認証でブラウザ操作を待つ秒数")
    sp.set_defaults(func=cmd_box_upload)

    sp = common(sub.add_parser("gdrive-upload", help="WAV を Google Drive にアップロードし共有リンクを発行"))
    sp.add_argument("--auth-timeout", type=float, default=None,
                    help="初回認証でブラウザ操作を待つ秒数(既定は無制限)")
    sp.set_defaults(func=cmd_gdrive_upload)

    sp = common(sub.add_parser("notify", help="通知文言を生成し、LINE で自分宛に送る"))
    sp.add_argument("--no-line", action="store_true", help="LINE への push を行わない")
    sp.set_defaults(func=cmd_notify)

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
    sp.add_argument("--no-tuning-first", dest="tuning_first", action="store_false")
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
