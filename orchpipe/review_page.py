"""境界レビューのページを組み立てる。

## なぜ必要か

現場前処理(`field.py`)では、母艦が境界を提案したあと、**人が移動中に iPhone で
境界を確かめて確定させる**必要がある。260829 で手作りしたページ(各ブロックの頭と
尻を聴いて判定するもの)を、日付とブロック構成を問わず作れる形にした。

## 判定だけでなく、その場で境界を動かせる

260829 のページは「よい / 切れている / 余分が長い」の判定を入れるだけで、実際に
境界を直すのは母艦へ戻ってからだった。現場経路ではそれでは遅い。そこで各境界に
**-10 / -5 / -2 / +2 / +5 / +10 秒**のボタンを置き、動かした結果の時刻をその場で
出すようにしてある。

そのため**クリップは境界の前後 25 秒ずつ(計 50 秒)を切り出す**。境界の位置は
クリップの中央で、ボタンで動かすと中央の印がずれる。±10 秒動かしてもクリップの
中に収まるので、動かしたあとの位置から聴き直せる。

## 音声の埋め込み

Artifact の CSP は外部からの音声読み込みを許さないので、MP3 を data URI で埋め込む。
50 秒 x 6 本を 128kbps で約 6MB(base64 化後)。上限 16MB に収まる。

## 保存

`artifact` capability でページ自身を publish し直す。ページは
`<script id="page-data">` の JSON から描画されるので、その JSON を差し替えて
`page-style` と `page-code` の outerHTML と繋ぎ直せば次の版になる。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from .util import FFMPEG, PipelineError, fmt_time, log, parse_time, run

# 境界の前後に含める長さ。いちばん大きいボタン(±30秒)で動かしてもクリップの中に
# 収まるようにしてある。`--tuning-first` の提案は開始がぴたりと合う一方、終了は
# guard のぶん最大2分ほど後ろに出るので、粗い刻みも要る。
PRE_S = 45.0
POST_S = 45.0
# 境界が切れているかを判断するだけなので、モノラルの 96kbps で足りる。
# 6本を data URI で埋め込むため、上限 16MiB に対する余裕を優先する。
CLIP_BITRATE = "96k"
CLIP_CHANNELS = 1
NUDGES = (-30, -15, -5, 5, 15, 30)


@dataclass
class Edge:
    edge_id: str        # "b1-head"
    block: str          # "前半"
    kind: str           # "head" | "tail"
    time: float         # 元音源での境界[秒]
    clip: Path


def _cut(src: Path, at: float, dst: Path, pre: float, post: float,
         bitrate: str, force: bool, channels: int = CLIP_CHANNELS) -> float:
    """境界 `at` の前後を切り出す。戻り値はクリップ内での境界位置[秒]。"""
    start = max(0.0, at - pre)
    offset = at - start
    if dst.exists() and not force:
        log(f"    スキップ(既存): {dst.name}")
        return offset
    run(
        [FFMPEG, "-hide_banner", "-v", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{offset + post:.3f}", "-i", str(src),
         "-ac", str(channels),
         "-c:a", "libmp3lame", "-b:a", bitrate, "-map_metadata", "-1", str(dst)],
        desc=f"    {dst.name}",
    )
    return offset


def collect_edges(blocks: list[dict]) -> list[tuple[str, str, float]]:
    """(edge_id, kind, 秒) の一覧。keep 区間の頭と尻。"""
    out = []
    for i, b in enumerate(blocks, start=1):
        out.append((f"b{i}-head", "head", b["start"]))
        out.append((f"b{i}-tail", "tail", b["end"]))
    return out


def build(
    outdir: Path,
    date: str,
    source: Path,
    blocks: list[dict],
    orchestra: str = "",
    pre_s: float = PRE_S,
    post_s: float = POST_S,
    bitrate: str = CLIP_BITRATE,
    force: bool = False,
) -> Path:
    """レビューページの HTML を書き出し、そのパスを返す。"""
    if not source.exists():
        raise PipelineError(f"{source} がありません")
    clipdir = outdir / "review_clips"
    clipdir.mkdir(parents=True, exist_ok=True)

    log(f"境界レビューのクリップを作成: {len(blocks)} ブロック x 2 = "
        f"{len(blocks) * 2} 本(前後 {pre_s:.0f}/{post_s:.0f} 秒)")

    clips: dict[str, str] = {}
    offsets: dict[str, float] = {}
    for edge_id, kind, at in collect_edges(blocks):
        dst = clipdir / f"{edge_id}.mp3"
        offsets[edge_id] = _cut(source, at, dst, pre_s, post_s, bitrate, force)
        clips[edge_id] = ("data:audio/mpeg;base64,"
                          + base64.b64encode(dst.read_bytes()).decode("ascii"))

    meta = {
        "date": date,
        "orchestra": orchestra,
        "source": source.name,
        "pre_s": pre_s,
        "post_s": post_s,
        "nudges": list(NUDGES),
        "blocks": [
            {
                "index": i,
                "name": b.get("name") or f"ブロック{i}",
                "label": b.get("label", ""),
                "start": b["start"],
                "end": b["end"],
                "head_offset": offsets[f"b{i}-head"],
                "tail_offset": offsets[f"b{i}-tail"],
            }
            for i, b in enumerate(blocks, start=1)
        ],
        "verdicts": [
            {"id": "ok", "label": "よい"},
            {"id": "short", "label": "切れている"},
            {"id": "long", "label": "余分が長い"},
        ],
    }
    state = {
        "updatedAt": None,
        "items": {eid: {"verdict": None, "note": "", "shift": 0}
                  for eid, _k, _t in collect_edges(blocks)},
    }

    html = _render(meta, clips, state)
    dst = outdir / "review_page.html"
    dst.write_text(html, encoding="utf-8")
    size = len(html.encode("utf-8")) / 2**20
    log(f"レビューページを書きました: {dst.name}  {size:.1f} MiB")
    if size > 15.0:
        log("  警告: 16MiB の上限に近づいています。--pre/--post を短くしてください")
    return dst


CSS = r"""
  *, *::before, *::after { box-sizing: border-box; }
  :root {
    --ground: #F2F0EB; --surface: #FFFFFF; --surface-2: #F7F5F1;
    --line: #D8D2C8; --line-soft: #E7E2DA;
    --ink: #1C1A17; --ink-2: #55504A; --ink-3: #8A837A;
    --accent: #2F4B7C; --accent-soft: #DEE6F2;
    --flag: #9C4A22; --flag-soft: #F3E3D9;
    --radius: 12px;
    --serif: "Zen Old Mincho", "Hiragino Mincho ProN", "Yu Mincho", serif;
    --sans: "Zen Kaku Gothic New", "Hiragino Sans", "Yu Gothic", system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #14161A; --surface: #1D2026; --surface-2: #252932;
      --line: #39404C; --line-soft: #2B3038;
      --ink: #E9E6E0; --ink-2: #ABA79F; --ink-3: #7C776F;
      --accent: #8FB0DC; --accent-soft: #1E2B3E;
      --flag: #DE9468; --flag-soft: #3A2A1F;
    }
  }
  :root[data-theme="dark"] {
    --ground: #14161A; --surface: #1D2026; --surface-2: #252932;
    --line: #39404C; --line-soft: #2B3038;
    --ink: #E9E6E0; --ink-2: #ABA79F; --ink-3: #7C776F;
    --accent: #8FB0DC; --accent-soft: #1E2B3E;
    --flag: #DE9468; --flag-soft: #3A2A1F;
  }

  body { margin: 0; background: var(--ground); color: var(--ink);
         font-family: var(--sans); line-height: 1.7;
         -webkit-font-smoothing: antialiased; -webkit-text-size-adjust: 100%; }
  .wrap { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1rem 7rem;
          display: flex; flex-direction: column; gap: 1.75rem; }

  .eyebrow { font-family: var(--mono); font-size: .72rem; letter-spacing: .14em;
             text-transform: uppercase; color: var(--ink-3); margin: 0 0 .5rem; }
  h1 { font-family: var(--serif); font-weight: 600;
       font-size: clamp(1.5rem, 1.1rem + 2vw, 2.1rem); line-height: 1.32;
       margin: 0 0 .6rem; text-wrap: balance; }
  .lede { margin: 0; color: var(--ink-2); font-size: .95rem; }

  .card { background: var(--surface); border: 1px solid var(--line-soft);
          border-radius: var(--radius); padding: 1.2rem 1.15rem;
          display: flex; flex-direction: column; gap: 1rem; }
  .card > h2 { font-family: var(--serif); font-weight: 600; font-size: 1.3rem;
               margin: 0; letter-spacing: .03em; display: flex;
               align-items: baseline; gap: .6rem; flex-wrap: wrap; }
  .pill { font-size: .74rem; font-weight: 400; color: var(--accent);
          background: var(--accent-soft); border-radius: 999px; padding: .1rem .6rem; }
  .span { font-family: var(--mono); font-size: .82rem; color: var(--ink-3);
          font-variant-numeric: tabular-nums; }

  .edge { background: var(--surface-2); border: 1px solid var(--line-soft);
          border-radius: 9px; padding: .85rem .8rem .9rem;
          display: flex; flex-direction: column; gap: .65rem; }
  .edge-top { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; }
  .edge-name { font-family: var(--serif); font-size: 1.05rem; font-weight: 600;
               color: var(--accent); }
  .edge-sub { font-size: .78rem; color: var(--ink-3); }
  .edge-time { margin-left: auto; font-family: var(--mono); font-size: 1.05rem;
               font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .edge-time.moved { color: var(--flag); font-weight: 500; }
  audio { width: 100%; height: 38px; display: block; }

  .row { display: flex; gap: .35rem; flex-wrap: wrap; align-items: center; }
  .row .lab { font-size: .74rem; color: var(--ink-3); letter-spacing: .06em;
              margin-right: .15rem; }
  button { font: inherit; cursor: pointer; }
  .chip { font-size: .82rem; padding: .3rem .7rem; background: var(--surface);
          color: var(--ink-2); border: 1px solid var(--line); border-radius: 999px; }
  .chip:hover { border-color: var(--accent); color: var(--ink); }
  .chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
                               color: #fff; font-weight: 500; }
  .nudge { font-family: var(--mono); font-size: .8rem; padding: .3rem .55rem;
           background: var(--surface); color: var(--ink-2);
           border: 1px solid var(--line); border-radius: 7px; min-width: 3rem; }
  .nudge:hover { border-color: var(--flag); color: var(--ink); }
  .seek { font-size: .8rem; padding: .3rem .7rem; background: transparent;
          color: var(--accent); border: 1px dashed var(--line); border-radius: 7px; }
  .reset { font-size: .78rem; padding: .3rem .6rem; background: transparent;
           color: var(--ink-3); border: 0; text-decoration: underline; }
  button:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent);
                                                 outline-offset: 2px; }
  textarea { font: inherit; font-size: .85rem; width: 100%; min-height: 2.4rem;
             resize: vertical; background: var(--surface); color: var(--ink);
             border: 1px solid var(--line); border-radius: 6px; padding: .4rem .55rem; }
  textarea::placeholder { color: var(--ink-3); }

  .savebar { position: sticky; bottom: 0; z-index: 5; background: var(--surface);
             border: 1px solid var(--line); border-radius: var(--radius);
             padding: .75rem .9rem; display: flex; align-items: center; gap: .8rem;
             flex-wrap: wrap; box-shadow: 0 -2px 14px rgba(0,0,0,.08); }
  .save { font-weight: 500; padding: .5rem 1.3rem; background: var(--accent);
          color: #fff; border: 1px solid var(--accent); border-radius: 7px; }
  .save[disabled] { opacity: .5; cursor: default; }
  .savemsg { font-size: .84rem; color: var(--ink-3); }
  .savemsg.err { color: var(--flag); }
  .tally { font-size: .84rem; color: var(--ink-2); margin-left: auto;
           font-variant-numeric: tabular-nums; }

  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; min-width: 26rem; }
  caption { text-align: left; font-size: .95rem; font-weight: 700; padding-bottom: .5rem; }
  th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line-soft); }
  thead th { font-size: .72rem; letter-spacing: .07em; color: var(--ink-3);
             font-weight: 500; border-bottom: 1px solid var(--line); }
  td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  td.moved { color: var(--flag); }

  footer { color: var(--ink-3); font-size: .82rem;
           border-top: 1px solid var(--line-soft); padding-top: .9rem; }
  footer p { margin: 0 0 .3rem; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = r"""
(function () {
  var data = JSON.parse(document.getElementById("page-data").textContent);
  var meta = data.meta, clips = data.clips, state = data.state;
  var dirty = false, saving = false, artifact = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function item(id) {
    if (!state.items[id]) state.items[id] = { verdict: null, note: "", shift: 0 };
    if (typeof state.items[id].shift !== "number") state.items[id].shift = 0;
    return state.items[id];
  }
  function hms(t) {
    t = Math.max(0, Math.round(t));
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    return h + ":" + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
  }
  function edgeTime(b, kind) {
    var base = kind === "head" ? b.start : b.end;
    return base + item("b" + b.index + "-" + kind).shift;
  }

  function edgeHtml(b, kind) {
    var id = "b" + b.index + "-" + kind, it = item(id);
    var isHead = kind === "head";
    var verdicts = meta.verdicts.map(function (v) {
      return '<button type="button" class="chip" data-act="verdict" data-id="' + id +
        '" data-verdict="' + v.id + '" aria-pressed="' +
        (it.verdict === v.id ? "true" : "false") + '">' + esc(v.label) + "</button>";
    }).join("");
    var nudges = meta.nudges.map(function (n) {
      return '<button type="button" class="nudge" data-act="nudge" data-id="' + id +
        '" data-n="' + n + '">' + (n > 0 ? "+" + n : n) + "秒</button>";
    }).join("");
    return '<div class="edge">' +
      '<div class="edge-top"><span class="edge-name">' + (isHead ? "頭" : "尻") + "</span>" +
      '<span class="edge-sub">' + (isHead ? "立ち上がりが切れていないか" : "終わりが切れていないか") +
      '</span><span class="edge-time' + (it.shift ? " moved" : "") + '" data-time="' + id + '">' +
      hms(edgeTime(b, kind)) + "</span></div>" +
      '<audio controls preload="none" data-audio="' + id + '" src="' + clips[id] + '"></audio>' +
      '<div class="row"><span class="lab">境界を動かす</span>' + nudges +
      '<button type="button" class="reset" data-act="reset" data-id="' + id + '">戻す</button></div>' +
      '<div class="row"><button type="button" class="seek" data-act="seek" data-id="' + id +
      '">境界の3秒前から聴く</button></div>' +
      '<div class="row">' + verdicts + "</div>" +
      '<textarea data-act="note" data-id="' + id + '" placeholder="気づいたこと(任意)">' +
      esc(it.note) + "</textarea></div>";
  }

  function cardHtml(b) {
    return '<article class="card"><h2>' + esc(b.name) +
      (b.label ? '<span class="pill">' + esc(b.label) + "</span>" : "") +
      '<span class="span">' + hms(b.start) + " – " + hms(b.end) + "</span></h2>" +
      edgeHtml(b, "head") + edgeHtml(b, "tail") + "</article>";
  }

  function tableHtml() {
    var rows = meta.blocks.map(function (b) {
      var hs = item("b" + b.index + "-head").shift, ts = item("b" + b.index + "-tail").shift;
      var a = edgeTime(b, "head"), z = edgeTime(b, "tail");
      return "<tr><td>" + esc(b.name) + "</td>" +
        '<td class="n' + (hs ? " moved" : "") + '">' + hms(a) + (hs ? " (" + (hs > 0 ? "+" : "") + hs + ")" : "") + "</td>" +
        '<td class="n' + (ts ? " moved" : "") + '">' + hms(z) + (ts ? " (" + (ts > 0 ? "+" : "") + ts + ")" : "") + "</td>" +
        '<td class="n">' + hms(z - a) + "</td></tr>";
    }).join("");
    return '<section class="card scroll"><table>' +
      "<caption>いまの境界</caption>" +
      "<thead><tr><th>ブロック</th><th>開始</th><th>終了</th><th>長さ</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table></section>";
  }

  function tally() {
    var ids = Object.keys(clips);
    var done = ids.filter(function (id) { return item(id).verdict; }).length;
    return done + " / " + ids.length + " 判定済み";
  }

  function render() {
    document.getElementById("app").innerHTML =
      '<div class="wrap"><header>' +
      '<p class="eyebrow">' + esc(meta.date) + (meta.orchestra ? " / " + esc(meta.orchestra) : "") + "</p>" +
      "<h1>ブロックの頭と尻を聴いて、境界を決める</h1>" +
      '<p class="lede">各境界の前後' + meta.pre_s + '秒を切り出しました。切れていたらその場でボタンで動かせます。' +
      "動かした結果は下の表に出ます。保存するとこのページ自体が新しい版になり、母艦がそれを読んで書き出しに進みます。</p>" +
      "</header>" +
      meta.blocks.map(cardHtml).join("") +
      tableHtml() +
      '<div class="savebar"><button type="button" class="save" data-act="save">保存する</button>' +
      '<span class="savemsg" id="savemsg"></span><span class="tally">' + tally() + "</span></div>" +
      "<footer><p>音源: " + esc(meta.source) + "</p>" +
      (state.updatedAt ? "<p>最終保存: " + esc(state.updatedAt) + "</p>" : "") +
      "</footer></div>";
    setMsg(dirty ? "未保存の変更があります" : "", false);
  }

  function setMsg(text, isErr) {
    var el = document.getElementById("savemsg");
    if (el) { el.textContent = text; el.className = "savemsg" + (isErr ? " err" : ""); }
  }

  function refreshEdge(id) {
    var parts = id.split("-"), idx = parseInt(parts[0].slice(1), 10), kind = parts[1];
    var b = meta.blocks.filter(function (x) { return x.index === idx; })[0];
    var el = document.querySelector('[data-time="' + id + '"]');
    if (el && b) {
      el.textContent = hms(edgeTime(b, kind));
      el.className = "edge-time" + (item(id).shift ? " moved" : "");
    }
    var tbl = document.querySelector(".scroll");
    if (tbl) tbl.outerHTML = tableHtml();
  }

  function buildDoc() {
    var out = { meta: meta, clips: clips, state: state };
    return '<!doctype html><html lang="ja"><head><meta charset="utf-8">' +
      '<meta name="viewport" content="width=device-width, initial-scale=1">' +
      "<title>" + esc(meta.date) + " 境界レビュー</title>" +
      '<link rel="preconnect" href="https://fonts.googleapis.com">' +
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' +
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Old+Mincho:wght@600&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">' +
      document.getElementById("page-style").outerHTML +
      '</head><body><div id="app"></div>' +
      '<script id="page-data" type="application/json">' + JSON.stringify(out) + "<\/script>" +
      document.getElementById("page-code").outerHTML +
      "</body></html>";
  }

  function save() {
    if (saving) return;
    if (!artifact) { setMsg("この画面からは保存できません(閲覧のみ)", true); return; }
    saving = true; setMsg("保存しています…", false);
    state.updatedAt = new Date().toLocaleString("ja-JP");
    artifact.publish(buildDoc()).then(function () {
      dirty = false; saving = false; setMsg("保存しました", false);
    }, function (e) {
      saving = false;
      var code = e && e.code;
      if (code === "conflict") { setMsg("別の版が先に保存されました。開き直してください", true); return; }
      if (code === "not_writer" || code === "not_granted") {
        artifact = null; setMsg("この画面は閲覧のみで保存できません", true); return;
      }
      setMsg("保存できませんでした(" + (code || "エラー") + ")", true);
    });
  }

  document.addEventListener("click", function (ev) {
    var el = ev.target.closest("[data-act]");
    if (!el) return;
    var act = el.dataset.act, id = el.dataset.id;
    if (act === "verdict") {
      var it = item(id);
      it.verdict = it.verdict === el.dataset.verdict ? null : el.dataset.verdict;
      dirty = true; render();
    } else if (act === "nudge") {
      item(id).shift += parseInt(el.dataset.n, 10);
      dirty = true; refreshEdge(id); setMsg("未保存の変更があります", false);
    } else if (act === "reset") {
      item(id).shift = 0; dirty = true; refreshEdge(id); setMsg("未保存の変更があります", false);
    } else if (act === "seek") {
      var au = document.querySelector('[data-audio="' + id + '"]');
      var parts = id.split("-"), idx = parseInt(parts[0].slice(1), 10);
      var b = meta.blocks.filter(function (x) { return x.index === idx; })[0];
      var off = (parts[1] === "head" ? b.head_offset : b.tail_offset) + item(id).shift;
      if (au) { au.currentTime = Math.max(0, off - 3); au.play().catch(function () {}); }
    } else if (act === "save") {
      save();
    }
  });
  document.addEventListener("input", function (ev) {
    var el = ev.target.closest('[data-act="note"]');
    if (!el) return;
    item(el.dataset.id).note = el.value; dirty = true;
    setMsg("未保存の変更があります", false);
  });
  window.addEventListener("beforeunload", function (e) {
    if (dirty) { e.preventDefault(); e.returnValue = ""; }
  });

  render();
  if (window.claude && window.claude.use) {
    window.claude.use("artifact").then(function (a) { artifact = a; }, function () {});
  }
})();
"""


def _render(meta: dict, clips: dict, state: dict) -> str:
    payload = json.dumps({"meta": meta, "clips": clips, "state": state},
                         ensure_ascii=False)
    return (
        '<title>' + meta["date"] + ' 境界レビュー</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Zen+Old+Mincho:wght@600&family=Zen+Kaku+Gothic+New:wght@400;500;700'
        '&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
        '<style id="page-style">' + CSS + '</style>\n'
        '<div id="app"></div>\n'
        '<script id="page-data" type="application/json">' + payload + '</script>\n'
        '<script id="page-code">' + JS + '</script>\n'
    )
