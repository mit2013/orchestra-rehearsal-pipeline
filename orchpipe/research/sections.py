"""ステージE: 曲目・楽章の切れ目候補の抽出。

書き起こしテキストからのキーワード検出**のみ**を行う(指示書 E-1)。
音楽的特徴の変化からの推定は今回のスコープ外。

これは「提案」であって断定ではない。各候補には、根拠となった書き起こしの
該当テキストをそのまま添える。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .asr import AsrSegment

# 漢数字・算用数字の両方を拾う
_NUM = r"[0-9０-９一二三四五六七八九十]"

PATTERNS: list[tuple[str, str, str]] = [
    # (種別, 説明, 正規表現)
    ("movement", "楽章の明示", rf"第?\s*{_NUM}+\s*楽章"),
    ("piece", "曲名の指示", r"(?:次は|次に|次、|つぎは)\s*\S{2,}"),
    ("restart", "頭からの再開", r"(?:頭から|最初から|冒頭から)"),
    ("restart", "もう一度の指示", r"(?:もう一回|もう一度|もういっかい)"),
    ("rehearsal_mark", "練習記号・小節番号", rf"(?:記号|レター|マーク|小節)\s*[A-Za-zＡ-Ｚ{_NUM}]+"),
    ("number", "番号の指示", rf"{_NUM}+\s*(?:番|曲目)"),
]

_COMPILED = [(kind, desc, re.compile(pat)) for kind, desc, pat in PATTERNS]


@dataclass
class SectionCandidate:
    time: float
    kind: str
    reason: str
    matched: str
    text: str

    def to_json(self) -> dict:
        return {
            "time": round(self.time, 2),
            "kind": self.kind,
            "reason": self.reason,
            "matched": self.matched,
            "text": self.text,
        }


def find_candidates(segments: list[AsrSegment]) -> list[SectionCandidate]:
    """採用済みの発話セグメントから切れ目候補を拾う。"""
    out: list[SectionCandidate] = []
    for s in segments:
        if not s.accepted:
            continue
        t = s.text.strip()
        for kind, desc, rx in _COMPILED:
            m = rx.search(t)
            if m:
                out.append(SectionCandidate(
                    time=s.start, kind=kind, reason=desc,
                    matched=m.group(0).strip(), text=t,
                ))
    out.sort(key=lambda c: c.time)
    return out


def dedupe(cands: list[SectionCandidate], window: float = 10.0) -> list[SectionCandidate]:
    """近接した同種の候補をまとめる(同じ指示を複数パターンが拾うため)。"""
    out: list[SectionCandidate] = []
    for c in cands:
        if out and c.kind == out[-1].kind and c.time - out[-1].time < window:
            continue
        out.append(c)
    return out
