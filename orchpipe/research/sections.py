"""ステージE: 曲目・楽章の切れ目候補の抽出。

書き起こしテキストからのキーワード検出**のみ**を行う(指示書 E-1)。
音楽的特徴の変化からの推定は今回のスコープ外。

これは「提案」であって断定ではない。各候補には、根拠となった書き起こしの
該当テキストをそのまま添える。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .asr import Utterance

# 漢数字・算用数字の両方を拾う
_NUM = r"[0-9０-９一二三四五六七八九十]"

PATTERNS: list[tuple[str, str, str]] = [
    # (種別, 説明, 正規表現)
    ("movement", "楽章の明示", rf"第?\s*{_NUM}+\s*楽章"),
    # 曲名の長さは上限を付ける。連結後テキストは空白がほとんど無いため、
    # `\S{2,}` のままだと発言の末尾まで一致してしまい候補が読めなくなる。
    ("piece", "曲名の指示", r"(?:次は|次に|次、|つぎは)\s*\S{2,12}"),
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


def find_candidates(utterances: list[Utterance]) -> list[SectionCandidate]:
    """連結後の発言から切れ目候補を拾う。

    検索対象を連結後テキストにしているのは、「次は」と曲名、「2楽」と「章」の
    ように、指示がセグメント境界をまたいで切れていると検出できないため。
    一方**候補の時刻は元セグメントの粒度を保つ**。連結後の発言は最長 45 秒
    ほどになるので、その先頭を一律に返すと切れ目の位置が最大でそれだけ
    ずれてしまう。`Utterance.time_at()` で一致位置の元時刻に引き直す。
    """
    out: list[SectionCandidate] = []
    for u in utterances:
        for kind, desc, rx in _COMPILED:
            m = rx.search(u.text)
            if m:
                out.append(SectionCandidate(
                    time=u.time_at(m.start()), kind=kind, reason=desc,
                    matched=m.group(0).strip(), text=u.text,
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
