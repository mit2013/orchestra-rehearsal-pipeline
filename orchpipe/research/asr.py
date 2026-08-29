"""音声認識(faster-whisper)のラッパ。

ステージA(speech/playing の判別)とステージD(書き起こし)の両方で使う。
1回の書き起こし結果を両方で共有し、モデルを二度回さない。

**この録音での実測にもとづく設計判断**:

- Silero VAD は管弦楽を 37% ほど発話候補と誤検出する(実測)。VAD だけでは
  speech/playing を分けられないので、書き起こし結果の妥当性で絞り込む。
- `no_speech_prob` / `avg_logprob` / `compression_ratio` は whisper の 30 秒窓
  単位で決まる値で、セグメントごとには変化しない(実測で同一窓内の全セグメントが
  同じ値だった)。したがってセグメント単位の判定には使えない。
- 代わりに **「1文字あたりの秒数」** が非常によく効く。実データでは
  正常な発話が 0.1〜0.6 秒/文字だったのに対し、破綻セグメント(131.9 秒に
  10 文字)は 13.19 秒/文字だった。
- 演奏を発話と誤判定すると、ダイジェスト(ステージC)から演奏が削られてしまう。
  これまでのパイプラインと同じく**安全側(playing 寄り)に倒す**。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import FFMPEG, PipelineError, log

ASR_SR = 16000

# --- 発話として採用する条件(実測にもとづく) -------------------------------
# 1文字あたりの秒数。これを超えると「引き伸ばされた幻聴」とみなす。
MAX_SEC_PER_CHAR = 1.2
# 短すぎるテキストは音楽の断片を拾っただけの可能性が高い。
MIN_TEXT_CHARS = 3
# 同一文字の連続や同一トークンの反復は幻聴の典型。
MAX_REPEAT_RATIO = 0.5
# 日本語らしさ(かな・漢字)の最低比率。記号や英字だけの出力を弾く。
MIN_JA_RATIO = 0.3
# VAD の検出量がこの割合を下回ったら「VAD が機能していない」とみなして再試行する。
MIN_SPEECH_RATIO = 0.01
# 隣接する発話セグメントをひとまとまりとして扱う最大の間隔(秒)。
# whisper のセグメント境界は息継ぎや語尾で細かく切れ、1〜3 秒の断片が並ぶ。
# 実測では採用セグメントの 69% が間隔 0 秒(完全に連続)だった。
MERGE_GAP_S = 2.0

_JA = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    accepted: bool
    reject_reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "no_speech_prob": round(self.no_speech_prob, 4),
            "avg_logprob": round(self.avg_logprob, 4),
            "compression_ratio": round(self.compression_ratio, 4),
        }


@dataclass
class Utterance:
    """近接する発話セグメントを連結した「ひとまとまりの発言」。

    whisper のセグメントは息継ぎや語尾ごとに細かく切れるため、そのまま並べると
    「5つ目で覚えあんくらいのメロディーが出てる」「ピアノ2つなんです」のように
    一続きの発言が分断されて読みにくい。表示とキーワード検索はこの連結後の
    テキストを対象にする。

    ただし**元セグメントの細かいタイムスタンプは捨てない**。`sources` に元の
    セグメントを、`offsets` に「そのテキストが連結後テキストの何文字目から
    始まるか」を保持しているので、`time_at()` で連結後の任意の文字位置から
    元の時刻に戻せる。ステージEの候補時刻はこれを使って精度を保つ。
    """

    start: float
    end: float
    text: str
    sources: list[AsrSegment]
    offsets: list[int]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def time_at(self, pos: int) -> float:
        """連結後テキストの文字位置 `pos` を含む元セグメントの開始秒。"""
        t = self.start
        for off, src in zip(self.offsets, self.sources):
            if off > pos:
                break
            t = src.start
        return t

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "n_segments": len(self.sources),
            # 元の細かいタイムスタンプ(連結前)をそのまま残す
            "segments": [{"start": round(s.start, 3), "end": round(s.end, 3),
                          "text": s.text} for s in self.sources],
        }


def merge_utterances(segments: list[AsrSegment],
                     max_gap: float = MERGE_GAP_S) -> list[Utterance]:
    """採用済みセグメントのうち、間隔が `max_gap` 以下のものを連結する。

    連結時の区切り文字は入れない。日本語は分かち書きしないので空白を挟むと
    かえって読みにくく、「2楽」+「章」のように語の途中で切れているときに
    ステージEの正規表現(`第?N+\\s*楽章` など)が一致しなくなるため。
    """
    out: list[Utterance] = []
    for s in segments:
        if not s.accepted:
            continue
        t = s.text.strip()
        if not t:
            continue
        if out and (s.start - out[-1].end) <= max_gap:
            u = out[-1]
            u.offsets.append(len(u.text))
            u.text += t
            u.sources.append(s)
            u.end = s.end
        else:
            out.append(Utterance(start=s.start, end=s.end, text=t,
                                 sources=[s], offsets=[0]))
    return out


def decode_mono16k(path: Path, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """ASR 用に 16kHz モノラル float32 で読み出す。"""
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(ASR_SR), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise PipelineError(
            f"ASR 用のデコードに失敗: {path}\n{proc.stderr.decode('utf-8','replace')[-500:]}"
        )
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def _repeat_ratio(text: str) -> float:
    """最頻の 2〜4 文字 n-gram が占める割合。幻聴の反復を捉える。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < 6:
        return 0.0
    worst = 0.0
    for n in (2, 3, 4):
        if len(t) < n * 2:
            continue
        grams = [t[i:i + n] for i in range(len(t) - n + 1)]
        if not grams:
            continue
        top = max(set(grams), key=grams.count)
        worst = max(worst, grams.count(top) * n / len(t))
    return worst


def judge(text: str, duration: float) -> tuple[bool, str]:
    """このセグメントを発話として採用してよいかを判定する。"""
    t = text.strip()
    if not t:
        return False, "空文字"
    if len(t) < MIN_TEXT_CHARS:
        return False, f"短すぎる({len(t)}文字)"
    ja = len(_JA.findall(t)) / len(t)
    if ja < MIN_JA_RATIO:
        return False, f"日本語比率が低い({ja:.2f})"
    spc = duration / len(t)
    if spc > MAX_SEC_PER_CHAR:
        return False, f"引き伸ばし({spc:.2f}秒/文字)"
    rr = _repeat_ratio(t)
    if rr > MAX_REPEAT_RATIO:
        return False, f"反復({rr:.2f})"
    return True, ""


class Transcriber:
    def __init__(self, model_size: str = "medium", compute_type: str = "int8", threads: int = 8):
        from faster_whisper import WhisperModel

        log(f"ASR モデルを読み込みます: {model_size} ({compute_type}, {threads} threads)")
        self.model_size = model_size
        self.model = WhisperModel(
            model_size, device="cpu", compute_type=compute_type, cpu_threads=threads
        )

    def _run(self, audio: np.ndarray, use_vad: bool):
        return self.model.transcribe(
            audio,
            language="ja",
            beam_size=1,
            vad_filter=use_vad,
            vad_parameters=(dict(min_silence_duration_ms=700, speech_pad_ms=200)
                            if use_vad else None),
            condition_on_previous_text=False,
        )

    def transcribe(self, audio: np.ndarray, offset: float = 0.0,
                   use_vad: bool = True) -> list[AsrSegment]:
        """書き起こす。`offset` は元音源での開始秒。

        既定では Silero VAD で発話候補に絞ってから処理する(高速)。
        ただし **VAD がこの録音で機能しない場合がある**。260726 では VAD が
        実質的に発話を検出しなかった(0〜0.05%)が、実際には指揮者の発言が
        存在していた(VAD を切ると「2楽章4番」等が正しく取れる)。この日は
        空調の吹き出し口との位置関係で指揮者とマイクの距離が遠く、近接マイク
        前提の VAD の想定から外れたと考えられる。編成(管セク)が原因と
        断定できるデータはない。
        そこで VAD の結果が「発話がほぼ無い」と言っている場合は、VAD なしで
        自動的にやり直す。「0件」ではなく「尺に対する割合」で判定するのは、
        260726 合奏2 で VAD が 1 件だけ返し(65分の音源に対し2秒)、
        0件判定のフォールバックをすり抜けたため。
        """
        segments, _info = self._run(audio, use_vad)
        segments = list(segments)
        covered = sum(s.end - s.start for s in segments)
        total = len(audio) / ASR_SR
        if use_vad and total > 0 and covered / total < MIN_SPEECH_RATIO:
            log(f"    VAD の発話検出が {covered:.0f} 秒 / {total:.0f} 秒 "
                f"({covered/total*100:.2f}%) と少なすぎます。"
                "VAD なしで再試行します(時間がかかります)")
            segments, _info = self._run(audio, False)
            segments = list(segments)
        out: list[AsrSegment] = []
        for s in segments:
            text = s.text.strip()
            ok, why = judge(text, s.end - s.start)
            out.append(AsrSegment(
                start=offset + s.start, end=offset + s.end, text=text,
                no_speech_prob=float(s.no_speech_prob),
                avg_logprob=float(s.avg_logprob),
                compression_ratio=float(s.compression_ratio),
                accepted=ok, reject_reason=why,
            ))
        return out
