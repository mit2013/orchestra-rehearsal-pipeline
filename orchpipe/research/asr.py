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

    def transcribe(self, audio: np.ndarray, offset: float = 0.0) -> list[AsrSegment]:
        """VAD で発話候補を絞ってから書き起こす。`offset` は元音源での開始秒。"""
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=200),
            condition_on_previous_text=False,
        )
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
