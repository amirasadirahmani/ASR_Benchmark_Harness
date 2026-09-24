"""
تشخیص فعالیت گفتاری (VAD) و دروازهٔ پایان‌یابی فرمان.

سه موتور قابل تعویض:
    webrtc  — پیش‌فرض. بدون هیچ فایل مدلی → آفلاین ۱۰۰٪، بسیار سبک روی RPi5.
    silero  — دقیق‌تر در محیط نویزی، نیازمند فایل ONNX محلی.
    energy  — fallback نهایی؛ وقتی هیچ کتابخانه‌ای در دسترس نیست.

دروازهٔ VADGate منطق بند ۹ را پیاده می‌کند:
    پس از تشخیص شروع گفتار، هر گاه SILENCE_DURATION (پیش‌فرض ۲ ثانیه)
    سکوت ممتد دیده شد، فرمان «نهایی» اعلام می‌شود.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.audio.audio_utils import bytes_to_float32, rms_dbfs
from backend.config.settings import VADSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class BaseVAD(abc.ABC):
    """قرارداد مشترک موتورهای VAD."""

    name: str = "base"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """آیا این فریم حاوی گفتار است؟"""

    def reset(self) -> None:
        """بازنشانی حالت داخلی (برای موتورهای stateful مثل silero)."""

    def describe(self) -> Dict[str, Any]:
        return {"engine": self.name, "sample_rate": self.sample_rate,
                "frame_ms": self.frame_ms}


# ---------------------------------------------------------------------------
class WebRTCVAD(BaseVAD):
    """موتور WebRTC — سریع، سبک، بدون فایل مدل."""

    name = "webrtc"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20,
                 aggressiveness: int = 2) -> None:
        super().__init__(sample_rate, frame_ms)
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD فقط نرخ ۸/۱۶/۳۲/۴۸ کیلوهرتز را می‌پذیرد.")
        if frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD فقط فریم ۱۰/۲۰/۳۰ میلی‌ثانیه را می‌پذیرد.")

        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "کتابخانهٔ webrtcvad نصب نیست. نصب: pip install webrtcvad-wheels"
            ) from exc

        self.aggressiveness = max(0, min(3, aggressiveness))
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._expected = int(sample_rate * frame_ms / 1000) * 2

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self._expected:
            if len(frame) < self._expected:
                frame = frame + b"\x00" * (self._expected - len(frame))
            else:
                frame = frame[: self._expected]
        try:
            return self._vad.is_speech(frame, self.sample_rate)
        except Exception:  # noqa: BLE001
            return False

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["aggressiveness"] = self.aggressiveness
        return d


class EnergyVAD(BaseVAD):
    """
    موتور انرژی‌محور با کف نویز تطبیقی.
    آستانه به‌آرامی با سکوت محیط تنظیم می‌شود تا در اتاق‌های مختلف کار کند.
    """

    name = "energy"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20,
                 threshold_dbfs: float = -45.0, adaptive: bool = True) -> None:
        super().__init__(sample_rate, frame_ms)
        self.threshold_dbfs = threshold_dbfs
        self.adaptive = adaptive
        self._noise_floor: Optional[float] = None
        self._margin = 8.0

    def is_speech(self, frame: bytes) -> bool:
        level = rms_dbfs(frame)
        if not self.adaptive:
            return level > self.threshold_dbfs

        if self._noise_floor is None:
            self._noise_floor = level
            return level > self.threshold_dbfs

        threshold = max(self.threshold_dbfs, self._noise_floor + self._margin)
        speech = level > threshold
        if not speech:  # فقط در سکوت، کف نویز را به‌روز کن
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * level
        return speech

    def reset(self) -> None:
        self._noise_floor = None


class SileroVAD(BaseVAD):
    """موتور Silero از طریق ONNX Runtime — فایل مدل باید محلی موجود باشد."""

    name = "silero"
    _WINDOW = 512  # نمونه، برای 16kHz

    def __init__(self, model_path: str | Path, sample_rate: int = 16000,
                 frame_ms: int = 20, threshold: float = 0.5) -> None:
        super().__init__(sample_rate, frame_ms)
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"فایل مدل Silero VAD یافت نشد: {path}\n"
                "اجرا کنید: python scripts/setup_models.py --vad silero"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("کتابخانهٔ onnxruntime نصب نیست.") from exc

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self._pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._last = False

    def is_speech(self, frame: bytes) -> bool:
        self._pending = np.concatenate([self._pending, bytes_to_float32(frame)])
        decided = self._last
        while self._pending.size >= self._WINDOW:
            chunk = self._pending[: self._WINDOW]
            self._pending = self._pending[self._WINDOW:]
            try:
                out, self._state = self._sess.run(
                    None,
                    {
                        "input": chunk.reshape(1, -1),
                        "state": self._state,
                        "sr": np.array(self.sample_rate, dtype=np.int64),
                    },
                )
                decided = float(out.squeeze()) >= self.threshold
            except Exception as exc:  # noqa: BLE001
                logger.warning("خطای Silero VAD: %s", exc)
                decided = False
        self._last = decided
        return decided


# ---------------------------------------------------------------------------
def create_vad(settings: VADSettings, sample_rate: int = 16000,
               frame_ms: int = 20) -> BaseVAD:
    """
    کارخانهٔ ساخت VAD با تنزل تدریجی امن:
        درخواستی → webrtc → energy
    هرگز خطا نمی‌دهد؛ در بدترین حالت موتور انرژی برمی‌گردد.
    """
    engine = settings.engine

    if engine == "silero":
        try:
            return SileroVAD(settings.silero_model_path, sample_rate, frame_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Silero VAD در دسترس نیست (%s) → webrtc", exc)
            engine = "webrtc"

    if engine == "webrtc":
        try:
            return WebRTCVAD(sample_rate, frame_ms, settings.aggressiveness)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebRTC VAD در دسترس نیست (%s) → energy", exc)
            engine = "energy"

    return EnergyVAD(sample_rate, frame_ms, settings.energy_threshold_dbfs)


# ---------------------------------------------------------------------------
# دروازهٔ پایان‌یابی
# ---------------------------------------------------------------------------
class VADEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_CONTINUE = "speech_continue"
    SILENCE = "silence"
    UTTERANCE_END = "utterance_end"     # ← SILENCE_DURATION تکمیل شد
    TIMEOUT = "timeout"                 # ← هیچ گفتاری نیامد
    MAX_DURATION = "max_duration"       # ← سقف ایمنی طول فرمان


@dataclass
class VADEvent:
    type: VADEventType
    timestamp: float = 0.0
    speech_duration: float = 0.0
    silence_duration: float = 0.0
    level_dbfs: float = -96.0
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": round(self.timestamp, 3),
            "speech_duration": round(self.speech_duration, 3),
            "silence_duration": round(self.silence_duration, 3),
            "level_dbfs": round(self.level_dbfs, 1),
            "is_final": self.is_final,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class VADGate:
    """
    ماشین حالت پایان‌یابی فرمان (بند ۹).

    جریان:
        هر فریم → push() → رویداد
        شروع گفتار پس از N فریم متوالی گفتار (ضد فعال‌سازی کاذب)
        پایان فرمان پس از SILENCE_DURATION سکوت ممتد
    """

    def __init__(
        self,
        vad: BaseVAD,
        *,
        frame_ms: int = 20,
        silence_duration: float = 2.0,
        speech_start_frames: int = 3,
        max_duration: float = 30.0,
        speech_timeout: float = 6.0,
    ) -> None:
        self.vad = vad
        self.frame_seconds = frame_ms / 1000.0
        self.silence_duration = silence_duration
        self.speech_start_frames = max(1, speech_start_frames)
        self.max_duration = max_duration
        self.speech_timeout = speech_timeout
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.vad.reset()
        self._elapsed = 0.0
        self._speech_time = 0.0
        self._silence_time = 0.0
        self._consecutive_speech = 0
        self._started = False
        self._finished = False
        self._frames_seen = 0

    # ------------------------------------------------------------------
    def push(self, frame: bytes) -> VADEvent:
        """پردازش یک فریم و برگرداندن رویداد متناظر."""
        if self._finished:
            return VADEvent(VADEventType.UTTERANCE_END, self._elapsed,
                            self._speech_time, self._silence_time, is_final=True)

        self._frames_seen += 1
        self._elapsed += self.frame_seconds
        level = rms_dbfs(frame)
        speech = self.vad.is_speech(frame)

        # --- سقف ایمنی طول فرمان ---
        if self._started and self._speech_time + self._silence_time >= self.max_duration:
            self._finished = True
            return VADEvent(VADEventType.MAX_DURATION, self._elapsed,
                            self._speech_time, self._silence_time, level, is_final=True)

        if speech:
            self._consecutive_speech += 1
            self._silence_time = 0.0
            if not self._started:
                if self._consecutive_speech >= self.speech_start_frames:
                    self._started = True
                    # فریم‌های تأییدکننده را هم جزو گفتار حساب کن
                    self._speech_time = self._consecutive_speech * self.frame_seconds
                    return VADEvent(VADEventType.SPEECH_START, self._elapsed,
                                    self._speech_time, 0.0, level)
                return VADEvent(VADEventType.SILENCE, self._elapsed,
                                0.0, self._silence_time, level)
            self._speech_time += self.frame_seconds
            return VADEvent(VADEventType.SPEECH_CONTINUE, self._elapsed,
                            self._speech_time, 0.0, level)

        # --- فریم سکوت ---
        self._consecutive_speech = 0
        self._silence_time += self.frame_seconds

        if not self._started:
            if self.speech_timeout > 0 and self._elapsed >= self.speech_timeout:
                self._finished = True
                return VADEvent(VADEventType.TIMEOUT, self._elapsed,
                                0.0, self._silence_time, level, is_final=True)
            return VADEvent(VADEventType.SILENCE, self._elapsed,
                            0.0, self._silence_time, level)

        if self._silence_time >= self.silence_duration:
            self._finished = True
            return VADEvent(VADEventType.UTTERANCE_END, self._elapsed,
                            self._speech_time, self._silence_time, level, is_final=True)

        return VADEvent(VADEventType.SILENCE, self._elapsed,
                        self._speech_time, self._silence_time, level)

    # ------------------------------------------------------------------
    @property
    def speech_started(self) -> bool:
        return self._started

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def speech_duration(self) -> float:
        return self._speech_time

    def status(self) -> Dict[str, Any]:
        return {
            "engine": self.vad.name,
            "started": self._started,
            "finished": self._finished,
            "elapsed": round(self._elapsed, 3),
            "speech_duration": round(self._speech_time, 3),
            "silence_duration": round(self._silence_time, 3),
            "silence_threshold": self.silence_duration,
            "frames_seen": self._frames_seen,
        }