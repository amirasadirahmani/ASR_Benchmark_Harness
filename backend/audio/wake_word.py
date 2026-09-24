"""
تشخیص کلمهٔ بیدارباش «آرینا» — موتور مستقل و Pluggable.

⚠️ قید معماری (بند ۸):
    این ماژول *هرگز* از مدل‌های تحت Benchmark استفاده نمی‌کند. اگر یکی از
    مدل‌های Whisper برای تشخیص Wake Word به کار می‌رفت:
      1. آن مدل دائماً در RAM می‌ماند و اندازه‌گیری Peak RSS بی‌معنا می‌شد.
      2. آن مدل مزیت «گرم بودن» پیدا می‌کرد و مقایسه ناعادلانه می‌شد.

سه موتور:
    vosk   — تشخیص متنی با مدل کوچک فارسی (~۵۰MB)، دقیق و کاملاً آفلاین.
    dtw    — تطبیق الگو با MFCC + Dynamic Time Warping. بدون هیچ مدلی؛
             کاربر ۳ نمونه از «آرینا» ضبط می‌کند. مناسب RPi5.
    manual — فعال‌سازی با دکمه در UI (همیشه در دسترس، fallback نهایی).
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.audio.audio_utils import bytes_to_float32, rms_dbfs
from backend.config.settings import WakeWordSettings
from backend.core.persian_normalizer import PersianNormalizer, get_normalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class WakeWordEvent:
    """نتیجهٔ یک تشخیص موفق."""
    detected: bool = False
    word: str = ""
    confidence: float = 0.0
    engine: str = ""
    timestamp: float = field(default_factory=time.time)
    matched_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "detected": self.detected,
            "word": self.word,
            "confidence": round(self.confidence, 4),
            "engine": self.engine,
            "timestamp": round(self.timestamp, 3),
        }
        if self.matched_text:
            d["matched_text"] = self.matched_text
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
class BaseWakeWordDetector(abc.ABC):
    """قرارداد مشترک موتورهای Wake Word."""

    name: str = "base"

    def __init__(self, settings: WakeWordSettings, sample_rate: int = 16000) -> None:
        self.settings = settings
        self.sample_rate = sample_rate
        self.word = settings.word
        self._last_trigger: float = 0.0

    @abc.abstractmethod
    def _process(self, frame: bytes) -> Optional[WakeWordEvent]: ...

    # ------------------------------------------------------------------
    def push(self, frame: bytes) -> Optional[WakeWordEvent]:
        """پردازش فریم با اعمال cooldown (ضد trigger مکرر)."""
        now = time.time()
        if now - self._last_trigger < self.settings.cooldown_seconds:
            return None
        event = self._process(frame)
        if event and event.detected:
            self._last_trigger = now
            self.reset()
        return event

    def reset(self) -> None:
        """بازنشانی حالت داخلی پس از trigger."""

    def close(self) -> None:
        """آزادسازی منابع."""

    def describe(self) -> Dict[str, Any]:
        return {"engine": self.name, "word": self.word,
                "sample_rate": self.sample_rate}


# ---------------------------------------------------------------------------
class ManualWakeWordDetector(BaseWakeWordDetector):
    """
    فعال‌سازی دستی از UI. هرگز خودکار trigger نمی‌شود؛
    API صریحاً trigger() را صدا می‌زند.
    """

    name = "manual"

    def _process(self, frame: bytes) -> Optional[WakeWordEvent]:
        return None

    def trigger(self) -> WakeWordEvent:
        self._last_trigger = time.time()
        return WakeWordEvent(True, self.word, 1.0, self.name,
                             metadata={"source": "ui_button"})


# ---------------------------------------------------------------------------
class VoskWakeWordDetector(BaseWakeWordDetector):
    """
    تشخیص متنی با Vosk (مدل کوچک فارسی، کاملاً آفلاین).

    از گرامر محدود (grammar-constrained decoding) استفاده می‌شود: دیکودر
    فقط بین «آرینا» و [unk] انتخاب می‌کند → بسیار سریع و کم‌مصرف،
    مناسب اجرای دائمی روی Raspberry Pi 5.
    """

    name = "vosk"

    def __init__(self, settings: WakeWordSettings, sample_rate: int = 16000) -> None:
        super().__init__(settings, sample_rate)
        path = Path(settings.vosk_model_path)
        if not path.is_absolute():
            from backend.config.settings import PROJECT_ROOT
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"مدل Vosk یافت نشد: {path}\n"
                "اجرا کنید: python scripts/setup_models.py --wakeword vosk"
            )
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            raise RuntimeError("کتابخانهٔ vosk نصب نیست: pip install vosk") from exc

        SetLogLevel(-1)
        self._model = Model(str(path))
        self._normalizer: PersianNormalizer = get_normalizer()
        self._aliases = {self._normalizer.normalize(a)
                         for a in settings.aliases if a.strip()}
        self._aliases.add(self._normalizer.normalize(settings.word))
        self._grammar = json.dumps(sorted(self._aliases) + ["[unk]"],
                                   ensure_ascii=False)
        self._rec = self._new_recognizer()

    def _new_recognizer(self):
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(self._model, self.sample_rate, self._grammar)
        rec.SetWords(False)
        return rec

    def _match(self, text: str) -> bool:
        norm = self._normalizer.normalize(text)
        if not norm:
            return False
        if norm in self._aliases:
            return True
        return any(f" {a} " in f" {norm} " for a in self._aliases)

    def _process(self, frame: bytes) -> Optional[WakeWordEvent]:
        try:
            final = self._rec.AcceptWaveform(frame)
            raw = self._rec.Result() if final else self._rec.PartialResult()
            data = json.loads(raw)
            text = data.get("text") if final else data.get("partial", "")
        except Exception as exc:  # noqa: BLE001
            logger.debug("خطای Vosk: %s", exc)
            return None

        if text and self._match(text):
            return WakeWordEvent(True, self.word, 1.0 if final else 0.75,
                                 self.name, matched_text=text)
        return None

    def reset(self) -> None:
        try:
            self._rec = self._new_recognizer()
        except Exception:  # noqa: BLE001
            pass

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["aliases"] = sorted(self._aliases)
        return d


# ---------------------------------------------------------------------------
# DTW — بدون نیاز به هیچ مدلی
# ---------------------------------------------------------------------------
def _mfcc(samples: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13,
          n_fft: int = 512, hop: int = 160, n_mels: int = 26) -> np.ndarray:
    """
    استخراج MFCC با numpy خالص (بدون librosa → سبک و آفلاین).
    خروجی: آرایهٔ (frames, n_mfcc) نرمال‌شده.
    """
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    # pre-emphasis
    emph = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])

    n_frames = 1 + (emph.size - n_fft) // hop
    if n_frames < 1:
        return np.zeros((1, n_mfcc), dtype=np.float32)

    idx = np.tile(np.arange(n_fft), (n_frames, 1)) + \
        np.tile(np.arange(0, n_frames * hop, hop), (n_fft, 1)).T
    frames = emph[idx.astype(np.int32)] * np.hamming(n_fft)

    power = (np.abs(np.fft.rfft(frames, n_fft)) ** 2) / n_fft

    # بانک فیلتر mel
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(0), hz2mel(sample_rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mel_pts) / sample_rate).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[m - 1, k] = (right - k) / (right - center)

    energy = np.maximum(power @ fbank.T, 1e-10)
    log_energy = np.log(energy)

    # DCT-II
    n = np.arange(n_mels)
    k = np.arange(n_mfcc).reshape(-1, 1)
    dct = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    mfcc = (log_energy @ dct.T).astype(np.float32)

    # نرمال‌سازی CMVN → مقاوم در برابر تغییر بلندی صدا و میکروفون
    mfcc -= mfcc.mean(axis=0, keepdims=True)
    std = mfcc.std(axis=0, keepdims=True)
    return mfcc / np.maximum(std, 1e-6)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """فاصلهٔ DTW نرمال‌شده بین دو دنبالهٔ ویژگی (کسینوسی)."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    cost = 1.0 - (an @ bn.T)          # (n, m) در بازهٔ [0, 2]

    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = ci[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


class DTWWakeWordDetector(BaseWakeWordDetector):
    """
    تطبیق الگو با MFCC + DTW.

    کاربر ۳ نمونه از «آرینا» ضبط می‌کند (scripts/enroll_wakeword.py).
    مزیت: هیچ مدلی لازم نیست، وابسته به گوینده → دقت بالا و مصرف ناچیز.
    """

    name = "dtw"

    def __init__(self, settings: WakeWordSettings, sample_rate: int = 16000) -> None:
        super().__init__(settings, sample_rate)
        self.threshold = settings.dtw_threshold
        self._templates: List[np.ndarray] = []
        self._load_templates()
        if not self._templates:
            raise FileNotFoundError(
                f"هیچ الگویی برای «{self.word}» در {settings.dtw_templates_dir} نیست.\n"
                "اجرا کنید: python scripts/enroll_wakeword.py"
            )
        # پنجرهٔ لغزان ۱.۲ ثانیه با گام ۰.۲ ثانیه
        self._win = int(1.2 * sample_rate)
        self._step = int(0.2 * sample_rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._since_check = 0

    def _load_templates(self) -> None:
        from backend.audio.audio_utils import read_wav
        from backend.config.settings import PROJECT_ROOT

        d = Path(self.settings.dtw_templates_dir)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        if not d.exists():
            return
        for wav in sorted(d.glob("*.wav")):
            try:
                pcm, rate, _ = read_wav(wav)
                self._templates.append(_mfcc(bytes_to_float32(pcm), rate))
            except Exception as exc:  # noqa: BLE001
                logger.warning("الگوی «%s» بارگذاری نشد: %s", wav.name, exc)

    def _process(self, frame: bytes) -> Optional[WakeWordEvent]:
        self._buf = np.concatenate([self._buf, bytes_to_float32(frame)])
        if self._buf.size > self._win:
            self._buf = self._buf[-self._win:]

        self._since_check += len(frame) // 2
        if self._buf.size < self._win or self._since_check < self._step:
            return None
        self._since_check = 0

        # صرفه‌جویی: روی سکوت اصلاً MFCC حساب نکن
        if rms_dbfs(frame) < -55.0:
            return None

        feats = _mfcc(self._buf, self.sample_rate)
        best = min((_dtw_distance(t, feats) for t in self._templates),
                   default=float("inf"))
        if best <= self.threshold:
            conf = max(0.0, min(1.0, 1.0 - best / max(self.threshold, 1e-6)))
            return WakeWordEvent(True, self.word, round(0.5 + conf / 2, 4),
                                 self.name, metadata={"dtw_distance": round(best, 4)})
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._since_check = 0

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"templates": len(self._templates), "threshold": self.threshold})
        return d


# ---------------------------------------------------------------------------
def create_wake_word_detector(settings: WakeWordSettings,
                              sample_rate: int = 16000) -> BaseWakeWordDetector:
    """
    کارخانهٔ ساخت با تنزل تدریجی امن:
        درخواستی → dtw → manual
    هرگز استثنا نمی‌دهد؛ در بدترین حالت فعال‌سازی دستی برمی‌گردد.
    """
    engine = settings.engine

    if engine == "vosk":
        try:
            return VoskWakeWordDetector(settings, sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("موتور Vosk در دسترس نیست (%s) → dtw", exc)
            engine = "dtw"

    if engine == "dtw":
        try:
            return DTWWakeWordDetector(settings, sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("موتور DTW در دسترس نیست (%s) → manual", exc)
            engine = "manual"

    logger.info("موتور Wake Word: فعال‌سازی دستی از رابط کاربری")
    return ManualWakeWordDetector(settings, sample_rate)