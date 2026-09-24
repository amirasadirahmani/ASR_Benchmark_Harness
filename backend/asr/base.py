"""
قرارداد آداپتور ASR — بند ۱۱ و ۲۵.

هر Runtime جدید فقط باید:
    1. از BaseASRAdapter ارث‌بری کند
    2. متدهای load() و transcribe() را پیاده کند
    3. با @register_runtime("نام") ثبت شود
هیچ تغییری در UI، Orchestrator، Metrics یا CSV لازم نخواهد بود.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TranscriptionOutput:
    """خروجی استاندارد رونویسی — مستقل از Runtime."""
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    audio_duration: float = 0.0
    inference_time: float = 0.0
    segments: list = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class ASRAdapterError(RuntimeError):
    """خطای عمومی لایهٔ آداپتور."""


class ModelNotFoundError(ASRAdapterError):
    """فایل‌های مدل روی دیسک موجود نیست (و دانلود در Runtime ممنوع است)."""


class RuntimeUnavailableError(ASRAdapterError):
    """کتابخانهٔ Runtime نصب نشده است."""


class BaseASRAdapter(abc.ABC):
    """کلاس پایهٔ همهٔ آداپتورهای ASR."""

    #: شناسهٔ Runtime — توسط دکوراتور register_runtime پر می‌شود
    runtime_name: str = "base"

    def __init__(
        self,
        model_id: str,
        model_path: str | Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        params: Optional[Dict[str, Any]] = None,
        cpu_threads: int = 4,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.device = device
        self.compute_type = compute_type
        self.params: Dict[str, Any] = dict(params or {})
        self.cpu_threads = cpu_threads

        self._model: Any = None
        self._loaded = False
        self.load_time: float = 0.0
        self.warmup_time: float = 0.0

    # ------------------------------------------------------------ abstract
    @abc.abstractmethod
    def _load_model(self) -> Any:
        """بارگذاری واقعی مدل. باید شیء مدل را برگرداند."""

    @abc.abstractmethod
    def _transcribe(self, audio_path: Path) -> TranscriptionOutput:
        """رونویسی یک فایل WAV با نرخ نمونه‌برداری 16kHz مونو."""

    # ------------------------------------------------------------ public
    def load(self) -> None:
        """بارگذاری مدل با اندازه‌گیری زمان (idempotent)."""
        if self._loaded:
            return
        self.ensure_model_exists()
        t0 = time.perf_counter()
        self._model = self._load_model()
        self.load_time = time.perf_counter() - t0
        self._loaded = True

    def warmup(self, audio_path: Optional[Path] = None) -> float:
        """
        اجرای یک رونویسی گرم‌کننده تا اثر lazy-init، تخصیص بافر و کش JIT
        از اندازه‌گیری اصلی حذف شود (بند ۱۳).
        """
        if audio_path is None or not Path(audio_path).exists():
            return 0.0
        t0 = time.perf_counter()
        try:
            self._transcribe(Path(audio_path))
        except Exception:
            # شکست warmup نباید اجرای اصلی را متوقف کند
            pass
        self.warmup_time = time.perf_counter() - t0
        return self.warmup_time

    def transcribe(self, audio_path: str | Path) -> TranscriptionOutput:
        """رونویسی با تضمین بارگذاری مدل و اندازه‌گیری زمان خالص استنتاج."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ASRAdapterError(f"فایل صوتی یافت نشد: {audio_path}")

        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        out = self._transcribe(audio_path)
        elapsed = time.perf_counter() - t0

        # اگر آداپتور خودش زمان نگذاشته بود، مقدار بیرونی را بنشان
        if not out.inference_time:
            out.inference_time = elapsed
        return out

    def unload(self) -> None:
        """
        آزادسازی مدل. در معماری Worker-per-model عملاً پروسه exit می‌کند،
        ولی برای حالت non-isolated و تست‌ها لازم است.
        """
        self._model = None
        self._loaded = False
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    # ------------------------------------------------------------ helpers
    def ensure_model_exists(self) -> None:
        """
        بررسی وجود فایل‌های مدل روی دیسک.
        ⚠️ دانلود در Runtime مطلقاً ممنوع است (بند ۴).
        """
        p = self.model_path
        if not p.exists() or (p.is_dir() and not any(p.iterdir())):
            raise ModelNotFoundError(
                f"فایل‌های مدل «{self.model_id}» در مسیر «{p}» یافت نشد. "
                f"ابتدا اجرا کنید: python scripts/setup_models.py --model {self.model_id}"
            )

    @staticmethod
    def require(module_name: str, install_hint: str = "") -> Any:
        """import ایمن کتابخانهٔ Runtime با پیام خطای فارسی و روشن."""
        import importlib
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            hint = f" ({install_hint})" if install_hint else ""
            raise RuntimeUnavailableError(
                f"کتابخانهٔ «{module_name}» نصب نشده است{hint}."
            ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def describe(self) -> Dict[str, Any]:
        """ابردادهٔ آداپتور — برای درج در گزارش نتایج."""
        return {
            "model_id": self.model_id,
            "runtime": self.runtime_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "model_path": str(self.model_path),
            "loaded": self._loaded,
            "load_time": round(self.load_time, 4),
            "warmup_time": round(self.warmup_time, 4),
        }

    # ------------------------------------------------------------ context
    def __enter__(self) -> "BaseASRAdapter":
        self.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} id={self.model_id!r} runtime={self.runtime_name!r} {state}>"