"""
Worker Process ایزوله — قلب متدولوژی اندازه‌گیری (بند ۱۷).

چرا هر مدل در یک پروسهٔ مجزا اجرا می‌شود؟

  1. اندازه‌گیری صادقانهٔ Peak RSS
     اگر همهٔ مدل‌ها در یک پروسه بارگذاری شوند، allocatorهای پایتون و
     PyTorch حافظه را به سیستم‌عامل بازنمی‌گردانند. مدل دوم روی حافظهٔ
     آزادشدهٔ مدل اول می‌نشیند و Peak RSS آن *کمتر از واقع* گزارش می‌شود.
     نتیجه: هر چه مدل دیرتر اجرا شود، «سبک‌تر» به نظر می‌رسد — سوگیری
     کاملاً ساختگی.

  2. ایزولاسیون خرابی
     Segfault در CTranslate2 یا OOM در یک مدل، کل سرور را از کار
     نمی‌اندازد؛ فقط همان مدل با وضعیت «ناموفق» ثبت می‌شود.

  3. تضمین شروع سرد یکسان
     هر مدل با یک مفسر پایتون تازه شروع می‌کند → load_time قابل مقایسه.

  4. آزادسازی قطعی حافظه
     خروج پروسه تنها راه ۱۰۰٪ مطمئن بازگرداندن RAM به سیستم‌عامل است —
     روی Raspberry Pi 5 با ۸ گیگ حیاتی است.

⚠️ روش spawn استفاده می‌شود (نه fork):
   fork روی macOS با کتابخانه‌های native (CoreML/Accelerate) ناپایدار است
   و علاوه بر آن، حافظهٔ پروسهٔ والد را به ارث می‌برد که baseline RSS را
   آلوده می‌کند.
"""

from __future__ import annotations

import faulthandler
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

WORKER_TIMEOUT_EXIT = -9


# ---------------------------------------------------------------------------
@dataclass
class WorkerRequest:
    """درخواست ارسالی به Worker (باید کاملاً picklable باشد)."""
    model_payload: Dict[str, Any]       # خروجی ModelConfig.to_worker_payload()
    audio_path: str
    warmup_path: Optional[str] = None
    runs: int = 1
    ram_sample_interval: float = 0.05
    cpu_threads_env: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerRunResult:
    """نتیجهٔ یک نوبت رونویسی."""
    run_index: int = 0
    text: str = ""
    language: Optional[str] = None
    language_probability: Optional[float] = None
    audio_duration: float = 0.0
    inference_time: float = 0.0
    segments: list = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    """خروجی کامل Worker برای یک مدل."""
    model_id: str = ""
    runtime: str = ""
    success: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None

    load_time: float = 0.0
    warmup_time: float = 0.0
    total_time: float = 0.0
    runs: list = field(default_factory=list)      # List[WorkerRunResult as dict]

    resource_usage: Dict[str, Any] = field(default_factory=dict)
    adapter_info: Dict[str, Any] = field(default_factory=dict)
    worker_pid: int = 0
    python_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
def _configure_worker_env(threads: int) -> None:
    """
    محدودسازی thread کتابخانه‌های عددی.

    بدون این تنظیم، OpenMP/BLAS به تعداد هسته‌ها thread می‌سازد و روی RPi5
    باعث thrash شدن و اعداد RTF بی‌ثبات می‌شود. همچنین برای عادلانه بودن
    مقایسه، همهٔ مدل‌ها باید *دقیقاً* به یک اندازه منابع CPU بگیرند.
    """
    value = str(max(1, threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = value

    # قفل آفلاین — حتی اگر کتابخانه‌ای تلاش به دانلود کند، رد می‌شود (بند ۴)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _worker_main(request_dict: Dict[str, Any], result_queue: "mp.Queue") -> None:
    """
    نقطهٔ ورود پروسهٔ فرزند.

    ⚠️ همهٔ importهای سنگین *داخل* این تابع انجام می‌شوند تا پروسهٔ والد
    (سرور FastAPI) هرگز torch/ctranslate2 را بارگذاری نکند و baseline RSS
    آن سبک بماند.
    """
    faulthandler.enable()   # در صورت segfault، stack trace چاپ می‌شود

    req = WorkerRequest(**request_dict)
    payload = req.model_payload
    result = WorkerResult(
        model_id=payload.get("id", "?"),
        runtime=payload.get("runtime", "?"),
        worker_pid=os.getpid(),
        python_version=sys.version.split()[0],
    )

    t_start = time.perf_counter()
    monitor = None

    try:
        if req.cpu_threads_env:
            _configure_worker_env(payload.get("cpu_threads", 4))

        from backend.asr.registry import create_adapter_from_payload
        from backend.benchmark.ram_monitor import RAMMonitor

        monitor = RAMMonitor(interval=req.ram_sample_interval)
        monitor.start()
        monitor.mark("worker_baseline")

        adapter = create_adapter_from_payload(payload)

        # --- بارگذاری ---
        adapter.load()
        result.load_time = round(adapter.load_time, 4)
        monitor.mark("model_loaded")

        # --- گرم‌کردن (از اندازه‌گیری اصلی مستثناست — بند ۱۳) ---
        if req.warmup_path:
            result.warmup_time = round(adapter.warmup(Path(req.warmup_path)), 4)
            monitor.mark("warmup_done")

        # --- اجراهای اصلی ---
        audio_path = Path(req.audio_path)
        for i in range(max(1, req.runs)):
            out = adapter.transcribe(audio_path)
            result.runs.append(asdict(WorkerRunResult(
                run_index=i,
                text=out.text,
                language=out.language,
                language_probability=out.language_probability,
                audio_duration=round(out.audio_duration, 4),
                inference_time=round(out.inference_time, 4),
                segments=out.segments,
                raw=out.raw,
            )))
            monitor.mark(f"run_{i}_done")

        result.adapter_info = adapter.describe()
        result.success = True

        try:
            adapter.unload()
        except Exception:  # noqa: BLE001
            pass

    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error = str(exc)
        result.error_type = type(exc).__name__
        result.traceback = traceback.format_exc(limit=12)

    finally:
        if monitor is not None:
            try:
                result.resource_usage = monitor.stop().to_dict()
            except Exception:  # noqa: BLE001
                pass
        result.total_time = round(time.perf_counter() - t_start, 4)
        try:
            result_queue.put(result.to_dict())
        except Exception:  # noqa: BLE001
            # اگر حتی ارسال نتیجه هم شکست خورد، حداقل خطا را برسان
            try:
                result_queue.put({
                    "model_id": result.model_id, "success": False,
                    "error": "ارسال نتیجه از Worker ناموفق بود.",
                    "error_type": "QueueError",
                })
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
class ModelWorker:
    """
    مدیریت چرخهٔ عمر یک Worker Process.

    Example:
        w = ModelWorker(timeout=180)
        result = w.run(WorkerRequest(payload, "cmd.wav", runs=3))
    """

    def __init__(self, timeout: float = 300.0, ctx_method: str = "spawn") -> None:
        self.timeout = timeout
        try:
            self._ctx = mp.get_context(ctx_method)
        except ValueError:
            logger.warning("روش «%s» پشتیبانی نمی‌شود → spawn", ctx_method)
            self._ctx = mp.get_context("spawn")

    # ------------------------------------------------------------------
    def run(self, request: WorkerRequest) -> WorkerResult:
        """
        اجرای همگام مدل در پروسهٔ مجزا.

        همیشه یک WorkerResult برمی‌گرداند — هرگز استثنا پرتاب نمی‌کند تا
        شکست یک مدل، زنجیرهٔ Benchmark را متوقف نکند (بند ۱۸).
        """
        model_id = request.model_payload.get("id", "?")
        q: "mp.Queue" = self._ctx.Queue(maxsize=1)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(request.to_dict(), q),
            name=f"asr-worker-{model_id}",
            daemon=False,
        )

        t0 = time.perf_counter()
        proc.start()
        logger.info("Worker مدل «%s» آغاز شد (pid=%s)", model_id, proc.pid)

        payload: Optional[Dict[str, Any]] = None
        try:
            payload = q.get(timeout=self.timeout)
        except queue.Empty:
            elapsed = time.perf_counter() - t0
            logger.error("مدل «%s» پس از %.1f ثانیه پاسخ نداد → خاتمه",
                         model_id, elapsed)
            self._kill(proc)
            return self._failure(
                request, "Timeout",
                f"مدل «{model_id}» در مهلت {self.timeout:.0f} ثانیه پاسخ نداد.",
                elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            self._kill(proc)
            return self._failure(request, type(exc).__name__, str(exc),
                                 time.perf_counter() - t0)
        finally:
            proc.join(timeout=10.0)
            if proc.is_alive():
                self._kill(proc)
            try:
                q.close()
                q.join_thread()
            except Exception:  # noqa: BLE001
                pass

        # تشخیص کرش پس از دریافت نتیجه (مثلاً OOM killer)
        exit_code = proc.exitcode
        result = WorkerResult(**_fill_defaults(payload))
        if result.success and exit_code not in (0, None):
            logger.warning("مدل «%s» با کد خروج %s پایان یافت",
                           model_id, exit_code)
            result.adapter_info["exit_code"] = exit_code

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _kill(proc: "mp.Process") -> None:
        """خاتمهٔ تدریجی: terminate → kill."""
        if not proc.is_alive():
            return
        try:
            proc.terminate()
            proc.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        if proc.is_alive():
            try:
                proc.kill()
                proc.join(timeout=3.0)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _failure(request: WorkerRequest, error_type: str,
                 message: str, elapsed: float) -> WorkerResult:
        return WorkerResult(
            model_id=request.model_payload.get("id", "?"),
            runtime=request.model_payload.get("runtime", "?"),
            success=False,
            error=message,
            error_type=error_type,
            total_time=round(elapsed, 4),
        )


def _fill_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """تکمیل کلیدهای غایب تا ساخت WorkerResult هرگز شکست نخورد."""
    allowed = set(WorkerResult.__dataclass_fields__.keys())
    return {k: v for k, v in d.items() if k in allowed}