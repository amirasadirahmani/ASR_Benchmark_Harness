"""
اندازه‌گیری مصرف حافظه و CPU در طول اجرای مدل.

متدولوژی (بند ۱۷ و ۲۱):
    1. Peak RSS با نمونه‌برداری با گام ۵۰ میلی‌ثانیه در یک thread جدا.
       چرا thread و نه اندازه‌گیری قبل/بعد؟ چون اوج مصرف حافظه در *میانهٔ*
       بارگذاری مدل رخ می‌دهد (buffer موقت تبدیل وزن‌ها) و اندازه‌گیری
       نقطه‌ای آن را از دست می‌دهد.

    2. Baseline RSS پیش از بارگذاری مدل ثبت می‌شود تا بتوان «حافظهٔ خالص
       مدل» را از حافظهٔ مفسر پایتون تفکیک کرد.

    3. روی لینوکس، `VmHWM` از /proc به‌عنوان مرجع دقیق‌تر خوانده می‌شود
       (High Water Mark هسته — هیچ نمونه‌ای را از دست نمی‌دهد). روی macOS
       این امکان نیست و به نمونه‌برداری اکتفا می‌شود.

⚠️ همهٔ این اندازه‌گیری‌ها داخل Worker Process انجام می‌شود تا حافظهٔ
   سرور FastAPI و مدل‌های قبلی در عدد نهایی دخیل نشوند.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

_IS_LINUX = platform.system() == "Linux"
_MB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
def get_process_rss_mb(pid: Optional[int] = None) -> float:
    """مصرف حافظهٔ فعلی پروسه (Resident Set Size) بر حسب مگابایت."""
    if _HAS_PSUTIL:
        try:
            return psutil.Process(pid or os.getpid()).memory_info().rss / _MB
        except Exception:  # noqa: BLE001
            pass
    if _IS_LINUX:
        try:
            with open(f"/proc/{pid or os.getpid()}/statm") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / _MB
        except Exception:  # noqa: BLE001
            pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # لینوکس: کیلوبایت | macOS: بایت
        return peak / 1024.0 if _IS_LINUX else peak / _MB
    except Exception:  # noqa: BLE001
        return 0.0


def get_kernel_peak_rss_mb() -> Optional[float]:
    """
    High Water Mark از هسته (فقط لینوکس / Raspberry Pi).
    دقیق‌ترین منبع Peak RSS؛ مستقل از نرخ نمونه‌برداری.
    """
    if not _IS_LINUX:
        return None
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def get_system_memory() -> Dict[str, float]:
    """وضعیت کلی حافظهٔ سیستم — برای هشدار پیش از بارگذاری مدل سنگین."""
    if not _HAS_PSUTIL:
        return {}
    try:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "total_mb": round(vm.total / _MB, 1),
            "available_mb": round(vm.available / _MB, 1),
            "used_mb": round(vm.used / _MB, 1),
            "percent": vm.percent,
            "swap_used_mb": round(sw.used / _MB, 1),
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
@dataclass
class ResourceSnapshot:
    t: float
    rss_mb: float
    cpu_percent: float = 0.0
    label: str = ""


@dataclass
class ResourceUsage:
    """نتیجهٔ نهایی پایش منابع."""
    baseline_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    final_rss_mb: float = 0.0
    model_rss_mb: float = 0.0          # peak − baseline
    kernel_peak_rss_mb: Optional[float] = None
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    samples: int = 0
    duration: float = 0.0
    sample_interval: float = 0.05
    marks: Dict[str, float] = field(default_factory=dict)
    system_memory: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAMMonitor:
    """
    پایشگر منابع مبتنی بر thread.

    Example:
        with RAMMonitor(interval=0.05) as mon:
            adapter.load()
            mon.mark("model_loaded")
            adapter.transcribe(path)
        usage = mon.usage
    """

    def __init__(self, interval: float = 0.05, track_cpu: bool = True,
                 keep_samples: bool = False) -> None:
        self.interval = max(0.01, interval)
        self.track_cpu = track_cpu and _HAS_PSUTIL
        self.keep_samples = keep_samples

        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._baseline = 0.0
        self._peak = 0.0
        self._final = 0.0
        self._cpu: List[float] = []
        self._count = 0
        self._start_t = 0.0
        self._end_t = 0.0
        self._marks: Dict[str, float] = {}
        self.samples: List[ResourceSnapshot] = []

    # ------------------------------------------------------------------
    def start(self) -> "RAMMonitor":
        if self._thread is not None:
            return self
        self._baseline = self._peak = self._read_rss()
        self._start_t = time.perf_counter()
        if self.track_cpu and self._proc:
            try:
                self._proc.cpu_percent(None)   # فراخوانی اول، مقداردهی اولیه
            except Exception:  # noqa: BLE001
                pass
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ram-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> ResourceUsage:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._end_t = time.perf_counter()
        self._final = self._read_rss()
        with self._lock:
            self._peak = max(self._peak, self._final)
        return self.usage

    def mark(self, label: str) -> float:
        """ثبت نقطهٔ عطف (مثلاً پایان بارگذاری مدل)."""
        rss = self._read_rss()
        with self._lock:
            self._peak = max(self._peak, rss)
            self._marks[label] = round(rss, 2)
            if self.keep_samples:
                self.samples.append(ResourceSnapshot(
                    time.perf_counter() - self._start_t, rss, label=label))
        return rss

    # ------------------------------------------------------------------
    def _read_rss(self) -> float:
        if self._proc is not None:
            try:
                return self._proc.memory_info().rss / _MB
            except Exception:  # noqa: BLE001
                pass
        return get_process_rss_mb()

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = self._read_rss()
            cpu = 0.0
            if self.track_cpu and self._proc:
                try:
                    cpu = self._proc.cpu_percent(None)
                except Exception:  # noqa: BLE001
                    pass
            with self._lock:
                self._peak = max(self._peak, rss)
                self._count += 1
                if cpu > 0:
                    self._cpu.append(cpu)
                if self.keep_samples:
                    self.samples.append(ResourceSnapshot(
                        time.perf_counter() - self._start_t, rss, cpu))
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            peak, baseline, final = self._peak, self._baseline, self._final
            cpu = list(self._cpu)
            count, marks = self._count, dict(self._marks)

        kernel_peak = get_kernel_peak_rss_mb()
        # روی لینوکس، مقدار هسته همیشه دقیق‌تر است
        effective_peak = max(peak, kernel_peak) if kernel_peak else peak

        return ResourceUsage(
            baseline_rss_mb=round(baseline, 2),
            peak_rss_mb=round(effective_peak, 2),
            final_rss_mb=round(final, 2),
            model_rss_mb=round(max(0.0, effective_peak - baseline), 2),
            kernel_peak_rss_mb=round(kernel_peak, 2) if kernel_peak else None,
            cpu_percent_avg=round(sum(cpu) / len(cpu), 2) if cpu else 0.0,
            cpu_percent_max=round(max(cpu), 2) if cpu else 0.0,
            samples=count,
            duration=round(max(0.0, self._end_t - self._start_t), 3),
            sample_interval=self.interval,
            marks=marks,
            system_memory=get_system_memory(),
        )

    # ------------------------------------------------------------------
    def __enter__(self) -> "RAMMonitor":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()