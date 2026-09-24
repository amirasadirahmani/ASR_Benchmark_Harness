"""
ارکستراتور Benchmark — هماهنگ‌کنندهٔ اجرای همهٔ مدل‌ها روی یک فایل صوتی.

تضمین‌های اصلی:

  * یکسانی ورودی (بند ۱۰)
    یک AudioArtifact واحد روی دیسک نوشته می‌شود و *همان مسیر* به تمام
    Workerها داده می‌شود. sha256 در گزارش ثبت می‌گردد تا قابل اثبات باشد.

  * اجرای ترتیبی (بند ۱۷)
    مدل‌ها هرگز موازی اجرا نمی‌شوند. اجرای هم‌زمان باعث رقابت بر سر CPU و
    حافظه می‌شود و هم RTF و هم Peak RSS را بی‌معنا می‌کند. روی RPi5 حتی
    منجر به OOM می‌شود.

  * تاب‌آوری (بند ۱۸)
    شکست یا timeout یک مدل، بقیه را متوقف نمی‌کند.

  * جریان زندهٔ نتایج (بند ۱۹)
    نتیجهٔ هر مدل بلافاصله پس از آماده شدن از طریق callback به UI می‌رود؛
    کاربر منتظر پایان همهٔ مدل‌ها نمی‌ماند.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.asr.registry import validate_registry
from backend.audio.audio_utils import (
    AudioArtifact, make_warmup_tone, write_wav,
)
from backend.benchmark.ram_monitor import get_system_memory
from backend.benchmark.worker import ModelWorker, WorkerRequest, WorkerResult
from backend.config.model_config import (
    ModelConfig, describe_environment, load_model_configs,
)
from backend.config.settings import AppSettings, get_settings
from backend.core.metrics import (
    AggregatedMetrics, MetricsEngine, TranscriptionMetrics,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """نتیجهٔ کامل یک مدل — واحد نمایش در UI و یک سطر در CSV."""
    model_id: str
    display_name: str
    runtime: str
    order: int = 100

    success: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None

    text: str = ""
    text_normalized: str = ""
    metrics: Optional[Dict[str, Any]] = None
    aggregated: Optional[Dict[str, Any]] = None

    load_time: float = 0.0
    warmup_time: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    rtf: Optional[float] = None

    peak_ram_mb: Optional[float] = None
    model_ram_mb: Optional[float] = None
    cpu_percent_avg: Optional[float] = None

    runs: int = 1
    device: str = "cpu"
    compute_type: str = "int8"
    worker_pid: int = 0
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """گزارش کامل یک نوبت Benchmark."""
    benchmark_id: str
    session_id: str = ""
    utterance_id: str = ""
    created_at: str = ""

    audio: Dict[str, Any] = field(default_factory=dict)
    reference_text: Optional[str] = None
    has_reference: bool = False
    no_reference_message: Optional[str] = None

    results: List[Dict[str, Any]] = field(default_factory=list)
    ranking: List[str] = field(default_factory=list)
    best_model: Optional[str] = None
    fastest_model: Optional[str] = None
    lightest_model: Optional[str] = None

    total_duration: float = 0.0
    models_total: int = 0
    models_succeeded: int = 0
    models_failed: int = 0

    environment: Dict[str, Any] = field(default_factory=dict)
    normalizer: Dict[str, Any] = field(default_factory=dict)
    settings_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
class BenchmarkOrchestrator:
    """
    اجراکنندهٔ Benchmark روی یک قطعهٔ صوتی.

    Example:
        orch = BenchmarkOrchestrator(on_progress=ws_send)
        report = await orch.run(artifact, reference_text="چراغ را روشن کن")
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        on_progress: Optional[ProgressCallback] = None,
        metrics_engine: Optional[MetricsEngine] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.on_progress = on_progress
        self.metrics = metrics_engine or MetricsEngine()
        self._worker = ModelWorker(
            timeout=self.settings.benchmark.model_timeout,
            ctx_method=self.settings.benchmark.mp_context,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="benchmark")
        self._cancel = asyncio.Event()
        self._running = False
        self._warmup_path: Optional[Path] = None

    # ------------------------------------------------------------------ API
    @property
    def is_running(self) -> bool:
        return self._running

    def cancel(self) -> None:
        """لغو Benchmark جاری (مدل در حال اجرا تا پایان ادامه می‌یابد)."""
        self._cancel.set()

    def list_models(self) -> List[ModelConfig]:
        return load_model_configs(enabled_only=True)

    def preflight(self) -> Dict[str, Any]:
        """بررسی آمادگی پیش از اجرا — برای نمایش در UI هنگام بالا آمدن."""
        configs = load_model_configs()
        report = validate_registry(configs)
        return {
            "models": [c.to_public_dict() for c in configs],
            "validation": report,
            "ready": len(report["ok"]) > 0,
            "environment": describe_environment(),
            "system_memory": get_system_memory(),
        }

    # ------------------------------------------------------------------
    async def run(
        self,
        artifact: AudioArtifact,
        reference_text: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """اجرای Benchmark روی artifact و برگرداندن گزارش کامل."""
        self._running = True
        self._cancel.clear()
        t_start = time.perf_counter()

        report = BenchmarkReport(
            benchmark_id=uuid.uuid4().hex[:12],
            session_id=artifact.session_id,
            utterance_id=artifact.utterance_id,
            created_at=artifact.created_at,
            audio=artifact.to_dict(),
            reference_text=reference_text,
            has_reference=bool(reference_text and reference_text.strip()),
            environment=describe_environment(),
            normalizer=self.metrics.normalizer.fingerprint(),
            settings_snapshot=self._settings_snapshot(),
        )
        if not report.has_reference:
            # بند ۱۵ — هیچ عدد دقتی ساخته نمی‌شود
            report.no_reference_message = (
                "جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت."
            )

        try:
            audio_path = self._ensure_audio_on_disk(artifact)
            configs = self._select_models(model_ids)
            report.models_total = len(configs)

            if not configs:
                await self._emit("benchmark_error", {
                    "benchmark_id": report.benchmark_id,
                    "message": "هیچ مدل فعال و موجودی برای اجرا یافت نشد.",
                })
                return report

            await self._emit("benchmark_started", {
                "benchmark_id": report.benchmark_id,
                "models": [c.to_public_dict() for c in configs],
                "audio": {"duration": round(artifact.duration, 3),
                          "sha256": artifact.sha256},
                "has_reference": report.has_reference,
                "runs_per_model": self.settings.benchmark.runs_per_model,
            })

            warmup = self._prepare_warmup()

            for idx, cfg in enumerate(configs, start=1):
                if self._cancel.is_set():
                    await self._emit("benchmark_cancelled", {
                        "benchmark_id": report.benchmark_id,
                        "completed": idx - 1, "total": len(configs),
                    })
                    break

                await self._emit("model_started", {
                    "benchmark_id": report.benchmark_id,
                    "model_id": cfg.id, "display_name": cfg.display_name,
                    "index": idx, "total": len(configs),
                })

                result = await self._run_single(cfg, audio_path, warmup,
                                                reference_text, artifact)
                report.results.append(result.to_dict())
                if result.success:
                    report.models_succeeded += 1
                else:
                    report.models_failed += 1

                # بند ۱۹ — نتیجه بلافاصله به UI می‌رود
                await self._emit("model_result", {
                    "benchmark_id": report.benchmark_id,
                    "index": idx, "total": len(configs),
                    "result": result.to_dict(),
                })

            self._finalize(report)
            report.total_duration = round(time.perf_counter() - t_start, 3)

            await self._emit("benchmark_completed", {
                "benchmark_id": report.benchmark_id,
                "summary": {
                    "total_duration": report.total_duration,
                    "succeeded": report.models_succeeded,
                    "failed": report.models_failed,
                    "ranking": report.ranking,
                    "best_model": report.best_model,
                    "fastest_model": report.fastest_model,
                    "lightest_model": report.lightest_model,
                },
                "no_reference_message": report.no_reference_message,
            })
            return report

        finally:
            self._running = False

    # ------------------------------------------------------- internals
    async def _run_single(
        self,
        cfg: ModelConfig,
        audio_path: Path,
        warmup_path: Optional[Path],
        reference_text: Optional[str],
        artifact: AudioArtifact,
    ) -> ModelResult:
        """اجرای یک مدل در Worker و تبدیل خروجی به ModelResult."""
        result = ModelResult(
            model_id=cfg.id,
            display_name=cfg.display_name,
            runtime=cfg.runtime,
            order=cfg.order,
            device=cfg.resolve_device(),
            compute_type=cfg.effective_compute_type(),
            runs=self.settings.benchmark.runs_per_model,
        )

        request = WorkerRequest(
            model_payload=cfg.to_worker_payload(),
            audio_path=str(audio_path),
            warmup_path=str(warmup_path) if warmup_path else None,
            runs=self.settings.benchmark.runs_per_model,
            ram_sample_interval=self.settings.benchmark.ram_sample_interval,
        )

        loop = asyncio.get_running_loop()
        wr: WorkerResult = await loop.run_in_executor(
            self._executor, self._worker.run, request)

        result.worker_pid = wr.worker_pid
        result.load_time = wr.load_time
        result.warmup_time = wr.warmup_time
        result.total_time = wr.total_time

        usage = wr.resource_usage or {}
        result.peak_ram_mb = usage.get("peak_rss_mb")
        result.model_ram_mb = usage.get("model_rss_mb")
        result.cpu_percent_avg = usage.get("cpu_percent_avg")
        baseline_ram = usage.get("baseline_rss_mb")

        if not wr.success or not wr.runs:
            result.success = False
            result.error = wr.error or "مدل خروجی معتبری تولید نکرد."
            result.error_type = wr.error_type
            logger.warning("مدل «%s» ناموفق: %s", cfg.id, result.error)
            return result

        # --- محاسبهٔ سنجه‌ها برای هر نوبت (بند ۱۴: یک خط‌کش واحد) ---
        per_run: List[TranscriptionMetrics] = []
        for run in wr.runs:
            per_run.append(self.metrics.evaluate(
                hypothesis=run.get("text", ""),
                reference=reference_text,
                audio_duration=run.get("audio_duration") or artifact.duration,
                load_time=wr.load_time,
                warmup_time=wr.warmup_time,
                inference_time=run.get("inference_time", 0.0),
                total_time=wr.total_time,
                peak_ram_mb=result.peak_ram_mb,
                baseline_ram_mb=baseline_ram,
                cpu_percent_avg=result.cpu_percent_avg,
                metadata={
                    "run_index": run.get("run_index", 0),
                    "language": run.get("language"),
                    "language_probability": run.get("language_probability"),
                },
            ))

        primary = per_run[0]
        aggregated: AggregatedMetrics = self.metrics.aggregate(per_run)

        result.success = True
        result.text = primary.hypothesis_raw
        result.text_normalized = primary.hypothesis_normalized
        result.metrics = primary.to_dict()
        result.aggregated = aggregated.to_dict()
        result.inference_time = (aggregated.inference_time_mean
                                 or primary.inference_time)
        result.rtf = aggregated.rtf_mean if aggregated.rtf_mean is not None else primary.rtf
        return result

    # ------------------------------------------------------------------
    def _select_models(self, model_ids: Optional[List[str]]) -> List[ModelConfig]:
        """انتخاب مدل‌های قابل اجرا؛ ترتیب ثابت برای تکرارپذیری."""
        configs = load_model_configs(enabled_only=True)
        if model_ids:
            wanted = set(model_ids)
            configs = [c for c in configs if c.id in wanted]

        runnable: List[ModelConfig] = []
        for c in configs:
            if c.runtime == "dummy" or c.exists():
                runnable.append(c)
            else:
                logger.warning("مدل «%s» روی دیسک موجود نیست → نادیده گرفته شد", c.id)
        return runnable

    def _ensure_audio_on_disk(self, artifact: AudioArtifact) -> Path:
        """
        نوشتن یک فایل واحد که *همهٔ* مدل‌ها می‌خوانند (بند ۱۰).
        هیچ مدلی نسخهٔ اختصاصی یا پیش‌پردازش‌شده دریافت نمی‌کند.
        """
        if artifact.path and Path(artifact.path).exists():
            return Path(artifact.path)
        tmp_dir = self.settings.paths.temp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        name = f"{artifact.utterance_id or artifact.sha256[:12]}.wav"
        path = write_wav(tmp_dir / name, artifact.pcm,
                         artifact.sample_rate, artifact.channels)
        artifact.path = path
        return path

    def _prepare_warmup(self) -> Optional[Path]:
        """ساخت یک‌بارهٔ فایل گرم‌کننده مشترک بین همهٔ مدل‌ها (بند ۱۳)."""
        if not self.settings.benchmark.warmup_enabled:
            return None
        if self._warmup_path and self._warmup_path.exists():
            return self._warmup_path
        d = self.settings.paths.temp_dir
        d.mkdir(parents=True, exist_ok=True)
        self._warmup_path = write_wav(
            d / "_warmup.wav",
            make_warmup_tone(self.settings.benchmark.warmup_seconds),
            self.settings.audio.sample_rate,
        )
        return self._warmup_path

    def _finalize(self, report: BenchmarkReport) -> None:
        """رتبه‌بندی و تعیین برندگان."""
        ok = [r for r in report.results if r.get("success")]
        if not ok:
            return

        # --- رتبه بر اساس دقت (فقط وقتی متن مرجع وجود دارد) ---
        if report.has_reference:
            def acc_key(r: Dict[str, Any]):
                agg = r.get("aggregated") or {}
                w = agg.get("wer_mean")
                rtf = agg.get("rtf_mean")
                return (w is None, w if w is not None else 9e9,
                        rtf if rtf is not None else 9e9)
            ranked = sorted(ok, key=acc_key)
            report.ranking = [r["model_id"] for r in ranked]
            report.best_model = ranked[0]["model_id"]
            for i, r in enumerate(ranked, start=1):
                for orig in report.results:
                    if orig["model_id"] == r["model_id"]:
                        orig["rank"] = i
        else:
            report.ranking = [r["model_id"] for r in ok]

        fastest = min((r for r in ok if r.get("rtf") is not None),
                      key=lambda r: r["rtf"], default=None)
        lightest = min((r for r in ok if r.get("peak_ram_mb") is not None),
                       key=lambda r: r["peak_ram_mb"], default=None)
        report.fastest_model = fastest["model_id"] if fastest else None
        report.lightest_model = lightest["model_id"] if lightest else None

    def _settings_snapshot(self) -> Dict[str, Any]:
        """تنظیمات مؤثر در نتیجه — برای تکرارپذیری (بند ۲۱)."""
        s = self.settings
        return {
            "runs_per_model": s.benchmark.runs_per_model,
            "warmup_enabled": s.benchmark.warmup_enabled,
            "model_timeout": s.benchmark.model_timeout,
            "mp_context": s.benchmark.mp_context,
            "ram_sample_interval": s.benchmark.ram_sample_interval,
            "vad_engine": s.vad.engine,
            "silence_duration": s.vad.silence_duration,
            "pre_roll_seconds": s.audio.pre_roll_seconds,
            "sample_rate": s.audio.sample_rate,
        }

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_progress:
            try:
                await self.on_progress({"event": event, **payload})
            except Exception as exc:  # noqa: BLE001
                logger.debug("ارسال رویداد «%s» ناموفق: %s", event, exc)

    def close(self) -> None:
        self._executor.shutdown(wait=False)