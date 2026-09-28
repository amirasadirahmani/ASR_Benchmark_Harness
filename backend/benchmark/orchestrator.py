"""اجرای ترتیبی و ایزولهٔ Benchmark روی یک AudioArtifact مشترک."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.asr.registry import validate_registry
from backend.audio.audio_utils import AudioArtifact, make_warmup_tone, write_wav
from backend.benchmark.ram_monitor import get_system_memory
from backend.benchmark.worker import ModelWorker, WorkerRequest, WorkerResult
from backend.config.model_config import ModelConfig, describe_environment, load_model_configs
from backend.config.settings import PROJECT_ROOT, AppSettings, get_settings
from backend.core.metrics import AggregatedMetrics, MetricsEngine, TranscriptionMetrics

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]

_GLOBAL_RUN_COUNTER = itertools.count(1)


@dataclass
class ModelResult:
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
    benchmark_id: str
    session_id: str = ""
    utterance_id: str = ""
    created_at: str = ""
    audio: Dict[str, Any] = field(default_factory=dict)
    reference_text: Optional[str] = None
    has_reference: bool = False
    no_reference_message: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    model_order: List[str] = field(default_factory=list)
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
    model_versions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkOrchestrator:
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark")
        self._cancel = asyncio.Event()
        self._running = False
        self._warmup_path: Optional[Path] = None
        self._owned_audio_path: Optional[Path] = None
        self._run_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def cancel(self) -> None:
        self._cancel.set()

    def list_models(self) -> List[ModelConfig]:
        return load_model_configs(enabled_only=True)

    def preflight(self) -> Dict[str, Any]:
        configs = load_model_configs()
        validation = validate_registry(configs)
        return {
            "models": [c.to_public_dict() for c in configs],
            "validation": validation,
            "ready": len(validation["ok"]) > 0,
            "environment": describe_environment(),
            "system_memory": get_system_memory(),
        }

    async def run(
        self,
        artifact: AudioArtifact,
        reference_text: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        self._running = True
        self._cancel.clear()
        self._run_count = next(_GLOBAL_RUN_COUNTER)
        started = time.perf_counter()

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
            report.no_reference_message = "جمله مبدا برای مقایسه و اعلام نتیجه وجود نداشت."

        try:
            audio_path = self._ensure_audio_on_disk(artifact)
            configs = self._select_models(model_ids)
            report.models_total = len(configs)
            report.model_order = [c.id for c in configs]
            report.model_versions = self._model_manifest_snapshot(report.model_order)

            if not configs:
                await self._emit("benchmark_error", {
                    "benchmark_id": report.benchmark_id,
                    "message": "هیچ مدل فعال و موجودی برای اجرا یافت نشد.",
                })
                return report

            await self._emit("benchmark_started", {
                "benchmark_id": report.benchmark_id,
                "models": [c.to_public_dict() for c in configs],
                "audio": {"duration": round(artifact.duration, 3), "sha256": artifact.sha256},
                "has_reference": report.has_reference,
                "runs_per_model": self.settings.benchmark.runs_per_model,
            })

            warmup = self._prepare_warmup()

            for index, cfg in enumerate(configs, start=1):
                if self._cancel.is_set():
                    await self._emit("benchmark_cancelled", {
                        "benchmark_id": report.benchmark_id,
                        "completed": index - 1,
                        "total": len(configs),
                    })
                    break

                await self._emit("model_started", {
                    "benchmark_id": report.benchmark_id,
                    "model_id": cfg.id,
                    "display_name": cfg.display_name,
                    "index": index,
                    "total": len(configs),
                })

                result = await self._run_single(
                    cfg, audio_path, warmup, reference_text, artifact
                )
                report.results.append(result.to_dict())
                if result.success:
                    report.models_succeeded += 1
                else:
                    report.models_failed += 1

                await self._emit("model_result", {
                    "benchmark_id": report.benchmark_id,
                    "index": index,
                    "total": len(configs),
                    "result": result.to_dict(),
                })

            self._finalize(report)
            report.total_duration = round(time.perf_counter() - started, 3)

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

    async def _run_single(
        self,
        cfg: ModelConfig,
        audio_path: Path,
        warmup_path: Optional[Path],
        reference_text: Optional[str],
        artifact: AudioArtifact,
    ) -> ModelResult:
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
            self._executor, self._worker.run, request
        )

        result.worker_pid = wr.worker_pid
        result.load_time = wr.load_time
        result.warmup_time = wr.warmup_time
        result.total_time = wr.total_time

        actual_device = (wr.adapter_info or {}).get("device")
        if actual_device:
            result.device = actual_device

        usage = wr.resource_usage or {}
        result.peak_ram_mb = usage.get("peak_rss_mb")
        result.model_ram_mb = usage.get("model_rss_mb")
        result.cpu_percent_avg = usage.get("cpu_percent_avg")
        baseline_ram = usage.get("baseline_rss_mb")

        if not wr.success or not wr.runs:
            result.error = wr.error or "مدل خروجی معتبری تولید نکرد."
            result.error_type = wr.error_type
            return result

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
        result.inference_time = aggregated.inference_time_mean or primary.inference_time
        result.rtf = aggregated.rtf_mean if aggregated.rtf_mean is not None else primary.rtf
        return result

    def _select_models(self, model_ids: Optional[List[str]]) -> List[ModelConfig]:
        configs = load_model_configs(enabled_only=True)
        if model_ids:
            wanted = set(model_ids)
            configs = [c for c in configs if c.id in wanted]

        runnable: List[ModelConfig] = []
        for config in configs:
            if config.runtime == "dummy" or config.exists():
                runnable.append(config)
            else:
                logger.warning("مدل «%s» روی دیسک موجود نیست → نادیده گرفته شد", config.id)

        if self.settings.benchmark.rotate_model_order and len(runnable) > 1:
            if self.settings.benchmark.rotation_strategy == "shuffle":
                random.shuffle(runnable)
            else:
                offset = self._run_count % len(runnable)
                runnable = runnable[offset:] + runnable[:offset]

        return runnable

    def _ensure_audio_on_disk(self, artifact: AudioArtifact) -> Path:
        if artifact.path and Path(artifact.path).exists():
            return Path(artifact.path)

        directory = self.settings.paths.temp_path
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{artifact.utterance_id or artifact.sha256[:12]}.wav"
        path = write_wav(
            directory / name,
            artifact.pcm,
            artifact.sample_rate,
            artifact.channels,
        )
        artifact.path = path
        self._owned_audio_path = path
        return path

    def _prepare_warmup(self) -> Optional[Path]:
        if not self.settings.benchmark.warmup_enabled:
            return None
        if self._warmup_path and self._warmup_path.exists():
            return self._warmup_path

        directory = self.settings.paths.temp_path
        directory.mkdir(parents=True, exist_ok=True)
        self._warmup_path = write_wav(
            directory / "_warmup.wav",
            make_warmup_tone(self.settings.benchmark.warmup_seconds),
            self.settings.audio.sample_rate,
        )
        return self._warmup_path

    def _finalize(self, report: BenchmarkReport) -> None:
        ok = [r for r in report.results if r.get("success")]
        if not ok:
            return

        if report.has_reference:
            def accuracy_key(item: Dict[str, Any]):
                aggregate = item.get("aggregated") or {}
                wer = aggregate.get("wer_mean")
                rtf = aggregate.get("rtf_mean")
                return (
                    wer is None,
                    wer if wer is not None else 9e9,
                    rtf if rtf is not None else 9e9,
                )

            ranked = sorted(ok, key=accuracy_key)
            report.ranking = [r["model_id"] for r in ranked]
            report.best_model = ranked[0]["model_id"]
            for rank, item in enumerate(ranked, start=1):
                for original in report.results:
                    if original["model_id"] == item["model_id"]:
                        original["rank"] = rank
        else:
            report.ranking = [r["model_id"] for r in ok]

        fastest = min(
            (r for r in ok if r.get("rtf") is not None),
            key=lambda r: r["rtf"],
            default=None,
        )
        lightest = min(
            (r for r in ok if r.get("peak_ram_mb") is not None),
            key=lambda r: r["peak_ram_mb"],
            default=None,
        )
        report.fastest_model = fastest["model_id"] if fastest else None
        report.lightest_model = lightest["model_id"] if lightest else None

    def _settings_snapshot(self) -> Dict[str, Any]:
        settings = self.settings
        return {
            "runs_per_model": settings.benchmark.runs_per_model,
            "warmup_enabled": settings.benchmark.warmup_enabled,
            "rotate_model_order": settings.benchmark.rotate_model_order,
            "rotation_strategy": settings.benchmark.rotation_strategy,
            "model_timeout": settings.benchmark.model_timeout,
            "mp_context": settings.benchmark.mp_context,
            "ram_sample_interval": settings.benchmark.ram_sample_interval,
            "vad_engine": settings.vad.engine,
            "silence_duration": settings.vad.silence_duration,
            "pre_roll_seconds": settings.audio.pre_roll_seconds,
            "sample_rate": settings.audio.sample_rate,
        }

    def _model_manifest_snapshot(self, model_ids: List[str]) -> Dict[str, Any]:
        manifest_path = PROJECT_ROOT / "model_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        models_used = {
            model_id: manifest.get("models", {}).get(model_id)
            for model_id in model_ids
            if manifest.get("models", {}).get(model_id)
        }
        snapshot: Dict[str, Any] = {"models": models_used}
        if self.settings.wake_word.engine == "vosk" and manifest.get("wakeword_vosk"):
            snapshot["wakeword_vosk"] = manifest["wakeword_vosk"]
        if self.settings.vad.engine == "silero" and manifest.get("vad_silero"):
            snapshot["vad_silero"] = manifest["vad_silero"]
        return snapshot

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_progress:
            try:
                await self.on_progress({"event": event, **payload})
            except Exception as exc:
                logger.debug("ارسال رویداد «%s» ناموفق: %s", event, exc)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        for path in (self._warmup_path, self._owned_audio_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug("حذف فایل موقت %s ناموفق: %s", path, exc)
