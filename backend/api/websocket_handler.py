"""
هندلر اتصال WebSocket — پل بین UI، AudioSession و BenchmarkOrchestrator.

جریان کار یک اتصال:
    1. کلاینت وصل می‌شود → AudioSession ساخته می‌شود → session_started ارسال می‌شود
    2. فریم‌های باینری → session.feed(pcm_bytes)
    3. پیام‌های متنی → پارس با schemas.parse_client_message → اعمال روی session
    4. AudioSession رویدادها را (wakeword_detected, recording_started, ...) از طریق
       callback همزمان (sync) صادر می‌کند → صف asyncio → ارسال ترتیبی به کلاینت
    5. وقتی یک گفتار کامل ضبط شد → on_utterance فراخوانی می‌شود →
       BenchmarkOrchestrator.run() در پس‌زمینه اجرا می‌شود → نتیجه ذخیره و پخش می‌شود

⚠️ چرا صف (Queue) به‌جای send مستقیم از callback؟
    AudioSession ممکن است رویدادها را از یک ترد/کانتکست غیر-async صادر کند
    (مثلاً از callback کتابخانهٔ VAD). WebSocket.send_json فقط در event loop
    اصلی امن است. صف، این دو دنیا را به‌درستی جدا می‌کند.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.api.schemas import (
    BenchmarkCompletedEvent,
    BenchmarkStartedEvent,
    ConfigOverrideMsg,
    ErrorEvent,
    ModelCompletedEvent,
    ModelStartedEvent,
    PongEvent,
    SelectModelsMsg,
    SessionEndedEvent,
    SetReferenceMsg,
    StopSessionMsg,
    TriggerWakeMsg,
    parse_client_message,
)
from backend.audio.session import AudioSession
from backend.benchmark.orchestrator import BenchmarkOrchestrator
from backend.config.settings import AppSettings
from backend.storage import ResultsStore

logger = logging.getLogger(__name__)
_MAX_QUEUE_SIZE = 200


class ConnectionHandler:
    """مدیریت چرخهٔ حیات یک اتصال WebSocket."""

    def __init__(
        self,
        websocket: WebSocket,
        settings: AppSettings,
        store: ResultsStore,
    ) -> None:
        self.ws = websocket
        self.settings = settings
        self.store = store
        self._loop = asyncio.get_event_loop()
        self._out_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(
            maxsize=_MAX_QUEUE_SIZE
        )
        self._reference_text: Optional[str] = None
        self._selected_model_ids: Optional[list[str]] = None
        self._current_artifact: Optional[Any] = None
        self._closed = False
        self.session = AudioSession(
            settings=settings,
            on_event=self._on_session_event,
            on_utterance=self._on_utterance,
        )
        self._bg_tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        await self.ws.accept()
        sender_task = asyncio.create_task(self._sender_loop())
        self._track(sender_task)

        try:
            await self.session.start()
        except Exception as exc:
            logger.exception("راه‌اندازی AudioSession ناموفق")
            await self._emit(ErrorEvent(
                error_type="session_start_failed",
                message=str(exc),
                recoverable=False,
            ))
            await self._shutdown("session_start_failed")
            return

        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("کلاینت قطع شد: session=%s", getattr(self.session, "session_id", "?"))
        except Exception:
            logger.exception("خطای غیرمنتظره در حلقهٔ دریافت")
        finally:
            await self._shutdown("disconnect")
            sender_task.cancel()

    async def _receive_loop(self) -> None:
        while not self._closed:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if (data := message.get("bytes")) is not None:
                await self._handle_audio_frame(data)
            elif (text := message.get("text")) is not None:
                await self._handle_text_message(text)

    async def _handle_audio_frame(self, data: bytes) -> None:
        try:
            await self.session.feed(data)
        except Exception as exc:
            logger.exception("خطا در feed صوت")
            await self._emit(ErrorEvent(
                error_type="audio_feed_failed",
                message=str(exc),
                recoverable=True,
            ))

    async def _handle_text_message(self, text: str) -> None:
        import json

        try:
            raw = json.loads(text)
            msg = parse_client_message(raw)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            await self._emit(ErrorEvent(
                error_type="invalid_message",
                message=str(exc),
                recoverable=True,
            ))
            return

        if isinstance(msg, SetReferenceMsg):
            self._reference_text = msg.text
            self.session.set_reference(msg.text)
        elif isinstance(msg, TriggerWakeMsg):
            await self.session.trigger_wake()
        elif isinstance(msg, StopSessionMsg):
            await self._shutdown("client_request")
        elif isinstance(msg, SelectModelsMsg):
            self._selected_model_ids = msg.model_ids or None
        elif isinstance(msg, ConfigOverrideMsg):
            self.session.apply_config_override(
                vad_silence_duration=msg.vad_silence_duration,
                wakeword_enabled=msg.wakeword_enabled,
            )
        else:
            await self._emit(PongEvent())

    async def _on_session_event(self, event: Any) -> None:
        payload = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        if "type" not in payload and "event" in payload:
            payload["type"] = payload["event"]
        try:
            self._loop.call_soon_threadsafe(self._enqueue_nowait, payload)
        except RuntimeError:
            pass

    def _enqueue_nowait(self, payload: Dict[str, Any]) -> None:
        try:
            self._out_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("صف خروجی پر شد — قدیمی‌ترین رویداد دور ریخته می‌شود")
            try:
                self._out_queue.get_nowait()
                self._out_queue.put_nowait(payload)
            except asyncio.QueueEmpty:
                pass

    async def _emit(self, event: Any) -> None:
        payload = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        await self._enqueue_async(payload)

    async def _enqueue_async(self, payload: Dict[str, Any]) -> None:
        try:
            self._out_queue.put_nowait(payload)
        except asyncio.QueueFull:
            await self._out_queue.get()
            await self._out_queue.put(payload)

    async def _sender_loop(self) -> None:
        try:
            while True:
                payload = await self._out_queue.get()
                await self.ws.send_json(payload)
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            logger.debug("اتصال WebSocket عادی قطع شد (sender_loop)")
        except Exception:
            logger.exception("خطا در sender_loop")

    async def _on_utterance(self, artifact: Any) -> None:
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._run_benchmark(artifact), self._loop
            )
            fut.add_done_callback(self._log_task_exception)
        except RuntimeError:
            pass

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "خطا در اجرای Benchmark پس‌زمینه",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _on_orchestrator_progress(self, payload: Dict[str, Any]) -> None:
        event = payload.get("event")

        if event == "benchmark_started":
            models = payload.get("models") or []
            await self._emit(BenchmarkStartedEvent(
                benchmark_id=payload.get("benchmark_id", ""),
                utterance_id=getattr(self._current_artifact, "utterance_id", "") or "",
                models_total=len(models),
                model_ids=[m.get("id") for m in models if m.get("id")],
            ))
            return

        if event == "model_started":
            await self._emit(ModelStartedEvent(
                benchmark_id=payload.get("benchmark_id", ""),
                model_id=payload.get("model_id", ""),
                display_name=payload.get("display_name", ""),
                index=payload.get("index", 0),
                total=payload.get("total", 0),
            ))
            return

        if event == "model_result":
            result = payload.get("result") or {}
            metrics = result.get("metrics") or {}
            await self._emit(ModelCompletedEvent(
                benchmark_id=payload.get("benchmark_id", ""),
                model_id=result.get("model_id", ""),
                display_name=result.get("display_name", ""),
                success=bool(result.get("success")),
                error_type=result.get("error_type"),
                error=result.get("error"),
                text=result.get("text"),
                wer=metrics.get("wer"),
                cer=metrics.get("cer"),
                rtf=result.get("rtf"),
                load_time=result.get("load_time"),
                inference_time=result.get("inference_time"),
                peak_ram_mb=result.get("peak_ram_mb"),
            ))
            return

        if event == "benchmark_error":
            await self._emit(ErrorEvent(
                error_type="benchmark_error",
                message=payload.get("message", "خطای نامشخص در Benchmark"),
                recoverable=True,
            ))
            return

        if event == "benchmark_completed":
            return

        await self._enqueue_async({**payload, "type": event or "progress"})

    async def _run_benchmark(self, artifact: Any) -> None:
        self._current_artifact = artifact
        orchestrator = BenchmarkOrchestrator(
            settings=self.settings, on_progress=self._on_orchestrator_progress,
        )
        try:
            report = await orchestrator.run(
                artifact,
                reference_text=self._reference_text,
                model_ids=self._selected_model_ids,
            )

            paths = self.store.save(report)
            logger.info("نتیجه ذخیره شد: %s", paths.json_path)

            data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            await self._emit(BenchmarkCompletedEvent(
                benchmark_id=data.get("benchmark_id", ""),
                utterance_id=data.get("utterance_id", ""),
                total_time=data.get("total_duration", 0.0),
                models_succeeded=data.get("models_succeeded", 0),
                models_failed=data.get("models_failed", 0),
                ranking=data.get("ranking", []),
                best_model=data.get("best_model"),
                fastest_model=data.get("fastest_model"),
                lightest_model=data.get("lightest_model"),
                has_reference=data.get("has_reference", False),
                reference_text=data.get("reference_text"),
                no_reference_message=data.get("no_reference_message"),
            ))

        except Exception as exc:
            logger.exception("اجرای Benchmark با خطا مواجه شد")
            await self._emit(ErrorEvent(
                error_type="benchmark_failed",
                message=str(exc),
                recoverable=True,
            ))
        finally:
            orchestrator.close()
            if not self._closed:
                await self.session.resume()

    def _track(self, task: asyncio.Task) -> None:
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _shutdown(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.close()
        except Exception:
            logger.exception("خطا در بستن AudioSession")

        try:
            await self._emit(SessionEndedEvent(reason=reason))
            await asyncio.sleep(0.05)
        except Exception:
            pass

        for t in list(self._bg_tasks):
            t.cancel()
