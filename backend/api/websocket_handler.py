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

# حداکثر پیام‌های صف‌شده قبل از ارسال — جلوگیری از مصرف نامحدود حافظه
# اگر کلاینت کند/قطع باشد و رویدادها انباشته شوند.
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
        self._closed = False

        self.session = AudioSession(
            settings=settings,
            on_event=self._on_session_event,   # ممکن است از هر تردی صدا زده شود
            on_utterance=self._on_utterance,   # این باید async باشد یا از _spawn استفاده کند
        )
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # چرخهٔ اصلی اتصال
    # ------------------------------------------------------------------
    async def run(self) -> None:
        await self.ws.accept()
        sender_task = asyncio.create_task(self._sender_loop())
        self._track(sender_task)

        try:
            await self.session.start()
        except Exception as exc:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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

    # ------------------------------------------------------------------
    # ورودی صوت
    # ------------------------------------------------------------------
    async def _handle_audio_frame(self, data: bytes) -> None:
        try:
            await self.session.feed(data)
        except Exception as exc:  # noqa: BLE001
            logger.exception("خطا در feed صوت")
            await self._emit(ErrorEvent(
                error_type="audio_feed_failed",
                message=str(exc),
                recoverable=True,
            ))

    # ------------------------------------------------------------------
    # ورودی متنی/کنترلی
    # ------------------------------------------------------------------
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

        else:  # PingMsg
            await self._emit(PongEvent())

    # ------------------------------------------------------------------
    # callback از AudioSession — ممکن است sync/از ترد دیگر صدا زده شود
    # ------------------------------------------------------------------
    async def _on_session_event(self, event: Any) -> None:
        """
        پل بین دنیای sync (AudioSession) و asyncio (WebSocket).

        `event` می‌تواند یک دیکشنری خام یا یک شیء ServerEvent باشد؛
        هر دو حالت پشتیبانی می‌شود تا با پیاده‌سازی فعلی AudioSession سازگار باشد.
        """
        payload = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        try:
            self._loop.call_soon_threadsafe(self._enqueue_nowait, payload)
        except RuntimeError:
            # event loop بسته شده — اتصال احتمالاً در حال shutdown است
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
        """ارسال مستقیم رویداد از داخل کانتکست async (بدون رفتن به صف ترد-امن)."""
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
        except Exception:  # noqa: BLE001
            logger.exception("خطا در sender_loop")

    # ------------------------------------------------------------------
    # وقتی یک گفتار کامل ضبط شد → Benchmark اجرا کن
    # ------------------------------------------------------------------
    async def _on_utterance(self, artifact: Any) -> None:
        """
        callback از AudioSession وقتی VAD تشخیص سکوت داد و گفتار کامل شد.

        این تابع sync است (فرض بر این‌که AudioSession از ترد جدا صدا می‌زند)
        و یک تسک async روی event loop اصلی زمان‌بندی می‌کند.
        """
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

    async def _run_benchmark(self, artifact: Any) -> None:
        orchestrator = BenchmarkOrchestrator(settings=self.settings)
        try:
            async def on_progress(kind: str, payload: Dict[str, Any]) -> None:
                # اگر orchestrator این callback را پشتیبانی نکند، پایین fallback داریم
                if kind == "model_started":
                    await self._emit(ModelStartedEvent(**payload))
                elif kind == "model_completed":
                    await self._emit(ModelCompletedEvent(**payload))

            await self._emit(BenchmarkStartedEvent(
                benchmark_id=getattr(artifact, "utterance_id", "unknown"),
                utterance_id=getattr(artifact, "utterance_id", "unknown"),
                models_total=0,
                model_ids=self._selected_model_ids or [],
            ))

            try:
                report = await orchestrator.run(
                    artifact,
                    reference_text=self._reference_text,
                    model_ids=self._selected_model_ids,
                    on_progress=on_progress,
                )
            except TypeError:
                # orchestrator.run فعلاً on_progress را پشتیبانی نمی‌کند —
                # بدون رویدادهای per-model، فقط رویداد نهایی را می‌فرستیم.
                logger.info("Orchestrator.run بدون on_progress صدا زده شد (fallback)")
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

        except Exception as exc:  # noqa: BLE001
            logger.exception("اجرای Benchmark با خطا مواجه شد")
            await self._emit(ErrorEvent(
                error_type="benchmark_failed",
                message=str(exc),
                recoverable=True,
            ))
        finally:
            orchestrator.close()

    # ------------------------------------------------------------------
    def _track(self, task: asyncio.Task) -> None:
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _shutdown(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            logger.exception("خطا در بستن AudioSession")

        try:
            await self._emit(SessionEndedEvent(reason=reason))
            # زمان کوتاه برای flush شدن صف قبل از قطع واقعی سوکت
            await asyncio.sleep(0.05)
        except Exception:  # noqa: BLE001
            pass

        for t in list(self._bg_tasks):
            t.cancel()