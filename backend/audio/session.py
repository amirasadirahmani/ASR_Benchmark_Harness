"""
مدیریت نشست صوتی — ماشین حالت مرکزی خط لولهٔ صدا.

جریان کامل (بندهای ۷ تا ۱۰):

    IDLE ──start()──► WAITING_WAKE ──[«آرینا»]──► LISTENING
                            ▲                          │
                            │                     [۲ ثانیه سکوت]
                            └──────────────────── FINALIZING
                                                       │
                                                       ▼
                                              AudioArtifact (sha256)
                                                       │
                                                       ▼
                                              BenchmarkOrchestrator

نکتهٔ کلیدی: در لحظهٔ trigger، محتوای PreRollBuffer به ابتدای ضبط الصاق
می‌شود تا هجاهای ابتدایی فرمان از دست نرود.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.audio.audio_utils import (
    AudioArtifact, SAMPLE_WIDTH, pcm16_duration, rms_dbfs, trim_pcm,
)
from backend.audio.ring_buffer import FrameAccumulator, PreRollBuffer
from backend.audio.vad import VADEvent, VADEventType, VADGate, create_vad
from backend.audio.wake_word import (
    BaseWakeWordDetector, ManualWakeWordDetector, WakeWordEvent,
    create_wake_word_detector,
)
from backend.config.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    IDLE = "idle"
    WAITING_WAKE = "waiting_wake"       # منتظر «آرینا»
    LISTENING = "listening"             # در حال ضبط فرمان
    FINALIZING = "finalizing"           # آماده‌سازی artifact
    PROCESSING = "processing"           # در اختیار Orchestrator
    ERROR = "error"


@dataclass
class SessionStats:
    frames_received: int = 0
    bytes_received: int = 0
    utterances: int = 0
    wake_events: int = 0
    timeouts: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_received": self.frames_received,
            "bytes_received": self.bytes_received,
            "audio_seconds": round(
                self.bytes_received / (16000 * SAMPLE_WIDTH), 2),
            "utterances": self.utterances,
            "wake_events": self.wake_events,
            "timeouts": self.timeouts,
            "uptime": round(time.time() - self.started_at, 1),
        }


EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]
UtteranceCallback = Callable[[AudioArtifact], Awaitable[None]]


class AudioSession:
    """
    یک نشست صوتی متناظر با یک اتصال WebSocket.

    مصرف‌کننده فقط `await feed(chunk)` را صدا می‌زند؛ همهٔ منطق
    Wake Word، VAD، Pre-Roll و نهایی‌سازی داخل همین کلاس است.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        settings: Optional[AppSettings] = None,
        *,
        on_event: Optional[EventCallback] = None,
        on_utterance: Optional[UtteranceCallback] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.on_event = on_event
        self.on_utterance = on_utterance

        audio = self.settings.audio
        self.sample_rate = audio.sample_rate
        self.frame_bytes = audio.frame_bytes

        # --- اجزا ---
        self._accumulator = FrameAccumulator(self.frame_bytes)
        self._pre_roll = PreRollBuffer(
            audio.pre_roll_seconds, audio.sample_rate, audio.channels)

        self._wake: BaseWakeWordDetector = create_wake_word_detector(
            self.settings.wake_word, self.sample_rate)

        self._vad_gate = VADGate(
            create_vad(self.settings.vad, self.sample_rate, audio.frame_ms),
            frame_ms=audio.frame_ms,
            silence_duration=self.settings.vad.silence_duration,
            speech_start_frames=self.settings.vad.speech_start_frames,
            max_duration=audio.max_command_seconds,
            speech_timeout=self.settings.vad.post_wake_speech_timeout,
        )

        # --- حالت ---
        self.state = SessionState.IDLE
        self.stats = SessionStats()
        self._recording = bytearray()
        self._pre_roll_used = 0.0
        self._utterance_id = ""
        self._wake_time = 0.0
        self._listen_start = 0.0
        self._last_level = -96.0

    # ------------------------------------------------------------------ API
    async def start(self) -> None:
        """آغاز گوش دادن (منتظر Wake Word)."""
        self.state = (
            SessionState.WAITING_WAKE
            if self.settings.wake_word.enabled
            else SessionState.LISTENING
        )
        if self.state == SessionState.LISTENING:
            self._begin_listening(from_wake=False)
        await self._emit("session_started", {
            "state": self.state.value,
            "wake_word": self.settings.wake_word.word,
            "wake_engine": self._wake.name,
            "wake_enabled": self.settings.wake_word.enabled,
            "vad_engine": self._vad_gate.vad.name,
            "silence_duration": self.settings.vad.silence_duration,
            "pre_roll_seconds": self.settings.audio.pre_roll_seconds,
        })

    async def stop(self) -> None:
        """توقف نشست؛ اگر در حال ضبط بودیم، همان را نهایی می‌کند."""
        if self.state == SessionState.LISTENING and self._has_enough_audio():
            await self._finalize(reason="manual_stop")
        self.state = SessionState.IDLE
        await self._emit("session_stopped", self.stats.to_dict())

    async def feed(self, chunk: bytes) -> None:
        """ورود داده از WebSocket (اندازهٔ دلخواه)."""
        if not chunk or self.state in (SessionState.IDLE, SessionState.ERROR):
            return
        self.stats.bytes_received += len(chunk)
        for frame in self._accumulator.push(chunk):
            self.stats.frames_received += 1
            await self._process_frame(frame)

    async def trigger_wake(self) -> None:
        """فعال‌سازی دستی از UI (دکمهٔ «شروع ضبط»)."""
        if self.state != SessionState.WAITING_WAKE:
            return
        ev = (self._wake.trigger()
              if isinstance(self._wake, ManualWakeWordDetector)
              else WakeWordEvent(True, self.settings.wake_word.word, 1.0, "manual"))
        await self._on_wake(ev)

    async def cancel_utterance(self) -> None:
        """لغو ضبط جاری و بازگشت به انتظار Wake Word."""
        self._reset_recording()
        self.state = (SessionState.WAITING_WAKE
                      if self.settings.wake_word.enabled else SessionState.LISTENING)
        if self.state == SessionState.LISTENING:
            self._begin_listening(from_wake=False)
        await self._emit("utterance_cancelled", {"state": self.state.value})

    async def resume(self) -> None:
        """بازگشت به گوش دادن پس از پایان Benchmark."""
        self._reset_recording()
        self.state = (SessionState.WAITING_WAKE
                      if self.settings.wake_word.enabled else SessionState.LISTENING)
        if self.state == SessionState.LISTENING:
            self._begin_listening(from_wake=False)
        await self._emit("listening_resumed", {"state": self.state.value})

    # ------------------------------------------------------- internals
    async def _process_frame(self, frame: bytes) -> None:
        self._last_level = rms_dbfs(frame)

        if self.state == SessionState.WAITING_WAKE:
            self._pre_roll.write(frame)          # ← همیشه در حال ضبط پس‌زمینه
            ev = self._wake.push(frame)
            if ev and ev.detected:
                await self._on_wake(ev)
            elif self.stats.frames_received % 10 == 0:
                await self._emit("level", {"dbfs": round(self._last_level, 1),
                                           "state": self.state.value})
            return

        if self.state == SessionState.LISTENING:
            self._recording.extend(frame)
            vad_event = self._vad_gate.push(frame)
            await self._handle_vad(vad_event)
            return

        # PROCESSING / FINALIZING → صدا دور ریخته می‌شود (عمدی)

    async def _on_wake(self, ev: WakeWordEvent) -> None:
        self.stats.wake_events += 1
        self._wake_time = time.time()
        self._begin_listening(from_wake=True)
        await self._emit("wake_detected", {
            **ev.to_dict(),
            "pre_roll_seconds": round(self._pre_roll_used, 3),
            "state": self.state.value,
        })

    def _begin_listening(self, from_wake: bool) -> None:
        """شروع ضبط فرمان + الصاق Pre-Roll (بند ۸)."""
        self._recording = bytearray()
        self._pre_roll_used = 0.0

        if from_wake:
            pre = self._pre_roll.snapshot()
            if pre:
                self._recording.extend(pre)
                self._pre_roll_used = pcm16_duration(pre, self.sample_rate)

        self._pre_roll.clear()
        self._vad_gate.reset()
        self._utterance_id = f"{self.session_id}-{uuid.uuid4().hex[:8]}"
        self._listen_start = time.time()
        self.state = SessionState.LISTENING

    async def _handle_vad(self, ev: VADEvent) -> None:
        if ev.type == VADEventType.SPEECH_START:
            await self._emit("speech_start", ev.to_dict())

        elif ev.type == VADEventType.SILENCE and self._vad_gate.speech_started:
            remaining = max(0.0, self.settings.vad.silence_duration - ev.silence_duration)
            await self._emit("silence", {**ev.to_dict(),
                                         "countdown": round(remaining, 2)})

        elif ev.type == VADEventType.SPEECH_CONTINUE:
            if self.stats.frames_received % 5 == 0:
                await self._emit("speech", {"level_dbfs": round(ev.level_dbfs, 1),
                                            "duration": round(ev.speech_duration, 2)})

        elif ev.type == VADEventType.TIMEOUT:
            self.stats.timeouts += 1
            await self._emit("wake_timeout", {
                **ev.to_dict(),
                "message": "پس از کلمهٔ بیدارباش، گفتاری دریافت نشد.",
            })
            await self.resume()

        elif ev.type in (VADEventType.UTTERANCE_END, VADEventType.MAX_DURATION):
            await self._finalize(reason=ev.type.value, vad_event=ev)

    # ------------------------------------------------------------------
    def _has_enough_audio(self) -> bool:
        duration = pcm16_duration(bytes(self._recording), self.sample_rate)
        return duration >= self.settings.audio.min_command_seconds

    async def _finalize(self, reason: str,
                        vad_event: Optional[VADEvent] = None) -> None:
        """ساخت AudioArtifact نهایی و تحویل به Orchestrator."""
        self.state = SessionState.FINALIZING
        pcm = bytes(self._recording)

        # حذف دُم سکوت (فقط انتها؛ ابتدا به‌خاطر Pre-Roll دست‌نخورده می‌ماند)
        if vad_event and vad_event.silence_duration > 0:
            keep = max(0.0, vad_event.silence_duration - 0.3)
            drop = int(keep * self.sample_rate) * SAMPLE_WIDTH
            if 0 < drop < len(pcm):
                pcm = pcm[:-drop]

        pcm = trim_pcm(pcm, self.settings.audio.max_command_seconds, self.sample_rate)
        duration = pcm16_duration(pcm, self.sample_rate)

        if duration < self.settings.audio.min_command_seconds:
            await self._emit("utterance_too_short", {
                "duration": round(duration, 3),
                "min_required": self.settings.audio.min_command_seconds,
                "message": "گفتار معتبری تشخیص داده نشد.",
            })
            await self.resume()
            return

        artifact = AudioArtifact(
            pcm=pcm,
            sample_rate=self.sample_rate,
            channels=self.settings.audio.channels,
            session_id=self.session_id,
            utterance_id=self._utterance_id,
            pre_roll_seconds=self._pre_roll_used,
            metadata={
                "reason": reason,
                "wake_engine": self._wake.name,
                "vad_engine": self._vad_gate.vad.name,
                "speech_duration": round(self._vad_gate.speech_duration, 3),
                "listen_latency": round(time.time() - self._listen_start, 3),
            },
        )

        if self.settings.audio.save_recordings:
            try:
                artifact.save(self.settings.paths.recordings_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ذخیرهٔ فایل صوتی ناموفق بود: %s", exc)

        self.stats.utterances += 1
        self.state = SessionState.PROCESSING

        await self._emit("utterance_ready", artifact.to_dict())
        if self.on_utterance:
            await self.on_utterance(artifact)

    def _reset_recording(self) -> None:
        self._recording = bytearray()
        self._pre_roll.clear()
        self._pre_roll_used = 0.0
        self._accumulator.reset()
        self._vad_gate.reset()
        self._wake.reset()

    async def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                await self.on_event({"event": event_type,
                                     "session_id": self.session_id, **payload})
            except Exception as exc:  # noqa: BLE001
                logger.debug("ارسال رویداد «%s» ناموفق بود: %s", event_type, exc)

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "level_dbfs": round(self._last_level, 1),
            "recording_seconds": round(
                pcm16_duration(bytes(self._recording), self.sample_rate), 2),
            "pre_roll_seconds": round(self._pre_roll_used, 3),
            "wake": self._wake.describe(),
            "vad": self._vad_gate.status(),
            "stats": self.stats.to_dict(),
        }

    def close(self) -> None:
        self._wake.close()
        self._reset_recording()
        self.state = SessionState.IDLE


class SessionManager:
    """رجیستری نشست‌های فعال (چند تب مرورگر هم‌زمان)."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self.settings = settings or get_settings()
        self._sessions: Dict[str, AudioSession] = {}

    def create(self, **kwargs: Any) -> AudioSession:
        s = AudioSession(settings=self.settings, **kwargs)
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[AudioSession]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        s = self._sessions.pop(session_id, None)
        if s:
            s.close()

    def close_all(self) -> None:
        for s in list(self._sessions.values()):
            s.close()
        self._sessions.clear()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def status(self) -> List[Dict[str, Any]]:
        return [s.status() for s in self._sessions.values()]