"""
قرارداد پیام‌های WebSocket بین UI و سرور.

دو مسیر پیام روی یک اتصال:
    1. باینری  → فریم‌های صوتی خام (PCM16LE mono) → مستقیم به AudioSession.feed()
    2. متنی/JSON → پیام‌های کنترلی دوطرفه (این فایل)

⚠️ چرا یک سوکت به‌جای دو سوکت (صوت/کنترل)؟
    مرورگرها اجازهٔ باز کردن دو WebSocket هم‌زمان با state مشترک را می‌دهند
    ولی هماهنگی ترتیب رویدادها (مثلاً «ضبط شروع شد» با اولین فریم صوتی)
    روی دو کانال جدا مستعد race condition است. یک کانال، ترتیب را تضمین می‌کند.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# پیام‌های ورودی: UI → سرور
# ============================================================

class ClientMessage(BaseModel):
    """پوستهٔ مشترک همهٔ پیام‌های متنی ورودی."""
    type: str


class SetReferenceMsg(ClientMessage):
    """کاربر متن مرجع را برای محاسبهٔ WER تنظیم می‌کند (اختیاری، بند ۱۵)."""
    type: Literal["set_reference"] = "set_reference"
    text: Optional[str] = None


class TriggerWakeMsg(ClientMessage):
    """فعال‌سازی دستی (بدون بیدارباش صوتی) — برای تست یا محیط‌های پرنویز."""
    type: Literal["trigger_wake"] = "trigger_wake"


class StopSessionMsg(ClientMessage):
    """پایان دادن صریح به جلسه از سمت کاربر."""
    type: Literal["stop_session"] = "stop_session"


class SelectModelsMsg(ClientMessage):
    """محدود کردن Benchmark به زیرمجموعه‌ای از مدل‌ها."""
    type: Literal["select_models"] = "select_models"
    model_ids: List[str] = Field(default_factory=list)


class ConfigOverrideMsg(ClientMessage):
    """override موقت تنظیمات VAD/wakeword برای همان جلسه (بدون تغییر app.yaml)."""
    type: Literal["config_override"] = "config_override"
    vad_silence_duration: Optional[float] = None
    wakeword_enabled: Optional[bool] = None


class PingMsg(ClientMessage):
    """keep-alive برای جلوگیری از قطع اتصال توسط پراکسی‌های میانی."""
    type: Literal["ping"] = "ping"


# نگاشت برای پارس دینامیک پیام‌های ورودی
INCOMING_MESSAGE_TYPES = {
    "set_reference": SetReferenceMsg,
    "trigger_wake": TriggerWakeMsg,
    "stop_session": StopSessionMsg,
    "select_models": SelectModelsMsg,
    "config_override": ConfigOverrideMsg,
    "ping": PingMsg,
}


def parse_client_message(raw: Dict[str, Any]) -> ClientMessage:
    """
    پارس امن پیام JSON ورودی.

    Raises:
        ValueError: اگر type ناشناخته یا payload نامعتبر باشد.
    """
    msg_type = raw.get("type")
    cls = INCOMING_MESSAGE_TYPES.get(msg_type)
    if cls is None:
        raise ValueError(f"نوع پیام ناشناخته: {msg_type!r}")
    return cls(**raw)


# ============================================================
# پیام‌های خروجی: سرور → UI  (بند ۱۹ — رویدادهای زندهٔ خط‌به‌خط)
# ============================================================

class ServerEvent(BaseModel):
    """پوستهٔ مشترک همهٔ رویدادهای خروجی."""
    type: str


class SessionStartedEvent(ServerEvent):
    type: Literal["session_started"] = "session_started"
    session_id: str
    wakeword_enabled: bool
    wakeword_phrase: Optional[str] = None


class ListeningEvent(ServerEvent):
    """در انتظار بیدارباش یا در انتظار گفتار (بسته به وضعیت)."""
    type: Literal["listening"] = "listening"
    awaiting_wakeword: bool


class WakewordDetectedEvent(ServerEvent):
    type: Literal["wakeword_detected"] = "wakeword_detected"
    engine: str          # "vosk" | "dtw" | "manual"
    confidence: Optional[float] = None


class RecordingStartedEvent(ServerEvent):
    type: Literal["recording_started"] = "recording_started"
    utterance_id: str


class SilenceDetectedEvent(ServerEvent):
    """VAD تشخیص داد کاربر صحبتش تمام شده — ضبط در حال بسته‌شدن است."""
    type: Literal["silence_detected"] = "silence_detected"
    silence_duration: float


class RecordingStoppedEvent(ServerEvent):
    type: Literal["recording_stopped"] = "recording_stopped"
    utterance_id: str
    duration: float
    audio_sha256: str


class BenchmarkStartedEvent(ServerEvent):
    type: Literal["benchmark_started"] = "benchmark_started"
    benchmark_id: str
    utterance_id: str
    models_total: int
    model_ids: List[str]


class ModelStartedEvent(ServerEvent):
    """یک مدل خاص شروع به پردازش کرد — برای نوار پیشرفت زنده در UI."""
    type: Literal["model_started"] = "model_started"
    benchmark_id: str
    model_id: str
    display_name: str
    index: int
    total: int


class ModelCompletedEvent(ServerEvent):
    """نتیجهٔ یک مدل به محض آماده‌شدن — پیش از تکمیل کل Benchmark (streaming)."""
    type: Literal["model_completed"] = "model_completed"
    benchmark_id: str
    model_id: str
    display_name: str
    success: bool
    error_type: Optional[str] = None
    error: Optional[str] = None
    text: Optional[str] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    rtf: Optional[float] = None
    load_time: Optional[float] = None
    inference_time: Optional[float] = None
    peak_ram_mb: Optional[float] = None


class BenchmarkCompletedEvent(ServerEvent):
    type: Literal["benchmark_completed"] = "benchmark_completed"
    benchmark_id: str
    utterance_id: str
    total_time: float
    models_succeeded: int
    models_failed: int
    ranking: List[str]
    best_model: Optional[str] = None
    fastest_model: Optional[str] = None
    lightest_model: Optional[str] = None
    has_reference: bool
    reference_text: Optional[str] = None
    no_reference_message: Optional[str] = None


class ErrorEvent(ServerEvent):
    """خطای سطح جلسه (نه یک مدل خاص) — مثلاً پیام JSON نامعتبر یا خطای داخلی."""
    type: Literal["error"] = "error"
    error_type: str
    message: str
    recoverable: bool = True


class PongEvent(ServerEvent):
    type: Literal["pong"] = "pong"


class SessionEndedEvent(ServerEvent):
    type: Literal["session_ended"] = "session_ended"
    reason: str   # "client_request" | "disconnect" | "server_shutdown"